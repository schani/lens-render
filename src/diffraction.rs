//! Fraunhofer diffraction at the iris: the point spread function of the
//! aperture shape (Airy pattern for a round stop, star bursts for straight
//! blades) is computed per wavelength and convolved with the film.

use crate::lens::Iris;
use crate::render::Film;
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const M: usize = 4096;
/// Oversampling: the nominal pupil diameter spans M / Q samples (Q = 2 is
/// Nyquist for the intensity).
const Q: f64 = 2.0;
const MAX_HALF: usize = 1024;

/// |FT(pupil)|² on an M×M grid as a summed-area table, so the integral over
/// any rectangle is exact in four lookups. Sample k (−M/2 ≤ k < M/2)
/// corresponds to an image-plane offset of k·λ·N/Q and covers [k − ½, k + ½).
pub struct PupilSpectrum {
    sat: Vec<f64>,
}

/// What the pupil spectrum depends on (it is independent of the f-number).
#[derive(Clone, PartialEq)]
struct ShapeKey {
    blades: u32,
    rotation: u64,
    curvature: u64,
    circular: bool,
    clip: u64,
}

static CACHE: std::sync::Mutex<Option<(ShapeKey, std::sync::Arc<PupilSpectrum>)>> = std::sync::Mutex::new(None);

impl PupilSpectrum {
    /// Cached per iris shape; recomputing takes a few hundred ms.
    pub fn get(iris: &Iris, stop_radius: f64) -> std::sync::Arc<PupilSpectrum> {
        let clip = (stop_radius / iris.radius).min(iris.max_radius / iris.radius * 1.01);
        let key = ShapeKey {
            blades: iris.blades,
            rotation: iris.rotation.to_bits(),
            curvature: iris.curvature.to_bits(),
            circular: iris.circular,
            clip: clip.to_bits(),
        };
        let mut cache = CACHE.lock().unwrap();
        if let Some((k, spec)) = cache.as_ref() {
            if *k == key {
                return spec.clone();
            }
        }
        let spec = std::sync::Arc::new(Self::compute(iris, stop_radius));
        *cache = Some((key, spec.clone()));
        spec
    }

    fn compute(iris: &Iris, stop_radius: f64) -> PupilSpectrum {
        let unit = M as f64 / (2.0 * Q); // samples per nominal radius
        let ss = 4;
        let mut buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); M * M];
        let reach = (iris.max_radius / iris.radius * unit).ceil() as i64 + 2;
        buf.par_chunks_mut(M).enumerate().for_each(|(row, line)| {
            let j = row as i64 - M as i64 / 2;
            if j.abs() > reach {
                return;
            }
            for i in -reach..=reach {
                let mut cover = 0;
                for sy in 0..ss {
                    for sx in 0..ss {
                        let x = (i as f64 + (sx as f64 + 0.5) / ss as f64 - 0.5) / unit * iris.radius;
                        let y = (j as f64 + (sy as f64 + 0.5) / ss as f64 - 0.5) / unit * iris.radius;
                        if iris.contains(x, y) && x * x + y * y <= stop_radius * stop_radius {
                            cover += 1;
                        }
                    }
                }
                line[(i + M as i64 / 2) as usize] = Complex::new(cover as f64 / (ss * ss) as f64, 0.0);
            }
        });
        fft2(&mut buf, M, M, false);
        let peak = buf[0].norm_sqr();
        // Shift DC to the centre, then prefix sums: sat[(j + 1) · (M + 1) + i + 1] = Σ over rows ≤ j, columns ≤ i.
        let n = M + 1;
        let mut sat = vec![0.0; n * n];
        let rows: Vec<Vec<f64>> = (0..M)
            .into_par_iter()
            .map(|j| {
                let sj = (j + M / 2) % M;
                let mut acc = 0.0;
                (0..M)
                    .map(|i| {
                        acc += buf[sj * M + (i + M / 2) % M].norm_sqr() / peak;
                        acc
                    })
                    .collect()
            })
            .collect();
        for j in 0..M {
            for i in 0..M {
                sat[(j + 1) * n + i + 1] = sat[j * n + i + 1] + rows[j][i];
            }
        }
        PupilSpectrum { sat }
    }

    /// Bilinear lookup in the table at continuous (u, v), clamped to the grid.
    fn at(&self, u: f64, v: f64) -> f64 {
        let n = M + 1;
        let u = u.clamp(0.0, M as f64);
        let v = v.clamp(0.0, M as f64);
        let (i, j) = ((u.floor() as usize).min(M - 1), (v.floor() as usize).min(M - 1));
        let (tx, ty) = (u - i as f64, v - j as f64);
        let s = |i: usize, j: usize| self.sat[j * n + i];
        (s(i, j) * (1.0 - tx) + s(i + 1, j) * tx) * (1.0 - ty) + (s(i, j + 1) * (1.0 - tx) + s(i + 1, j + 1) * tx) * ty
    }

    /// Integral over samples in [x0, x1) × [y0, y1) (sample coordinates).
    fn integral(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        let o = M as f64 / 2.0 + 0.5;
        let (u0, u1, v0, v1) = (x0 + o, x1 + o, y0 + o, y1 + o);
        self.at(u1, v1) - self.at(u0, v1) - self.at(u1, v0) + self.at(u0, v0)
    }
}

/// Kernel half size (pixels) for wavelength `l`, working f-number `n_w`, pitch in mm.
pub fn kernel_half(l: f64, n_w: f64, pitch: f64) -> usize {
    let dxi = l * 1e-6 * n_w / Q / pitch;
    ((M as f64 / 2.0 - 1.0) * dxi).floor().clamp(1.0, MAX_HALF as f64) as usize
}

/// Pixel-integrated, normalised PSF kernel of size (2·half+1)² for wavelength
/// `l` (nm), averaged over ±2 % in wavelength.
pub fn kernel(spec: &PupilSpectrum, l: f64, n_w: f64, pitch: f64, half: usize) -> Vec<f64> {
    let size = 2 * half + 1;
    let mut k: Vec<f64> = (0..size * size)
        .into_par_iter()
        .map(|idx| {
            let (dx, dy) = ((idx % size) as f64 - half as f64, (idx / size) as f64 - half as f64);
            let mut acc = 0.0;
            for s in [-1.0, 0.0, 1.0] {
                let dxi = l * (1.0 + s * 0.02) * 1e-6 * n_w / Q / pitch; // PSF sample spacing in pixels
                // Film columns run along −x on the sensor.
                acc += spec.integral(-(dx + 0.5) / dxi, -(dx - 0.5) / dxi, (dy - 0.5) / dxi, (dy + 0.5) / dxi) * dxi * dxi;
            }
            // Circular taper over the outer third so the truncated wings leave no square edge.
            let r = dx.hypot(dy) / half as f64;
            let t = ((r - 2.0 / 3.0) * 3.0).clamp(0.0, 1.0);
            acc * (0.5 + 0.5 * (std::f64::consts::PI * t).cos())
        })
        .collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

fn fft2<T: rustfft::FftNum>(buf: &mut [Complex<T>], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::<T>::new();
    let fw = if inverse { planner.plan_fft_inverse(w) } else { planner.plan_fft_forward(w) };
    let fh = if inverse { planner.plan_fft_inverse(h) } else { planner.plan_fft_forward(h) };
    buf.par_chunks_mut(w).for_each(|row| fw.process(row));
    let mut t = transpose(buf, w, h);
    t.par_chunks_mut(h).for_each(|col| fh.process(col));
    let back = transpose(&t, h, w);
    buf.copy_from_slice(&back);
}

fn transpose<T: Copy + Send + Sync>(src: &[T], w: usize, h: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w * h);
    // Column-major read of `src`, parallel by output row.
    let rows: Vec<Vec<T>> = (0..w).into_par_iter().map(|x| (0..h).map(|y| src[y * w + x]).collect()).collect();
    for r in rows {
        out.extend(r);
    }
    out
}

/// Smallest 2^a 3^b 5^c ≥ n.
fn smooth(n: usize) -> usize {
    let mut m = n;
    loop {
        let mut k = m;
        for p in [2, 3, 5] {
            while k % p == 0 {
                k /= p;
            }
        }
        if k == 1 {
            return m;
        }
        m += 1;
    }
}

/// Convolves every wavelength bin of `film` with the iris' diffraction PSF.
/// `n_w` is the image-space working f-number, `pitch` the pixel pitch in mm;
/// the opening is the iris clipped by the stop's housing of `stop_radius`.
pub fn apply(film: &mut Film, iris: &Iris, stop_radius: f64, n_w: f64, pitch: f64) {
    let spec = PupilSpectrum::get(iris, stop_radius);
    let (w, h, bins) = (film.w, film.h, film.bins);
    let halves: Vec<usize> = (0..bins).map(|b| kernel_half(film.bin_center(b), n_w, pitch)).collect();
    let hmax = *halves.iter().max().unwrap();
    let pw = smooth(w + hmax + 1);
    let ph = smooth(h + hmax + 1);
    let kernels: Vec<Vec<f64>> = (0..bins).map(|b| kernel(&spec, film.bin_center(b), n_w, pitch, halves[b])).collect();
    for b in 0..bins {
        let half = halves[b];
        let k = &kernels[b];
        let size = 2 * half + 1;
        // f64: f32 rounding of very bright lights would spread noise over the frame.
        let mut kb: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); pw * ph];
        for j in 0..size {
            for i in 0..size {
                let x = (i as i64 - half as i64).rem_euclid(pw as i64) as usize;
                let y = (j as i64 - half as i64).rem_euclid(ph as i64) as usize;
                kb[y * pw + x] = Complex::new(k[j * size + i], 0.0);
            }
        }
        let mut img: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); pw * ph];
        for y in 0..h {
            for x in 0..w {
                img[y * pw + x] = Complex::new(film.data[(y * w + x) * bins + b] as f64, 0.0);
            }
        }
        fft2(&mut kb, pw, ph, false);
        fft2(&mut img, pw, ph, false);
        img.par_iter_mut().zip(kb.par_iter()).for_each(|(a, k)| *a *= *k);
        fft2(&mut img, pw, ph, true);
        let s = 1.0 / (pw * ph) as f64;
        for y in 0..h {
            for x in 0..w {
                film.data[(y * w + x) * bins + b] = (img[y * pw + x].re * s).max(0.0) as f32;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel equals a brute-force box integration of the PSF samples.
    #[test]
    fn kernel_is_pixel_box_integral() {
        let mut lens = crate::lens::Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap();
        lens.set_aperture(8.0, 6, 10.0, 0.0).unwrap();
        let iris = lens.iris.clone();
        let spec = PupilSpectrum::compute(&iris, 100.0);
        let (l, n_w, pitch) = (450.0, 8.2, 0.03);
        let half = 20;
        let k = kernel(&spec, l, n_w, pitch, half);
        // Brute force at the central wavelength only, compared with a kernel built the same way.
        let size = 2 * half + 1;
        let dxi = l * 1e-6 * n_w / Q / pitch;
        let mut exact = vec![0.0; size * size];
        let reach = (half as f64 / dxi) as i64 + 2;
        let p = |i: i64, j: i64| {
            let (u, v) = (i + M as i64 / 2, j + M as i64 / 2);
            let n = M + 1;
            let s = &spec.sat;
            let (u, v) = (u as usize, v as usize);
            s[(v + 1) * n + u + 1] - s[v * n + u + 1] - s[(v + 1) * n + u] + s[v * n + u]
        };
        for j in -reach..=reach {
            for i in -reach..=reach {
                let px = (-(i as f64) * dxi + half as f64 + 0.5).floor();
                let py = (j as f64 * dxi + half as f64 + 0.5).floor();
                if px >= 0.0 && py >= 0.0 && px < size as f64 && py < size as f64 {
                    exact[py as usize * size + px as usize] += p(i, j);
                }
            }
        }
        let se: f64 = exact.iter().sum();
        let c = half * size + half;
        let ks: f64 = (0..size * size).filter(|&i| (i % size).abs_diff(half).max((i / size).abs_diff(half)) <= 5).map(|i| k[i]).sum();
        let es: f64 = (0..size * size).filter(|&i| (i % size).abs_diff(half).max((i / size).abs_diff(half)) <= 5).map(|i| exact[i] / se).sum();
        // ±2 % wavelength averaging and the taper make small differences.
        assert!((k[c] - exact[c] / se).abs() < 0.01, "{} vs {}", k[c], exact[c] / se);
        assert!((ks - es).abs() < 0.01, "{ks} vs {es}");
    }
}

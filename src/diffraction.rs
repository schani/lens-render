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

/// |FT(pupil)|² on an M×M grid, DC at index 0. Sample k corresponds to an
/// image-plane offset of k·λ·N/Q.
fn pupil_spectrum(iris: &Iris) -> Vec<f64> {
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
                    if iris.contains(x, y) {
                        cover += 1;
                    }
                }
            }
            line[(i + M as i64 / 2) as usize] = Complex::new(cover as f64 / (ss * ss) as f64, 0.0);
        }
    });
    fft2(&mut buf, M, M, false);
    buf.par_iter().map(|c| c.norm_sqr()).collect()
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

/// Pixel-integrated, normalised PSF kernel of size (2·half+1)² for wavelength `l` (nm).
fn kernel(spec: &[f64], l: f64, n_w: f64, pitch: f64, half: usize) -> Vec<f64> {
    let size = 2 * half + 1;
    let mut k = vec![0.0; size * size];
    let peak = spec[0];
    let mut add = |lambda: f64, weight: f64| {
        let dxi = lambda * 1e-6 * n_w / Q / pitch; // PSF sample spacing in pixels
        for j in 0..M {
            let kj = if j < M / 2 { j as f64 } else { j as f64 - M as f64 };
            let py = kj * dxi + half as f64;
            if py < 0.0 || py >= (size - 1) as f64 {
                continue;
            }
            for i in 0..M {
                let v = spec[j * M + i];
                if v < peak * 1e-14 {
                    continue;
                }
                let ki = if i < M / 2 { i as f64 } else { i as f64 - M as f64 };
                let px = ki * dxi + half as f64;
                if px < 0.0 || px >= (size - 1) as f64 {
                    continue;
                }
                // Kernel pixel centres sit at integer coordinates: bilinear splat.
                let x0 = px.floor() as usize;
                let y0 = py.floor() as usize;
                let tx = px - x0 as f64;
                let ty = py - y0 as f64;
                let w = v * weight;
                k[y0 * size + x0] += w * (1.0 - tx) * (1.0 - ty);
                k[y0 * size + x0 + 1] += w * tx * (1.0 - ty);
                k[(y0 + 1) * size + x0] += w * (1.0 - tx) * ty;
                k[(y0 + 1) * size + x0 + 1] += w * tx * ty;
            }
        }
    };
    for s in [-1.0, 0.0, 1.0] {
        add(l * (1.0 + s * 0.02), 1.0);
    }
    // Circular taper over the outer third so the truncated wings leave no square edge.
    for j in 0..size {
        for i in 0..size {
            let r = (i as f64 - half as f64).hypot(j as f64 - half as f64) / half as f64;
            let t = ((r - 2.0 / 3.0) * 3.0).clamp(0.0, 1.0);
            k[j * size + i] *= 0.5 + 0.5 * (std::f64::consts::PI * t).cos();
        }
    }
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

/// Convolves every wavelength bin of `film` with the iris' diffraction PSF.
/// `n_w` is the image-space working f-number, `pitch` the pixel pitch in mm.
pub fn apply(film: &mut Film, iris: &Iris, n_w: f64, pitch: f64) {
    let spec = pupil_spectrum(iris);
    let (w, h, bins) = (film.w, film.h, film.bins);
    let halves: Vec<usize> = (0..bins)
        .map(|b| {
            let dxi = film.bin_center(b) * 1e-6 * n_w / Q / pitch;
            ((M as f64 / 2.0 - 1.0) * dxi).floor().clamp(1.0, MAX_HALF as f64) as usize
        })
        .collect();
    let hmax = *halves.iter().max().unwrap();
    let pw = smooth(w + hmax + 1);
    let ph = smooth(h + hmax + 1);
    let kernels: Vec<Vec<f64>> = (0..bins).into_par_iter().map(|b| kernel(&spec, film.bin_center(b), n_w, pitch, halves[b])).collect();
    for b in 0..bins {
        let half = halves[b];
        let k = &kernels[b];
        let size = 2 * half + 1;
        let mut kb: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); pw * ph];
        for j in 0..size {
            for i in 0..size {
                let x = (i as i64 - half as i64).rem_euclid(pw as i64) as usize;
                let y = (j as i64 - half as i64).rem_euclid(ph as i64) as usize;
                kb[y * pw + x] = Complex::new(k[j * size + i] as f32, 0.0);
            }
        }
        let mut img: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); pw * ph];
        for y in 0..h {
            for x in 0..w {
                img[y * pw + x] = Complex::new(film.data[(y * w + x) * bins + b], 0.0);
            }
        }
        fft2(&mut kb, pw, ph, false);
        fft2(&mut img, pw, ph, false);
        img.par_iter_mut().zip(kb.par_iter()).for_each(|(a, k)| *a *= *k);
        fft2(&mut img, pw, ph, true);
        let s = 1.0 / (pw * ph) as f32;
        for y in 0..h {
            for x in 0..w {
                film.data[(y * w + x) * bins + b] = img[y * pw + x].re * s;
            }
        }
    }
}

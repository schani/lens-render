//! Wave-optical point spread functions and MTF, computed the way optical
//! design programs do it ("FFT PSF"):
//!
//! 1. Aim the chief ray from the source through the centre of the stop.
//! 2. Trace a uniform grid of rays over the part of the entrance pupil that
//!    passes the lens, accumulating optical path lengths.
//! 3. Continue each ray to the reference sphere centred on the chief ray's
//!    sensor point and passing through the exit pupil; the optical path
//!    difference (OPD) to the chief ray is the wavefront error.
//! 4. The pupil function √T · exp(2πi·OPD/λ) is Fourier transformed. Image
//!    coordinates follow from the rays' exit direction cosines, so tilted
//!    (off-axis) image planes are handled.
//!
//! Diffraction and all aberrations are therefore treated together. The
//! direction cosines are fitted affinely across the pupil (the usual FFT-PSF
//! approximation; pupil distortion beyond that is neglected).

use crate::glass::LAMBDA_D;
use crate::lens::{Lens, Media};
use crate::math::{Ray, Vec3};
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

#[derive(Clone, Copy, Debug)]
pub enum Source {
    /// Point source in lens coordinates (mm, in front of the lens).
    Point(Vec3),
    /// Collimated light travelling along the unit vector `dir` (dir.z > 0).
    Infinity(Vec3),
}

impl Source {
    /// Object point (or direction) whose chief ray has field angle `theta`
    /// (radians, in the y–z plane) seen from the entrance pupil, at `distance`
    /// mm in front of the front vertex (None = infinity).
    pub fn at_field(lens: &Lens, theta: f64, distance: Option<f64>) -> Source {
        match distance {
            None => Source::Infinity(Vec3::new(0.0, theta.sin(), theta.cos())),
            Some(d) => {
                let z_ep = lens.paraxial_data(LAMBDA_D).entrance_pupil_z;
                Source::Point(Vec3::new(0.0, -(z_ep + d) * theta.tan(), -d))
            }
        }
    }

    /// Ray from the source through `q`, and the optical path from the
    /// source's wavefront (point, or plane through the origin) to its origin.
    fn ray_through(&self, q: Vec3) -> (Ray, f64) {
        match *self {
            Source::Point(p) => (Ray { o: p, d: (q - p).normalize() }, 0.0),
            Source::Infinity(d) => {
                let o = q - d * 1000.0;
                (Ray { o, d }, o.dot(d))
            }
        }
    }
}

/// Sampled wavefront in the exit pupil for one source point and wavelength.
pub struct Wavefront {
    pub lambda: f64,
    pub n: usize,
    /// Grid spacing on the entrance-pupil plane (mm).
    pub du: f64,
    /// Row-major n × n: (amplitude, OPD in mm), None where blocked.
    pub samples: Vec<Option<(f64, f64)>>,
    /// Exit direction cosines ≈ c0 + a · (u, v), (u, v) from the grid centre.
    pub c0: [f64; 2],
    pub a: [[f64; 2]; 2],
    /// Chief ray's point on the sensor plane (sensor coordinates, mm).
    pub reference: [f64; 2],
    /// Σ transmittance · du² (relative illumination × pupil area).
    pub power: f64,
    /// Geometric ray hits on the sensor, relative to `reference`.
    pub spot: Vec<[f64; 2]>,
}

/// Intersection of the sensor-side chief ray with the stop plane, ignoring apertures.
fn stop_hit(lens: &Lens, ray: Ray, l: f64, m: &Media) -> Option<[f64; 2]> {
    let s = lens.stop_index;
    let r = if s == 0 { ray } else { lens.trace_path_opl(ray, l, m, &lens.forward_path[..s], false)?.0 };
    let stop = &lens.surfaces[s];
    let t = (stop.z - r.o.z) / r.d.z;
    Some([r.o.x + r.d.x * t - stop.decenter.0, r.o.y + r.d.y * t - stop.decenter.1])
}

/// Point on the entrance-pupil plane the chief ray aims at (Newton iteration).
fn aim_chief(lens: &Lens, src: &Source, l: f64, m: &Media, z_ep: f64) -> Option<Vec3> {
    let mut q = Vec3::new(0.0, 0.0, z_ep);
    let h = 1e-5;
    for _ in 0..30 {
        let f = stop_hit(lens, src.ray_through(q).0, l, m)?;
        if f[0].hypot(f[1]) < 1e-11 {
            return Some(q);
        }
        let fx = stop_hit(lens, src.ray_through(q + Vec3::new(h, 0.0, 0.0)).0, l, m)?;
        let fy = stop_hit(lens, src.ray_through(q + Vec3::new(0.0, h, 0.0)).0, l, m)?;
        let j = [[(fx[0] - f[0]) / h, (fy[0] - f[0]) / h], [(fx[1] - f[1]) / h, (fy[1] - f[1]) / h]];
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() < 1e-18 {
            return None;
        }
        q.x -= (j[1][1] * f[0] - j[0][1] * f[1]) / det;
        q.y -= (-j[1][0] * f[0] + j[0][0] * f[1]) / det;
    }
    None
}

/// Where the chief ray from `src` meets the sensor plane.
pub fn chief_image(lens: &Lens, src: &Source, l: f64) -> Option<[f64; 2]> {
    let m = lens.media(l);
    let q = aim_chief(lens, src, l, &m, lens.paraxial_data(LAMBDA_D).entrance_pupil_z)?;
    let (out, _, _) = lens.trace_path_opl(src.ray_through(q).0, l, &m, &lens.forward_path, false)?;
    let p = out.o + out.d * ((lens.sensor_z - out.o.z) / out.d.z);
    Some([p.x, p.y])
}

/// Field angle whose chief ray lands at image height `h` (mm, along +y).
pub fn field_for_height(lens: &Lens, h: f64, distance: Option<f64>) -> Option<f64> {
    if h == 0.0 {
        return Some(0.0);
    }
    let f = |th: f64| chief_image(lens, &Source::at_field(lens, th, distance), LAMBDA_D).map(|p| p[1] - h);
    let mut a = (h / lens.focal_length()).atan();
    let mut b = a * 1.02;
    let (mut fa, mut fb) = (f(a)?, f(b)?);
    for _ in 0..40 {
        if fb.abs() < 1e-9 || fb == fa {
            break;
        }
        let c = b - fb * (b - a) / (fb - fa);
        (a, fa) = (b, fb);
        b = c;
        fb = f(b)?;
    }
    Some(b)
}

/// One traced pupil ray: amplitude², OPL to the reference sphere, exit direction, sensor hit.
struct PupilRay {
    t: f64,
    opl: f64,
    dir: Vec3,
    hit: [f64; 2],
}

struct Reference {
    centre: Vec3,
    pupil: Vec3,
    radius: f64,
}

fn trace_to_sphere(lens: &Lens, src: &Source, q: Vec3, l: f64, m: &Media, rf: &Reference, clip: bool) -> Option<PupilRay> {
    let (ray, opl0) = src.ray_through(q);
    let (out, t, opl) = lens.trace_path_opl(ray, l, m, &lens.forward_path, clip)?;
    if out.d.z <= 0.0 {
        return None;
    }
    let n_img = m.n[lens.surfaces.len()];
    let oc = out.o - rf.centre;
    let b = oc.dot(out.d);
    let disc = b * b - (oc.dot(oc) - rf.radius * rf.radius);
    if disc < 0.0 {
        return None;
    }
    // The root on the exit-pupil side of the sphere.
    let s1 = -b - disc.sqrt();
    let s2 = -b + disc.sqrt();
    let near = |s: f64| (out.o + out.d * s - rf.pupil).length();
    let s = if near(s1) < near(s2) { s1 } else { s2 };
    let k = (lens.sensor_z - out.o.z) / out.d.z;
    let hit = out.o + out.d * k;
    Some(PupilRay { t, opl: opl0 + opl + n_img * s, dir: out.d, hit: [hit.x - rf.centre.x, hit.y - rf.centre.y] })
}

impl Wavefront {
    /// Wavefront for `src` at wavelength `l` (nm) on an n × n pupil grid; `n`
    /// None chooses it from the geometric spot size (Nyquist for the
    /// wavefront slope, between 64 and `max_n`).
    pub fn compute(lens: &Lens, src: &Source, l: f64, n: Option<usize>, max_n: usize) -> Option<Wavefront> {
        let m = lens.media(l);
        let par = lens.paraxial_data(LAMBDA_D);
        let q_c = aim_chief(lens, src, l, &m, par.entrance_pupil_z)?;
        // Chief ray: reference point on the sensor and exit-pupil point.
        let (ray, _) = src.ray_through(q_c);
        let (out, _, _) = lens.trace_path_opl(ray, l, &m, &lens.forward_path, false)?;
        let centre = out.o + out.d * ((lens.sensor_z - out.o.z) / out.d.z);
        let pupil = out.o + out.d * ((par.exit_pupil_z - out.o.z) / out.d.z);
        let rf = Reference { centre, pupil, radius: (centre - pupil).length() };
        let chief = trace_to_sphere(lens, src, q_c, l, &m, &rf, false)?;

        // Part of the entrance-pupil plane whose rays pass.
        let half = 2.5 * lens.iris.max_radius / par.entrance_pupil_mag + 1.0;
        let g = 96;
        let cell = 2.0 * half / g as f64;
        let mut bb = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut extent: f64 = 0.0;
        for j in 0..g {
            for i in 0..g {
                let x = q_c.x - half + (i as f64 + 0.5) * cell;
                let y = q_c.y - half + (j as f64 + 0.5) * cell;
                if let Some(r) = trace_to_sphere(lens, src, Vec3::new(x, y, q_c.z), l, &m, &rf, true) {
                    bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
                    extent = extent.max(r.hit[0].hypot(r.hit[1]));
                }
            }
        }
        if bb[0] > bb[2] {
            return None;
        }
        let side = (bb[2] - bb[0]).max(bb[3] - bb[1]) + 2.0 * cell;
        let (cx, cy) = (0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3]));
        let sample_grid = |n: usize| -> Vec<Option<PupilRay>> {
            let du = side / n as f64;
            (0..n * n)
                .into_par_iter()
                .map(|k| {
                    let q = Vec3::new(cx - side / 2.0 + ((k % n) as f64 + 0.5) * du, cy - side / 2.0 + ((k / n) as f64 + 0.5) * du, q_c.z);
                    trace_to_sphere(lens, src, q, l, &m, &rf, true)
                })
                .collect()
        };
        let n = match n {
            Some(n) => n,
            None => {
                // Pupil width in direction cosines, from a coarse grid.
                let coarse = sample_grid(16);
                let dirs: Vec<Vec3> = coarse.iter().flatten().map(|r| r.dir).collect();
                let span = |f: fn(&Vec3) -> f64| dirs.iter().map(f).fold(f64::NEG_INFINITY, f64::max) - dirs.iter().map(f).fold(f64::INFINITY, f64::min);
                let width = span(|d| d.x).max(span(|d| d.y)).max(1e-6) * 16.0 / 15.0;
                // The PSF window n·λ/width must hold the spot with margin.
                let need = 3.0 * (extent + 2.0 * l * 1e-6 / width) * width / (l * 1e-6);
                (need.ceil() as usize).next_power_of_two().clamp(64, max_n)
            }
        };
        let rays = sample_grid(n);
        let du = side / n as f64;
        // Least-squares affine fit of exit direction cosines over the pupil.
        let (mut s, mut su, mut sv, mut suu, mut suv, mut svv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut sl, mut sm, mut slu, mut slv, mut smu, mut smv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (k, r) in rays.iter().enumerate() {
            if let Some(r) = r {
                let u = ((k % n) as f64 + 0.5) * du - side / 2.0;
                let v = ((k / n) as f64 + 0.5) * du - side / 2.0;
                s += 1.0;
                su += u;
                sv += v;
                suu += u * u;
                suv += u * v;
                svv += v * v;
                sl += r.dir.x;
                sm += r.dir.y;
                slu += r.dir.x * u;
                slv += r.dir.x * v;
                smu += r.dir.y * u;
                smv += r.dir.y * v;
            }
        }
        // Solve the 3×3 normal equations for (c0, a_u, a_v) for L and for M.
        let solve = |b0: f64, bu: f64, bv: f64| -> [f64; 3] {
            let mat = [[s, su, sv], [su, suu, suv], [sv, suv, svv]];
            let det = |m: [[f64; 3]; 3]| {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            };
            let d = det(mat);
            let rhs = [b0, bu, bv];
            let mut x = [0.0; 3];
            for c in 0..3 {
                let mut mc = mat;
                for r in 0..3 {
                    mc[r][c] = rhs[r];
                }
                x[c] = det(mc) / d;
            }
            x
        };
        let fl = solve(sl, slu, slv);
        let fm = solve(sm, smu, smv);
        let samples = rays.iter().map(|r| r.as_ref().map(|r| (r.t.sqrt(), r.opl - chief.opl))).collect();
        let power = rays.iter().flatten().map(|r| r.t).sum::<f64>() * du * du;
        let spot = rays.iter().flatten().map(|r| r.hit).collect();
        Some(Wavefront {
            lambda: l,
            n,
            du,
            samples,
            c0: [fl[0], fm[0]],
            a: [[fl[1], fl[2]], [fm[1], fm[2]]],
            reference: [centre.x, centre.y],
            power,
            spot,
        })
    }

    /// RMS wavefront error in waves (piston removed), over the open pupil.
    pub fn rms_waves(&self) -> f64 {
        let w: Vec<f64> = self.samples.iter().flatten().map(|(_, o)| o / (self.lambda * 1e-6)).collect();
        let mean = w.iter().sum::<f64>() / w.len() as f64;
        (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / w.len() as f64).sqrt()
    }
}

/// A PSF as energy samples on the sensor plane (sensor coordinates relative
/// to the wavefront's reference point), normalised to unit total energy.
pub struct Psf {
    pub lambda: f64,
    pub reference: [f64; 2],
    pub power: f64,
    /// (dx, dy, energy).
    pub samples: Vec<[f64; 3]>,
    /// Spacing of the underlying grid along each grid axis (mm).
    pub spacing: f64,
}

impl Psf {
    /// Fraunhofer diffraction of the wavefront: zero-padded FFT of the pupil function.
    pub fn from_wavefront(wf: &Wavefront, pad: usize) -> Psf {
        let n = wf.n;
        let nn = n * pad;
        let k = 2.0 * std::f64::consts::PI / (wf.lambda * 1e-6);
        let mut buf = vec![Complex::new(0.0, 0.0); nn * nn];
        for (idx, s) in wf.samples.iter().enumerate() {
            if let Some((a, opd)) = s {
                buf[(idx / n) * nn + idx % n] = Complex::from_polar(*a, k * opd);
            }
        }
        fft2(&mut buf, nn);
        let total: f64 = buf.iter().map(|c| c.norm_sqr()).sum();
        // Δ = −λ/(N du) · A^{-T} (kx, ky)
        let a = wf.a;
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let inv_t = [[a[1][1] / det, -a[1][0] / det], [-a[0][1] / det, a[0][0] / det]];
        let scale = -(wf.lambda * 1e-6) / (nn as f64 * wf.du);
        let peak = buf.iter().map(|c| c.norm_sqr()).fold(0.0, f64::max);
        let samples = buf
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let e = c.norm_sqr();
                if e < peak * 1e-12 {
                    return None;
                }
                let kx = signed(i % nn, nn);
                let ky = signed(i / nn, nn);
                let dx = scale * (inv_t[0][0] * kx + inv_t[0][1] * ky);
                let dy = scale * (inv_t[1][0] * kx + inv_t[1][1] * ky);
                Some([dx, dy, e / total])
            })
            .collect();
        let spacing = (wf.lambda * 1e-6) / (nn as f64 * wf.du) / det.abs().sqrt();
        Psf { lambda: wf.lambda, reference: wf.reference, power: wf.power, samples, spacing }
    }

    /// Integrates the PSF over sensor pixels: a `size` × `size` grid whose
    /// pixel (0, 0) covers [origin, origin + pitch) in sensor coordinates.
    pub fn pixels(&self, origin: [f64; 2], pitch: f64, size: usize) -> Vec<f64> {
        let mut out = vec![0.0; size * size];
        for s in &self.samples {
            let x = ((self.reference[0] + s[0] - origin[0]) / pitch).floor();
            let y = ((self.reference[1] + s[1] - origin[1]) / pitch).floor();
            if x >= 0.0 && y >= 0.0 && (x as usize) < size && (y as usize) < size {
                out[y as usize * size + x as usize] += s[2];
            }
        }
        out
    }

    /// Energy-weighted centroid relative to the reference point.
    pub fn centroid(&self) -> [f64; 2] {
        let e: f64 = self.samples.iter().map(|s| s[2]).sum();
        [self.samples.iter().map(|s| s[0] * s[2]).sum::<f64>() / e, self.samples.iter().map(|s| s[1] * s[2]).sum::<f64>() / e]
    }
}

fn signed(i: usize, n: usize) -> f64 {
    if i < n / 2 { i as f64 } else { i as f64 - n as f64 }
}

fn fft2(buf: &mut [Complex<f64>], n: usize) {
    let mut planner = FftPlanner::new();
    let f = planner.plan_fft_forward(n);
    buf.par_chunks_mut(n).for_each(|row| f.process(row));
    let mut t = vec![Complex::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..n {
            t[i * n + j] = buf[j * n + i];
        }
    }
    t.par_chunks_mut(n).for_each(|col| f.process(col));
    for j in 0..n {
        for i in 0..n {
            buf[j * n + i] = t[i * n + j];
        }
    }
}

/// Polychromatic PSF: per-wavelength PSFs combined with spectral weights,
/// placed relative to a common reference point (lateral colour included).
pub struct PolyPsf {
    pub reference: [f64; 2],
    pub samples: Vec<[f64; 3]>,
}

impl PolyPsf {
    /// White light (CIE ȳ weighting of an equal-energy spectrum) over `wavelengths`.
    pub fn compute(lens: &Lens, src: &Source, wavelengths: &[f64], max_n: usize) -> Option<PolyPsf> {
        let psfs: Vec<Psf> = wavelengths
            .iter()
            .filter_map(|&l| Wavefront::compute(lens, src, l, None, max_n).map(|wf| Psf::from_wavefront(&wf, 2)))
            .collect();
        if psfs.is_empty() {
            return None;
        }
        let reference = psfs[psfs.len() / 2].reference;
        let weights: Vec<f64> = psfs.iter().map(|p| crate::spectrum::cie_xyz(p.lambda)[1] * p.power).collect();
        let total: f64 = weights.iter().sum();
        let mut samples = Vec::new();
        for (p, w) in psfs.iter().zip(&weights) {
            let off = [p.reference[0] - reference[0], p.reference[1] - reference[1]];
            samples.extend(p.samples.iter().map(|s| [s[0] + off[0], s[1] + off[1], s[2] * w / total]));
        }
        Some(PolyPsf { reference, samples })
    }

    /// Modulation transfer at spatial frequencies `freqs` (cycles/mm) along
    /// the unit direction `dir` on the sensor.
    pub fn mtf(&self, freqs: &[f64], dir: [f64; 2]) -> Vec<f64> {
        let total: f64 = self.samples.iter().map(|s| s[2]).sum();
        freqs
            .par_iter()
            .map(|&f| {
                let (mut re, mut im) = (0.0, 0.0);
                for s in &self.samples {
                    let ph = -2.0 * std::f64::consts::PI * f * (s[0] * dir[0] + s[1] * dir[1]);
                    re += s[2] * ph.cos();
                    im += s[2] * ph.sin();
                }
                re.hypot(im) / total
            })
            .collect()
    }
}

/// Diffraction-limited MTF of a circular aperture at working f-number `n_w`.
pub fn diffraction_limit_mtf(f: f64, lambda_nm: f64, n_w: f64) -> f64 {
    let x = f * lambda_nm * 1e-6 * n_w;
    if x >= 1.0 {
        return 0.0;
    }
    2.0 / std::f64::consts::PI * (x.acos() - x * (1.0 - x * x).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triplet(fstop: f64) -> Lens {
        let mut l = Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap();
        l.set_aperture(fstop, 0, 0.0, 0.0).unwrap();
        l.sensor_z = l.best_focus(None).unwrap();
        l
    }

    /// Stopped down on axis the lens is diffraction limited: the MTF follows
    /// the Airy formula and the wavefront error is small.
    #[test]
    fn diffraction_limited_on_axis() {
        let lens = triplet(22.0);
        let src = Source::Infinity(Vec3::new(0.0, 0.0, 1.0));
        let wf = Wavefront::compute(&lens, &src, 550.0, Some(256), 256).unwrap();
        assert!(wf.rms_waves() < 0.07, "rms {}", wf.rms_waves());
        let psf = Psf::from_wavefront(&wf, 4);
        let poly = PolyPsf { reference: psf.reference, samples: psf.samples.clone() };
        let n_w = lens.working_fnumber();
        for f in [10.0, 30.0, 60.0] {
            let m = poly.mtf(&[f], [1.0, 0.0])[0];
            let ideal = diffraction_limit_mtf(f, 550.0, n_w);
            assert!(m <= ideal + 0.01 && m > ideal - 0.06, "{f} lp/mm: {m} vs ideal {ideal}");
        }
    }

    /// Strongly aberrated: the wave PSF converges to the geometric spot
    /// (same centroid and similar RMS radius), which also checks signs and
    /// orientation of the image-plane mapping.
    #[test]
    fn large_aberrations_match_geometry() {
        let lens = triplet(3.5);
        let src = Source::at_field(&lens, 18f64.to_radians(), None);
        let wf = Wavefront::compute(&lens, &src, 550.0, None, 1024).unwrap();
        let psf = Psf::from_wavefront(&wf, 2);
        let gc = {
            let n = wf.spot.len() as f64;
            [wf.spot.iter().map(|p| p[0]).sum::<f64>() / n, wf.spot.iter().map(|p| p[1]).sum::<f64>() / n]
        };
        let wc = psf.centroid();
        let rms = |pts: Vec<[f64; 3]>, c: [f64; 2]| {
            let e: f64 = pts.iter().map(|p| p[2]).sum();
            (pts.iter().map(|p| p[2] * ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2))).sum::<f64>() / e).sqrt()
        };
        let g_rms = rms(wf.spot.iter().map(|p| [p[0], p[1], 1.0]).collect(), gc);
        let w_rms = rms(psf.samples.clone(), wc);
        println!("n {}: geometric centroid {gc:?} rms {g_rms}, wave centroid {wc:?} rms {w_rms}", wf.n);
        assert!((gc[1] - wc[1]).abs() < 0.15 * g_rms, "centroid");
        assert!(w_rms > 0.85 * g_rms && w_rms < 1.25 * g_rms, "rms");
    }
}

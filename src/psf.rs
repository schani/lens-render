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

    /// Like `at_field`, with the chief ray's image at azimuth `alpha` (radians,
    /// measured in the sensor plane from +x; `at_field` is alpha = π/2).
    pub fn at(lens: &Lens, theta: f64, alpha: f64, distance: Option<f64>) -> Source {
        let (s, c) = alpha.sin_cos();
        match distance {
            None => Source::Infinity(Vec3::new(theta.sin() * c, theta.sin() * s, theta.cos())),
            Some(d) => {
                let z_ep = lens.paraxial_data(LAMBDA_D).entrance_pupil_z;
                let r = (z_ep + d) * theta.tan();
                Source::Point(Vec3::new(-r * c, -r * s, -d))
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
    /// Exit direction cosines ≈ const + a · (u, v), (u, v) from the grid centre.
    pub a: [[f64; 2]; 2],
    /// Chief ray's point on the sensor plane (sensor coordinates, mm).
    pub reference: [f64; 2],
    /// Σ transmittance · du² (relative illumination × pupil area).
    pub power: f64,
    /// Geometric ray hits on the sensor, relative to `reference`.
    #[allow(dead_code)]
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

/// `count` rays from `src` in its meridional plane, spread evenly across
/// `spread` times the entrance pupil around the chief ray (aimed at the
/// d line). None if no chief ray reaches the stop.
pub fn meridional_fan(lens: &Lens, src: &Source, count: usize, spread: f64) -> Option<Vec<Ray>> {
    let m = lens.media(LAMBDA_D);
    let par = lens.paraxial_data(LAMBDA_D);
    let q = aim_chief(lens, src, LAMBDA_D, &m, par.entrance_pupil_z)?;
    let r = spread * lens.iris.max_radius / par.entrance_pupil_mag;
    // The meridional plane holds the axis and the source.
    let dir = match *src {
        Source::Point(p) => Vec3::new(p.x, p.y, 0.0),
        Source::Infinity(d) => Vec3::new(d.x, d.y, 0.0),
    };
    let dir = if dir.length() > 0.0 { dir.normalize() } else { Vec3::new(0.0, 1.0, 0.0) };
    Some((0..count).map(|k| src.ray_through(q + dir * (r * (2.0 * k as f64 / (count - 1).max(1) as f64 - 1.0))).0).collect())
}

/// Rays from `src` through a g × g grid over a disk of `spread` times the
/// entrance pupil radius around the chief ray (aimed at the d line).
pub fn pupil_rays(lens: &Lens, src: &Source, g: usize, spread: f64) -> Option<Vec<Ray>> {
    let m = lens.media(LAMBDA_D);
    let par = lens.paraxial_data(LAMBDA_D);
    let q = aim_chief(lens, src, LAMBDA_D, &m, par.entrance_pupil_z)?;
    let r = spread * lens.iris.max_radius / par.entrance_pupil_mag;
    let cell = 2.0 * r / g as f64;
    Some(
        (0..g * g)
            .map(|k| (-r + ((k % g) as f64 + 0.5) * cell, -r + ((k / g) as f64 + 0.5) * cell))
            .filter(|(x, y)| x * x + y * y <= r * r)
            .map(|(x, y)| src.ray_through(q + Vec3::new(x, y, 0.0)).0)
            .collect(),
    )
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

/// Chief-ray aim point on the entrance-pupil plane, and the reference
/// sphere: centred on the chief ray's sensor point, through the exit pupil.
fn chief(lens: &Lens, src: &Source, l: f64, m: &Media) -> Option<(Vec3, Reference)> {
    let par = lens.paraxial_data(LAMBDA_D);
    let q_c = aim_chief(lens, src, l, m, par.entrance_pupil_z)?;
    let (ray, _) = src.ray_through(q_c);
    let (out, _, _) = lens.trace_path_opl(ray, l, m, &lens.forward_path, false)?;
    let centre = out.o + out.d * ((lens.sensor_z - out.o.z) / out.d.z);
    let pupil = out.o + out.d * ((par.exit_pupil_z - out.o.z) / out.d.z);
    Some((q_c, Reference { centre, pupil, radius: (centre - pupil).length() }))
}

/// Square on the entrance-pupil plane holding the rays that pass, with a
/// margin, and the geometric spot radius. It barely depends on wavelength,
/// so one frame serves all wavelengths of a source.
#[derive(Clone, Copy)]
pub struct PupilFrame {
    cx: f64,
    cy: f64,
    side: f64,
    extent: f64,
}

impl PupilFrame {
    pub fn find(lens: &Lens, src: &Source, l: f64) -> Option<PupilFrame> {
        let m = lens.media(l);
        let par = lens.paraxial_data(LAMBDA_D);
        let (q_c, rf) = chief(lens, src, l, &m)?;
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
        Some(PupilFrame { cx: 0.5 * (bb[0] + bb[2]), cy: 0.5 * (bb[1] + bb[3]), side, extent })
    }
}

/// Chief ray, reference sphere, pupil frame and grid size of a wavefront.
struct Setup {
    m: Media,
    q_c: Vec3,
    rf: Reference,
    chief: PupilRay,
    cx: f64,
    cy: f64,
    side: f64,
    n: usize,
}

impl Setup {
    fn new(lens: &Lens, src: &Source, l: f64, n: Option<usize>, max_n: usize, frame: Option<&PupilFrame>) -> Option<Setup> {
        let m = lens.media(l);
        let (q_c, rf) = chief(lens, src, l, &m)?;
        let chief = trace_to_sphere(lens, src, q_c, l, &m, &rf, false)?;
        let own;
        let frame = match frame {
            Some(f) => f,
            None => {
                own = PupilFrame::find(lens, src, l)?;
                &own
            }
        };
        let PupilFrame { cx, cy, side, extent } = *frame;
        let n = match n {
            Some(n) => n,
            None => {
                // Pupil width in direction cosines, from a coarse grid.
                let du = side / 16.0;
                let coarse: Vec<PupilRay> = (0..256)
                    .filter_map(|k| {
                        let q = Vec3::new(cx - side / 2.0 + ((k % 16) as f64 + 0.5) * du, cy - side / 2.0 + ((k / 16) as f64 + 0.5) * du, q_c.z);
                        trace_to_sphere(lens, src, q, l, &m, &rf, true)
                    })
                    .collect();
                let extent = coarse.iter().map(|r| r.hit[0].hypot(r.hit[1])).fold(extent, f64::max);
                let span = |f: fn(&Vec3) -> f64| coarse.iter().map(|r| f(&r.dir)).fold(f64::NEG_INFINITY, f64::max) - coarse.iter().map(|r| f(&r.dir)).fold(f64::INFINITY, f64::min);
                let width = span(|d| d.x).max(span(|d| d.y)).max(1e-6) * 16.0 / 15.0;
                // The PSF window n·λ/width must hold the spot with margin.
                let need = 3.0 * (extent + 2.0 * l * 1e-6 / width) * width / (l * 1e-6);
                (need.ceil() as usize).next_power_of_two().clamp(64, max_n)
            }
        };
        Some(Setup { m, q_c, rf, chief, cx, cy, side, n })
    }
}

/// Linear part of the least-squares affine fit of exit direction cosines
/// (L, M) over pupil points (u, v).
fn affine_fit(rays: impl Iterator<Item = (f64, f64, Vec3)>) -> [[f64; 2]; 2] {
    let (mut s, mut su, mut sv, mut suu, mut suv, mut svv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sl, mut sm, mut slu, mut slv, mut smu, mut smv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (u, v, dir) in rays {
        s += 1.0;
        su += u;
        sv += v;
        suu += u * u;
        suv += u * v;
        svv += v * v;
        sl += dir.x;
        sm += dir.y;
        slu += dir.x * u;
        slv += dir.x * v;
        smu += dir.y * u;
        smv += dir.y * v;
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
    [[fl[1], fl[2]], [fm[1], fm[2]]]
}

/// Sensor offset (mm) per frequency index of an `nn`-point transform of
/// pupil samples `du` apart: Δ = −λ/(nn du) · A^{-T} (kx, ky).
fn fft_map(a: [[f64; 2]; 2], lambda: f64, nn: usize, du: f64) -> [[f64; 2]; 2] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let scale = -(lambda * 1e-6) / (nn as f64 * du);
    [[scale * a[1][1] / det, -scale * a[1][0] / det], [-scale * a[0][1] / det, scale * a[0][0] / det]]
}

/// What a GPU needs to sample a wavefront: the pupil grid and the OPD at
/// every `step`-th sample. The OPD nodes are traced without apertures
/// (extrapolated where even that fails), so the wavefront continues smoothly
/// past the pupil edge and can be interpolated anywhere; which samples pass,
/// and the affine fit of exit directions, are left to the GPU.
pub struct WavefrontPlan {
    pub lambda: f64,
    pub n: usize,
    pub step: usize,
    /// Sample (i, j) lies at (x0 + (i + ½) du, y0 + (j + ½) du, z).
    pub x0: f64,
    pub y0: f64,
    pub z: f64,
    pub du: f64,
    pub source: Source,
    /// (n/step + 3)² OPDs (mm), row-major; node (I, J) is sample step · (I − 1, J − 1).
    pub nodes: Vec<f32>,
    /// Chief ray's point on the sensor plane.
    pub reference: [f64; 2],
}

impl WavefrontPlan {
    pub fn new(lens: &Lens, src: &Source, l: f64, max_n: usize, frame: &PupilFrame) -> Option<WavefrontPlan> {
        let Setup { m, q_c, rf, chief, cx, cy, side, n } = Setup::new(lens, src, l, None, max_n, Some(frame))?;
        let du = side / n as f64;
        let (x0, y0) = (cx - side / 2.0, cy - side / 2.0);
        let step = (n / 32).clamp(1, 16);
        let nc = n / step + 3;
        let traced: Vec<Option<f64>> = (0..nc * nc)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (((k % nc) as f64 - 1.0) * step as f64, ((k / nc) as f64 - 1.0) * step as f64);
                let q = Vec3::new(x0 + (i + 0.5) * du, y0 + (j + 0.5) * du, q_c.z);
                trace_to_sphere(lens, src, q, l, &m, &rf, false).map(|r| r.opl - chief.opl)
            })
            .collect();
        // Nodes even unaperture-traced rays miss: extrapolate linearly from
        // their neighbours, a ring at a time.
        let mut opd = traced;
        for _ in 0..4 {
            let prev = opd.clone();
            let at = |x: isize, y: isize| if x < 0 || y < 0 || x >= nc as isize || y >= nc as isize { None } else { prev[y as usize * nc + x as usize] };
            for k in 0..nc * nc {
                if prev[k].is_some() {
                    continue;
                }
                let (x, y) = ((k % nc) as isize, (k / nc) as isize);
                let est: Vec<f64> = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .filter_map(|&(dx, dy)| {
                        let v1 = at(x + dx, y + dy)?;
                        Some(at(x + 2 * dx, y + 2 * dy).map_or(v1, |v2| 2.0 * v1 - v2))
                    })
                    .collect();
                if !est.is_empty() {
                    opd[k] = Some(est.iter().sum::<f64>() / est.len() as f64);
                }
            }
        }
        Some(WavefrontPlan {
            lambda: l,
            n,
            step,
            x0,
            y0,
            z: q_c.z,
            du,
            source: *src,
            nodes: opd.iter().map(|v| v.unwrap_or(0.0) as f32).collect(),
            reference: [rf.centre.x, rf.centre.y],
        })
    }
}

impl Wavefront {
    /// Wavefront for `src` at wavelength `l` (nm) on an n × n pupil grid; `n`
    /// None chooses it from the geometric spot size (Nyquist for the
    /// wavefront slope, between 64 and `max_n`).
    pub fn compute(lens: &Lens, src: &Source, l: f64, n: Option<usize>, max_n: usize) -> Option<Wavefront> {
        Self::compute_sampled(lens, src, l, n, max_n, None, true)
    }

    /// `compute` within a pupil frame found at another wavelength.
    pub fn compute_in(lens: &Lens, src: &Source, l: f64, max_n: usize, frame: &PupilFrame) -> Option<Wavefront> {
        Self::compute_sampled(lens, src, l, None, max_n, Some(frame), true)
    }

    /// `compute`, optionally tracing every pupil sample instead of interpolating.
    fn compute_sampled(lens: &Lens, src: &Source, l: f64, n: Option<usize>, max_n: usize, frame: Option<&PupilFrame>, interpolate: bool) -> Option<Wavefront> {
        let Setup { m, q_c, rf, chief, cx, cy, side, n } = Setup::new(lens, src, l, n, max_n, frame)?;
        let du = side / n as f64;
        let trace = |i: f64, j: f64| {
            let q = Vec3::new(cx - side / 2.0 + (i + 0.5) * du, cy - side / 2.0 + (j + 0.5) * du, q_c.z);
            trace_to_sphere(lens, src, q, l, &m, &rf, true)
        };
        // Rays are traced every `step` samples (plus a border node for the
        // stencil). The wavefront is smooth, so samples whose whole 4 × 4
        // stencil passes are interpolated bicubically; only samples near the
        // pupil edge are traced. The open pupil is convex and large compared
        // with a stencil, so a sample inside a passing stencil passes and one
        // inside a blocked stencil is blocked.
        let step = if interpolate { (n / 64).clamp(1, 8) } else { 1 };
        let nc = n / step + 3;
        let coarse: Vec<Option<PupilRay>> = (0..nc * nc)
            .into_par_iter()
            .map(|k| trace(((k % nc) as f64 - 1.0) * step as f64, ((k / nc) as f64 - 1.0) * step as f64))
            .collect();
        let node = |ci: usize, cj: usize| coarse[(cj + 1) * nc + ci + 1].as_ref();
        // Per sample: amplitude, OPD and exit direction cosines (for the fit).
        let sample = |r: &PupilRay| [r.t.sqrt(), r.opl - chief.opl, r.dir.x, r.dir.y];
        let full: Vec<Option<[f64; 4]>> = (0..n * n)
            .into_par_iter()
            .map(|k| {
                let (i, j) = (k % n, k / n);
                let (ci, cj) = (i / step, j / step);
                if i % step == 0 && j % step == 0 {
                    return node(ci, cj).map(sample);
                }
                let mut st = [[[0.0; 4]; 4]; 4];
                let mut open = 0;
                for dj in 0..4 {
                    for di in 0..4 {
                        if let Some(r) = coarse[(cj + dj) * nc + ci + di].as_ref() {
                            for (c, v) in sample(r).into_iter().enumerate() {
                                st[c][dj][di] = v;
                            }
                            open += 1;
                        }
                    }
                }
                match open {
                    0 => return None,
                    16 => {}
                    _ => return trace(i as f64, j as f64).as_ref().map(sample),
                }
                let (fx, fy) = ((i % step) as f64 / step as f64, (j % step) as f64 / step as f64);
                Some(st.map(|c| bicubic(&c, fx, fy)))
            })
            .collect();
        // Least-squares affine fit of exit direction cosines over the pupil.
        let fit = full.iter().enumerate().filter_map(|(k, s)| {
            s.map(|s| (((k % n) as f64 + 0.5) * du - side / 2.0, ((k / n) as f64 + 0.5) * du - side / 2.0, Vec3::new(s[2], s[3], 0.0)))
        });
        let a = affine_fit(fit);
        let samples: Vec<Option<(f64, f64)>> = full.iter().map(|s| s.map(|s| (s[0], s[1]))).collect();
        let spot = coarse.iter().flatten().map(|r| r.hit).collect();
        let power = samples.iter().flatten().map(|(a, _)| a * a).sum::<f64>() * du * du;
        Some(Wavefront {
            lambda: l,
            n,
            du,
            samples,
            a,
            reference: [rf.centre.x, rf.centre.y],
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
}

impl Psf {
    /// Fraunhofer diffraction of the wavefront: zero-padded FFT of the pupil function.
    pub fn from_wavefront(wf: &Wavefront, pad: usize) -> Psf {
        let g = PsfGrid::from_wavefront(wf, pad);
        let peak = g.e.iter().fold(0f32, |a, &b| a.max(b));
        let mut samples = Vec::new();
        g.for_each(|dx, dy, e| {
            if e >= peak * 1e-12 {
                samples.push([dx, dy, e as f64]);
            }
        });
        Psf { lambda: wf.lambda, reference: wf.reference, power: wf.power, samples }
    }

    /// Integrates the PSF over sensor pixels: a `size` × `size` grid whose
    /// pixel (0, 0) covers [origin, origin + pitch) in sensor coordinates.
    #[cfg(test)]
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
    #[cfg(test)]
    pub fn centroid(&self) -> [f64; 2] {
        let e: f64 = self.samples.iter().map(|s| s[2]).sum();
        [self.samples.iter().map(|s| s[0] * s[2]).sum::<f64>() / e, self.samples.iter().map(|s| s[1] * s[2]).sum::<f64>() / e]
    }
}

/// A PSF as the intensity grid of the zero-padded FFT: sample (i, j) lies at
/// `map · (kx, ky)` from the reference point, (kx, ky) the signed frequency
/// indices.
pub struct PsfGrid {
    pub reference: [f64; 2],
    nn: usize,
    map: [[f64; 2]; 2],
    /// Row-major nn × nn, normalised to unit sum.
    e: Vec<f32>,
}

impl PsfGrid {
    pub fn from_wavefront(wf: &Wavefront, pad: usize) -> PsfGrid {
        let n = wf.n;
        let nn = n * pad;
        let k = 2.0 * std::f64::consts::PI / (wf.lambda * 1e-6);
        let mut buf = vec![Complex::new(0f32, 0.0); nn * nn];
        for (idx, s) in wf.samples.iter().enumerate() {
            if let Some((a, opd)) = s {
                let (sin, cos) = (k * opd).sin_cos();
                buf[(idx / n) * nn + idx % n] = Complex::new((a * cos) as f32, (a * sin) as f32);
            }
        }
        fft2(&mut buf, nn, n);
        let e: Vec<f32> = buf.iter().map(|c| c.norm_sqr()).collect();
        let total: f64 = e.iter().map(|&v| v as f64).sum();
        let e = e.into_iter().map(|v| (v as f64 / total) as f32).collect();
        let map = fft_map(wf.a, wf.lambda, nn, wf.du);
        PsfGrid { reference: wf.reference, nn, map, e }
    }

    /// Calls `f(dx, dy, energy)` for every sample.
    fn for_each(&self, mut f: impl FnMut(f64, f64, f32)) {
        let nn = self.nn;
        for (j, row) in self.e.chunks(nn).enumerate() {
            let ky = signed(j, nn);
            for (i, &e) in row.iter().enumerate() {
                let kx = signed(i, nn);
                f(self.map[0][0] * kx + self.map[0][1] * ky, self.map[1][0] * kx + self.map[1][1] * ky, e);
            }
        }
    }

    /// Radii (mm) containing the given fractions of the energy (radial histogram).
    pub fn energy_radii<const N: usize>(&self, fractions: [f64; N]) -> [f64; N] {
        let h = self.nn as f64 / 2.0;
        let r_max = [[h, h], [h, -h]].iter().map(|[x, y]| (self.map[0][0] * x + self.map[0][1] * y).hypot(self.map[1][0] * x + self.map[1][1] * y)).fold(0.0, f64::max);
        let bins = 4096;
        let mut hist = vec![0.0; bins];
        let to_bin = bins as f64 / r_max;
        self.for_each(|dx, dy, e| hist[(((dx * dx + dy * dy).sqrt() * to_bin) as usize).min(bins - 1)] += e as f64);
        let total: f64 = hist.iter().sum();
        fractions.map(|fraction| {
            let mut acc = 0.0;
            for (k, e) in hist.iter().enumerate() {
                acc += e;
                if acc >= fraction * total {
                    return (k + 1) as f64 / to_bin;
                }
            }
            r_max
        })
    }

    /// Integrates over sensor pixels: a `size` × `size` grid whose pixel
    /// (0, 0) covers [origin, origin + pitch) in sensor coordinates.
    pub fn pixels(&self, origin: [f64; 2], pitch: f64, size: usize) -> Vec<f64> {
        let mut out = vec![0.0; size * size];
        let (ox, oy) = ((self.reference[0] - origin[0]) / pitch, (self.reference[1] - origin[1]) / pitch);
        self.for_each(|dx, dy, e| {
            let (x, y) = ((ox + dx / pitch).floor(), (oy + dy / pitch).floor());
            if x >= 0.0 && y >= 0.0 && (x as usize) < size && (y as usize) < size {
                out[y as usize * size + x as usize] += e as f64;
            }
        });
        out
    }
}

/// Catmull-Rom bicubic interpolation inside the centre cell of a 4 × 4 stencil.
fn bicubic(p: &[[f64; 4]; 4], fx: f64, fy: f64) -> f64 {
    let cr = |a: f64, b: f64, c: f64, d: f64, t: f64| b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)));
    let r = p.map(|row| cr(row[0], row[1], row[2], row[3], fx));
    cr(r[0], r[1], r[2], r[3], fy)
}

fn signed(i: usize, n: usize) -> f64 {
    if i < n / 2 { i as f64 } else { i as f64 - n as f64 }
}

/// In-place 2-D FFT of an n × n grid whose rows from `rows` on are zero.
fn fft2(buf: &mut [Complex<f32>], n: usize, rows: usize) {
    thread_local!(static PLANNER: std::cell::RefCell<FftPlanner<f32>> = std::cell::RefCell::new(FftPlanner::new()));
    let f = PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n));
    buf[..rows * n].par_chunks_mut(n).for_each(|row| f.process(row));
    // In 16-row blocks: each block of destination rows reads 16-wide strips of the source.
    let b = 16.min(n);
    let transpose = |src: &[Complex<f32>], dst: &mut [Complex<f32>]| {
        dst.par_chunks_mut(b * n).enumerate().for_each(|(bi, block)| {
            for j in 0..n {
                let row = &src[j * n + bi * b..j * n + bi * b + b];
                for (k, &c) in row.iter().enumerate() {
                    block[k * n + j] = c;
                }
            }
        })
    };
    let mut t = vec![Complex::new(0.0, 0.0); n * n];
    transpose(buf, &mut t);
    t.par_chunks_mut(n).for_each(|col| f.process(col));
    transpose(&t, buf);
}

/// Polychromatic PSF: per-wavelength PSFs combined with spectral weights,
/// placed relative to a common reference point (lateral colour included).
pub struct PolyPsf {
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
        Some(PolyPsf { samples })
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
        let poly = PolyPsf { samples: psf.samples.clone() };
        let n_w = lens.working_fnumber();
        for f in [10.0, 30.0, 60.0] {
            let m = poly.mtf(&[f], [1.0, 0.0])[0];
            let ideal = diffraction_limit_mtf(f, 550.0, n_w);
            assert!(m <= ideal + 0.01 && m > ideal - 0.06, "{f} lp/mm: {m} vs ideal {ideal}");
        }
    }

    /// Interpolating the wavefront between traced rays gives the same PSF as
    /// tracing every pupil sample: in focus, defocused, vignetted, aspheric,
    /// polygonal iris.
    #[test]
    fn interpolated_wavefront_matches_traced() {
        let mut lux = Lens::parse(include_str!("../lenses/summilux-50-asph.lens")).unwrap();
        lux.set_aperture(1.4, 0, 0.0, 0.0).unwrap();
        lux.sensor_z = lux.best_focus(None).unwrap();
        let mut poly = triplet(8.0);
        poly.set_aperture(8.0, 6, 0.0, 0.0).unwrap();
        let cases = [
            (triplet(3.5), 0.0, None, 550.0),
            (triplet(3.5), 14.0, Some(1500.0), 450.0),
            (poly, 10.0, Some(700.0), 650.0),
            (lux.clone(), 0.0, Some(2000.0), 550.0),
            (lux, 20.0, None, 430.0),
        ];
        for (lens, deg, dist, l) in cases {
            let src = Source::at_field(&lens, f64::to_radians(deg), dist);
            let a = Wavefront::compute_sampled(&lens, &src, l, None, 1024, None, true).unwrap();
            let b = Wavefront::compute_sampled(&lens, &src, l, None, 1024, None, false).unwrap();
            let (pa, pb) = (Psf::from_wavefront(&a, 2), Psf::from_wavefront(&b, 2));
            let pitch = 0.003;
            let size = 201;
            let origin = [pb.reference[0] - pitch * size as f64 / 2.0, pb.reference[1] - pitch * size as f64 / 2.0];
            let (ka, kb) = (pa.pixels(origin, pitch, size), pb.pixels(origin, pitch, size));
            let l1: f64 = ka.iter().zip(&kb).map(|(x, y)| (x - y).abs()).sum::<f64>() / kb.iter().sum::<f64>();
            println!("{deg}° {dist:?} {l} nm, n {}: L1 {l1:.5}, power {:.5}", a.n, a.power / b.power);
            assert!(l1 < 0.01 && (a.power / b.power - 1.0).abs() < 0.002, "L1 {l1}");
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

//! Lens optimisation by damped least squares (Levenberg–Marquardt) with
//! finite-difference Jacobians.
//!
//! Merit: the polychromatic RMS spot radius from real rays at a few image
//! heights, for an object at the focus distance, on the sensor plane the
//! renderer focuses to (least-squares blur of the axial point). The clear
//! apertures are variables too (soft-edged while optimising), so designs may
//! vignette towards the corner as far as the job allows. Constraints (glass
//! and air thicknesses, back focus, vignetting) are penalty terms; the focal
//! length is held by scaling all curvatures. Random restarts around the best
//! design so far help out of local minima. A job
//! file (TOML) records the start design, variables, constraints and merit, so
//! a design can be regenerated with `lens-render optimize <job>`.

use crate::glass::LAMBDA_D;
use crate::lens::{Iris, Lens};
use crate::math::{Ray, Vec3};
use crate::psf::{Source, chief_aim};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;

/// Half-diagonal of the 36 × 24 frame (mm).
pub const HALF_DIAG: f64 = 21.63;
/// Wavelengths (nm) and weights of the polychromatic spot: F, e, C.
pub const SPOT_WAVELENGTHS: [(f64, f64); 3] = [(486.13, 1.0), (546.07, 2.0), (656.27, 1.0)];
/// Penalty per mm of constraint violation, in µm of spot radius.
const PENALTY: f64 = 1000.0;
/// Penalty factor on missing light (fraction of the iris' light, × PENALTY).
const LIGHT: f64 = 10.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    /// `@name` of the output lens.
    pub name: String,
    /// Comment written at the top of the output lens file.
    #[serde(default)]
    pub header: String,
    /// Start prescription (lens-file rows).
    pub start: String,
    /// Output lens file, relative to the job file.
    pub output: String,
    pub fstop: f64,
    /// Object distance in m.
    #[serde(default = "d_focus")]
    pub focus: f64,
    #[serde(default = "d_efl")]
    pub efl: f64,
    /// Image heights as fractions of the half-diagonal, and their weights.
    #[serde(default = "d_fields")]
    pub fields: Vec<f64>,
    #[serde(default)]
    pub field_weights: Vec<f64>,
    /// One wavelength (nm) instead of F, e, C.
    #[serde(default)]
    pub mono: Option<f64>,
    /// Rows (1-based, as in the lens file) whose curvature / thickness vary.
    #[serde(default)]
    pub curvatures: Vec<usize>,
    #[serde(default)]
    pub thicknesses: Vec<usize>,
    /// Mirror the rows in front of the stop to the rows behind it (only the
    /// front rows are listed as variables).
    #[serde(default)]
    pub symmetric: bool,
    /// Allowed mechanical vignetting: at image height h (fraction of the
    /// half-diagonal) the apertures pass at least 1 − vignetting · h² of the
    /// light the iris passes.
    #[serde(default)]
    pub vignetting: f64,
    #[serde(default = "d_centre")]
    pub min_glass_centre: f64,
    #[serde(default = "d_edge")]
    pub min_glass_edge: f64,
    #[serde(default = "d_max_glass")]
    pub max_glass: f64,
    #[serde(default = "d_air")]
    pub min_air: f64,
    #[serde(default = "d_bfd")]
    pub min_back_focus: f64,
    #[serde(default = "d_iterations")]
    pub iterations: usize,
    /// Optional weights (µm per wave) on the Seidel sums W040 … W111 at full
    /// field, e.g. to target coma when the field points cannot isolate it.
    #[serde(default)]
    pub seidel_weights: Vec<f64>,
    /// Random restarts after the first descent: variables perturbed by up
    /// to ± `restart_scale` (relative) around the best design so far.
    #[serde(default)]
    pub restarts: usize,
    #[serde(default = "d_restart_scale")]
    pub restart_scale: f64,
    /// Extra limits on single thicknesses: [row, min, max].
    #[serde(default)]
    pub thickness_limits: Vec<[f64; 3]>,
}

fn d_focus() -> f64 {
    2.5
}
fn d_efl() -> f64 {
    50.0
}
fn d_fields() -> Vec<f64> {
    vec![0.0, 0.5, 0.7, 1.0]
}
fn d_centre() -> f64 {
    1.5
}
fn d_edge() -> f64 {
    1.0
}
fn d_restart_scale() -> f64 {
    0.15
}
fn d_air() -> f64 {
    0.1
}
fn d_max_glass() -> f64 {
    20.0
}
fn d_bfd() -> f64 {
    15.0
}
fn d_iterations() -> usize {
    200
}

#[derive(Clone, Copy, Debug)]
enum Var {
    Curvature(usize),
    Thickness(usize),
    SemiDiameter(usize),
}

struct Field {
    frac: f64,
    theta: f64,
    /// Weight in the merit (0: only the vignetting limit applies).
    weight: f64,
}

/// Field angle (from the entrance pupil) imaged near `frac` × the
/// half-diagonal, for an object `dist` mm away and focal length `efl`.
fn field_angle(frac: f64, dist: f64, efl: f64) -> f64 {
    (frac * HALF_DIAG * (dist - efl) / (efl * dist)).atan()
}

/// Rays from `src` through a g × g grid over the entrance pupil, `spread`
/// times its paraxial radius, around the chief ray.
fn pupil_grid(lens: &Lens, src: &Source, g: usize, spread: f64) -> Vec<Option<Ray>> {
    let par = lens.paraxial_data(LAMBDA_D);
    let r = spread * lens.iris.max_radius / par.entrance_pupil_mag;
    let q = chief_aim(lens, src, LAMBDA_D);
    let cell = 2.0 / g as f64;
    (0..g * g)
        .map(|k| (-1.0 + ((k % g) as f64 + 0.5) * cell, -1.0 + ((k / g) as f64 + 0.5) * cell))
        .filter(|(x, y)| x * x + y * y <= 1.0)
        .map(|(x, y)| q.map(|q| src.ray_through(q + Vec3::new(x * r, y * r, 0.0)).0))
        .collect()
}

/// A traced bundle: exit rays (None if lost) with their weights.
struct Bundle {
    rays: Vec<(Option<Ray>, f64)>,
}

/// Traces a pupil grid at each wavelength. Without `clip`, apertures do not
/// block and `heights` gets the largest radius reached at each surface.
fn trace_bundle(lens: &Lens, src: &Source, wls: &[(f64, f64)], g: usize, spread: f64, clip: bool, heights: &mut [f64]) -> Bundle {
    let grid = pupil_grid(lens, src, g, spread);
    let mut rays = Vec::with_capacity(grid.len() * wls.len());
    for &(l, w) in wls {
        let m = lens.media(l);
        for ray in &grid {
            let out = ray.and_then(|ray| {
                if clip {
                    return lens.trace_forward(ray, l, &m).map(|r| r.0);
                }
                let mut cur = ray;
                for (i, ev) in lens.forward_path.iter().enumerate() {
                    // The stop does not refract; a ray already past its plane is fine.
                    if lens.surfaces[i].is_stop {
                        continue;
                    }
                    cur = lens.trace_path_opl(cur, l, &m, std::slice::from_ref(ev), false)?.0;
                    heights[i] = heights[i].max(cur.o.x.hypot(cur.o.y));
                }
                Some(cur)
            });
            rays.push((out, w));
        }
    }
    Bundle { rays }
}

/// Smooth step from 0 (d ≤ −e) to 1 (d ≥ e).
fn smooth_step(d: f64, e: f64) -> f64 {
    let t = ((d + e) / (2.0 * e)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Traces a pupil grid with soft apertures (edges blurred over a few % of
/// their radius, so the merit stays smooth). Per ray: the exit ray, its
/// weight through all apertures, and through the iris alone (wavelength
/// weights included).
fn trace_soft(lens: &Lens, src: &Source, wls: &[(f64, f64)], g: usize, spread: f64) -> Vec<(Option<Ray>, f64, f64)> {
    let grid = pupil_grid(lens, src, g, spread);
    let k = lens.stop_index;
    let r_iris = lens.iris.max_radius;
    let mut out = Vec::with_capacity(grid.len() * wls.len());
    for &(l, wl) in wls {
        let m = lens.media(l);
        for ray in &grid {
            let (mut w_iris, mut w_all) = (0.0, 0.0);
            let exit = ray.and_then(|ray| {
                let (mut cur, mut w) = (ray, 1.0);
                for (i, ev) in lens.forward_path.iter().enumerate() {
                    let s = &lens.surfaces[i];
                    if i == k {
                        // The stop does not refract; where the ray line crosses its plane.
                        let t = (s.z - cur.o.z) / cur.d.z;
                        w_iris = smooth_step(r_iris - (cur.o.x + cur.d.x * t).hypot(cur.o.y + cur.d.y * t), 0.01 * r_iris);
                        continue;
                    }
                    cur = lens.trace_path_opl(cur, l, &m, std::slice::from_ref(ev), false)?.0;
                    w *= smooth_step(s.semi_diameter - cur.o.x.hypot(cur.o.y), 0.02 * s.semi_diameter + 0.05);
                }
                w_all = w * w_iris;
                Some(cur)
            });
            out.push((exit, wl * w_all, wl * w_iris));
        }
    }
    out
}

/// Plane minimising the weighted mean squared distance of the rays from the axis.
fn axial_focus(b: &Bundle) -> Option<f64> {
    let (mut num, mut den) = (0.0, 0.0);
    for (r, w) in &b.rays {
        let Some(r) = r else { continue };
        let (a, c) = (r.d.x / r.d.z, r.d.y / r.d.z);
        num += w * ((r.o.z * a - r.o.x) * a + (r.o.z * c - r.o.y) * c);
        den += w * (a * a + c * c);
    }
    (den > 0.0).then(|| num / den)
}

/// Sensor plane the renderer focuses to (`Lens::best_focus`: luminance-weighted
/// least-squares blur of the axial point, 420–660 nm), from a coarser grid.
fn render_focus(lens: &Lens, dist: f64) -> Option<f64> {
    let wls: Vec<(f64, f64)> = (0..13).map(|i| 420.0 + 20.0 * i as f64).map(|l| (l, crate::spectrum::cie_xyz(l)[1])).collect();
    let mut heights = vec![0.0; lens.surfaces.len()];
    axial_focus(&trace_bundle(lens, &Source::at_field(lens, 0.0, Some(dist)), &wls, 16, 1.0, false, &mut heights))
}

/// Weighted spot on the plane `z`: offsets (mm) from the centroid of the rays
/// that get through, with normalised weights (lost rays: None).
fn spot(b: &Bundle, z: f64) -> Vec<(Option<(f64, f64)>, f64)> {
    let pts: Vec<(Option<(f64, f64)>, f64)> = b
        .rays
        .iter()
        .map(|(r, w)| {
            let p = r.map(|r| {
                let t = (z - r.o.z) / r.d.z;
                (r.o.x + r.d.x * t, r.o.y + r.d.y * t)
            });
            (p, *w)
        })
        .collect();
    let (mut cx, mut cy, mut sw) = (0.0, 0.0, 0.0);
    for (p, w) in &pts {
        if let Some((x, y)) = p {
            cx += x * w;
            cy += y * w;
            sw += w;
        }
    }
    if sw > 0.0 {
        (cx, cy) = (cx / sw, cy / sw);
    }
    let total: f64 = pts.iter().map(|(_, w)| w).sum();
    pts.into_iter().map(|(p, w)| (p.map(|(x, y)| (x - cx, y - cy)), w / total)).collect()
}

/// RMS spot radius in µm of the object point imaged at `h` mm, with
/// apertures, at wavelengths `wls` (nm, weight) on the plane `z` (None: the
/// plane of least RMS for this point).
pub fn rms_spot_um(lens: &Lens, h: f64, dist: Option<f64>, wls: &[(f64, f64)], z: Option<f64>) -> Option<f64> {
    let theta = crate::psf::field_for_height(lens, h, dist)?;
    let b = trace_bundle(lens, &Source::at_field(lens, theta, dist), wls, 64, 1.6, true, &mut []);
    let b = Bundle { rays: b.rays.into_iter().filter(|r| r.0.is_some()).collect() };
    if b.rays.is_empty() {
        return None;
    }
    let rms = |z: f64| spot(&b, z).iter().map(|(p, w)| p.map_or(0.0, |(x, y)| w * (x * x + y * y))).sum::<f64>().sqrt() * 1000.0;
    Some(match z {
        Some(z) => rms(z),
        None => {
            // Golden-section search around the sensor.
            let (mut a, mut c) = (lens.sensor_z - 3.0, lens.sensor_z + 3.0);
            for _ in 0..60 {
                let (m1, m2) = (c - 0.618 * (c - a), a + 0.618 * (c - a));
                if rms(m1) < rms(m2) { c = m2 } else { a = m1 }
            }
            rms(0.5 * (a + c))
        }
    })
}

pub struct Problem {
    job: Job,
    base: Lens,
    vars: Vec<Var>,
    dist: f64,
    fields: Vec<Field>,
    wls: Vec<(f64, f64)>,
}

/// One evaluated design.
pub struct Eval {
    pub lens: Lens,
    pub residuals: Vec<f64>,
    /// RMS spot radius (µm) per merit field.
    pub rms: Vec<f64>,
    /// Constraints that are violated.
    pub violations: Vec<String>,
}

impl Problem {
    pub fn new(job: Job) -> Result<Problem> {
        let base = Lens::parse(&job.start).context("start design")?;
        let n = base.surfaces.len();
        let k = base.stop_index;
        let mut vars = Vec::new();
        for &r in &job.curvatures {
            if r == 0 || r > n || base.surfaces[r - 1].is_stop {
                bail!("curvature row {r} is not a lens surface");
            }
            vars.push(Var::Curvature(r - 1));
        }
        for &r in &job.thicknesses {
            if r == 0 || r >= n {
                bail!("thickness row {r} does not exist (the last row's is the image distance)");
            }
            vars.push(Var::Thickness(r - 1));
        }
        if let Some(l) = job.thickness_limits.iter().find(|l| l[0] < 1.0 || l[0] as usize > n) {
            bail!("thickness limit on row {} that does not exist", l[0]);
        }
        if job.symmetric {
            if n != 2 * k + 1 {
                bail!("a symmetric design needs as many rows behind the stop as in front");
            }
            if vars.iter().any(|v| matches!(v, Var::Curvature(i) | Var::Thickness(i) if *i >= k)) {
                bail!("symmetric: list only rows in front of the stop");
            }
        }
        // Every lens surface's clear aperture varies.
        vars.extend((0..n).filter(|&i| !base.surfaces[i].is_stop).map(Var::SemiDiameter));
        let fw = |i: usize| job.field_weights.get(i).copied().unwrap_or(1.0);
        let dist = job.focus * 1000.0;
        let mut fields: Vec<Field> = job.fields.iter().enumerate().map(|(i, &frac)| Field { frac, theta: field_angle(frac, dist, job.efl), weight: fw(i) }).collect();
        for frac in [0.0, 0.5, 0.7, 1.0] {
            if !job.fields.contains(&frac) {
                fields.push(Field { frac, theta: field_angle(frac, dist, job.efl), weight: 0.0 });
            }
        }
        let wls = match job.mono {
            Some(l) => vec![(l, 1.0)],
            None => SPOT_WAVELENGTHS.to_vec(),
        };
        Ok(Problem { job, base, vars, dist, fields, wls })
    }

    fn start(&self) -> Vec<f64> {
        self.read(&self.base)
    }

    fn read(&self, lens: &Lens) -> Vec<f64> {
        self.vars
            .iter()
            .map(|v| match *v {
                Var::Curvature(i) => lens.surfaces[i].curvature(),
                Var::Thickness(i) => lens.surfaces[i].thickness,
                Var::SemiDiameter(i) => lens.surfaces[i].semi_diameter,
            })
            .collect()
    }

    /// The design for variables `x`: mirrored, scaled to the focal length,
    /// iris at the design f-number. None if it has no positive focal length.
    pub fn lens(&self, x: &[f64]) -> Option<Lens> {
        let mut lens = self.base.clone();
        for (v, &x) in self.vars.iter().zip(x) {
            match *v {
                Var::Curvature(i) => lens.surfaces[i].radius = if x == 0.0 { 0.0 } else { 1.0 / x },
                Var::Thickness(i) => lens.surfaces[i].thickness = x,
                Var::SemiDiameter(i) => lens.surfaces[i].semi_diameter = x.abs(),
            }
        }
        let k = lens.stop_index;
        if self.job.symmetric {
            for i in 0..k {
                lens.surfaces[2 * k - i].radius = -lens.surfaces[i].radius;
                lens.surfaces[2 * k - 1 - i].thickness = lens.surfaces[i].thickness;
            }
        }
        for _ in 0..30 {
            let mut z = 0.0;
            for s in lens.surfaces.iter_mut() {
                s.z = z;
                z += s.thickness;
            }
            let efl = lens.focal_length();
            if !(efl > 0.0 && efl < 1e4) {
                return None;
            }
            let s = self.job.efl / efl;
            if (s - 1.0).abs() < 1e-13 {
                break;
            }
            lens.surfaces.iter_mut().for_each(|srf| srf.radius *= s);
        }
        let r = lens.stop_radius_for_fstop(self.job.fstop);
        lens.surfaces[k].semi_diameter = r;
        lens.iris = Iris::new(0, 0.0, 0.0, r, true);
        lens.name = self.job.name.clone();
        Some(lens)
    }

    pub fn eval(&self, x: &[f64]) -> Option<Eval> {
        let lens = self.lens(x)?;
        let n = lens.surfaces.len();
        let z = render_focus(&lens, self.dist)?;
        let mut res = Vec::new();
        let mut rms = Vec::new();
        let mut light = Vec::new();
        for f in &self.fields {
            let rays = trace_soft(&lens, &Source::at_field(&lens, f.theta, Some(self.dist)), &self.wls, 20, 1.3);
            let (w_all, w_iris) = rays.iter().fold((0.0, 0.0), |a, r| (a.0 + r.1, a.1 + r.2));
            light.push((f.frac, if w_iris > 0.0 { w_all / w_iris } else { 0.0 }));
            if f.weight == 0.0 {
                continue;
            }
            let pts: Vec<(f64, f64, f64)> = rays
                .iter()
                .map(|(r, w, _)| match r {
                    Some(r) if *w > 0.0 => {
                        let t = (z - r.o.z) / r.d.z;
                        (r.o.x + r.d.x * t, r.o.y + r.d.y * t, w / w_all)
                    }
                    _ => (0.0, 0.0, 0.0),
                })
                .collect();
            let cx: f64 = pts.iter().map(|p| p.0 * p.2).sum();
            let cy: f64 = pts.iter().map(|p| p.1 * p.2).sum();
            let mut sum = 0.0;
            for (x, y, w) in pts {
                let s = (f.weight * w).sqrt() * 1000.0;
                res.push(s * (x - cx));
                res.push(s * (y - cy));
                sum += w * ((x - cx).powi(2) + (y - cy).powi(2));
            }
            rms.push(sum.sqrt() * 1000.0);
        }
        let s = &lens.surfaces;
        let sd = |i: usize| s[i].semi_diameter;
        let mut violations = Vec::new();
        let mut pen = |what: String, v: f64| {
            res.push(PENALTY * v.max(0.0));
            if v > 1e-3 {
                violations.push(format!("{what} by {v:.3} mm"));
            }
        };
        for (frac, pass) in light {
            pen(format!("vignetting at {frac}"), LIGHT * (0.99 - self.job.vignetting * frac * frac - pass));
        }
        // Sag at radius r, or how far r lies outside the sphere.
        let sag = |i: usize, r: f64| -> Result<f64, f64> {
            let c = s[i].curvature();
            if c * c * r * r >= 1.0 { Err(r - 1.0 / c.abs()) } else { Ok(s[i].sag(r)) }
        };
        for i in 0..n - 1 {
            let glass = s[i].material.is_some();
            // Edge: glass at the element's rim, air where both surfaces exist.
            let r = if glass { sd(i).max(sd(i + 1)) } else { sd(i).min(sd(i + 1)) };
            let edge = match (sag(i, r), sag(i + 1, r)) {
                (Ok(a), Ok(b)) => s[i].thickness + b - a,
                (Err(e), _) | (_, Err(e)) => -1.0 - e,
            };
            if glass {
                pen(format!("row {} glass centre", i + 1), self.job.min_glass_centre - s[i].thickness);
                pen(format!("row {} glass edge", i + 1), self.job.min_glass_edge - edge);
                pen(format!("row {} glass max", i + 1), s[i].thickness - self.job.max_glass);
            } else {
                pen(format!("row {} air centre", i + 1), self.job.min_air - s[i].thickness);
                pen(format!("row {} air edge", i + 1), self.job.min_air - edge);
            }
        }
        for &[row, lo, hi] in &self.job.thickness_limits {
            let t = s[row as usize - 1].thickness;
            pen(format!("row {row} min"), lo - t);
            pen(format!("row {row} max"), t - hi);
        }
        let last = n - 1;
        pen("back focus".into(), self.job.min_back_focus - (z - s[last].z));
        match sag(last, sd(last)) {
            Ok(a) => pen("rear rim to sensor".into(), 1.0 - (z - s[last].z - a)),
            Err(e) => pen("rear rim to sensor".into(), 1.0 + e),
        }
        if !self.job.seidel_weights.is_empty() {
            let w = crate::seidel::seidel(&lens, field_angle(1.0, self.dist, self.job.efl), Some(self.dist));
            res.extend(self.job.seidel_weights.iter().zip(w.sum).map(|(k, v)| k * v));
        }
        Some(Eval { lens, residuals: res, rms, violations })
    }

    /// Levenberg–Marquardt from the start design; `log` gets progress lines.
    /// With `restarts`, LM is rerun from random perturbations of the best design.
    pub fn optimize(&self, log: &mut dyn FnMut(&str)) -> Result<Eval> {
        let start = self.eval(&self.start()).context("the start design cannot be traced")?;
        log(&format!("start: merit {:.4e}, RMS {:?} µm", merit(&start), round1(&start.rms)));
        if self.vars.is_empty() {
            return Ok(start);
        }
        let mut best = self.lm(self.start(), start, log)?;
        log(&format!("descent: merit {:.4e}, RMS {:?} µm", merit(&best), round1(&best.rms)));
        // Deterministic xorshift, uniform in [-1, 1).
        let mut seed = 0x9e3779b97f4a7c15_u64;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        for r in 0..self.job.restarts {
            let x: Vec<f64> = self.read(&best.lens).iter().map(|v| v * (1.0 + self.job.restart_scale * rand())).collect();
            let Some(e) = self.eval(&x) else { continue };
            let Ok(e) = self.lm(x, e, &mut |_| {}) else { continue };
            log(&format!("restart {r}: merit {:.4e}, RMS {:?} µm", merit(&e), round1(&e.rms)));
            if merit(&e) < merit(&best) {
                best = e;
            }
        }
        log(&format!("end: merit {:.4e}, RMS {:?} µm", merit(&best), round1(&best.rms)));
        if !best.violations.is_empty() {
            log(&format!("violated: {}", best.violations.join(", ")));
        }
        Ok(best)
    }

    fn lm(&self, mut x: Vec<f64>, mut cur: Eval, log: &mut dyn FnMut(&str)) -> Result<Eval> {
        let mut f = merit(&cur);
        let mut lambda = 1e-3;
        let mut stalled = 0;
        for it in 0..self.job.iterations {
            // Forward-difference Jacobian, columns in parallel.
            let cols: Vec<Option<Vec<f64>>> = (0..x.len())
                .into_par_iter()
                .map(|j| {
                    let h = match self.vars[j] {
                        Var::Curvature(_) => 1e-7,
                        Var::Thickness(_) | Var::SemiDiameter(_) => 1e-5,
                    };
                    let mut xh = x.clone();
                    xh[j] += h;
                    let e = self.eval(&xh)?;
                    (e.residuals.len() == cur.residuals.len()).then(|| e.residuals.iter().zip(&cur.residuals).map(|(a, b)| (a - b) / h).collect())
                })
                .collect();
            let Some(cols) = cols.into_iter().collect::<Option<Vec<_>>>() else { bail!("Jacobian: a perturbed design cannot be traced") };
            let n = x.len();
            let mut a = vec![vec![0.0; n]; n];
            let mut g = vec![0.0; n];
            for i in 0..n {
                g[i] = cols[i].iter().zip(&cur.residuals).map(|(j, r)| j * r).sum();
                for k in 0..=i {
                    a[i][k] = cols[i].iter().zip(&cols[k]).map(|(p, q)| p * q).sum();
                    a[k][i] = a[i][k];
                }
            }
            let mut accepted = false;
            while lambda < 1e12 {
                let mut m = a.clone();
                let scale = (0..n).map(|i| a[i][i]).fold(0.0, f64::max);
                for i in 0..n {
                    m[i][i] += lambda * a[i][i] + 1e-12 * scale;
                }
                let step = solve(m, g.iter().map(|v| -v).collect());
                let xt: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
                if let Some(e) = self.eval(&xt).filter(|e| e.residuals.len() == cur.residuals.len()) {
                    let ft = merit(&e);
                    if ft < f {
                        // Read back, so curvature scaling for the focal length is absorbed.
                        x = self.read(&e.lens);
                        stalled = if ft > f * (1.0 - 1e-5) { stalled + 1 } else { 0 };
                        f = ft;
                        cur = e;
                        lambda = (lambda / 3.0).max(1e-9);
                        accepted = true;
                        break;
                    }
                }
                lambda *= 4.0;
            }
            if it % 10 == 0 || !accepted {
                log(&format!("iteration {it}: merit {f:.4e}, RMS {:?} µm, λ {lambda:.1e}", round1(&cur.rms)));
            }
            if !accepted || stalled >= 5 {
                break;
            }
        }
        Ok(cur)
    }

    /// The optimised lens as lens-file text with the job's header.
    pub fn finish(&self, e: &Eval, job_file: &str) -> String {
        let mut lens = e.lens.clone();
        for s in lens.surfaces.iter_mut() {
            if s.is_stop {
                // Just above the iris at the design f-number.
                s.semi_diameter = (s.semi_diameter * 1000.0 + 1.0).ceil() / 1000.0;
            } else {
                s.semi_diameter = (s.semi_diameter * 100.0).ceil() / 100.0;
            }
        }
        let mut t = format!("@name {}\n", lens.name);
        for line in self.job.header.trim_end().lines() {
            t += &format!("# {line}\n").replace("# \n", "#\n");
        }
        let fields: Vec<String> = self.job.fields.iter().zip(&e.rms).map(|(f, r)| format!("{f}: {r:.1}")).collect();
        t += &format!(
            "#\n# Generated by `lens-render optimize {job_file}`.\n# RMS spot ({}) at f/{}, object at {} m, by image height\n# (fraction of 21.63 mm): {} µm. Units: mm.\n#\n",
            self.job.mono.map_or("F, e, C".to_string(), |l| format!("{l} nm")),
            self.job.fstop,
            self.job.focus,
            fields.join(", ")
        );
        t += "# radius      thickness   material       semi-diameter\n";
        for s in &lens.surfaces {
            let radius = if s.is_stop { "stop".to_string() } else if s.radius == 0.0 { "inf".to_string() } else { format!("{:.4}", s.radius) };
            let material = s.material.as_ref().map_or("air", |g| g.name());
            t += &format!("  {radius:<11} {:<11.4} {material:<14} {:.3}\n", s.thickness, s.semi_diameter);
        }
        t
    }
}

fn merit(e: &Eval) -> f64 {
    e.residuals.iter().map(|r| r * r).sum()
}

fn round1(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 10.0).round() / 10.0).collect()
}

/// Solves m x = b by Gaussian elimination with partial pivoting.
fn solve(mut m: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs())).unwrap();
        m.swap(c, p);
        b.swap(c, p);
        if m[c][c] == 0.0 {
            continue;
        }
        for r in c + 1..n {
            let f = m[r][c] / m[c][c];
            for k in c..n {
                m[r][k] -= f * m[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| m[r][k] * x[k]).sum();
        x[r] = if m[r][r] == 0.0 { 0.0 } else { (b[r] - s) / m[r][r] };
    }
    x
}

/// Runs a job file and writes its output lens next to it.
pub fn run(job_path: &str) -> Result<()> {
    let job: Job = toml::from_str(&std::fs::read_to_string(job_path).with_context(|| format!("reading {job_path}"))?)?;
    let out = Path::new(job_path).parent().unwrap_or(Path::new(".")).join(&job.output);
    let p = Problem::new(job)?;
    let e = p.optimize(&mut |s| eprintln!("{s}"))?;
    std::fs::write(&out, p.finish(&e, job_path))?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A detuned achromat improves several-fold and keeps its focal length.
    #[test]
    fn improves_detuned_achromat() {
        let job: Job = toml::from_str(
            r#"
            name = "test"
            output = "unused"
            fstop = 8
            fields = [0.0]
            curvatures = [2, 3, 4]
            iterations = 40
            start = """
            stop   0.5  air    4
            25     4.0  N-BK7  6
            -50    2.0  F2     6
            -200   45   air    6
            """
            "#,
        )
        .unwrap();
        let p = Problem::new(job).unwrap();
        let start = p.eval(&p.start()).unwrap();
        let end = p.optimize(&mut |_| {}).unwrap();
        assert!((end.lens.focal_length() / 50.0 - 1.0).abs() < 1e-6, "efl {}", end.lens.focal_length());
        assert!(end.rms[0] < start.rms[0] / 3.0, "axis RMS {} → {}", start.rms[0], end.rms[0]);
        assert!(end.residuals.iter().map(|r| r * r).sum::<f64>() < 0.2 * start.residuals.iter().map(|r| r * r).sum::<f64>());
    }
}


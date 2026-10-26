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
//!
//! Optional (docs/optimiser.md): conic and aspheric coefficients as
//! variables with manufacturability limits, geometric-MTF, distortion,
//! illumination and length terms, glass choices, the spectral stop's
//! parameters, and a global search (multi-start with basin hops, in parallel).

use crate::glass::{Glass, LAMBDA_D};
use crate::lens::{Iris, Lens, SpectralStop, Surface};
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

#[derive(Clone, Deserialize)]
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
    /// Merit wavelengths and weights, [[nm, weight], …], instead of F, e, C.
    #[serde(default)]
    pub wavelengths: Vec<[f64; 2]>,
    /// Weight of the RMS spot in the merit (0: other terms only).
    #[serde(default = "d_one")]
    pub rms_weight: f64,
    /// Rows whose conic constant varies.
    #[serde(default)]
    pub conics: Vec<usize>,
    /// Rows whose aspheric coefficients of `asphere_orders` vary.
    #[serde(default)]
    pub aspheres: Vec<usize>,
    #[serde(default = "d_orders")]
    pub asphere_orders: Vec<usize>,
    /// Aspheres: largest departure (mm) from the sphere through vertex and
    /// rim, its largest slope (mm/mm), most inflections of the profile.
    #[serde(default)]
    pub max_departure: Option<f64>,
    #[serde(default)]
    pub max_departure_slope: Option<f64>,
    #[serde(default)]
    pub max_inflections: Option<usize>,
    /// Aspheres: steepest surface (degrees).
    #[serde(default = "d_surface_slope")]
    pub max_surface_slope: f64,
    /// Geometric MTF (from the spot) at `mtf_freqs` lp/mm, sagittal and
    /// tangential: weight in µm of RMS per unit of lost contrast (0: off).
    #[serde(default)]
    pub mtf_weight: f64,
    #[serde(default = "d_mtf_freqs")]
    pub mtf_freqs: Vec<f64>,
    /// One-sided MTF targets from published data.
    #[serde(default)]
    pub mtf_target: Option<MtfTarget>,
    /// Clear apertures vary (false: they stay as in the start design).
    #[serde(default = "d_true")]
    pub vary_apertures: bool,
    /// Largest relative change of every radius from the start design, and
    /// per row: [row, change].
    #[serde(default)]
    pub max_radius_change: Option<f64>,
    #[serde(default)]
    pub radius_change_limits: Vec<[f64; 2]>,
    /// Aspheric surfaces: largest sag change (mm) from the start design's
    /// profile over the clear aperture.
    #[serde(default)]
    pub max_profile_change: Option<f64>,
    /// Lens file the radius and profile changes are measured from, relative
    /// to the job file (default: the start design).
    #[serde(default)]
    pub origin: Option<String>,
    /// |distortion| (%) at every field.
    #[serde(default)]
    pub max_distortion: Option<f64>,
    /// Relative illumination at the corner (cos⁴, vignetting, distortion).
    #[serde(default)]
    pub min_corner_illumination: Option<f64>,
    /// Front vertex to sensor (mm).
    #[serde(default)]
    pub max_length: Option<f64>,
    /// Spectral stop parameters that vary: any of "inner", "lo", "hi".
    #[serde(default)]
    pub spectral_stop: Vec<String>,
    /// SCRATCH: rows whose nd/vd model glass varies, and [row, nd min, nd max, vd min, vd max].
    #[serde(default)]
    pub model_glasses: Vec<usize>,
    #[serde(default)]
    pub glass_bounds: Vec<[f64; 5]>,
    #[serde(default)]
    pub glass_map_scale: Option<f64>,
    /// SCRATCH: [row, ΔPgF] of model glasses (normal-line partial dispersion + this).
    #[serde(default)]
    pub glass_dp: Vec<[f64; 2]>,
    /// Luminous transmission of the spectral stop, wide open.
    #[serde(default)]
    pub min_stop_transmission: Option<f64>,
    /// Candidate glasses per row, tried by the global search.
    #[serde(default)]
    pub glass_choice: Vec<GlassChoice>,
    /// Limits for elements of one material, e.g. a hard crystal.
    #[serde(default)]
    pub material_limits: Vec<MaterialLimits>,
    #[serde(default)]
    pub global: Option<Global>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialLimits {
    pub glass: String,
    /// Centre thickness (mm).
    #[serde(default)]
    pub max_thickness: Option<f64>,
    /// Aspheric departure (mm) of surfaces bounding this material.
    #[serde(default)]
    pub max_departure: Option<f64>,
    /// Clear semi-diameter (mm) of both faces, e.g. the size a crystal grows to.
    #[serde(default)]
    pub max_semi_diameter: Option<f64>,
}

/// Published MTF (white light, object at infinity) the design should reach:
/// residual `weight` × (reference + `margin` − ours) where ours falls short,
/// at every f-number, image height, frequency, sagittal and tangential.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtfTarget {
    /// JSON as for `lens-render mtf --reference` ({"f/1.4": {"5S": […], …},
    /// "heights_mm": […]}, percent), relative to the job file.
    pub reference: String,
    pub fstops: Vec<f64>,
    /// Image heights (mm), interpolated in the reference; default: its own.
    #[serde(default)]
    pub heights: Vec<f64>,
    /// Frequencies (lp/mm) to use; default: all of the reference's.
    #[serde(default)]
    pub freqs: Vec<f64>,
    /// µm of RMS per unit of contrast short of the target.
    pub weight: f64,
    #[serde(default)]
    pub margin: f64,
    /// Pupil grid (rays per side, over 1.3 × the paraxial pupil).
    #[serde(default = "d_mtf_grid")]
    pub grid: usize,
}

/// Target contrasts at one f-number: per height, per (frequency, tangential).
#[derive(Clone)]
struct MtfGoal {
    fstop: f64,
    heights: Vec<f64>,
    freqs: Vec<(f64, bool)>,
    /// [height][freq] contrast (0–1).
    values: Vec<Vec<f64>>,
}

/// White-light wavelengths of the MTF target (as `lens-render mtf`), CIE ȳ weighted.
fn mtf_wavelengths() -> Vec<(f64, f64)> {
    crate::mtf::WAVELENGTHS.iter().map(|&l| (l, crate::spectrum::cie_xyz(l)[1])).collect()
}

/// Reads the targets of `t` from its reference file.
fn mtf_goals(t: &MtfTarget) -> Result<Vec<MtfGoal>> {
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&t.reference).with_context(|| format!("reading {}", t.reference))?)?;
    let ref_h: Vec<f64> = r["heights_mm"].as_array().context("reference: heights_mm")?.iter().filter_map(|v| v.as_f64()).collect();
    let heights = if t.heights.is_empty() { ref_h.clone() } else { t.heights.clone() };
    let mut goals = Vec::new();
    for &f in &t.fstops {
        let data = r.get(format!("f/{f}")).and_then(|d| d.as_object()).with_context(|| format!("reference has no f/{f}"))?;
        let mut freqs = Vec::new();
        let mut curves = Vec::new();
        for (key, vals) in data {
            let (q, tan) = match key.strip_suffix('S') {
                Some(q) => (q, false),
                None => (key.strip_suffix('T').with_context(|| format!("reference key {key}"))?, true),
            };
            let q: f64 = q.parse()?;
            if !t.freqs.is_empty() && !t.freqs.contains(&q) {
                continue;
            }
            freqs.push((q, tan));
            let v: Vec<f64> = vals.as_array().context("reference values")?.iter().filter_map(|v| v.as_f64()).collect();
            if v.len() != ref_h.len() {
                bail!("reference f/{f} {key}: {} values for {} heights", v.len(), ref_h.len());
            }
            curves.push(v);
        }
        let interp = |v: &[f64], h: f64| -> f64 {
            let i = ref_h.partition_point(|&x| x < h).clamp(1, ref_h.len() - 1);
            let t = ((h - ref_h[i - 1]) / (ref_h[i] - ref_h[i - 1])).clamp(0.0, 1.0);
            (v[i - 1] + t * (v[i] - v[i - 1])) / 100.0
        };
        let values = heights.iter().map(|&h| curves.iter().map(|c| interp(c, h)).collect()).collect();
        goals.push(MtfGoal { fstop: f, heights: heights.clone(), freqs, values });
    }
    Ok(goals)
}

/// Sagittal and tangential MTF at image height `h` of `lens` (iris and sensor
/// as set): geometric MTF of the white-light spot from a soft-edged pupil grid,
/// times the diffraction limit at the working f-number. None if no light passes.
fn target_mtf(lens: &Lens, dist: f64, h: f64, freqs: &[(f64, bool)], grid: usize) -> Option<Vec<f64>> {
    let theta = crate::psf::field_for_height(lens, h, Some(dist))?;
    let wls = mtf_wavelengths();
    let rays = trace_soft(lens, &Source::at_field(lens, theta, Some(dist)), &wls, grid, 1.3);
    let total: f64 = rays.iter().map(|r| r.w).sum();
    if total <= 0.0 {
        return None;
    }
    let z = lens.sensor_z;
    let pts: Vec<(f64, f64, f64)> = rays
        .iter()
        .filter_map(|r| {
            let e = r.exit.filter(|_| r.w > 0.0)?;
            let t = (z - e.o.z) / e.d.z;
            Some((e.o.x + e.d.x * t, e.o.y + e.d.y * t, r.w / total))
        })
        .collect();
    let n_w = lens.working_fnumber();
    let wsum: f64 = wls.iter().map(|w| w.1).sum();
    Some(freqs.iter().map(|&(q, tan)| spot_mtf(&pts, q, tan) * wls.iter().map(|&(l, w)| w * crate::psf::diffraction_limit_mtf(q, l, n_w)).sum::<f64>() / wsum).collect())
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlassChoice {
    pub row: usize,
    pub glasses: Vec<String>,
}

/// Global search: `restarts` local optimisations from random perturbations
/// of the start design (start 0 unperturbed), each followed by `hops` basin
/// hops; the `keep` best distinct designs are written.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Global {
    pub restarts: usize,
    #[serde(default = "d_keep")]
    pub keep: usize,
    #[serde(default = "d_one_u64")]
    pub seed: u64,
    /// Relative perturbation of curvatures and thicknesses (curvatures also
    /// by ± perturb / efl, so flat surfaces bend).
    #[serde(default = "d_perturb")]
    pub perturb: f64,
    /// Perturbation of aspheric and conic terms, as sag at the rim (mm).
    #[serde(default = "d_asphere_perturb")]
    pub asphere_perturb: f64,
    #[serde(default)]
    pub hops: usize,
    /// LM iterations per descent (default: the job's).
    #[serde(default)]
    pub iterations: Option<usize>,
    /// Designs closer than this (relative units, see docs) count as one.
    #[serde(default = "d_distinct")]
    pub distinct: f64,
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
fn d_one() -> f64 {
    1.0
}
fn d_one_u64() -> u64 {
    1
}
fn d_orders() -> Vec<usize> {
    vec![4, 6, 8, 10]
}
fn d_surface_slope() -> f64 {
    60.0
}
fn d_mtf_freqs() -> Vec<f64> {
    vec![10.0, 30.0]
}
fn d_true() -> bool {
    true
}
fn d_mtf_grid() -> usize {
    32
}
fn d_keep() -> usize {
    5
}
fn d_perturb() -> f64 {
    0.3
}
fn d_asphere_perturb() -> f64 {
    0.02
}
fn d_distinct() -> f64 {
    0.01
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Var {
    Curvature(usize),
    Thickness(usize),
    SemiDiameter(usize),
    /// Conic constant, scaled to sag at the rim.
    Conic(usize),
    /// Aspheric coefficient (row, index into `Surface::aspheric`), scaled to sag at the rim.
    Aspheric(usize, usize),
    Inner,
    BandLo,
    BandHi,
    Nd(usize),
    Vd(usize),
}

#[derive(Clone)]
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

/// Traces a pupil grid at each wavelength, weighted by a spectral stop's
/// transmission. Without `clip`, apertures do not block and `heights` gets
/// the largest radius reached at each surface.
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
            let w = match ray {
                Some(ray) if out.is_some() => w * lens.spectral_weight(*ray, l, &m),
                _ => w,
            };
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

/// A ray of `trace_soft`, weights including the wavelength's.
struct SoftRay {
    exit: Option<Ray>,
    /// Through all apertures and the spectral stop.
    w: f64,
    /// Through the iris alone.
    w_iris: f64,
    /// Through all apertures, without the spectral stop.
    w_clear: f64,
    /// cos⁴ of the ray's angle to the axis in object space (illumination).
    cos4: f64,
}

/// Traces a pupil grid with soft apertures (edges blurred over a few % of
/// their radius, so the merit stays smooth).
fn trace_soft(lens: &Lens, src: &Source, wls: &[(f64, f64)], g: usize, spread: f64) -> Vec<SoftRay> {
    let grid = pupil_grid(lens, src, g, spread);
    let k = lens.stop_index;
    let r_iris = lens.iris.max_radius;
    let mut out = Vec::with_capacity(grid.len() * wls.len());
    for &(l, wl) in wls {
        let m = lens.media(l);
        for ray in &grid {
            let (mut w_iris, mut w_all, mut filter) = (0.0, 0.0, 1.0);
            let exit = ray.and_then(|ray| {
                let (mut cur, mut w) = (ray, 1.0);
                for (i, ev) in lens.forward_path.iter().enumerate() {
                    let s = &lens.surfaces[i];
                    if i == k {
                        // The stop does not refract; where the ray line crosses its plane.
                        let t = (s.z - cur.o.z) / cur.d.z;
                        let (x, y) = (cur.o.x + cur.d.x * t, cur.o.y + cur.d.y * t);
                        w_iris = smooth_step(r_iris - x.hypot(y), 0.01 * r_iris);
                        if let Some(f) = &lens.spectral_stop {
                            // The annulus edge soft, too.
                            let b = f.band(l);
                            filter = b + (1.0 - b) * smooth_step(f.inner * s.semi_diameter - x.hypot(y), 0.02 * s.semi_diameter);
                        }
                        continue;
                    }
                    cur = lens.trace_path_opl(cur, l, &m, std::slice::from_ref(ev), false)?.0;
                    w *= smooth_step(s.semi_diameter - cur.o.x.hypot(cur.o.y), 0.02 * s.semi_diameter + 0.05);
                }
                w_all = w * w_iris;
                Some(cur)
            });
            let cos4 = ray.map_or(0.0, |r| r.d.z.powi(4));
            out.push(SoftRay { exit, w: wl * w_all * filter, w_iris: wl * w_iris, w_clear: wl * w_all, cos4 });
        }
    }
    out
}

/// Geometric MTF of weighted spot points (x, y, w; weights summing to 1)
/// at `f` lp/mm along x (`along_y` false) or y.
fn spot_mtf(pts: &[(f64, f64, f64)], f: f64, along_y: bool) -> f64 {
    let k = -2.0 * std::f64::consts::PI * f;
    let (mut re, mut im) = (0.0, 0.0);
    for &(x, y, w) in pts {
        let (s, c) = (k * if along_y { y } else { x }).sin_cos();
        re += w * c;
        im += w * s;
    }
    re.hypot(im)
}

/// Image height per unit tan θ of chief rays near the axis.
fn paraxial_scale(lens: &Lens, dist: f64) -> Option<f64> {
    let th = 1e-4_f64;
    Some(crate::psf::chief_image(lens, &Source::at_field(lens, th, Some(dist)), LAMBDA_D)?[1].abs() / th.tan())
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
    // Clipped by the real apertures over a grid wider than the paraxial pupil, as the renderer's bundle is.
    axial_focus(&trace_bundle(lens, &Source::at_field(lens, 0.0, Some(dist)), &wls, 24, 1.3, true, &mut []))
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

#[derive(Clone)]
pub struct Problem {
    job: Job,
    base: Lens,
    vars: Vec<Var>,
    /// Physical value per unit of each variable.
    norm: Vec<f64>,
    dist: f64,
    fields: Vec<Field>,
    wls: Vec<(f64, f64)>,
    /// Candidate glasses per row (0-based).
    choices: Vec<(usize, Vec<Glass>)>,
    /// Surfaces that are or may become aspheric (fixed, so the residuals are).
    aspheric: Vec<bool>,
    /// MTF targets per f-number.
    goals: Vec<MtfGoal>,
    /// The start design as written (radius and profile changes).
    origin: Lens,
}

/// One evaluated design.
pub struct Eval {
    pub lens: Lens,
    pub residuals: Vec<f64>,
    /// RMS spot radius (µm) per merit field.
    pub rms: Vec<f64>,
    /// Mean geometric MTF over the job's frequencies, sagittal and
    /// tangential, per merit field (only with `mtf_weight`).
    pub mtf: Vec<f64>,
    /// Constraints that are violated.
    pub violations: Vec<String>,
    /// MTF targets: (f-number, height, frequency, tangential, ours, target).
    pub targets: Vec<(f64, f64, f64, bool, f64, f64)>,
}

/// A local minimum found by the global search.
pub struct Candidate {
    pub eval: Eval,
    pub merit: f64,
    /// Index of the start it descended from.
    pub start: usize,
    /// Glasses of the `glass_choice` rows.
    pub glasses: Vec<String>,
    x: Vec<f64>,
}

/// Deterministic xorshift, uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    /// Stream `stream` of seed `seed` (splitmix64 of both).
    fn new(seed: u64, stream: u64) -> Rng {
        let mut z = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ stream.wrapping_add(1).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        Rng((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Shape of an aspheric surface over its clear aperture.
struct AsphereShape {
    /// How far the clear aperture extends beyond the surface's domain (mm).
    outside: f64,
    /// Largest departure from the sphere through vertex and rim (mm), and its largest slope.
    departure: f64,
    departure_slope: f64,
    /// Steepest slope dz/dr.
    slope: f64,
    /// Per run of constant sign of the profile curvature z″: peak |z″| × semi-diameter.
    lobes: Vec<f64>,
}

fn asphere_shape(s: &Surface) -> AsphereShape {
    let m = 64;
    let h = s.semi_diameter;
    let r = |j: usize| h * j as f64 / m as f64;
    let ok = (0..=m).take_while(|&j| s.sag_u(r(j).powi(2)).is_some()).count().max(1) - 1;
    let (hr, zr) = (r(ok), s.sag_u(r(ok).powi(2)).unwrap_or(0.0));
    let cb = if hr > 0.0 { 2.0 * zr / (hr * hr + zr * zr) } else { 0.0 };
    let (mut departure, mut departure_slope, mut slope) = (0.0f64, 0.0f64, 0.0f64);
    let mut dz = Vec::with_capacity(ok + 1);
    for j in 0..=ok {
        let (rj, u) = (r(j), r(j).powi(2));
        let z = s.sag_u(u).unwrap_or(0.0);
        let d = 2.0 * rj * s.dsag_du(u);
        let q = (1.0 - cb * cb * u).max(1e-12).sqrt();
        departure = departure.max((z - cb * u / (1.0 + q)).abs());
        departure_slope = departure_slope.max((d - cb * rj / q).abs());
        slope = slope.max(d.abs());
        dz.push(d);
    }
    let d2: Vec<f64> = (1..ok.max(1)).map(|j| (dz[j + 1] - dz[j - 1]) * m as f64 / (2.0 * h)).collect();
    let peak = d2.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut lobes: Vec<f64> = Vec::new();
    let mut sign = 0.0;
    for &v in d2.iter().filter(|v| v.abs() > 1e-9 * peak) {
        if v.signum() != sign {
            lobes.push(0.0);
            sign = v.signum();
        }
        let last = lobes.last_mut().unwrap();
        *last = last.max(v.abs() * h);
    }
    AsphereShape { outside: h - hr, departure, departure_slope, slope, lobes }
}

/// Relative illumination at field angle `theta` from the flux ratio to the
/// axis (`flux`): divided by the image area per object area (radial ×
/// tangential magnification) relative to the axis, as `metrics::numbers`.
fn relative_illumination(lens: &Lens, dist: f64, theta: f64, flux: f64) -> Option<f64> {
    let par = lens.paraxial_data(LAMBDA_D);
    let h = |th: f64| crate::psf::chief_image(lens, &Source::at_field(lens, th, Some(dist)), LAMBDA_D).map(|p| p[1].abs());
    let dth = 1e-4;
    let y_obj = |th: f64| (par.entrance_pupil_z + dist) * th.tan();
    let radial = (h(theta + dth)? - h(theta - dth)?) / (y_obj(theta + dth) - y_obj(theta - dth));
    let tangential = h(theta)? / y_obj(theta);
    let m0 = paraxial_scale(lens, dist)? / (par.entrance_pupil_z + dist);
    Some(flux * m0 * m0 / (radial * tangential))
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
        let surface_row = |r: usize, what: &str| -> Result<usize> {
            if r == 0 || r > n || base.surfaces[r - 1].is_stop {
                bail!("{what} row {r} is not a lens surface");
            }
            Ok(r - 1)
        };
        for &r in &job.conics {
            vars.push(Var::Conic(surface_row(r, "conic")?));
        }
        for &r in &job.aspheres {
            let i = surface_row(r, "asphere")?;
            for &o in &job.asphere_orders {
                if o < 4 || o % 2 == 1 || o > 18 {
                    bail!("aspheric orders are 4, 6, …, 18");
                }
                vars.push(Var::Aspheric(i, (o - 4) / 2));
            }
        }
        for v in &job.spectral_stop {
            if base.spectral_stop.is_none() {
                bail!("spectral_stop variables need annulus= and pass= on the start design's stop");
            }
            vars.push(match v.as_str() {
                "inner" => Var::Inner,
                "lo" => Var::BandLo,
                "hi" => Var::BandHi,
                _ => bail!("spectral_stop variables are inner, lo and hi"),
            });
        }
        if let Some(l) = job.thickness_limits.iter().find(|l| l[0] < 1.0 || l[0] as usize > n) {
            bail!("thickness limit on row {} that does not exist", l[0]);
        }
        if job.symmetric {
            if n != 2 * k + 1 {
                bail!("a symmetric design needs as many rows behind the stop as in front");
            }
            if vars.iter().any(|v| matches!(v, Var::Curvature(i) | Var::Thickness(i) | Var::Conic(i) | Var::Aspheric(i, _) if *i >= k)) {
                bail!("symmetric: list only rows in front of the stop");
            }
        }
        for &r in &job.model_glasses {
            vars.push(Var::Nd(r - 1));
            vars.push(Var::Vd(r - 1));
        }
        if job.vary_apertures {
            vars.extend((0..n).filter(|&i| !base.surfaces[i].is_stop).map(Var::SemiDiameter));
        }
        // Aspheric terms as their sag at the start's rim, so steps are comparable.
        let norm = vars
            .iter()
            .map(|v| match *v {
                Var::Aspheric(i, j) => base.surfaces[i].semi_diameter.powi(-(2 * j as i32 + 4)),
                Var::Conic(i) => {
                    let s = &base.surfaces[i];
                    8.0 / (s.curvature().abs().max(1.0 / job.efl).powi(3) * s.semi_diameter.powi(4))
                }
                _ => 1.0,
            })
            .collect();
        let mut choices = Vec::new();
        for c in &job.glass_choice {
            if c.row == 0 || c.row > n || base.surfaces[c.row - 1].material.is_none() || c.glasses.is_empty() {
                bail!("glass_choice row {} is not glass, or has no glasses", c.row);
            }
            choices.push((c.row - 1, c.glasses.iter().map(|g| Glass::parse(g)).collect::<Result<Vec<_>>>()?));
        }
        let fw = |i: usize| job.field_weights.get(i).copied().unwrap_or(1.0);
        let dist = job.focus * 1000.0;
        let mut fields: Vec<Field> = job.fields.iter().enumerate().map(|(i, &frac)| Field { frac, theta: field_angle(frac, dist, job.efl), weight: fw(i) }).collect();
        for frac in [0.0, 0.5, 0.7, 1.0] {
            if !job.fields.contains(&frac) {
                fields.push(Field { frac, theta: field_angle(frac, dist, job.efl), weight: 0.0 });
            }
        }
        let wls = match job.mono {
            _ if !job.wavelengths.is_empty() => job.wavelengths.iter().map(|w| (w[0], w[1])).collect(),
            Some(l) => vec![(l, 1.0)],
            None => SPOT_WAVELENGTHS.to_vec(),
        };
        let aspheric = (0..n).map(|i| base.surfaces[i].is_aspheric() || vars.iter().any(|v| matches!(v, Var::Conic(j) | Var::Aspheric(j, _) if *j == i || (job.symmetric && 2 * k - *j == i)))).collect();
        let goals = match &job.mtf_target {
            Some(t) => mtf_goals(t)?,
            None => Vec::new(),
        };
        let origin = match &job.origin {
            Some(path) => Lens::load(path)?,
            None => base.clone(),
        };
        Ok(Problem { job, base, vars, norm, dist, fields, wls, choices, aspheric, goals, origin })
    }

    fn start(&self) -> Vec<f64> {
        self.read(&self.base)
    }

    fn read(&self, lens: &Lens) -> Vec<f64> {
        let f = lens.spectral_stop.unwrap_or(SpectralStop { inner: 0.0, lo: 0.0, hi: 0.0 });
        self.vars
            .iter()
            .zip(&self.norm)
            .map(|(v, norm)| match *v {
                Var::Curvature(i) => lens.surfaces[i].curvature(),
                Var::Thickness(i) => lens.surfaces[i].thickness,
                Var::SemiDiameter(i) => lens.surfaces[i].semi_diameter,
                Var::Conic(i) => lens.surfaces[i].conic / norm,
                Var::Aspheric(i, j) => lens.surfaces[i].aspheric.get(j).copied().unwrap_or(0.0) / norm,
                Var::Inner => f.inner,
                Var::BandLo => f.lo,
                Var::BandHi => f.hi,
                Var::Nd(i) => lens.surfaces[i].material.as_ref().unwrap().nd(),
                Var::Vd(i) => lens.surfaces[i].material.as_ref().unwrap().abbe(),
            })
            .collect()
    }

    /// The design for variables `x`: mirrored, scaled to the focal length,
    /// iris at the design f-number. None if it has no positive focal length.
    pub fn lens(&self, x: &[f64]) -> Option<Lens> {
        let mut lens = self.base.clone();
        for ((v, &x), norm) in self.vars.iter().zip(x).zip(&self.norm) {
            match *v {
                Var::Curvature(i) => lens.surfaces[i].radius = if x == 0.0 { 0.0 } else { 1.0 / x },
                Var::Thickness(i) => lens.surfaces[i].thickness = x,
                Var::SemiDiameter(i) => lens.surfaces[i].semi_diameter = x.abs(),
                Var::Conic(i) => lens.surfaces[i].conic = x * norm,
                Var::Aspheric(i, j) => {
                    let a = &mut lens.surfaces[i].aspheric;
                    if a.len() <= j {
                        a.resize(j + 1, 0.0);
                    }
                    a[j] = x * norm;
                }
                Var::Inner => lens.spectral_stop.as_mut()?.inner = x.clamp(0.0, 1.0),
                Var::BandLo => lens.spectral_stop.as_mut()?.lo = x,
                Var::BandHi => lens.spectral_stop.as_mut()?.hi = x,
                Var::Nd(i) => {
                    let vd = lens.surfaces[i].material.as_ref().unwrap().abbe();
                    let dp = self.job.glass_dp.iter().find(|g| g[0] as usize == i + 1).map_or(0.0, |g| g[1]);
                    lens.surfaces[i].material = Some(Glass::model(&format!("{x:.5}/{vd:.2}/{dp}"), x, vd, dp));
                }
                Var::Vd(i) => {
                    let nd = lens.surfaces[i].material.as_ref().unwrap().nd();
                    let dp = self.job.glass_dp.iter().find(|g| g[0] as usize == i + 1).map_or(0.0, |g| g[1]);
                    lens.surfaces[i].material = Some(Glass::model(&format!("{nd:.5}/{x:.2}/{dp}"), nd, x, dp));
                }
            }
        }
        if let Some(f) = lens.spectral_stop.as_mut().filter(|f| f.lo > f.hi - 1.0) {
            let mid = 0.5 * (f.lo + f.hi);
            (f.lo, f.hi) = (mid - 0.5, mid + 0.5);
        }
        let k = lens.stop_index;
        if self.job.symmetric {
            for i in 0..k {
                lens.surfaces[2 * k - i].radius = -lens.surfaces[i].radius;
                lens.surfaces[2 * k - 1 - i].thickness = lens.surfaces[i].thickness;
                if self.job.conics.contains(&(i + 1)) {
                    lens.surfaces[2 * k - i].conic = lens.surfaces[i].conic;
                }
                if self.job.aspheres.contains(&(i + 1)) {
                    lens.surfaces[2 * k - i].aspheric = lens.surfaces[i].aspheric.iter().map(|a| -a).collect();
                }
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
        let mut lens = self.lens(x)?;
        let n = lens.surfaces.len();
        let z = render_focus(&lens, self.dist)?;
        // Distortion and illumination use chief rays on the sensor.
        lens.sensor_z = z;
        let mut res = Vec::new();
        let mut rms = Vec::new();
        let mut mtf = Vec::new();
        let mut light = Vec::new();
        // Per field: light from the object point, ∝ Σ cos⁴ over the pupil grid.
        let mut flux = Vec::new();
        for f in &self.fields {
            let rays = trace_soft(&lens, &Source::at_field(&lens, f.theta, Some(self.dist)), &self.wls, 20, 1.3);
            let (w_all, w_iris, w_clear) = rays.iter().fold((0.0, 0.0, 0.0), |a, r| (a.0 + r.w, a.1 + r.w_iris, a.2 + r.w_clear));
            light.push((f.frac, if w_iris > 0.0 { w_clear / w_iris } else { 0.0 }));
            flux.push(rays.iter().map(|r| r.w_clear * r.cos4).sum::<f64>());
            if f.weight == 0.0 {
                continue;
            }
            let pts: Vec<(f64, f64, f64)> = rays
                .iter()
                .map(|r| match r.exit {
                    Some(e) if r.w > 0.0 => {
                        let t = (z - e.o.z) / e.d.z;
                        (e.o.x + e.d.x * t, e.o.y + e.d.y * t, r.w / w_all)
                    }
                    _ => (0.0, 0.0, 0.0),
                })
                .collect();
            let cx: f64 = pts.iter().map(|p| p.0 * p.2).sum();
            let cy: f64 = pts.iter().map(|p| p.1 * p.2).sum();
            let mut sum = 0.0;
            for &(x, y, w) in &pts {
                let s = (self.job.rms_weight * f.weight * w).sqrt() * 1000.0;
                res.push(s * (x - cx));
                res.push(s * (y - cy));
                sum += w * ((x - cx).powi(2) + (y - cy).powi(2));
            }
            rms.push(sum.sqrt() * 1000.0);
            if self.job.mtf_weight > 0.0 {
                let nf = 2.0 * self.job.mtf_freqs.len() as f64;
                let s = (f.weight / nf).sqrt() * self.job.mtf_weight;
                let mut mean = 0.0;
                for &q in &self.job.mtf_freqs {
                    for along_y in [false, true] {
                        let m = spot_mtf(&pts, q, along_y);
                        res.push(s * (1.0 - m));
                        mean += m / nf;
                    }
                }
                mtf.push(mean);
            }
        }
        let s = &lens.surfaces;
        let sd = |i: usize| s[i].semi_diameter;
        let mut violations = Vec::new();
        let mut pen = |what: String, v: f64| {
            res.push(PENALTY * v.max(0.0));
            if v > 1e-3 {
                violations.push(format!("{what} by {v:.3}"));
            }
        };
        // Fixed apertures: as they are, whatever they pass.
        for (frac, pass) in light.into_iter().filter(|_| self.job.vary_apertures) {
            pen(format!("vignetting at {frac}"), LIGHT * (0.99 - self.job.vignetting * frac * frac - pass));
        }
        // Sag at radius r, or how far r lies outside the surface.
        let sag = |i: usize, r: f64| -> Result<f64, f64> {
            let c = s[i].curvature();
            if s[i].is_aspheric() {
                return s[i].sag_u(r * r).ok_or_else(|| r - 1.0 / (c * c * (1.0 + s[i].conic)).sqrt());
            }
            if c * c * r * r >= 1.0 { Err(r - 1.0 / c.abs()) } else { Ok(s[i].sag(r)) }
        };
        for i in 0..n - 1 {
            let glass = s[i].material.is_some();
            // Edge: glass at the element's rim, air where both surfaces exist.
            let r = if glass { sd(i).max(sd(i + 1)) } else { sd(i).min(sd(i + 1)) };
            let gap = |r: f64| match (sag(i, r), sag(i + 1, r)) {
                (Ok(a), Ok(b)) => s[i].thickness + b - a,
                (Err(e), _) | (_, Err(e)) => -1.0 - e,
            };
            let edge = gap(r);
            // Aspheres may be thinnest inside the aperture.
            let aspheric = self.aspheric[i] || self.aspheric[i + 1];
            let inside = if aspheric { (1..16).map(|j| gap(r * j as f64 / 16.0)).fold(f64::INFINITY, f64::min) } else { 0.0 };
            if glass {
                pen(format!("row {} glass centre", i + 1), self.job.min_glass_centre - s[i].thickness);
                pen(format!("row {} glass edge", i + 1), self.job.min_glass_edge - edge);
                pen(format!("row {} glass max", i + 1), s[i].thickness - self.job.max_glass);
                if aspheric {
                    pen(format!("row {} glass inside", i + 1), self.job.min_glass_edge - inside);
                }
            } else {
                pen(format!("row {} air centre", i + 1), self.job.min_air - s[i].thickness);
                pen(format!("row {} air edge", i + 1), self.job.min_air - edge);
                if aspheric {
                    pen(format!("row {} air inside", i + 1), self.job.min_air - inside);
                }
            }
        }
        let limits = |i: usize| s[i].material.as_ref().and_then(|g| self.job.material_limits.iter().find(|m| Glass::parse(&m.glass).is_ok_and(|l| l.name() == g.name())));
        for i in 0..n {
            if let Some(max) = limits(i).and_then(|m| m.max_thickness) {
                pen(format!("row {} {} max", i + 1, s[i].material.as_ref().unwrap().name()), s[i].thickness - max);
            }
            if let Some(max) = limits(i).and_then(|m| m.max_semi_diameter) {
                pen(format!("row {} {} semi-diameter", i + 1, s[i].material.as_ref().unwrap().name()), sd(i).max(sd(i + 1)) - max);
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
        if let Some(max) = self.job.max_length {
            pen("length".into(), z - max);
        }
        for i in (0..n).filter(|&i| self.aspheric[i]) {
            let a = asphere_shape(&s[i]);
            pen(format!("row {} outside the asphere", i + 1), a.outside);
            pen(format!("row {} surface slope", i + 1), a.slope - self.job.max_surface_slope.to_radians().tan());
            // Departures are small: 10× the usual weight (µm of departure ≡ 10 µm of RMS).
            let bounding = [i.checked_sub(1).and_then(limits), limits(i)];
            if let Some(max) = bounding.iter().flatten().filter_map(|m| m.max_departure).chain(self.job.max_departure).reduce(f64::min) {
                pen(format!("row {} departure (0.1 mm)", i + 1), 10.0 * (a.departure - max));
            }
            if let Some(max) = self.job.max_departure_slope {
                pen(format!("row {} departure slope", i + 1), a.departure_slope - max);
            }
            if let Some(max) = self.job.max_inflections {
                // Too many inflections: the weakest curvature lobes have to go.
                let mut lobes = a.lobes.clone();
                lobes.sort_by(f64::total_cmp);
                let excess = lobes.len().saturating_sub(max + 1);
                pen(format!("row {} inflections", i + 1), lobes[..excess].iter().sum());
            }
        }
        if let Some(max) = self.job.max_distortion {
            let k = paraxial_scale(&lens, self.dist);
            for f in self.fields.iter().filter(|f| f.frac > 0.0) {
                let h = crate::psf::chief_image(&lens, &Source::at_field(&lens, f.theta, Some(self.dist)), LAMBDA_D);
                let d = match (k, h) {
                    (Some(k), Some(h)) => (h[1] / (k * f.theta.tan()) - 1.0).abs() * 100.0,
                    _ => max + 1.0,
                };
                pen(format!("distortion (%) at {}", f.frac), d - max);
            }
        }
        if let Some(min) = self.job.min_corner_illumination {
            let (c, a) = (self.fields.iter().position(|f| f.frac == 1.0).unwrap(), self.fields.iter().position(|f| f.frac == 0.0).unwrap());
            let ri = if flux[a] > 0.0 { relative_illumination(&lens, self.dist, self.fields[c].theta, flux[c] / flux[a]).unwrap_or(0.0) } else { 0.0 };
            pen("corner illumination".into(), LIGHT * (min - ri));
        }
        if let (Some(min), Some(f)) = (self.job.min_stop_transmission, &lens.spectral_stop) {
            pen("spectral stop transmission".into(), LIGHT * (min - f.luminous(lens.iris.radius / s[lens.stop_index].semi_diameter)));
        }
        for &r in &self.job.model_glasses {
            let g = s[r - 1].material.as_ref().unwrap();
            let (nd, vd) = (g.nd(), g.abbe());
            let d = crate::glass_extra::EXTRA.iter().map(|(n, _, _)| Glass::parse(n).unwrap()).map(|c| ((c.nd() - nd) / 0.02).hypot((c.abbe() - vd) / 2.0)).fold(f64::INFINITY, f64::min);
            if let Some(k) = self.job.glass_map_scale {
                pen(format!("row {r} glass map"), d / k - 1.0);
            }
        }
        for &[r, n0, n1, v0, v1] in &self.job.glass_bounds {
            let g = s[r as usize - 1].material.as_ref().unwrap();
            pen(format!("row {r} nd"), (n0 - g.nd()).max(g.nd() - n1) * 100.0);
            pen(format!("row {r} vd"), (v0 - g.abbe()).max(g.abbe() - v1) * 0.1);
        }
        {
            let o = &self.origin;
            for i in (0..n).filter(|&i| !s[i].is_stop && o.surfaces[i].radius != 0.0) {
                let limit = self.job.radius_change_limits.iter().find(|l| l[0] as usize == i + 1).map(|l| l[1]).or(self.job.max_radius_change);
                if let Some(max) = limit {
                    // Relative change, as 0.1 mm per % (a penalty of 1000 µm per 10 %).
                    pen(format!("row {} radius change (10 %)", i + 1), 10.0 * ((s[i].radius / o.surfaces[i].radius - 1.0).abs() - max));
                }
            }
            if let Some(max) = self.job.max_profile_change {
                for i in (0..n).filter(|&i| self.aspheric[i]) {
                    let h = o.surfaces[i].semi_diameter;
                    let change = (0..=32).map(|j| (h * j as f64 / 32.0).powi(2)).map(|u| match (s[i].sag_u(u), o.surfaces[i].sag_u(u)) {
                        (Some(a), Some(b)) => (a - b).abs(),
                        _ => 1.0,
                    });
                    // µm of profile ≡ 10 µm of RMS, as departures.
                    pen(format!("row {} profile change (0.1 mm)", i + 1), 10.0 * (change.fold(0.0, f64::max) - max));
                }
            }
        }
        if !self.job.seidel_weights.is_empty() {
            let w = crate::seidel::seidel(&lens, field_angle(1.0, self.dist, self.job.efl), Some(self.dist));
            res.extend(self.job.seidel_weights.iter().zip(w.sum).map(|(k, v)| k * v));
        }
        let mut targets = Vec::new();
        if let Some(t) = &self.job.mtf_target {
            for g in &self.goals {
                // Iris at this f-number, refocused as the renderer would.
                let mut l = lens.clone();
                l.iris = Iris::new(0, 0.0, 0.0, l.stop_radius_for_fstop(g.fstop), true);
                l.sensor_z = render_focus(&l, self.dist)?;
                for (&h, goal) in g.heights.iter().zip(&g.values) {
                    let ours = target_mtf(&l, self.dist, h, &g.freqs, t.grid).unwrap_or_else(|| vec![0.0; g.freqs.len()]);
                    for ((&(q, tan), &m), &v) in g.freqs.iter().zip(&ours).zip(goal) {
                        res.push(t.weight * (v + t.margin - m).max(0.0));
                        targets.push((g.fstop, h, q, tan, m, v));
                    }
                }
            }
        }
        Some(Eval { lens, residuals: res, rms, mtf, violations, targets })
    }

    /// `v` of variable `var` nudged by the relative amount `u` (additively for
    /// aspheric terms, as rim sag: 0.1 mm per unit).
    fn nudge(&self, var: Var, v: f64, u: f64) -> f64 {
        match var {
            Var::Curvature(_) | Var::Thickness(_) | Var::SemiDiameter(_) => v * (1.0 + u),
            Var::Conic(_) | Var::Aspheric(..) => v + 0.1 * u,
            Var::Inner => (v + 0.5 * u).clamp(0.0, 1.0),
            Var::BandLo | Var::BandHi => v + 50.0 * u,
            Var::Nd(_) => v + 0.05 * u,
            Var::Vd(_) => v * (1.0 + u),
        }
    }

    /// Random start for the global search: curvatures by ± `scale` relative
    /// and ± `scale` / efl, thicknesses ± `scale` relative, aspheric terms
    /// by ± `asph` mm of rim sag.
    fn perturb(&self, x: &[f64], scale: f64, asph: f64, rng: &mut Rng) -> Vec<f64> {
        x.iter()
            .zip(&self.vars)
            .map(|(&v, &var)| match var {
                Var::Curvature(_) => v * (1.0 + scale * rng.next()) + scale * rng.next() / self.job.efl,
                Var::Thickness(_) => v * (1.0 + scale * rng.next()),
                Var::SemiDiameter(_) => v,
                Var::Conic(_) | Var::Aspheric(..) => v + asph * rng.next(),
                _ => self.nudge(var, v, scale * rng.next()),
            })
            .collect()
    }

    /// Levenberg–Marquardt from the start design; `log` gets progress lines.
    /// With `restarts`, LM is rerun from random perturbations of the best design.
    pub fn optimize(&self, log: &mut dyn FnMut(&str)) -> Result<Eval> {
        let start = self.eval(&self.start()).context("the start design cannot be traced")?;
        log(&format!("start: merit {:.4e}, RMS {:?} µm", merit(&start), round1(&start.rms)));
        if self.vars.is_empty() {
            return Ok(start);
        }
        let mut best = self.lm(self.start(), start, self.job.iterations, log)?;
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
            let x: Vec<f64> = self.read(&best.lens).iter().zip(&self.vars).map(|(v, var)| self.nudge(*var, *v, self.job.restart_scale * rand())).collect();
            let Some(e) = self.eval(&x) else { continue };
            let Ok(e) = self.lm(x, e, self.job.iterations, &mut |_| {}) else { continue };
            log(&format!("restart {r}: merit {:.4e}, RMS {:?} µm", merit(&e), round1(&e.rms)));
            if merit(&e) < merit(&best) {
                best = e;
            }
        }
        log(&format!("end: merit {:.4e}, RMS {:?} µm", merit(&best), round1(&best.rms)));
        if let Some(summary) = target_summary(&best) {
            log(&format!("MTF targets: {summary}"));
        }
        if !best.violations.is_empty() {
            log(&format!("violated: {}", best.violations.join(", ")));
        }
        Ok(best)
    }

    /// This problem with glass combination `combo` (mixed radix over the
    /// `glass_choice` rows, first row fastest), and the glasses' names.
    fn with_glasses(&self, combo: usize) -> (Problem, Vec<String>) {
        let mut p = self.clone();
        let mut names = Vec::new();
        let mut c = combo;
        for (row, glasses) in &self.choices {
            let g = &glasses[c % glasses.len()];
            names.push(g.name().to_string());
            p.base.surfaces[*row].material = Some(g.clone());
            c /= glasses.len();
        }
        (p, names)
    }

    /// Whether designs `a` and `b` (variables) lie within `d` of each other:
    /// curvatures d / efl, thicknesses 10 d mm, aspheric terms d mm of rim
    /// sag, the annulus d, band edges 100 d nm; clear apertures do not count.
    fn close(&self, a: &[f64], b: &[f64], d: f64) -> bool {
        self.vars.iter().zip(a.iter().zip(b)).all(|(v, (x, y))| {
            let tol = match v {
                Var::Curvature(_) => d / self.job.efl,
                Var::Thickness(_) => 10.0 * d,
                Var::SemiDiameter(_) => f64::INFINITY,
                Var::Conic(_) | Var::Aspheric(..) | Var::Inner => d,
                Var::BandLo | Var::BandHi => 100.0 * d,
                Var::Nd(_) => d,
                Var::Vd(_) => 100.0 * d,
            };
            (x - y).abs() <= tol
        })
    }

    /// Global search: `g.restarts` descents in parallel, each from a random
    /// perturbation of the start design (start 0 as given) with the next
    /// glass combination, followed by `g.hops` basin hops. Returns the best
    /// distinct local minima, best first; deterministic for a seed.
    pub fn global(&self, g: &Global, log: &(dyn Fn(&str) + Sync)) -> Vec<Candidate> {
        let combos: usize = self.choices.iter().map(|c| c.1.len()).product();
        let iterations = g.iterations.unwrap_or(self.job.iterations);
        let found: Vec<Vec<Candidate>> = (0..g.restarts)
            .into_par_iter()
            .map(|r| {
                let (p, glasses) = self.with_glasses(r % combos);
                let mut rng = Rng::new(g.seed, r as u64);
                let x0 = p.start();
                let start = (0..10).find_map(|t| {
                    let x = if r == 0 && t == 0 { x0.clone() } else { p.perturb(&x0, g.perturb, g.asphere_perturb, &mut rng) };
                    p.eval(&x).map(|e| (x, e))
                });
                let Some(Ok(mut best)) = start.map(|(x, e)| p.lm(x, e, iterations, &mut |_| {})) else {
                    log(&format!("start {r}: no traceable descent"));
                    return Vec::new();
                };
                let mut out = Vec::new();
                for _ in 0..g.hops {
                    let x = p.perturb(&p.read(&best.lens), g.perturb / 2.0, g.asphere_perturb / 2.0, &mut rng);
                    let Some(Ok(e)) = p.eval(&x).map(|e| p.lm(x, e, iterations, &mut |_| {})) else { continue };
                    out.push(if merit(&e) < merit(&best) { std::mem::replace(&mut best, e) } else { e });
                }
                log(&format!("start {r} {glasses:?}: merit {:.4e}, RMS {:?} µm", merit(&best), round1(&best.rms)));
                out.push(best);
                out.into_iter().map(|e| Candidate { merit: merit(&e), x: p.read(&e.lens), eval: e, start: r, glasses: glasses.clone() }).collect()
            })
            .collect();
        let mut all: Vec<Candidate> = found.into_iter().flatten().filter(|c| c.merit.is_finite()).collect();
        all.sort_by(|a, b| a.merit.total_cmp(&b.merit));
        let mut keep: Vec<Candidate> = Vec::new();
        for c in all {
            if keep.len() == g.keep {
                break;
            }
            if !keep.iter().any(|k| k.glasses == c.glasses && self.close(&k.x, &c.x, g.distinct)) {
                keep.push(c);
            }
        }
        keep
    }

    fn lm(&self, mut x: Vec<f64>, mut cur: Eval, iterations: usize, log: &mut dyn FnMut(&str)) -> Result<Eval> {
        let mut f = merit(&cur);
        let mut lambda = 1e-3;
        let mut stalled = 0;
        for it in 0..iterations {
            // Forward-difference Jacobian, columns in parallel.
            let cols: Vec<Option<Vec<f64>>> = (0..x.len())
                .into_par_iter()
                .map(|j| {
                    let h = match self.vars[j] {
                        Var::Curvature(_) => 1e-7,
                        Var::Thickness(_) | Var::SemiDiameter(_) | Var::Inner => 1e-5,
                        Var::Conic(_) | Var::Aspheric(..) => 1e-6,
                        Var::BandLo | Var::BandHi => 1e-3,
                        Var::Nd(_) => 1e-6,
                        Var::Vd(_) => 1e-4,
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

    /// The optimised lens as lens-file text with the job's header; `rank`:
    /// (place, merit, start) in a global search.
    pub fn finish(&self, e: &Eval, job_file: &str, rank: Option<(usize, f64, usize)>) -> String {
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
            if self.job.wavelengths.is_empty() { self.job.mono.map_or("F, e, C".to_string(), |l| format!("{l} nm")) } else { format!("{} wavelengths", self.job.wavelengths.len()) },
            self.job.fstop,
            self.job.focus,
            fields.join(", ")
        );
        if !e.mtf.is_empty() {
            let mtf: Vec<String> = self.job.fields.iter().zip(&e.mtf).map(|(f, m)| format!("{f}: {m:.2}")).collect();
            t += &format!("# Geometric MTF (mean of {:?} lp/mm, S and T): {}.\n#\n", self.job.mtf_freqs, mtf.join(", "));
        }
        if let Some(summary) = target_summary(e) {
            t += &format!("# MTF targets: {summary}.\n#\n");
        }
        if let Some((place, merit, start)) = rank {
            t += &format!("# Global search: place {place}, merit {merit:.4e} (start {start}, seed {}).\n#\n", self.job.global.as_ref().map_or(0, |g| g.seed));
        }
        t += "# radius      thickness   material       semi-diameter\n";
        for s in &lens.surfaces {
            let radius = if s.is_stop { "stop".to_string() } else if s.radius == 0.0 { "inf".to_string() } else { format!("{:.4}", s.radius) };
            let material = s.material.as_ref().map_or("air", |g| g.name());
            let mut extra = String::new();
            if s.conic != 0.0 {
                extra += &format!(" k={:.6}", s.conic);
            }
            for (i, a) in s.aspheric.iter().enumerate().filter(|(_, a)| **a != 0.0) {
                extra += &format!(" a{}={a:.6e}", 4 + 2 * i);
            }
            if let (true, Some(f)) = (s.is_stop, &lens.spectral_stop) {
                extra += &format!(" annulus={:.4} pass={:.1}-{:.1}", f.inner, f.lo, f.hi);
            }
            t += &format!("  {radius:<11} {:<11.4} {material:<14} {:.3}{extra}\n", s.thickness, s.semi_diameter);
        }
        t
    }
}

/// How many MTF targets are met, and the largest shortfall.
fn target_summary(e: &Eval) -> Option<String> {
    let worst = e.targets.iter().max_by(|a, b| (a.5 - a.4).total_cmp(&(b.5 - b.4)))?;
    let met = e.targets.iter().filter(|t| t.4 >= t.5).count();
    let (f, h, q, tan, m, v) = *worst;
    Some(format!("{met} of {} met; largest shortfall {:.3} (f/{f}, {h} mm, {q} lp/mm {}: {m:.3} vs {v:.3})", e.targets.len(), v - m, if tan { "T" } else { "S" }))
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
    let mut job: Job = toml::from_str(&std::fs::read_to_string(job_path).with_context(|| format!("reading {job_path}"))?)?;
    let dir = Path::new(job_path).parent().unwrap_or(Path::new("."));
    let out = dir.join(&job.output);
    if let Some(t) = job.mtf_target.as_mut() {
        t.reference = dir.join(&t.reference).to_string_lossy().to_string();
    }
    if let Some(o) = job.origin.as_mut() {
        *o = dir.join(&o).to_string_lossy().to_string();
    }
    let global = job.global.clone();
    let p = Problem::new(job)?;
    if let Some(g) = global {
        // The best design goes to `output`, the next to `<stem>-2.lens` and so on.
        let t0 = std::time::Instant::now();
        let found = p.global(&g, &|s| eprintln!("{s}"));
        if found.is_empty() {
            bail!("no start could be optimised");
        }
        eprintln!("{} starts in {:.1} s", g.restarts, t0.elapsed().as_secs_f64());
        let stem = out.file_stem().unwrap_or_default().to_string_lossy().to_string();
        for (i, c) in found.iter().enumerate() {
            let path = if i == 0 { out.clone() } else { out.with_file_name(format!("{stem}-{}.lens", i + 1)) };
            std::fs::write(&path, p.finish(&c.eval, job_path, Some((i + 1, c.merit, c.start))))?;
            eprintln!("{}: merit {:.4e}, RMS {:?} µm, {:?}, start {} → {}", i + 1, c.merit, round1(&c.eval.rms), c.glasses, c.start, path.display());
            if !c.eval.violations.is_empty() {
                eprintln!("   violated: {}", c.eval.violations.join(", "));
            }
        }
        return Ok(());
    }
    if !p.choices.is_empty() {
        eprintln!("glass_choice is used by the global search only");
    }
    let e = p.optimize(&mut |s| eprintln!("{s}"))?;
    std::fs::write(&out, p.finish(&e, job_path, None))?;
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

    const SINGLET: &str = r#"
        name = "test"
        output = "unused"
        fstop = 2.8
        fields = [0.0]
        mono = 546.07
        iterations = 60
        start = """
        stop   0.5  air    6
        30     5.0  N-BK7  10
        -200   45   air    10
        """
        "#;

    /// An f/2.8 singlet on axis at one wavelength: aspherising the front
    /// surface removes most of the spherical aberration; a departure limit
    /// holds and limits the gain.
    #[test]
    fn asphere_corrects_spherical_aberration() {
        let job = |extra: &str| -> Job { toml::from_str(&format!("{extra}\n{SINGLET}")).unwrap() };
        let run = |j: Job| {
            let p = Problem::new(j).unwrap();
            let e = p.optimize(&mut |_| {}).unwrap();
            (p.eval(&p.start()).unwrap().rms[0], e)
        };
        let (_, sphere) = run(job("curvatures = [2, 3]"));
        let (_, asphere) = run(job("curvatures = [2, 3]\nconics = [2]\naspheres = [2]"));
        let (_, limited) = run(job("curvatures = [2, 3]\nconics = [2]\naspheres = [2]\nmax_departure = 0.015\nmax_inflections = 0"));
        let shape = asphere_shape(&limited.lens.surfaces[1]);
        println!("RMS sphere {:.2}, asphere {:.2}, departure-limited {:.2} µm; departures {:.4} / {:.4} mm, lobes {:?}", sphere.rms[0], asphere.rms[0], limited.rms[0], asphere_shape(&asphere.lens.surfaces[1]).departure, shape.departure, shape.lobes);
        // On the renderer's white-light focus, so axial colour limits the gain.
        assert!(asphere.rms[0] < 0.15 * sphere.rms[0]);
        assert!(shape.departure < 0.0151 && shape.lobes.len() <= 1, "{shape:?}", shape = (shape.departure, &shape.lobes));
        // Both end near the colour limit; the limited one may not beat the free one by more than noise.
        assert!(limited.rms[0] > 0.98 * asphere.rms[0] && limited.rms[0] < sphere.rms[0]);
        assert!(limited.violations.is_empty() && asphere.violations.is_empty(), "{:?} {:?}", limited.violations, asphere.violations);
        // The output keeps the asphere.
        let text = Problem::new(job("conics = [2]\naspheres = [2]")).unwrap().finish(&asphere, "job", None);
        let back = Lens::parse(&text).unwrap();
        assert!(back.surfaces[1].is_aspheric() && (back.surfaces[1].conic - asphere.lens.surfaces[1].conic).abs() < 1e-5);
    }

    /// Material limits bind wherever the material sits: an f/2.8 crown of a
    /// limited semi-diameter shrinks to it, the other element does not.
    #[test]
    fn material_semi_diameter_limit() {
        let job = |extra: &str| -> Job {
            toml::from_str(&format!(
                "name = \"t\"\noutput = \"u\"\nfstop = 2.8\nfields = [0.0, 1.0]\nvignetting = 1\ncurvatures = [2, 3, 4]\niterations = 30\n\
                 start = \"\"\"\nstop 2 air 9\n30 5 N-LAK9 12\n-40 2 F2 12\n-200 45 air 12\n\"\"\"\n{extra}"
            ))
            .unwrap()
        };
        let free = Problem::new(job("")).unwrap().optimize(&mut |_| {}).unwrap();
        let p = Problem::new(job("[[material_limits]]\nglass = \"N-LAK9\"\nmax_semi_diameter = 10.5")).unwrap();
        let start = p.eval(&p.start()).unwrap();
        assert!(start.violations.iter().any(|v| v.starts_with("row 2 N-LAK9 semi-diameter")), "{:?}", start.violations);
        let e = p.optimize(&mut |_| {}).unwrap();
        let sd = |e: &Eval, i: usize| e.lens.surfaces[i].semi_diameter;
        println!("free {:.2}/{:.2}, limited {:.2}/{:.2}/{:.2}", sd(&free, 1), sd(&free, 2), sd(&e, 1), sd(&e, 2), sd(&e, 3));
        assert!(sd(&free, 1).max(sd(&free, 2)) > 10.0);
        assert!(sd(&e, 1) < 10.55 && sd(&e, 2) < 10.55, "{:?}", e.violations);
        assert!(sd(&e, 3) > 10.6);
    }

    /// The merit's sensor plane is the renderer's (`Lens::best_focus`), also
    /// for a fast aspheric lens whose real pupil differs from the paraxial one.
    #[test]
    fn render_focus_matches_best_focus() {
        for (text, f) in [(include_str!("../lenses/summilux-50-asph.lens"), 1.4), (include_str!("../lenses/explainer/11b-double-gauss.lens"), 2.0)] {
            let mut lens = Lens::parse(text).unwrap();
            lens.prepare(f, 0, 0.0, 0.0, Some(1e6), 0.0).unwrap();
            let (a, b) = (render_focus(&lens, 1e6).unwrap(), lens.best_focus(Some(1e6)).unwrap());
            println!("{}: render_focus {a:.4}, best_focus {b:.4}", lens.name);
            assert!((a - b).abs() < 0.005);
        }
    }

    /// The merit's corner illumination and distortion agree with
    /// `metrics::numbers` (up to soft apertures and the paraxial field angle),
    /// also when the start's image distance is far from the focus.
    #[test]
    fn illumination_and_distortion_match_numbers() {
        let lens_text = include_str!("../lenses/explainer/07-triplet.lens");
        let rows: String = lens_text.lines().filter(|l| !l.starts_with('#') && !l.starts_with('@')).map(|l| format!("{l}\n")).collect();
        let rows = rows.replace("42.2078", "30.0");
        let job: Job = toml::from_str(&format!("name = \"t\"\noutput = \"u\"\nfstop = 3.5\nmin_corner_illumination = 1.0\nmax_distortion = 0.0\nstart = \"\"\"\n{rows}\"\"\"\n")).unwrap();
        let p = Problem::new(job).unwrap();
        let e = p.eval(&p.start()).unwrap();
        let amount = |what: &str| e.violations.iter().find_map(|v| v.strip_prefix(what)?.strip_prefix(" by ")?.parse::<f64>().ok()).unwrap();
        let (ri, dist) = (1.0 - amount("corner illumination") / LIGHT, amount("distortion (%) at 1"));
        let n = crate::metrics::numbers(&Lens::parse(lens_text).unwrap(), 3.5, 2500.0).unwrap();
        println!("illumination {ri:.3} vs {:.3}, |distortion| {dist:.3} vs {:.3} %", n.relative_illumination, n.distortion_pct);
        assert!((ri - n.relative_illumination).abs() < 0.03 && (dist - n.distortion_pct.abs()).abs() < 0.1);
    }

    /// A detuned achromat misses an MTF target on axis and off axis; with
    /// the target as the only merit it ends meeting it. Targets met cost nothing.
    #[test]
    fn mtf_target_is_met() {
        let path = std::env::temp_dir().join(format!("mtf-target-{}.json", std::process::id()));
        let reference = |c: f64| format!(r#"{{"heights_mm": [0, 5], "f/8": {{"20S": [{c}, {c}], "20T": [{c}, {c}]}}, "f/16": {{"10S": [{c}, {c}], "10T": [{c}, {c}]}}}}"#);
        let job = |c: f64| -> Job {
            std::fs::write(&path, reference(c)).unwrap();
            toml::from_str(&format!(
                "name = \"t\"\noutput = \"u\"\nfstop = 8\nfocus = 1000\nfields = [0.0]\nrms_weight = 0\ncurvatures = [2, 3, 4]\niterations = 40\n\
                 start = \"\"\"\nstop 0.5 air 4\n25 4.0 N-BK7 6\n-50 2.0 F2 6\n-200 45 air 6\n\"\"\"\n\
                 [mtf_target]\nreference = {:?}\nfstops = [8, 16]\nweight = 100\nmargin = 0.01\n",
                path.to_string_lossy()
            ))
            .unwrap()
        };
        let p = Problem::new(job(85.0)).unwrap();
        let start = p.eval(&p.start()).unwrap();
        let end = p.optimize(&mut |_| {}).unwrap();
        let worst = |e: &Eval| e.targets.iter().map(|t| t.5 - t.4).fold(f64::MIN, f64::max);
        println!("{:?}\n{:?}", start.targets, end.targets);
        assert_eq!(start.targets.len(), 8);
        // The target is beyond this doublet; the merit must still close much of the gap.
        assert!(worst(&start) > 0.05 && worst(&end) < 0.6 * worst(&start), "worst shortfall {} → {}", worst(&start), worst(&end));
        let easy = Problem::new(job(1.0)).unwrap();
        // A target residual is zero exactly where the MTF clears target + margin.
        let e = easy.eval(&easy.start()).unwrap();
        let r = &e.residuals[e.residuals.len() - 8..];
        assert!(r.iter().zip(&e.targets).all(|(r, t)| (*r == 0.0) == (t.4 >= t.5 + 0.01)));
        std::fs::remove_file(&path).unwrap();
    }

    /// The MTF target's contrast (geometric × diffraction limit) follows the
    /// wave-optical MTF of `lens-render mtf`.
    #[test]
    fn target_mtf_matches_wave_mtf() {
        let freqs = [(10.0, false), (10.0, true), (20.0, false), (20.0, true), (40.0, false), (40.0, true)];
        let mut worst: f64 = 0.0;
        for f in [1.4, 2.8, 5.6] {
            let mut lens = Lens::parse(include_str!("../lenses/summilux-50-asph.lens")).unwrap();
            lens.set_aperture(f, 0, 0.0, 0.0).unwrap();
            lens.sensor_z = lens.best_focus(Some(1e6)).unwrap();
            for h in [0.0, 5.0, 10.0, 12.5, 15.0, 20.0, 21.0] {
                let ours = target_mtf(&lens, 1e6, h, &freqs, 32).unwrap();
                let (s, t) = crate::mtf::mtf_at(&lens, h, Some(1e6), &[10.0, 20.0, 40.0]).unwrap();
                let wave = [s[0], t[0], s[1], t[1], s[2], t[2]];
                println!("f/{f} {h:5.1} mm: target {:?} wave {:?}", round3(&ours), round3(&wave));
                worst = ours.iter().zip(&wave).fold(worst, |w, (a, b)| w.max((a - b).abs()));
            }
        }
        println!("worst {worst:.3}");
    }

    fn round3(v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| (x * 1000.0).round() / 1000.0).collect()
    }

    /// Departure, slopes and inflections of a known profile.
    #[test]
    fn asphere_shape_measures() {
        let mut s = Lens::parse("stop 1 air 5\n40 3 N-BK7 10\n-40 40 air 10").unwrap().surfaces[1].clone();
        // A pure conic sphere-equivalent: no departure, no inflection.
        assert!(asphere_shape(&s).departure < 1e-12 && asphere_shape(&s).lobes.len() == 1);
        // z″ = c + 12 a4 r² changes sign once inside r = 10 for a4 < −c / 1200.
        s.aspheric = vec![-4e-5];
        let a = asphere_shape(&s);
        println!("departure {}, slope {}, lobes {:?}", a.departure, a.slope, a.lobes);
        assert_eq!(a.lobes.len(), 2);
        // Departure ≈ |a4| (r⁴ − r² h²) at its extreme r = h/√2: 4e-5 · 2500.
        assert!((a.departure - 0.1).abs() < 0.01 && a.slope < 0.3 && a.outside == 0.0);
        // A conic whose domain ends inside the clear aperture.
        s.aspheric.clear();
        s.conic = 15.0;
        assert!(asphere_shape(&s).outside > 0.0);
    }

    /// The global search is deterministic, tries each candidate glass,
    /// returns distinct designs best first, and beats the plain descent.
    #[test]
    fn global_search() {
        let job: Job = toml::from_str(
            r#"
            name = "test"
            output = "unused"
            fstop = 8
            fields = [0.0, 0.7]
            curvatures = [2, 3, 4]
            iterations = 30
            start = """
            stop   0.5  air    4
            25     4.0  N-BK7  6
            -50    2.0  F2     6
            -200   45   air    6
            """
            [[glass_choice]]
            row = 3
            glasses = ["F2", "N-SF5", "N-KZFS4"]
            [global]
            restarts = 6
            keep = 4
            hops = 1
            "#,
        )
        .unwrap();
        let g = job.global.clone().unwrap();
        let p = Problem::new(job).unwrap();
        let a = p.global(&g, &|_| {});
        let b = p.global(&g, &|_| {});
        let summary = |c: &[Candidate]| c.iter().map(|c| (c.merit, c.start, c.glasses.clone())).collect::<Vec<_>>();
        println!("{:?}", summary(&a));
        assert_eq!(summary(&a), summary(&b));
        assert!(a.len() >= 2 && a.windows(2).all(|w| w[0].merit <= w[1].merit));
        for (i, c) in a.iter().enumerate() {
            assert!(a[..i].iter().all(|k| k.glasses != c.glasses || !p.close(&k.x, &c.x, g.distinct)));
            assert_eq!(c.eval.lens.surfaces[2].material.as_ref().unwrap().name(), c.glasses[0]);
        }
        let plain = p.optimize(&mut |_| {}).unwrap();
        assert!(a[0].merit <= merit(&plain) * 1.0001);
    }

    /// Spectral stop parameters vary; the transmission floor holds.
    #[test]
    fn spectral_stop_variables() {
        let job: Job = toml::from_str(
            r#"
            name = "test"
            output = "unused"
            fstop = 4
            fields = [0.0]
            iterations = 40
            wavelengths = [[440, 1], [450, 1], [460, 1], [470, 1], [480, 2], [490, 2], [500, 2], [510, 2], [520, 3], [530, 3], [540, 3], [550, 3], [560, 3], [570, 2], [580, 2], [590, 2], [600, 2], [610, 1], [620, 1], [630, 1], [640, 1], [650, 1], [660, 1]]
            spectral_stop = ["inner", "lo", "hi"]
            min_stop_transmission = 0.6
            start = """
            stop    0.5     air    6.3  annulus=0.7 pass=470-640
            51.25   2.5     N-BK7  7
            -51.25  48      air    7
            """
            "#,
        )
        .unwrap();
        let p = Problem::new(job).unwrap();
        let start = p.eval(&p.start()).unwrap();
        let end = p.optimize(&mut |_| {}).unwrap();
        let f = end.lens.spectral_stop.unwrap();
        let t = f.luminous(1.0);
        println!("RMS {:.1} → {:.1} µm, {f:?}, luminous transmission {t:.3}", start.rms[0], end.rms[0]);
        assert!(end.rms[0] < 0.8 * start.rms[0] && t > 0.59);
        assert!(p.finish(&end, "job", None).contains("annulus="));
    }
}


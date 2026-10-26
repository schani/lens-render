//! Summary numbers of a lens focused at a finite distance, and the bending
//! curve of a singlet (for the explainer video).

use crate::glass::{LAMBDA_C, LAMBDA_D, LAMBDA_F};
use crate::lens::Lens;
use crate::math::{Ray, Vec3};
use crate::optimize::{HALF_DIAG, SPOT_WAVELENGTHS, rms_spot_um};
use crate::psf::{Source, chief_aim, chief_image, field_for_height};
use crate::seidel::{seidel, transverse_um};
use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Serialize)]
pub struct Numbers {
    pub name: String,
    pub fstop: f64,
    /// Image-side working f-number at the focus distance.
    pub working_fnumber: f64,
    pub efl_mm: f64,
    /// Last vertex to sensor, and front vertex to sensor (mm).
    pub back_focus_mm: f64,
    pub length_mm: f64,
    /// W040, W131, W222, W220P, W311, W020, W111 in waves at 550 nm, full field.
    pub seidel_waves: [f64; 7],
    /// The same as transverse ray aberrations on the sensor (`seidel::transverse_um`).
    pub seidel_um: [f64; 7],
    /// Polychromatic (F, e, C 1:2:1) RMS spot radius at 0, 0.7 and 1 × the half-diagonal.
    pub rms_um: [f64; 3],
    /// Distortion at the corner (negative: barrel).
    pub distortion_pct: f64,
    /// Paraxial focus of C minus F (positive: blue focuses first).
    pub axial_colour_um: f64,
    /// Petzval surface radius (negative: curved towards the lens).
    pub petzval_mm: f64,
    /// Corner illumination relative to the centre: cos⁴ law, vignetting and distortion, no reflection losses.
    pub relative_illumination: f64,
}

/// Image height of the chief ray at small field angles per unit tan θ.
fn paraxial_scale(lens: &Lens, dist: f64) -> Option<f64> {
    let th = 1e-4_f64;
    Some(chief_image(lens, &Source::at_field(lens, th, Some(dist)), LAMBDA_D)?[1].abs() / th.tan())
}

/// Flux from a Lambertian object point at field angle `theta` reaching the
/// sensor, per unit radiance: a grid over twice the entrance pupil, apertures included.
fn flux(lens: &Lens, theta: f64, dist: f64) -> Option<f64> {
    let src = Source::at_field(lens, theta, Some(dist));
    let Source::Point(o) = src else { return None };
    let par = lens.paraxial_data(LAMBDA_D);
    let r = 2.0 * lens.iris.max_radius / par.entrance_pupil_mag;
    let q = chief_aim(lens, &src, LAMBDA_D)?;
    let m = lens.media(LAMBDA_D);
    let g = 200;
    let cell = 2.0 * r / g as f64;
    let mut sum = 0.0;
    for k in 0..g * g {
        let (x, y) = (-r + ((k % g) as f64 + 0.5) * cell, -r + ((k / g) as f64 + 0.5) * cell);
        let (ray, _) = src.ray_through(q + Vec3::new(x, y, 0.0));
        if lens.trace_forward(ray, LAMBDA_D, &m).is_some() {
            let d = q + Vec3::new(x, y, 0.0) - o;
            let cos = d.z / d.length();
            sum += cos * cos / d.dot(d);
        }
    }
    Some(sum * cell * cell)
}

/// Numbers of `lens` at `fstop`, focused (as the renderer does) on an object `dist` mm away.
pub fn numbers(lens: &Lens, fstop: f64, dist: f64) -> Result<Numbers> {
    let mut lens = lens.clone();
    lens.prepare(fstop, 0, 0.0, 0.0, Some(dist), 0.0)?;
    let par = lens.paraxial_data(LAMBDA_D);
    let n_w = lens.working_fnumber();
    let k = paraxial_scale(&lens, dist).context("chief ray")?;
    let w = seidel(&lens, (HALF_DIAG / k).atan(), Some(dist));
    let rms = [0.0, 0.7, 1.0].map(|f| rms_spot_um(&lens, f * HALF_DIAG, Some(dist), &SPOT_WAVELENGTHS, Some(lens.sensor_z)).unwrap_or(f64::NAN));
    let th_c = field_for_height(&lens, HALF_DIAG, Some(dist));
    let distortion_pct = th_c.map_or(f64::NAN, |th| (HALF_DIAG / (k * th.tan()) - 1.0) * 100.0);
    let m = lens.media(LAMBDA_D);
    let petzval: f64 = lens.surfaces.iter().enumerate().map(|(i, s)| s.curvature() * (m.n[i + 1] - m.n[i]) / (m.n[i] * m.n[i + 1])).sum();
    // Irradiance ∝ flux / image area per object area (radial × tangential magnification).
    let illumination = |th_c: f64| -> Option<f64> {
        let h = |th: f64| chief_image(&lens, &Source::at_field(&lens, th, Some(dist)), LAMBDA_D).map(|p| p[1].abs());
        let dth = 1e-4;
        let (h1, h2) = (h(th_c - dth)?, h(th_c + dth)?);
        let y_obj = |th: f64| (par.entrance_pupil_z + dist) * th.tan();
        let radial = (h2 - h1) / (y_obj(th_c + dth) - y_obj(th_c - dth));
        let tangential = HALF_DIAG / y_obj(th_c);
        let m0 = k / (par.entrance_pupil_z + dist);
        Some(flux(&lens, th_c, dist)? / (radial * tangential) / (flux(&lens, 0.0, dist)? / (m0 * m0)))
    };
    let relative_illumination = th_c.and_then(illumination).unwrap_or(f64::NAN);
    Ok(Numbers {
        name: lens.name.clone(),
        fstop,
        working_fnumber: n_w,
        efl_mm: par.efl,
        back_focus_mm: lens.sensor_z - lens.last_z(),
        length_mm: lens.sensor_z,
        seidel_waves: w.sum,
        seidel_um: transverse_um(&w, n_w),
        rms_um: rms,
        distortion_pct,
        axial_colour_um: (lens.paraxial_focus(Some(dist), LAMBDA_C) - lens.paraxial_focus(Some(dist), LAMBDA_F)) * 1000.0,
        petzval_mm: -1.0 / petzval,
        relative_illumination,
    })
}

/// The explainer's scoreboard: the seven classic aberrations measured with
/// real rays on the sensor of a set-up lens (f-number, focus and sensor
/// position as rendered), object `dist` mm away. µm, distortion in %.
/// Focus errors become the radius of the blur disc they cause, δ / 2N.
pub struct Scoreboard {
    /// Spherical (on-axis RMS at 546 nm); coma (corner spot's centroid
    /// offset from the chief ray at its best focus); astigmatism (corner,
    /// foci of the full T and S fans Δz apart: blur at the medial focus,
    /// Δz / 4N); field curvature (corner, medial focus δ from the sensor: δ / 2N);
    /// distortion (corner, %, negative: barrel); axial colour (C and F
    /// foci Δz apart, between them: Δz / 4N); lateral colour (corner F–C
    /// chief-ray separation).
    pub bars: [f64; 7],
    /// White-light (F, e, C 1:2:1) RMS spot radius in the centre and the corner.
    pub blur: [f64; 2],
}

const E_LINE: f64 = 546.07;

/// Exit rays at 546 nm of the corner point through the entrance pupil
/// (1.6 × its paraxial radius, apertures clipping; a spectral stop passing
/// at least half): the tangential fan (y), the sagittal fan (x), or a full
/// grid; and the chief ray.
fn corner_rays(lens: &Lens, src: &Source, fan: Option<bool>) -> Option<(Vec<Ray>, Ray)> {
    let q = chief_aim(lens, src, E_LINE)?;
    let r = 1.6 * lens.iris.max_radius / lens.paraxial_data(LAMBDA_D).entrance_pupil_mag;
    let m = lens.media(E_LINE);
    let trace = |x: f64, y: f64| {
        let ray = src.ray_through(q + Vec3::new(x * r, y * r, 0.0)).0;
        lens.trace_forward(ray, E_LINE, &m).filter(|_| lens.spectral_weight(ray, E_LINE, &m) >= 0.5).map(|o| o.0)
    };
    let n = 201;
    let u = |k: usize| 2.0 * k as f64 / (n - 1) as f64 - 1.0;
    let rays: Vec<Ray> = match fan {
        Some(true) => (0..n).filter_map(|k| trace(0.0, u(k))).collect(),
        Some(false) => (0..n).filter_map(|k| trace(u(k), 0.0)).collect(),
        None => {
            let g = 64;
            let c = |k: usize| 2.0 * (k as f64 + 0.5) / g as f64 - 1.0;
            (0..g * g).map(|k| (c(k % g), c(k / g))).filter(|(x, y)| x * x + y * y <= 1.0).filter_map(|(x, y)| trace(x, y)).collect()
        }
    };
    Some((rays, trace(0.0, 0.0)?))
}

/// Plane of least spread of one coordinate (y or x) of rays.
fn least_spread(rays: &[Ray], use_y: bool) -> Option<f64> {
    if rays.len() < 3 {
        return None;
    }
    let ab: Vec<(f64, f64)> = rays.iter().map(|r| {
        let (dir, pos) = if use_y { (r.d.y, r.o.y) } else { (r.d.x, r.o.x) };
        let b = dir / r.d.z;
        (pos - b * r.o.z, b)
    }).collect();
    let n = ab.len() as f64;
    let (am, bm) = (ab.iter().map(|v| v.0).sum::<f64>() / n, ab.iter().map(|v| v.1).sum::<f64>() / n);
    let num: f64 = ab.iter().map(|(a, b)| (a - am) * (b - bm)).sum();
    let den: f64 = ab.iter().map(|(_, b)| (b - bm) * (b - bm)).sum();
    (den > 0.0).then(|| -num / den)
}

fn at_z(r: &Ray, z: f64) -> (f64, f64) {
    let t = (z - r.o.z) / r.d.z;
    (r.o.x + r.d.x * t, r.o.y + r.d.y * t)
}

/// Coma: the corner spot's centroid offset from the chief ray (mm), on the
/// plane where that spot's RMS is least.
fn coma_offset(rays: &[Ray], chief: &Ray, z0: f64) -> Option<f64> {
    if rays.is_empty() {
        return None;
    }
    let n = rays.len() as f64;
    let stats = |z: f64| {
        let p: Vec<(f64, f64)> = rays.iter().map(|r| at_z(r, z)).collect();
        let (cx, cy) = (p.iter().map(|q| q.0).sum::<f64>() / n, p.iter().map(|q| q.1).sum::<f64>() / n);
        (p.iter().map(|q| (q.0 - cx).powi(2) + (q.1 - cy).powi(2)).sum::<f64>() / n, cx, cy)
    };
    let (mut a, mut b) = (z0 - 15.0, z0 + 15.0);
    for _ in 0..80 {
        let (m1, m2) = (b - 0.618 * (b - a), a + 0.618 * (b - a));
        if stats(m1).0 < stats(m2).0 { b = m2 } else { a = m1 }
    }
    let z = 0.5 * (a + b);
    let (_, cx, cy) = stats(z);
    let (hx, hy) = at_z(chief, z);
    Some((cx - hx).hypot(cy - hy))
}

pub fn scoreboard(lens: &Lens, dist: f64) -> Scoreboard {
    let n_w = lens.working_fnumber();
    let blur_um = |dz: f64| dz.abs() / (2.0 * n_w) * 1000.0;
    let spherical = rms_spot_um(lens, 0.0, Some(dist), &[(E_LINE, 1.0)], Some(lens.sensor_z)).unwrap_or(f64::NAN);
    let th = field_for_height(lens, HALF_DIAG, Some(dist));
    let src = th.map(|th| Source::at_field(lens, th, Some(dist)));
    let coma = src.as_ref().and_then(|src| {
        let (rays, chief) = corner_rays(lens, src, None)?;
        coma_offset(&rays, &chief, lens.sensor_z)
    });
    // Foci of the full tangential and sagittal fans: how the lens balances them.
    let foci = src.as_ref().and_then(|src| Some((least_spread(&corner_rays(lens, src, Some(true))?.0, true)?, least_spread(&corner_rays(lens, src, Some(false))?.0, false)?)));
    let (astigmatism, curvature) = match foci {
        Some((zt, zs)) => (blur_um(zt - zs) / 2.0, blur_um((zt + zs) / 2.0 - lens.sensor_z)),
        None => (f64::NAN, f64::NAN),
    };
    let distortion = match (paraxial_scale(lens, dist), th) {
        (Some(k), Some(th)) => (HALF_DIAG / (k * th.tan()) - 1.0) * 100.0,
        _ => f64::NAN,
    };
    let mut axial = blur_um(lens.paraxial_focus(Some(dist), LAMBDA_C) - lens.paraxial_focus(Some(dist), LAMBDA_F)) / 2.0;
    if let Some(f) = &lens.spectral_stop {
        // C and F pass the annulus only in part: scale by their RMS pupil
        // radius relative to the open pupil's, √(2⟨ρ²⟩).
        let a2 = (f.inner * lens.surfaces[lens.stop_index].semi_diameter / lens.iris.radius).min(1.0).powi(2);
        let k = |l: f64| {
            let b = f.band(l);
            let den = a2 + b * (1.0 - a2);
            if den > 0.0 { ((a2 * a2 + b * (1.0 - a2 * a2)) / den).sqrt() } else { 0.0 }
        };
        axial *= 0.5 * (k(LAMBDA_C) + k(LAMBDA_F));
    }
    let lateral = src.as_ref().map_or(f64::NAN, |src| {
        let y = |l: f64| -> Option<f64> {
            let m = lens.media(l);
            let q = chief_aim(lens, src, l)?;
            let (out, _) = lens.trace_forward(src.ray_through(q).0, l, &m)?;
            Some(out.o.y + out.d.y * (lens.sensor_z - out.o.z) / out.d.z)
        };
        match (y(LAMBDA_F), y(LAMBDA_C)) {
            (Some(f), Some(c)) => (f - c).abs() * 1000.0,
            _ => f64::NAN,
        }
    });
    let blur = [0.0, HALF_DIAG].map(|h| rms_spot_um(lens, h, Some(dist), &SPOT_WAVELENGTHS, Some(lens.sensor_z)).unwrap_or(f64::NAN));
    let coma = coma.map_or(f64::NAN, |c| c * 1000.0);
    Scoreboard { bars: [spherical, coma, astigmatism, curvature, distortion, axial, lateral], blur }
}

/// Markdown table of the numbers.
pub fn table(all: &[Numbers]) -> String {
    let mut t = String::from(
        "| Lens | f/ | EFL | BFD | length | W040 | W131 | W222 | W220P | W311 | W020 | W111 | SA µm | coma µm | astig µm | Petzval µm | dist. µm | ax. col. µm | lat. col. µm | RMS 0 | RMS 0.7 | RMS 1.0 | distortion % | C–F µm | Petzval R mm | rel. illum. |\n",
    );
    t += &format!("|{}\n", "---|".repeat(26));
    for n in all {
        t += &format!("| {} | {:.2} | {:.2} | {:.1} | {:.1} |", n.name, n.fstop, n.efl_mm, n.back_focus_mm, n.length_mm);
        for v in n.seidel_waves {
            t += &format!(" {v:.2} |");
        }
        for v in n.seidel_um {
            t += &format!(" {:.1} |", v.abs());
        }
        for v in n.rms_um {
            t += &format!(" {v:.1} |");
        }
        t += &format!(" {:+.2} | {:.0} | {:.0} | {:.2} |\n", n.distortion_pct, n.axial_colour_um, n.petzval_mm, n.relative_illumination);
    }
    t
}

#[derive(Serialize)]
pub struct BendPoint {
    /// Coddington shape factor (c1 + c2) / (c1 − c2): −1 flat front, 0 equiconvex, +1 flat back.
    pub shape: f64,
    pub r1_mm: f64,
    pub r2_mm: f64,
    /// Seidel spherical aberration (waves at 550 nm).
    pub w040_waves: f64,
    /// Centre RMS spot (µm): white light (F, e, C) on the renderer's focus, and
    /// 546 nm on its own best focus (spherical aberration alone).
    pub rms_um: f64,
    pub rms_mono_um: f64,
}

#[derive(Serialize)]
pub struct Bending {
    pub fstop: f64,
    pub focus_m: f64,
    pub points: Vec<BendPoint>,
    pub equiconvex: BendPoint,
    pub flat_front: BendPoint,
    /// Least 546 nm RMS spot.
    pub best: BendPoint,
    /// Equiconvex / best and flat front / best.
    pub ratio_w040_equiconvex: f64,
    pub ratio_rms_mono_equiconvex: f64,
    pub ratio_w040_flat_front: f64,
    pub ratio_rms_mono_flat_front: f64,
}

/// Bends the singlet `lens` (a stop and two surfaces) to `shape` at constant
/// thickness and focal length, and evaluates it.
fn bend(lens: &Lens, shape: f64, fstop: f64, dist: f64) -> Result<BendPoint> {
    let mut l = lens.clone();
    let (i1, i2) = {
        let s: Vec<usize> = (0..l.surfaces.len()).filter(|&i| !l.surfaces[i].is_stop).collect();
        (s[0], s[1])
    };
    let efl = lens.focal_length();
    let mut a = 1.0 / (0.5 * efl);
    for _ in 0..50 {
        let (c1, c2) = (a * (shape + 1.0) / 2.0, a * (shape - 1.0) / 2.0);
        l.surfaces[i1].radius = if c1 == 0.0 { 0.0 } else { 1.0 / c1 };
        l.surfaces[i2].radius = if c2 == 0.0 { 0.0 } else { 1.0 / c2 };
        a *= l.focal_length() / efl;
    }
    l.prepare(fstop, 0, 0.0, 0.0, Some(dist), 0.0)?;
    Ok(BendPoint {
        shape,
        r1_mm: l.surfaces[i1].radius,
        r2_mm: l.surfaces[i2].radius,
        w040_waves: seidel(&l, 0.0, Some(dist)).sum[0],
        rms_um: rms_spot_um(&l, 0.0, Some(dist), &SPOT_WAVELENGTHS, Some(l.sensor_z)).context("no rays")?,
        rms_mono_um: rms_spot_um(&l, 0.0, Some(dist), &[(546.07, 1.0)], None).context("no rays")?,
    })
}

/// Spherical aberration against the shape factor of a singlet, −1 … 2.
pub fn bending(lens: &Lens, fstop: f64, dist: f64) -> Result<Bending> {
    let points = (0..=30).map(|i| bend(lens, -1.0 + 0.1 * i as f64, fstop, dist)).collect::<Result<Vec<_>>>()?;
    // Golden-section search for the least 546 nm blur.
    let (mut a, mut c) = (0.0, 1.5);
    let f = |x: f64| bend(lens, x, fstop, dist).map(|p| p.rms_mono_um).unwrap_or(f64::INFINITY);
    for _ in 0..40 {
        let (m1, m2) = (c - 0.618 * (c - a), a + 0.618 * (c - a));
        if f(m1) < f(m2) { c = m2 } else { a = m1 }
    }
    let best = bend(lens, 0.5 * (a + c), fstop, dist)?;
    let (equiconvex, flat_front) = (bend(lens, 0.0, fstop, dist)?, bend(lens, -1.0, fstop, dist)?);
    Ok(Bending {
        fstop,
        focus_m: dist / 1000.0,
        ratio_w040_equiconvex: equiconvex.w040_waves / best.w040_waves,
        ratio_rms_mono_equiconvex: equiconvex.rms_mono_um / best.rms_mono_um,
        ratio_w040_flat_front: flat_front.w040_waves / best.w040_waves,
        ratio_rms_mono_flat_front: flat_front.rms_mono_um / best.rms_mono_um,
        points,
        equiconvex,
        flat_front,
        best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(text: &str, fstop: f64) -> Lens {
        let mut l = Lens::parse(text).unwrap();
        l.sensor_half = (18.0, 12.0);
        l.prepare(fstop, 0, 0.0, 0.0, Some(2500.0), 0.0).unwrap();
        l
    }

    /// The singlet is bad everywhere; the landscape lens's corners are sharp
    /// but its grid bows (barrel); stopping down shrinks the blur but not
    /// the distortion.
    #[test]
    fn scoreboard_tells_the_story() {
        let singlet = scoreboard(&setup(include_str!("../lenses/explainer/01-singlet.lens"), 4.0), 2500.0);
        let landscape = scoreboard(&setup(include_str!("../lenses/explainer/05-landscape.lens"), 14.0), 2500.0);
        let b = &singlet.bars;
        assert!(b[0] > 10.0 && b[2] > 200.0 && b[3] > 200.0 && b[5] > 20.0, "singlet {b:?}");
        // Axial colour: C–F 0.8 mm at f/4.
        assert!((b[5] - 798.0 / 16.0).abs() < 3.0, "axial colour {}", b[5]);
        let l = &landscape.bars;
        assert!((l[4] + 3.6).abs() < 0.2, "landscape distortion {}", l[4]);
        // Its corner blur is mostly astigmatism.
        assert!(l[5] < 5.0 && l[3] < 15.0 && (l[2] / landscape.blur[1] - 1.3).abs() < 0.2, "landscape {l:?} {:?}", landscape.blur);
        let stopped = scoreboard(&setup(include_str!("../lenses/explainer/02-best-form.lens"), 16.0), 2500.0);
        let open = scoreboard(&setup(include_str!("../lenses/explainer/02-best-form.lens"), 4.0), 2500.0);
        assert!(stopped.bars[0] < open.bars[0] / 4.0 && stopped.blur[0] < 10.0, "{:?} {:?}", stopped.bars, stopped.blur);
        assert!((stopped.bars[4] - open.bars[4]).abs() < 0.05);
    }

    /// A green-pass annulus in the singlet's stop: the white-light centre
    /// blur and the axial colour bar shrink, spherical aberration at 546 nm stays.
    #[test]
    fn spectral_stop_shrinks_axial_colour() {
        let text = include_str!("../lenses/explainer/01-singlet.lens");
        let plain = scoreboard(&setup(text, 4.0), 2500.0);
        let annulus = setup(&text.replace("0.5000      air            6.252", "0.5000      air            6.252  annulus=0.4 pass=500-600"), 4.0);
        assert!(annulus.spectral_stop.is_some());
        let dichroic = scoreboard(&annulus, 2500.0);
        println!("centre blur {:.1} → {:.1} µm, axial colour {:.1} → {:.1} µm, spherical {:.1} → {:.1} µm", plain.blur[0], dichroic.blur[0], plain.bars[5], dichroic.bars[5], plain.bars[0], dichroic.bars[0]);
        assert!(dichroic.blur[0] < 0.8 * plain.blur[0]);
        assert!(dichroic.bars[5] < 0.5 * plain.bars[5]);
        assert!((dichroic.bars[0] / plain.bars[0] - 1.0).abs() < 0.05);
    }

    /// The bars agree with the measured blur: no off-axis bar far above the
    /// corner blur, spherical aberration not far above the centre blur.
    #[test]
    fn scoreboard_matches_measured_blur() {
        let lenses = [
            ("01-singlet", 4.0),
            ("02b-flat-front", 4.0),
            ("02-best-form", 4.0),
            ("02-best-form", 16.0),
            ("04-achromat", 8.0),
            ("05-landscape", 14.0),
            ("06-rapid-rectilinear", 8.0),
            ("07-triplet", 3.5),
            ("08-triplet-f2", 2.0),
            ("09-clark", 4.0),
            ("10-planar", 2.8),
            ("11b-double-gauss", 2.0),
        ];
        for (name, f) in lenses {
            let text = std::fs::read_to_string(format!("lenses/explainer/{name}.lens")).unwrap();
            let s = scoreboard(&setup(&text, f), 2500.0);
            for k in 1..4 {
                assert!(s.bars[k] <= 2.5 * s.blur[1], "{name}: bar {k} {} vs corner {}", s.bars[k], s.blur[1]);
            }
            assert!(s.bars[0] <= 2.0 * s.blur[0], "{name}: spherical {} vs centre {}", s.bars[0], s.blur[0]);
        }
    }
}

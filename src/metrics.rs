//! Summary numbers of a lens focused at a finite distance, and the bending
//! curve of a singlet (for the explainer video).

use crate::glass::{LAMBDA_C, LAMBDA_D, LAMBDA_F};
use crate::lens::Lens;
use crate::math::Vec3;
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
    /// Spherical (on-axis RMS at 546 nm); coma (corner, tangential:
    /// mid-point of the rim rays from the chief ray); astigmatism (corner,
    /// T and S foci Δz apart: blur at the medial focus, Δz / 4N); field
    /// curvature (corner, medial focus δ from the sensor: δ / 2N);
    /// distortion (corner, %, negative: barrel); axial colour (C and F
    /// foci Δz apart, between them: Δz / 4N); lateral colour (corner F–C
    /// chief-ray separation).
    pub bars: [f64; 7],
    /// White-light (F, e, C 1:2:1) RMS spot radius in the centre and the corner.
    pub blur: [f64; 2],
}

const E_LINE: f64 = 546.07;

/// Tangential fan of the corner point at 546 nm: (pupil fraction, sensor height)
/// of the rays that pass, from −1.5 to 1.5 times the paraxial pupil radius.
fn corner_fan(lens: &Lens, src: &Source) -> Vec<(f64, f64)> {
    let Some(q) = chief_aim(lens, src, E_LINE) else { return Vec::new() };
    let par = lens.paraxial_data(LAMBDA_D);
    let r = lens.iris.max_radius / par.entrance_pupil_mag;
    let m = lens.media(E_LINE);
    (0..=600)
        .filter_map(|k| {
            let u = 1.5 * (k as f64 / 300.0 - 1.0);
            let (out, _) = lens.trace_forward(src.ray_through(q + Vec3::new(0.0, u * r, 0.0)).0, E_LINE, &m)?;
            Some((u, out.o.y + out.d.y * (lens.sensor_z - out.o.z) / out.d.z))
        })
        .collect()
}

/// Real-ray tangential and sagittal foci (z) of a narrow bundle around the chief ray.
fn ts_foci(lens: &Lens, src: &Source) -> Option<(f64, f64)> {
    let q = chief_aim(lens, src, E_LINE)?;
    let r = 0.25 * lens.iris.max_radius / lens.paraxial_data(LAMBDA_D).entrance_pupil_mag;
    let m = lens.media(E_LINE);
    // z where a pair of rays symmetric about the chief ray cross, in y (T) or x (S).
    let cross = |d: Vec3, y: bool| -> Option<f64> {
        let a = lens.trace_forward(src.ray_through(q + d).0, E_LINE, &m)?.0;
        let b = lens.trace_forward(src.ray_through(q - d).0, E_LINE, &m)?.0;
        let (pa, sa, pb, sb) = if y { (a.o.y, a.d.y / a.d.z, b.o.y, b.d.y / b.d.z) } else { (a.o.x, a.d.x / a.d.z, b.o.x, b.d.x / b.d.z) };
        let (pa, pb) = (pa - sa * a.o.z, pb - sb * b.o.z);
        Some((pb - pa) / (sa - sb))
    };
    // Narrower pairs where vignetting cuts the wider ones.
    let at = |y: bool| [1.0, 0.5, 0.2].iter().find_map(|f| cross(if y { Vec3::new(0.0, f * r, 0.0) } else { Vec3::new(f * r, 0.0, 0.0) }, y));
    Some((at(true)?, at(false)?))
}

pub fn scoreboard(lens: &Lens, dist: f64) -> Scoreboard {
    let n_w = lens.working_fnumber();
    let blur_um = |dz: f64| dz.abs() / (2.0 * n_w) * 1000.0;
    let spherical = rms_spot_um(lens, 0.0, Some(dist), &[(E_LINE, 1.0)], Some(lens.sensor_z)).unwrap_or(f64::NAN);
    let th = field_for_height(lens, HALF_DIAG, Some(dist));
    let src = th.map(|th| Source::at_field(lens, th, Some(dist)));
    let coma = src.as_ref().map_or(f64::NAN, |src| {
        let fan = corner_fan(lens, src);
        // The rays that pass on either side of the chief ray, without gaps.
        let chief = fan.iter().min_by(|a, b| a.0.abs().total_cmp(&b.0.abs())).copied();
        match (chief, fan.first(), fan.last()) {
            (Some((u0, y0)), Some(lo), Some(hi)) if u0.abs() < 0.01 => ((lo.1 + hi.1) / 2.0 - y0).abs() * 1000.0,
            _ => f64::NAN,
        }
    });
    let (astigmatism, curvature) = match src.as_ref().and_then(|src| ts_foci(lens, src)) {
        Some((zt, zs)) => (blur_um(zt - zs) / 2.0, blur_um((zt + zs) / 2.0 - lens.sensor_z)),
        None => (f64::NAN, f64::NAN),
    };
    let distortion = match (paraxial_scale(lens, dist), th) {
        (Some(k), Some(th)) => (HALF_DIAG / (k * th.tan()) - 1.0) * 100.0,
        _ => f64::NAN,
    };
    let axial = blur_um(lens.paraxial_focus(Some(dist), LAMBDA_C) - lens.paraxial_focus(Some(dist), LAMBDA_F)) / 2.0;
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
        assert!(l[5] < 5.0 && l[3] < 10.0 && (l[2] / landscape.blur[1] - 1.4).abs() < 0.2, "landscape {l:?} {:?}", landscape.blur);
        let stopped = scoreboard(&setup(include_str!("../lenses/explainer/02-best-form.lens"), 16.0), 2500.0);
        let open = scoreboard(&setup(include_str!("../lenses/explainer/02-best-form.lens"), 4.0), 2500.0);
        assert!(stopped.bars[0] < open.bars[0] / 4.0 && stopped.blur[0] < 10.0, "{:?} {:?}", stopped.bars, stopped.blur);
        assert!((stopped.bars[4] - open.bars[4]).abs() < 0.05);
    }
}

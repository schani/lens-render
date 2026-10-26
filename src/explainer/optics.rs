//! What the explainer shows about a lens besides its image: rays for the
//! cross-section, third-order aberrations as blur on the sensor, measured
//! blur, image surfaces, and the undistorted chart grid.

use crate::glass::LAMBDA_D;
use crate::lens::Lens;
use crate::math::{Ray, Vec3};
use crate::psf::{Source, chief_aim, chief_image, field_for_height, meridional_fan};
use crate::scene::{Scene, Texture};

/// Wavelengths (nm) for polychromatic blur, weighted by luminance.
const SPOT_WAVELENGTHS: [f64; 7] = [440.0, 480.0, 520.0, 555.0, 590.0, 630.0, 670.0];
/// Wavelengths and their names for rays drawn by colour.
pub const RGB_WAVELENGTHS: [f64; 3] = [460.0, 550.0, 640.0];

/// Lens `a` with curvatures, thicknesses and semi-diameters moved towards
/// `b` (same rows) by `t`; glasses stay `a`'s.
pub fn morph(a: &Lens, b: &Lens, t: f64) -> Lens {
    if t >= 1.0 {
        return b.clone();
    }
    let mut l = a.clone();
    if a.surfaces.len() != b.surfaces.len() {
        return l;
    }
    let mix = |x: f64, y: f64| x + (y - x) * t;
    let mut z = 0.0;
    for (s, sb) in l.surfaces.iter_mut().zip(&b.surfaces) {
        let c = mix(s.curvature(), sb.curvature());
        s.radius = if c.abs() < 1e-12 { 0.0 } else { 1.0 / c };
        s.thickness = mix(s.thickness, sb.thickness);
        s.semi_diameter = mix(s.semi_diameter, sb.semi_diameter);
        s.z = z;
        z += s.thickness;
    }
    l
}

/// Fraction of the paraxial entrance pupil the real marginal ray of an
/// axial point passes through (pupil aberration makes it a little less than 1).
fn rim(lens: &Lens, object: Option<f64>) -> f64 {
    let src = Source::at_field(lens, 0.0, object);
    let m = lens.media(LAMBDA_D);
    let passes = |s: f64| meridional_fan(lens, &src, 2, s).is_some_and(|r| r.iter().all(|r| lens.trace_forward(*r, LAMBDA_D, &m).is_some()));
    let (mut lo, mut hi) = (0.5, 0.995);
    if passes(hi) {
        return hi;
    }
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        if passes(mid) { lo = mid } else { hi = mid }
    }
    lo * 0.998
}

/// A ray's path through the lens to the sensor plane, and whether it gets there.
pub struct RayPath {
    pub pts: Vec<Vec3>,
    pub passed: bool,
    /// Wavelength (nm) and position across the pupil (−1…1).
    pub lambda: f64,
    pub pupil: f64,
}

/// `count` meridional rays from the object point whose image is at
/// `frac` × `half_diag`, started at `z_start`, at each wavelength.
pub fn fan(lens: &Lens, object: Option<f64>, frac: f64, half_diag: f64, count: usize, wavelengths: &[f64], z_start: f64) -> Vec<RayPath> {
    let Some(theta) = field_for_height(lens, frac * half_diag, object) else { return Vec::new() };
    let Some(rays) = meridional_fan(lens, &Source::at_field(lens, theta, object), count, rim(lens, object)) else { return Vec::new() };
    let mut out = Vec::new();
    for (k, ray) in rays.iter().enumerate() {
        let mut r = *ray;
        r.o = r.o + r.d * ((z_start - r.o.z) / r.d.z);
        for &l in wavelengths {
            let (mut pts, exit) = lens.trace_points(r, l);
            if let Some(e) = exit.filter(|e| e.d.z > 0.0) {
                pts.push(e.o + e.d * ((lens.sensor_z - e.o.z) / e.d.z));
            }
            out.push(RayPath { pts, passed: exit.is_some(), lambda: l, pupil: 2.0 * k as f64 / (count - 1).max(1) as f64 - 1.0 });
        }
    }
    out
}

/// Rays leaving the lens from the object point imaged at height `h` (mm),
/// through a g × g grid over the entrance pupil, with their wavelengths.
fn bundle(lens: &Lens, object: Option<f64>, h: f64, g: usize, wavelengths: &[f64]) -> Vec<(Ray, f64)> {
    let Some(theta) = field_for_height(lens, h, object) else { return Vec::new() };
    let src = Source::at_field(lens, theta, object);
    let Some(q) = chief_aim(lens, &src, LAMBDA_D) else { return Vec::new() };
    let par = lens.paraxial_data(LAMBDA_D);
    let r = 1.05 * lens.iris.max_radius / par.entrance_pupil_mag;
    let mut out = Vec::new();
    for &l in wavelengths {
        let m = lens.media(l);
        for k in 0..g * g {
            let (x, y) = ((2.0 * ((k % g) as f64 + 0.5) / g as f64 - 1.0) * r, (2.0 * ((k / g) as f64 + 0.5) / g as f64 - 1.0) * r);
            if x * x + y * y > r * r {
                continue;
            }
            if let Some((ray, _)) = lens.trace_forward(src.ray_through(q + Vec3::new(x, y, 0.0)).0, l, &m) {
                out.push((ray, l));
            }
        }
    }
    out
}

/// Luminance-weighted RMS radius (mm) of rays on the plane z.
fn rms(rays: &[(Ray, f64)], z: f64) -> f64 {
    let pts: Vec<(f64, f64, f64)> = rays
        .iter()
        .map(|(r, l)| {
            let t = (z - r.o.z) / r.d.z;
            (r.o.x + r.d.x * t, r.o.y + r.d.y * t, crate::spectrum::cie_xyz(*l)[1])
        })
        .collect();
    let w: f64 = pts.iter().map(|p| p.2).sum();
    if w <= 0.0 {
        return f64::NAN;
    }
    let cx = pts.iter().map(|p| p.0 * p.2).sum::<f64>() / w;
    let cy = pts.iter().map(|p| p.1 * p.2).sum::<f64>() / w;
    (pts.iter().map(|p| p.2 * ((p.0 - cx).powi(2) + (p.1 - cy).powi(2))).sum::<f64>() / w).sqrt()
}

/// Sensor shift (mm) from its position that minimises the RMS blur at image height `h`.
pub fn best_offset(lens: &Lens, object: Option<f64>, h: f64) -> f64 {
    let rays = bundle(lens, object, h, 16, &SPOT_WAVELENGTHS);
    let (mut a, mut b) = (-12.0, 12.0);
    let g = 0.5 * (5f64.sqrt() - 1.0);
    for _ in 0..50 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if rms(&rays, lens.sensor_z + c) < rms(&rays, lens.sensor_z + d) {
            b = d;
        } else {
            a = c;
        }
    }
    0.5 * (a + b)
}

/// Plane minimising the spread of one coordinate of rays (x or y).
fn variance_focus(rays: &[Ray], use_y: bool) -> Option<f64> {
    let ab: Vec<(f64, f64)> = rays
        .iter()
        .map(|r| {
            let (dir, pos) = if use_y { (r.d.y, r.o.y) } else { (r.d.x, r.o.x) };
            let b = dir / r.d.z;
            (pos - b * r.o.z, b)
        })
        .collect();
    if ab.len() < 4 {
        return None;
    }
    let n = ab.len() as f64;
    let am = ab.iter().map(|v| v.0).sum::<f64>() / n;
    let bm = ab.iter().map(|v| v.1).sum::<f64>() / n;
    let num: f64 = ab.iter().map(|(a, b)| (a - am) * (b - bm)).sum();
    let den: f64 = ab.iter().map(|(_, b)| (b - bm) * (b - bm)).sum();
    (den > 0.0).then(|| -num / den)
}

/// The drawn fan's rays as they leave the lens (the ones that pass).
fn exit_rays(paths: &[RayPath]) -> Vec<Ray> {
    paths
        .iter()
        .filter(|r| r.passed && r.pts.len() >= 2)
        .map(|r| {
            let (a, b) = (r.pts[r.pts.len() - 2], r.pts[r.pts.len() - 1]);
            Ray { o: a, d: (b - a).normalize() }
        })
        .collect()
}

/// The sagittal counterpart of `fan`: rays across the pupil perpendicular
/// to the meridional plane, as they leave the lens.
fn sagittal_rays(lens: &Lens, object: Option<f64>, h: f64, count: usize) -> Vec<Ray> {
    let Some(theta) = field_for_height(lens, h, object) else { return Vec::new() };
    let src = Source::at_field(lens, theta, object);
    let Some(q) = chief_aim(lens, &src, LAMBDA_D) else { return Vec::new() };
    let r = rim(lens, object) * lens.iris.max_radius / lens.paraxial_data(LAMBDA_D).entrance_pupil_mag;
    let m = lens.media(LAMBDA_D);
    (0..count)
        .filter_map(|k| {
            let x = r * (2.0 * k as f64 / (count - 1) as f64 - 1.0);
            lens.trace_forward(src.ray_through(q + Vec3::new(x, 0.0, 0.0)).0, LAMBDA_D, &m).map(|o| o.0)
        })
        .collect()
}

/// Rays per fan in the cross-section, and in the fans that define the image surfaces.
pub const FAN_RAYS: usize = 11;

/// Image surfaces against image height h: [h, T, S, Petzval] as focus
/// shifts (mm) from the sensor, then the heights (mm) at which the T and S
/// fans actually converge (their foci lie on the inclined chief ray, not at h).
/// T and S are the least-spread planes of the full-aperture meridional fan
/// drawn in the cross-section and of its sagittal counterpart (d line);
/// Petzval is third-order theory.
pub fn field_curves(lens: &Lens, object: Option<f64>, half_diag: f64) -> Vec<[f64; 6]> {
    let m = lens.media(LAMBDA_D);
    let petzval: f64 = lens.surfaces.iter().enumerate().map(|(i, s)| s.curvature() * (m.n[i + 1] - m.n[i]) / (m.n[i] * m.n[i + 1])).sum();
    (0..=12)
        .filter_map(|k| {
            let frac = k as f64 / 12.0;
            let h = half_diag * frac;
            let t = exit_rays(&fan(lens, object, frac, half_diag, FAN_RAYS, &[LAMBDA_D], lens.sensor_z - 300.0));
            let s = sagittal_rays(lens, object, h, FAN_RAYS);
            let (zt, zs) = (variance_focus(&t, true)?, variance_focus(&s, false)?);
            let mean_y = |rays: &[Ray], z: f64| rays.iter().map(|r| r.o.y + r.d.y * (z - r.o.z) / r.d.z).sum::<f64>() / rays.len() as f64;
            Some([h, zt - lens.sensor_z, zs - lens.sensor_z, -petzval * h * h / 2.0, mean_y(&t, zt), mean_y(&s, zs)])
        })
        .collect()
}

/// Where a drawn fan (passing rays) comes to its tangential focus: (z, y).
pub fn fan_focus(paths: &[RayPath]) -> Option<(f64, f64)> {
    let rays = exit_rays(paths);
    let z = variance_focus(&rays, true)?;
    let y = rays.iter().map(|r| r.o.y + r.d.y * (z - r.o.z) / r.d.z).sum::<f64>() / rays.len() as f64;
    Some((z, y))
}

/// Shape factor (c1 + c2) / (c1 − c2) of the first element.
pub fn shape_factor(lens: &Lens) -> Option<f64> {
    let i = lens.surfaces.iter().position(|s| s.material.is_some())?;
    let (c1, c2) = (lens.surfaces[i].curvature(), lens.surfaces.get(i + 1)?.curvature());
    (c1 != c2).then(|| (c1 + c2) / (c1 - c2))
}

/// The chart's grid lines as they would be imaged without distortion, as
/// segments in normalised image coordinates (x right, y down).
pub fn ideal_grid(lens: &Lens, scene: &Scene, sensor: (f64, f64)) -> Vec<[[f32; 2]; 2]> {
    let Some(r) = scene.rects.iter().filter(|r| matches!(r.texture, Texture::Chart { .. })).max_by(|a, b| (a.w * a.h).total_cmp(&(b.w * b.h))) else { return Vec::new() };
    // Paraxial magnification for the chart's plane, from a nearly axial chief ray.
    let eps = 1.0;
    let Some(img) = chief_image(lens, &Source::Point(Vec3::new(0.0, eps, r.center.z)), LAMBDA_D) else { return Vec::new() };
    let mag = img[1] / eps;
    let to_image = |p: Vec3| -> [f32; 2] { [(0.5 - mag * p.x / sensor.0) as f32, (0.5 + mag * p.y / sensor.1) as f32] };
    // Chart texture: 16 lines per chart height, square cells (see scene::chart).
    let at = |x: f64, y: f64| r.center + r.u * (x * r.h - 0.5 * r.w) + r.v * ((y - 0.5) * r.h);
    let aspect = r.w / r.h;
    let mut out = Vec::new();
    for k in 0..=(16.0 * aspect) as usize {
        let x = k as f64 / 16.0;
        out.push([to_image(at(x, 0.0)), to_image(at(x, 1.0))]);
    }
    for k in 0..=16 {
        let y = k as f64 / 16.0;
        out.push([to_image(at(0.0, y)), to_image(at(aspect, y))]);
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The tangential image surface at the corner is where the drawn
    /// meridional fan comes to focus.
    #[test]
    fn tangential_curve_meets_the_drawn_fan() {
        for (name, f) in [("01-singlet", 4.0), ("04-achromat", 8.0), ("05-landscape", 14.0), ("06-rapid-rectilinear", 8.0), ("11b-double-gauss", 2.0)] {
            let mut l = Lens::load(&format!("lenses/explainer/{name}.lens")).unwrap();
            l.prepare(f, 0, 0.0, 0.0, Some(2500.0), 0.0).unwrap();
            let corner = field_curves(&l, Some(2500.0), 21.63).last().copied().unwrap();
            let drawn = fan(&l, Some(2500.0), 1.0, 21.63, FAN_RAYS, &[LAMBDA_D], l.sensor_z - 300.0);
            let (z, y) = fan_focus(&drawn).unwrap();
            assert!((corner[1] - (z - l.sensor_z)).abs() < 0.1, "{name}: T {} vs fan focus {}", corner[1], z - l.sensor_z);
            assert!((corner[4] - y).abs() < 0.1, "{name}: T drawn at height {} vs fan focus {y}", corner[4]);
        }
    }
}

//! Pupil footprints (vignetting) and transmission losses per surface.

use crate::glass::LAMBDA_D;
use crate::lens::{Coating, Lens};
use crate::math::Vec3;
use crate::psf::{Source, chief_aim, field_for_height};
use rayon::prelude::*;

/// Image heights of the footprints, as fractions of the half diagonal.
pub const FIELD_FRACTIONS: [f64; 5] = [0.0, 0.5, 0.7, 0.85, 1.0];

/// Rays through a square grid on the entrance-pupil plane for one field.
pub struct Footprint {
    /// Image height as a fraction of the half diagonal, and in mm.
    pub fraction: f64,
    pub height: f64,
    /// Field angle (radians).
    pub theta: f64,
    pub grid: usize,
    /// Grid centre (the chief ray's aim point) and cell size on the entrance-pupil plane (mm).
    pub centre: [f64; 2],
    pub cell: f64,
    /// Row-major `grid` × `grid`, row 0 at +y, column 0 at −x: None where the
    /// ray passes, otherwise the surface blocking it.
    pub cells: Vec<Option<usize>>,
    /// Passing area relative to the on-axis footprint.
    pub relative_area: f64,
}

pub struct Footprints {
    pub fields: Vec<Footprint>,
    /// Paraxial entrance pupil radius (mm).
    pub pupil_radius: f64,
    pub stop_index: usize,
}

/// Surface that stopped a ray, from the points `Lens::trace_points` reached.
fn blocking_surface(lens: &Lens, pts: &[Vec3]) -> usize {
    if pts.len() >= 2 {
        let k = pts.len() - 2;
        let s = &lens.surfaces[k];
        let p = pts[k + 1];
        let (x, y) = (p.x - s.decenter.0, p.y - s.decenter.1);
        if x * x + y * y > s.semi_diameter * s.semi_diameter || (s.is_stop && !lens.iris.contains(x, y)) {
            return k;
        }
    }
    // The ray missed the next surface (or was totally reflected there).
    (pts.len() - 1).min(lens.surfaces.len() - 1)
}

/// Footprints for image heights `FIELD_FRACTIONS` × `half_diag` mm, object
/// at `focus` mm (None = infinity), on a `grid` × `grid` raster spanning
/// ±1.3 paraxial entrance pupil radii around the chief ray (d line). Fields
/// whose chief ray cannot be found are left out.
pub fn footprints(lens: &Lens, focus: Option<f64>, half_diag: f64, grid: usize) -> Footprints {
    let par = lens.paraxial_data(LAMBDA_D);
    let pupil_radius = lens.iris.max_radius / par.entrance_pupil_mag;
    let half = 1.3 * pupil_radius;
    let cell = 2.0 * half / grid as f64;
    let mut fields: Vec<Footprint> = FIELD_FRACTIONS
        .iter()
        .filter_map(|&fraction| {
            let height = fraction * half_diag;
            let theta = field_for_height(lens, height, focus)?;
            let src = Source::at_field(lens, theta, focus);
            let q = chief_aim(lens, &src, LAMBDA_D)?;
            let cells = (0..grid * grid)
                .into_par_iter()
                .map(|k| {
                    let (i, j) = (k % grid, k / grid);
                    let p = Vec3::new(q.x - half + (i as f64 + 0.5) * cell, q.y + half - (j as f64 + 0.5) * cell, q.z);
                    let (pts, out) = lens.trace_points(src.ray_through(p).0, LAMBDA_D);
                    if out.is_some() { None } else { Some(blocking_surface(lens, &pts)) }
                })
                .collect();
            Some(Footprint { fraction, height, theta, grid, centre: [q.x, q.y], cell, cells, relative_area: 0.0 })
        })
        .collect();
    let passing = |f: &Footprint| f.cells.iter().filter(|c| c.is_none()).count() as f64;
    let axis = fields.iter().find(|f| f.fraction == 0.0).map_or(0.0, passing);
    for f in &mut fields {
        f.relative_area = if axis > 0.0 { passing(f) / axis } else { 0.0 };
    }
    Footprints { fields, pupil_radius, stop_index: lens.stop_index }
}

/// Throughput along one ray.
#[derive(Default)]
pub struct Transmission {
    /// Cumulative transmission after each surface.
    pub cumulative: Vec<f64>,
    /// Reflectance of each surface.
    pub reflectance: Vec<f64>,
}

impl Transmission {
    pub fn total(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or(0.0)
    }
}

/// Names of the rays in `LightLoss`.
pub const LIGHT_LOSS_RAYS: [&str; 2] = ["axis, 0.7 pupil", "full-field chief"];

/// Wavelength (nm) of the light-loss traces.
pub const LIGHT_LOSS_LAMBDA: f64 = 550.0;

/// Fresnel losses with the lens's coating and uncoated, for the rays of
/// `LIGHT_LOSS_RAYS` (an empty `Transmission` if a ray cannot be aimed).
pub struct LightLoss {
    pub coated: [Transmission; 2],
    pub uncoated: [Transmission; 2],
    pub stop_index: usize,
}

fn transmission(lens: &Lens, src: &Source, q: Option<Vec3>) -> Transmission {
    let Some(q) = q else { return Transmission::default() };
    let cumulative = lens.surface_throughput(src.ray_through(q).0, LIGHT_LOSS_LAMBDA);
    let reflectance = cumulative.iter().scan(1.0, |prev, &t| {
        let r = if *prev > 0.0 { 1.0 - t / *prev } else { 0.0 };
        *prev = t;
        Some(r)
    });
    Transmission { reflectance: reflectance.collect(), cumulative }
}

/// Per-surface transmission of the on-axis ray through 0.7 of the entrance
/// pupil radius and of the chief ray at image height `half_diag` mm, object
/// at `focus` mm (None = infinity), at `LIGHT_LOSS_LAMBDA`.
pub fn light_loss(lens: &Lens, focus: Option<f64>, half_diag: f64) -> LightLoss {
    let par = lens.paraxial_data(LAMBDA_D);
    let axis = Source::at_field(lens, 0.0, focus);
    let q_axis = Some(Vec3::new(0.0, 0.7 * lens.iris.max_radius / par.entrance_pupil_mag, par.entrance_pupil_z));
    let field = field_for_height(lens, half_diag, focus).map(|th| Source::at_field(lens, th, focus));
    let q_field = field.as_ref().and_then(|s| chief_aim(lens, s, LAMBDA_D));
    let field = field.unwrap_or(axis);
    let mut bare = lens.clone();
    bare.coating = Coating::None;
    let rays = |l: &Lens| [transmission(l, &axis, q_axis), transmission(l, &field, q_field)];
    LightLoss { coated: rays(lens), uncoated: rays(&bare), stop_index: lens.stop_index }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Ray;

    fn triplet() -> Lens {
        Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap()
    }

    #[test]
    fn triplet_footprints() {
        let lens = triplet();
        let fp = footprints(&lens, None, 21.63, 64);
        assert_eq!(fp.fields.len(), FIELD_FRACTIONS.len());
        let axis = &fp.fields[0];
        assert_eq!(axis.relative_area, 1.0);
        // The on-axis beam fills the paraxial entrance pupil.
        let area = axis.cells.iter().filter(|c| c.is_none()).count() as f64 * axis.cell * axis.cell;
        let ideal = std::f64::consts::PI * fp.pupil_radius * fp.pupil_radius;
        assert!((area / ideal - 1.0).abs() < 0.05, "{area} vs {ideal}");
        // Its edge lies at the paraxial pupil radius; the iris is among the apertures bounding it.
        for (k, c) in axis.cells.iter().enumerate() {
            let (x, y) = ((k % 64) as f64 + 0.5 - 32.0, 32.0 - (k / 64) as f64 - 0.5);
            let r = x.hypot(y) * axis.cell / fp.pupil_radius;
            assert!(r > 0.95 || c.is_none(), "blocked at {r}");
            assert!(r < 1.03 || c.is_some(), "passes at {r}");
        }
        assert!(axis.cells.contains(&Some(lens.stop_index)));
        let corner = fp.fields.last().unwrap();
        assert!(corner.relative_area < 0.9, "{}", corner.relative_area);
        // At the corner the front and rear rims cut the beam as well.
        assert!(corner.cells.iter().any(|c| matches!(c, Some(s) if *s != lens.stop_index)));
    }

    #[test]
    fn blocking_surfaces() {
        let lens = triplet();
        let (pts, out) = lens.trace_points(Ray { o: Vec3::new(0.0, 12.0, -1.0), d: Vec3::new(0.0, 0.0, 1.0) }, LAMBDA_D);
        assert!(out.is_none());
        assert_eq!(blocking_surface(&lens, &pts), 0);
        let r_ep = lens.iris.max_radius / lens.paraxial_data(LAMBDA_D).entrance_pupil_mag;
        let (pts, out) = lens.trace_points(Ray { o: Vec3::new(0.0, 1.05 * r_ep, -1.0), d: Vec3::new(0.0, 0.0, 1.0) }, LAMBDA_D);
        assert!(out.is_none());
        assert_eq!(blocking_surface(&lens, &pts), lens.stop_index);
    }

    #[test]
    fn triplet_light_loss() {
        let lens = triplet();
        let ll = light_loss(&lens, None, 21.63);
        let ns = lens.surfaces.len();
        let m = lens.media(LIGHT_LOSS_LAMBDA);
        // Uncoated, near normal incidence: Π (1 − ((n − n') / (n + n'))²).
        let normal: f64 = (0..ns).map(|i| 1.0 - ((m.n[i] - m.n[i + 1]) / (m.n[i] + m.n[i + 1])).powi(2)).product();
        for k in 0..2 {
            let (c, u) = (&ll.coated[k], &ll.uncoated[k]);
            assert_eq!((c.cumulative.len(), u.reflectance.len()), (ns, ns));
            assert_eq!(c.reflectance[lens.stop_index], 0.0);
            assert!(c.total() > u.total() + 0.1, "{} vs {}", c.total(), u.total());
            // Oblique incidence reflects a little more.
            let tol = [0.002, 0.03][k];
            assert!(u.total() < normal && u.total() > normal * (1.0 - tol), "{} vs {normal}", u.total());
            let product: f64 = u.reflectance.iter().map(|r| 1.0 - r).product();
            assert!((product - u.total()).abs() < 1e-12);
        }
    }
}

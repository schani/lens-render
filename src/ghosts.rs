//! Ghost images: light reflected at one surface, back again at an earlier
//! one, and on to the sensor.

use crate::lens::Lens;
use crate::math::Vec3;
use crate::psf::{Source, meridional_fan, pupil_rays};

pub struct Ghost {
    /// Surfaces (0-based) of the first and second reflection, a > b.
    pub a: usize,
    pub b: usize,
    /// Energy reaching the sensor relative to the direct image.
    pub energy: f64,
    /// RMS radius of the ghost on the sensor (mm).
    pub spread: f64,
    /// A meridional fan of rays along the ghost path, for drawing.
    pub fan: Vec<Vec<Vec3>>,
}

/// The ghosts of light from `src` at `l` nm, strongest first.
pub fn ghosts(lens: &Lens, src: &Source, l: f64) -> Vec<Ghost> {
    let Some(rays) = pupil_rays(lens, src, 32, 1.1) else { return Vec::new() };
    let fan = meridional_fan(lens, src, 9, 1.0).unwrap_or_default();
    let direct: f64 = rays.iter().filter_map(|r| lens.trace_path_points(*r, l, &lens.forward_path).1).map(|(_, t)| t).sum();
    if direct <= 0.0 {
        return Vec::new();
    }
    let mut out: Vec<Ghost> = lens
        .ghost_paths()
        .into_iter()
        .map(|(a, b, path)| {
            let hits: Vec<(Vec3, f64)> = rays
                .iter()
                .filter_map(|r| {
                    let (o, t) = lens.trace_path_points(*r, l, &path).1?;
                    (o.d.z > 0.0).then(|| (o.o + o.d * ((lens.sensor_z - o.o.z) / o.d.z), t))
                })
                .collect();
            let energy: f64 = hits.iter().map(|h| h.1).sum();
            let spread = if energy > 0.0 {
                let c = hits.iter().fold(Vec3::new(0.0, 0.0, 0.0), |acc, h| acc + h.0 * h.1) * (1.0 / energy);
                (hits.iter().map(|h| h.1 * ((h.0.x - c.x).powi(2) + (h.0.y - c.y).powi(2))).sum::<f64>() / energy).sqrt()
            } else {
                0.0
            };
            let fan = fan
                .iter()
                .filter_map(|r| {
                    let (mut pts, out) = lens.trace_path_points(*r, l, &path);
                    let (o, _) = out?;
                    pts.push(o.o + o.d * ((lens.sensor_z - o.o.z) / o.d.z));
                    Some(pts)
                })
                .collect();
            Ghost { a, b, energy: energy / direct, spread, fan }
        })
        .filter(|g| g.energy > 0.0)
        .collect();
    out.sort_by(|x, y| y.energy.total_cmp(&x.energy));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uncoated glass reflects ~4–6 % per surface, so a two-reflection ghost
    /// carries ~0.2–0.4 %; a coating cuts that by about an order of magnitude.
    #[test]
    fn ghost_energy_is_two_reflections() {
        let mut lens = Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap();
        lens.prepare(8.0, 0, 0.0, 0.0, None, 0.0).unwrap();
        let src = Source::Infinity(Vec3::new(0.0, 0.0, 1.0));
        lens.coating = crate::lens::Coating::None;
        let bare = ghosts(&lens, &src, 550.0);
        lens.coating = crate::lens::Coating::Single { n: 1.38, lambda0: 550.0 };
        let coated = ghosts(&lens, &src, 550.0);
        println!("bare {:.5} ({}↔{}), coated {:.5}", bare[0].energy, bare[0].a, bare[0].b, coated[0].energy);
        assert!(bare[0].energy > 1e-3 && bare[0].energy < 6e-3);
        assert!(coated[0].energy < 0.3 * bare[0].energy);
        assert!(bare.iter().all(|g| g.a > g.b));
    }
}

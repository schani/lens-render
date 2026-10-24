//! Wavefront maps and colour PSFs at a few image heights, for display.

use crate::lens::Lens;
use crate::psf::{PsfGrid, Source, Wavefront, field_for_height};
use crate::spectrum::{cie_xyz, xyz_to_linear_srgb};
use rayon::prelude::*;

pub struct FieldPsf {
    /// Image height (mm).
    pub height: f64,
    /// OPD in waves at 550 nm (piston removed) on a `map` × `map` pupil grid; None outside.
    pub opd: Vec<Option<f32>>,
    pub map: usize,
    pub rms: f64,
    pub pv: f64,
    /// Linear RGB of the white-light PSF on a `size` × `size` grid around the
    /// 550 nm chief ray, normalised to a peak of 1.
    pub rgb: Vec<[f32; 3]>,
    pub size: usize,
    /// Half width of the PSF grid (mm).
    pub half: f64,
}

const WAVELENGTHS: [f64; 7] = [440.0, 475.0, 510.0, 550.0, 590.0, 625.0, 660.0];

/// For image heights 0, 0.7 and 1 × `half_diag`, objects at `focus` (mm, None = ∞).
pub fn field_psfs(lens: &Lens, focus: Option<f64>, half_diag: f64) -> Vec<FieldPsf> {
    [0.0, 0.7, 1.0]
        .par_iter()
        .filter_map(|&frac| {
            let height = frac * half_diag;
            let src = Source::at_field(lens, field_for_height(lens, height, focus)?, focus);
            let wfs: Vec<Wavefront> = WAVELENGTHS.par_iter().filter_map(|&l| Wavefront::compute(lens, &src, l, None, 512)).collect();
            let green = wfs.iter().find(|w| w.lambda == 550.0)?;
            let grids: Vec<(f64, PsfGrid)> = wfs.iter().map(|w| (w.lambda, PsfGrid::from_wavefront(w, 2))).collect();
            let g550 = &grids.iter().find(|g| g.0 == 550.0)?.1;
            let centre = g550.reference;
            // Enough to hold the 95 % energy radius of every wavelength, lateral colour included.
            let half = grids
                .iter()
                .map(|(_, g)| {
                    let [r] = g.energy_radii([0.95]);
                    r + (g.reference[0] - centre[0]).hypot(g.reference[1] - centre[1])
                })
                .fold(0.0, f64::max)
                * 1.3;
            let size = 96;
            let cell = 2.0 * half / size as f64;
            let origin = [centre[0] - half, centre[1] - half];
            let mut rgb = vec![[0f32; 3]; size * size];
            for (l, g) in &grids {
                let c = xyz_to_linear_srgb(cie_xyz(*l)).map(|v| v.max(0.0) as f32);
                for (p, v) in rgb.iter_mut().zip(g.pixels(origin, cell, size)) {
                    for k in 0..3 {
                        p[k] += c[k] * v as f32;
                    }
                }
            }
            let peak = rgb.iter().flatten().fold(0f32, |a, &b| a.max(b)).max(1e-30);
            rgb.iter_mut().flatten().for_each(|v| *v /= peak);
            // The 550 nm wavefront, resampled.
            let map = 64;
            let waves: Vec<Option<f64>> = green.samples.iter().map(|s| s.map(|(_, o)| o / (550e-6))).collect();
            let mean = waves.iter().flatten().sum::<f64>() / waves.iter().flatten().count().max(1) as f64;
            let opd: Vec<Option<f32>> = (0..map * map)
                .map(|k| waves[(k / map * green.n / map) * green.n + k % map * green.n / map].map(|w| (w - mean) as f32))
                .collect();
            let (lo, hi) = waves.iter().flatten().fold((f64::MAX, f64::MIN), |(a, b), &w| (a.min(w), b.max(w)));
            Some(FieldPsf { height, opd, map, rms: green.rms_waves(), pv: hi - lo, rgb, size, half })
        })
        .collect()
}

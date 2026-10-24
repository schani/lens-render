//! A bank of wave-optical PSFs for image formation: field angle × object
//! distance (vergence) × iris azimuth × wavelength bin, each integrated over
//! sensor pixels and stored as a cumulative distribution for sampling.
//!
//! Scene points within the bank's depth range get their light distributed
//! by the interpolated PSF; points further out of focus keep geometric
//! optics, which is accurate once the blur is many Airy discs wide.

use crate::glass::LAMBDA_D;
use crate::lens::Lens;
use crate::psf::{Psf, Source, Wavefront};
use crate::spectrum::{LAMBDA_MIN, LAMBDA_RANGE};
use rayon::prelude::*;

pub struct PsfBank {
    /// Field samples are uniform in tan θ from 0 to `tan_max`.
    pub n_theta: usize,
    pub tan_max: f64,
    /// Vergence (1/mm, object distance measured from the entrance pupil) samples, uniform.
    pub n_verg: usize,
    pub verg0: f64,
    pub dverg: f64,
    /// Image azimuths α_a = a · period / n_az of the samples.
    pub n_az: usize,
    pub period: f64,
    pub bins: usize,
    /// Largest kernel half size in pixels (kernels cover (2·half + 1)² pixels).
    pub half: usize,
    /// Per entry, two kernels: a core at sub-pixel resolution and a wing (the
    /// tail outside the core square) at pixel resolution:
    /// [core half (px), core cells per pixel, core CDF offset, wing half (px),
    /// wing CDF offset, core energy fraction (f32 bits)].
    pub layout: Vec<[u32; 6]>,
    pub pitch: f64,
    /// Per entry: chief-ray image radius (mm).
    pub refs: Vec<f32>,
    /// Per entry: cumulative distribution over the kernel's pixels.
    pub cdf: Vec<f32>,
}

impl PsfBank {
    /// Builds the bank for the lens as configured (aperture, focus, sensor
    /// position), for objects around `focus` (mm from the front vertex).
    pub fn build(lens: &Lens, focus: Option<f64>, bins: usize, pitch: f64, half_diag: f64, progress: &(dyn Fn(f64) + Sync)) -> PsfBank {
        let par = lens.paraxial_data(LAMBDA_D);
        let ep_z = par.entrance_pupil_z;
        let n_w = lens.working_fnumber();
        let f = par.efl;
        // Depth range: until the defocus blur is ~40 Airy diameters, where
        // geometric optics takes over (the needed pupil sampling is about
        // constant in Airy units, so this costs the same at any aperture).
        let dv = 40.0 * 2.44 * 550e-6 * n_w * n_w / (f * f);
        let vf = focus.map_or(0.0, |d| 1.0 / (d + ep_z));
        let n_verg = 9;
        let v_lo = (vf - dv).max(0.0);
        let v_hi = vf + dv;
        let dverg = (v_hi - v_lo) / (n_verg - 1) as f64;
        let n_theta = 16;
        let tan_max = 1.05 * half_diag / (f * (1.0 + f * vf));
        let (n_az, period) = if lens.iris.circular { (1, std::f64::consts::TAU) } else { (4, std::f64::consts::TAU / lens.iris.blades as f64) };
        let entries: Vec<(usize, usize, usize, usize)> = (0..n_theta)
            .flat_map(|t| (0..n_verg).flat_map(move |v| (0..n_az).flat_map(move |a| (0..bins).map(move |b| (t, v, a, b)))))
            .collect();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let psfs: Vec<Option<Psf>> = entries
            .par_iter()
            .map(|&(t, v, a, b)| {
                let theta = (tan_max * t as f64 / (n_theta - 1) as f64).atan();
                let verg = v_lo + dverg * v as f64;
                let distance = if verg <= 0.0 { None } else { Some(1.0 / verg - ep_z) };
                let alpha = std::f64::consts::FRAC_PI_2 + period * a as f64 / n_az as f64;
                let l = LAMBDA_MIN + LAMBDA_RANGE * (b as f64 + 0.5) / bins as f64;
                let src = Source::at(lens, theta, alpha, distance);
                let psf = Wavefront::compute(lens, &src, l, None, 1024).map(|wf| Psf::from_wavefront(&wf, 2));
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n % 64 == 0 {
                    progress(n as f64 / entries.len() as f64);
                }
                psf
            })
            .collect();
        // Per entry: a core holding 99 % of the energy at sub-pixel resolution
        // (so that binning into film pixels is the only pixel integration),
        // and a pixel-resolution wing out to 99.9 % for the diffraction tails.
        let mut refs = Vec::with_capacity(psfs.len());
        let mut cdf = Vec::new();
        let mut layout = Vec::with_capacity(psfs.len());
        let mut max_half = 1;
        let push_cdf = |k: &[f64], cdf: &mut Vec<f32>| -> u32 {
            let off = cdf.len() as u32;
            let total: f64 = k.iter().sum();
            let mut acc = 0.0;
            cdf.extend(k.iter().map(|v| {
                acc += v / total.max(1e-300);
                acc as f32
            }));
            off
        };
        for p in &psfs {
            let Some(p) = p else {
                // Nothing passes: a delta at the reference (its weight is zero anyway).
                refs.push(0.0);
                let off = push_cdf(&[1.0], &mut cdf);
                layout.push([0, 1, off, 0, off, 1f32.to_bits()]);
                continue;
            };
            let hc = ((energy_radius(&p.samples, 0.99) / pitch).ceil() as usize + 1).min(24);
            let hw = ((energy_radius(&p.samples, 0.999) / pitch).ceil() as usize + 1).clamp(hc, 64);
            let sub = (128 / (2 * hc + 1)).clamp(1, 16);
            max_half = max_half.max(hw);
            refs.push(p.reference[0].hypot(p.reference[1]) as f32);
            let corner = |h: usize| [p.reference[0] - (h as f64 + 0.5) * pitch, p.reference[1] - (h as f64 + 0.5) * pitch];
            let core = p.pixels(corner(hc), pitch / sub as f64, (2 * hc + 1) * sub);
            let mut wing = p.pixels(corner(hw), pitch, 2 * hw + 1);
            let size = 2 * hw + 1;
            for j in hw - hc..=hw + hc {
                for i in hw - hc..=hw + hc {
                    wing[j * size + i] = 0.0;
                }
            }
            let (ec, ew): (f64, f64) = (core.iter().sum(), wing.iter().sum());
            let core_off = push_cdf(&core, &mut cdf);
            let wing_off = if ew > 0.0 { push_cdf(&wing, &mut cdf) } else { core_off };
            let fraction = if ew > 0.0 { ec / (ec + ew) } else { 1.0 };
            layout.push([hc as u32, sub as u32, core_off, hw as u32, wing_off, (fraction as f32).to_bits()]);
        }
        PsfBank { n_theta, tan_max, n_verg, verg0: v_lo, dverg, n_az, period, bins, half: max_half, layout, pitch, refs, cdf }
    }
}

/// Radius (mm) containing `fraction` of the PSF's energy (radial histogram).
fn energy_radius(samples: &[[f64; 3]], fraction: f64) -> f64 {
    let r_max = samples.iter().map(|s| s[0].hypot(s[1])).fold(0.0, f64::max);
    if r_max == 0.0 {
        return 0.0;
    }
    let n = 4096;
    let mut hist = vec![0.0; n];
    for s in samples {
        let k = ((s[0].hypot(s[1]) / r_max * n as f64) as usize).min(n - 1);
        hist[k] += s[2];
    }
    let total: f64 = hist.iter().sum();
    let mut acc = 0.0;
    for (k, e) in hist.iter().enumerate() {
        acc += e;
        if acc >= fraction * total {
            return (k + 1) as f64 / n as f64 * r_max;
        }
    }
    r_max
}

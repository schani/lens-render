//! A bank of wave-optical PSFs for image formation: field angle × object
//! distance (vergence) × iris azimuth × wavelength bin, each integrated over
//! sensor pixels and stored as a cumulative distribution for sampling.
//!
//! Scene points within the bank's depth range get their light distributed
//! by the interpolated PSF; points further out of focus keep geometric
//! optics, which is accurate once the blur is many Airy discs wide.

use crate::glass::LAMBDA_D;
use crate::lens::Lens;
use crate::psf::{PsfGrid, PupilFrame, Source, Wavefront};
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
        let sources: Vec<(usize, usize, usize)> = (0..n_theta).flat_map(|t| (0..n_verg).flat_map(move |v| (0..n_az).map(move |a| (t, v, a)))).collect();
        let total = sources.len() * bins;
        let done = std::sync::atomic::AtomicUsize::new(0);
        // Per entry: a core holding 99 % of the energy at sub-pixel resolution
        // (so that binning into film pixels is the only pixel integration),
        // and a pixel-resolution wing out to 99.9 % for the diffraction tails.
        let kernels: Vec<Option<Kernels>> = sources
            .par_iter()
            .flat_map(|&(t, v, a)| {
                let theta = (tan_max * t as f64 / (n_theta - 1) as f64).atan();
                let verg = v_lo + dverg * v as f64;
                let distance = if verg <= 0.0 { None } else { Some(1.0 / verg - ep_z) };
                let alpha = std::f64::consts::FRAC_PI_2 + period * a as f64 / n_az as f64;
                let src = Source::at(lens, theta, alpha, distance);
                let frame = PupilFrame::find(lens, &src, LAMBDA_D);
                (0..bins)
                    .into_par_iter()
                    .map(|b| {
                        let l = LAMBDA_MIN + LAMBDA_RANGE * (b as f64 + 0.5) / bins as f64;
                        let k = frame
                            .as_ref()
                            .and_then(|f| Wavefront::compute_in(lens, &src, l, 1024, f))
                            .map(|wf| Kernels::new(&PsfGrid::from_wavefront(&wf, 2), pitch));
                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if n % 64 == 0 {
                            progress(n as f64 / total as f64);
                        }
                        k
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut refs = Vec::with_capacity(kernels.len());
        let mut cdf = Vec::new();
        let mut layout = Vec::with_capacity(kernels.len());
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
        for k in &kernels {
            let Some(k) = k else {
                // Nothing passes: a delta at the reference (its weight is zero anyway).
                refs.push(0.0);
                let off = push_cdf(&[1.0], &mut cdf);
                layout.push([0, 1, off, 0, off, 1f32.to_bits()]);
                continue;
            };
            max_half = max_half.max(k.hw);
            refs.push(k.reference);
            let (ec, ew): (f64, f64) = (k.core.iter().sum(), k.wing.iter().sum());
            let core_off = push_cdf(&k.core, &mut cdf);
            let wing_off = if ew > 0.0 { push_cdf(&k.wing, &mut cdf) } else { core_off };
            let fraction = if ew > 0.0 { ec / (ec + ew) } else { 1.0 };
            layout.push([k.hc as u32, k.sub as u32, core_off, k.hw as u32, wing_off, (fraction as f32).to_bits()]);
        }
        PsfBank { n_theta, tan_max, n_verg, verg0: v_lo, dverg, n_az, period, bins, half: max_half, layout, pitch, refs, cdf }
    }
}

/// One entry's kernels, before packing into the bank.
struct Kernels {
    reference: f32,
    hc: usize,
    sub: usize,
    hw: usize,
    core: Vec<f64>,
    wing: Vec<f64>,
}

impl Kernels {
    fn new(p: &PsfGrid, pitch: f64) -> Kernels {
        let [r99, r999] = p.energy_radii([0.99, 0.999]);
        let hc = ((r99 / pitch).ceil() as usize + 1).min(24);
        let hw = ((r999 / pitch).ceil() as usize + 1).clamp(hc, 64);
        let sub = (128 / (2 * hc + 1)).clamp(1, 16);
        let corner = |h: usize| [p.reference[0] - (h as f64 + 0.5) * pitch, p.reference[1] - (h as f64 + 0.5) * pitch];
        let core = p.pixels(corner(hc), pitch / sub as f64, (2 * hc + 1) * sub);
        let mut wing = p.pixels(corner(hw), pitch, 2 * hw + 1);
        let size = 2 * hw + 1;
        for j in hw - hc..=hw + hc {
            for i in hw - hc..=hw + hc {
                wing[j * size + i] = 0.0;
            }
        }
        Kernels { reference: p.reference[0].hypot(p.reference[1]) as f32, hc, sub, hw, core, wing }
    }
}

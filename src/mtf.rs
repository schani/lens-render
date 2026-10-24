//! MTF charts from wave-optical PSFs (white light, CIE ȳ weighted).

use crate::lens::Lens;
use crate::plot::{Chart, Series, save};
use crate::psf::{PolyPsf, Source, diffraction_limit_mtf, field_for_height};
use anyhow::{Context, Result};
use std::fmt::Write;

pub const WAVELENGTHS: [f64; 7] = [430.0, 470.0, 510.0, 550.0, 590.0, 630.0, 670.0];
const CHART_FREQS: [f64; 4] = [5.0, 10.0, 20.0, 40.0];
const COLORS: [&str; 4] = ["#c0392b", "#2471a3", "#1e8449", "#b9770e"];

/// Sagittal and tangential MTF at `freqs` for image height `h` (mm).
pub fn mtf_at(lens: &Lens, h: f64, distance: Option<f64>, freqs: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let theta = field_for_height(lens, h, distance)?;
    let psf = PolyPsf::compute(lens, &Source::at_field(lens, theta, distance), &WAVELENGTHS, 1024)?;
    // The field lies along +y: tangential = radial (y), sagittal = x.
    Some((psf.mtf(freqs, [1.0, 0.0]), psf.mtf(freqs, [0.0, 1.0])))
}

/// Writes `<prefix>-field.{svg,png}` (contrast vs image height, Leica style)
/// and `<prefix>-freq.{svg,png}` (contrast vs frequency) for each f-number.
pub fn run(base: &Lens, fstops: &[f64], blades: u32, distance: Option<f64>, half_diag: f64, prefix: &str) -> Result<String> {
    let mut report = String::new();
    let mut field_charts = Vec::new();
    let mut freq_charts = Vec::new();
    let dist_label = distance.map_or("∞".to_string(), |d| format!("{:.2} m", d / 1000.0));
    for &f in fstops {
        let mut lens = base.clone();
        lens.set_aperture(f, blades, 0.0, 0.0)?;
        lens.set_floating(distance);
        lens.sensor_z = lens.best_focus(distance)?;
        let steps = 24;
        let heights: Vec<f64> = (0..=steps).map(|i| half_diag * i as f64 / steps as f64).collect();
        writeln!(report, "\n{} at f/{f}, focus {dist_label}", lens.name)?;
        writeln!(report, "  height(mm)   sagittal {CHART_FREQS:?} lp/mm      tangential")?;
        let mut sag = vec![Vec::new(); CHART_FREQS.len()];
        let mut tan = vec![Vec::new(); CHART_FREQS.len()];
        for &h in &heights {
            let Some((s, t)) = mtf_at(&lens, h, distance, &CHART_FREQS) else { continue };
            writeln!(report, "  {h:6.2}   {}   {}", fmt4(&s), fmt4(&t))?;
            for k in 0..CHART_FREQS.len() {
                sag[k].push((h, s[k]));
                tan[k].push((h, t[k]));
            }
        }
        let mut series = Vec::new();
        for k in 0..CHART_FREQS.len() {
            series.push(Series { label: format!("{} lp/mm sag.", CHART_FREQS[k]), points: sag[k].clone(), color: COLORS[k], dashed: false });
            series.push(Series { label: format!("{} lp/mm tan.", CHART_FREQS[k]), points: tan[k].clone(), color: COLORS[k], dashed: true });
        }
        field_charts.push(Chart {
            title: format!("{}  f/{f}  ({dist_label})", lens.name),
            x_label: "image height (mm)".into(),
            y_label: "contrast".into(),
            x_range: (0.0, half_diag),
            y_range: (0.0, 1.0),
            x_ticks: (0..=(half_diag / 5.0) as usize).map(|i| i as f64 * 5.0).collect(),
            y_ticks: (0..=5).map(|i| i as f64 * 0.2).collect(),
            series,
        });

        let freqs: Vec<f64> = (0..=40).map(|i| i as f64 * 2.5).collect();
        let mut series = Vec::new();
        for (k, frac) in [0.0, 0.5, 0.75, 1.0].iter().enumerate() {
            let h = half_diag * frac;
            let (s, t) = mtf_at(&lens, h, distance, &freqs).context("no rays reach the sensor")?;
            let pts = |v: &[f64]| freqs.iter().cloned().zip(v.iter().cloned()).collect::<Vec<_>>();
            series.push(Series { label: format!("{h:.1} mm sag."), points: pts(&s), color: COLORS[k], dashed: false });
            if k > 0 {
                series.push(Series { label: format!("{h:.1} mm tan."), points: pts(&t), color: COLORS[k], dashed: true });
            }
        }
        let n_w = lens.working_fnumber();
        series.push(Series {
            label: "diffraction limit".into(),
            points: freqs.iter().map(|&q| (q, diffraction_limit_mtf(q, 550.0, n_w))).collect(),
            color: "#888888",
            dashed: false,
        });
        freq_charts.push(Chart {
            title: format!("{}  f/{f}  ({dist_label})", lens.name),
            x_label: "spatial frequency (lp/mm)".into(),
            y_label: "contrast".into(),
            x_range: (0.0, 100.0),
            y_range: (0.0, 1.0),
            x_ticks: (0..=10).map(|i| i as f64 * 10.0).collect(),
            y_ticks: (0..=5).map(|i| i as f64 * 0.2).collect(),
            series,
        });
    }
    save(&field_charts, &format!("{prefix}-field"))?;
    save(&freq_charts, &format!("{prefix}-freq"))?;
    writeln!(report, "\nWrote {prefix}-field.png and {prefix}-freq.png (and .svg)")?;
    Ok(report)
}

fn fmt4(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:5.3}")).collect::<Vec<_>>().join(" ")
}

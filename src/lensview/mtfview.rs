//! MTF charts: contrast against image height (as lens makers publish it)
//! and against spatial frequency.

use crate::mtf::{CHART_FREQS, Curves};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, pos2, vec2};

const COLORS: [Color32; 4] = [Color32::from_rgb(192, 57, 43), Color32::from_rgb(36, 113, 163), Color32::from_rgb(30, 132, 73), Color32::from_rgb(185, 119, 14)];
const SWEEP_COLORS: [Color32; 3] = [Color32::from_rgb(20, 140, 40), Color32::from_rgb(200, 40, 40), Color32::from_rgb(40, 90, 220)];

/// Published contrast (0–1) for the chart frequencies at image heights:
/// per frequency, (sagittal, tangential) points.
pub type Published = Vec<[Vec<(f64, f64)>; 2]>;

/// Reads a reference file ({"f/1.4": {"5S": [...], "5T": [...], …}, "heights_mm": [...]},
/// contrast in percent) for the f-number nearest `fstop`, if within 0.05.
pub fn published(json: &serde_json::Value, fstop: f64) -> Option<Published> {
    let heights: Vec<f64> = json["heights_mm"].as_array()?.iter().filter_map(|v| v.as_f64()).collect();
    let (_, data) = json.as_object()?.iter().find(|(k, _)| k.strip_prefix("f/").and_then(|v| v.parse::<f64>().ok()).is_some_and(|v| (v - fstop).abs() < 0.05))?;
    Some(
        CHART_FREQS
            .iter()
            .map(|f| {
                ["S", "T"].map(|tag| {
                    let vals = data[format!("{f}{tag}")].as_array().cloned().unwrap_or_default();
                    heights.iter().zip(vals).filter_map(|(h, v)| v.as_f64().map(|v| (*h, v / 100.0))).step_by(2).collect()
                })
            })
            .collect(),
    )
}

struct Plot {
    rect: Rect,
    x: (f64, f64),
}

impl Plot {
    fn new(ui: &mut egui::Ui, size: egui::Vec2, x: (f64, f64), x_ticks: &[f64], x_label: &str, title: &str) -> (Plot, egui::Painter) {
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::WHITE);
        let plot = Plot { rect: Rect::from_min_max(rect.min + vec2(36.0, 22.0), rect.max - vec2(8.0, 30.0)), x };
        let font = egui::FontId::proportional(11.0);
        let grid = Stroke::new(1.0, Color32::from_gray(225));
        for k in 0..=5 {
            let v = k as f64 * 0.2;
            painter.line_segment([plot.at(x.0, v), plot.at(x.1, v)], grid);
            painter.text(plot.at(x.0, v) - vec2(4.0, 0.0), egui::Align2::RIGHT_CENTER, format!("{v:.1}"), font.clone(), Color32::GRAY);
        }
        for &t in x_ticks {
            painter.line_segment([plot.at(t, 0.0), plot.at(t, 1.0)], grid);
            painter.text(plot.at(t, 0.0) + vec2(0.0, 3.0), egui::Align2::CENTER_TOP, format!("{t:.0}"), font.clone(), Color32::GRAY);
        }
        painter.text(plot.rect.center_bottom() + vec2(0.0, 16.0), egui::Align2::CENTER_TOP, x_label, font.clone(), Color32::DARK_GRAY);
        painter.text(rect.left_top() + vec2(6.0, 4.0), egui::Align2::LEFT_TOP, title, egui::FontId::proportional(12.0), Color32::DARK_GRAY);
        (plot, painter)
    }

    fn at(&self, x: f64, y: f64) -> Pos2 {
        pos2(
            self.rect.left() + ((x - self.x.0) / (self.x.1 - self.x.0)) as f32 * self.rect.width(),
            self.rect.bottom() - (y.clamp(0.0, 1.0) as f32) * self.rect.height(),
        )
    }

    /// Solid, or dashed in segments of the data spacing.
    fn line(&self, painter: &egui::Painter, pts: &[(f64, f64)], color: Color32, dashed: bool) {
        let p: Vec<Pos2> = pts.iter().map(|&(x, y)| self.at(x, y)).collect();
        if !dashed {
            painter.line(p, Stroke::new(1.8, color));
            return;
        }
        for w in p.windows(2) {
            let (a, b) = (w[0], w[1]);
            let n = ((b - a).length() / 6.0).ceil().max(1.0) as usize;
            for k in (0..n).step_by(2) {
                painter.line_segment([a + (b - a) * (k as f32 / n as f32), a + (b - a) * (((k + 1) as f32 / n as f32).min(1.0))], Stroke::new(1.8, color));
            }
        }
    }
}

/// Both charts side by side, with `published` values as markers.
pub fn mtf_ui(ui: &mut egui::Ui, c: &Curves, published: Option<&Published>) {
    ui.horizontal_wrapped(|ui| {
        for (k, f) in CHART_FREQS.iter().enumerate() {
            ui.colored_label(COLORS[k], format!("{f} lp/mm"));
        }
        ui.label("— solid sagittal, dashed tangential");
        if published.is_some() {
            ui.label("· dots / rings: published");
        }
    });
    let w = (ui.available_width() - 8.0) / 2.0;
    let h = ui.available_height();
    ui.horizontal(|ui| {
        let half = c.heights.last().copied().unwrap_or(1.0);
        let ticks: Vec<f64> = (0..=(half / 5.0) as usize).map(|i| i as f64 * 5.0).collect();
        let (plot, painter) = Plot::new(ui, vec2(w, h), (0.0, half), &ticks, "image height (mm)", "contrast against image height");
        for k in 0..4 {
            let pts = |v: &[[f64; 4]]| -> Vec<(f64, f64)> { c.heights.iter().zip(v).map(|(h, s)| (*h, s[k])).collect() };
            plot.line(&painter, &pts(&c.sag), COLORS[k], false);
            plot.line(&painter, &pts(&c.tan), COLORS[k], true);
            if let Some(p) = published {
                for &(x, y) in &p[k][0] {
                    painter.circle_filled(plot.at(x, y), 2.6, COLORS[k]);
                }
                for &(x, y) in &p[k][1] {
                    painter.circle(plot.at(x, y), 2.6, Color32::WHITE, Stroke::new(1.2, COLORS[k]));
                }
            }
        }
        let fmax = c.freqs.last().copied().unwrap_or(80.0);
        let ticks: Vec<f64> = (0..=(fmax / 20.0) as usize).map(|i| i as f64 * 20.0).collect();
        let (plot, painter) = Plot::new(ui, vec2(w, h), (0.0, fmax), &ticks, "spatial frequency (lp/mm)", "contrast against frequency");
        let limit: Vec<(f64, f64)> = c.freqs.iter().copied().zip(c.limit.iter().copied()).collect();
        plot.line(&painter, &limit, Color32::from_gray(90), true);
        let font = egui::FontId::proportional(11.0);
        for (i, (hgt, s, t)) in c.sweep.iter().enumerate() {
            let color = SWEEP_COLORS[i % 3];
            plot.line(&painter, &c.freqs.iter().copied().zip(s.iter().copied()).collect::<Vec<_>>(), color, false);
            plot.line(&painter, &c.freqs.iter().copied().zip(t.iter().copied()).collect::<Vec<_>>(), color, true);
            painter.text(plot.rect.right_top() + vec2(-4.0, 4.0 + 14.0 * i as f32), egui::Align2::RIGHT_TOP, format!("{hgt:.1} mm"), font.clone(), color);
        }
        painter.text(plot.rect.right_top() + vec2(-4.0, 4.0 + 14.0 * c.sweep.len() as f32), egui::Align2::RIGHT_TOP, "diffraction limit", font, Color32::from_gray(90));
    });
}

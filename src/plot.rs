//! Minimal line charts as SVG, rasterised to PNG with resvg.

use anyhow::{Context, Result};
use std::fmt::Write;

#[derive(Clone, Copy, PartialEq)]
pub enum Style {
    Solid,
    Dashed,
    /// Markers only: filled for `Dots`, hollow for `Rings`.
    Dots,
    Rings,
}

pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
    pub color: &'static str,
    pub style: Style,
}

pub struct Chart {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub x_ticks: Vec<f64>,
    pub y_ticks: Vec<f64>,
    pub series: Vec<Series>,
}

const W: f64 = 640.0;
const H: f64 = 440.0;
const L: f64 = 64.0;
const R: f64 = 170.0;
const T: f64 = 40.0;
const B: f64 = 52.0;

impl Chart {
    fn svg(&self, ox: f64, out: &mut String) -> std::fmt::Result {
        let px = |x: f64| ox + L + (x - self.x_range.0) / (self.x_range.1 - self.x_range.0) * (W - L - R);
        let py = |y: f64| T + (1.0 - (y - self.y_range.0) / (self.y_range.1 - self.y_range.0)) * (H - T - B);
        writeln!(out, r#"<text x="{}" y="24" font-size="15" font-weight="bold">{}</text>"#, ox + L, esc(&self.title))?;
        for &x in &self.x_ticks {
            writeln!(out, r##"<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#ddd"/>"##, px(x), py(self.y_range.0), py(self.y_range.1))?;
            writeln!(out, r#"<text x="{}" y="{}" font-size="11" text-anchor="middle">{}</text>"#, px(x), py(self.y_range.0) + 16.0, fmt(x))?;
        }
        for &y in &self.y_ticks {
            writeln!(out, r##"<line x1="{1}" y1="{0}" x2="{2}" y2="{0}" stroke="#ddd"/>"##, py(y), px(self.x_range.0), px(self.x_range.1))?;
            writeln!(out, r#"<text x="{}" y="{}" font-size="11" text-anchor="end">{}</text>"#, px(self.x_range.0) - 6.0, py(y) + 4.0, fmt(y))?;
        }
        writeln!(
            out,
            r##"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#444"/>"##,
            px(self.x_range.0),
            py(self.y_range.1),
            px(self.x_range.1) - px(self.x_range.0),
            py(self.y_range.0) - py(self.y_range.1)
        )?;
        writeln!(out, r#"<text x="{}" y="{}" font-size="12" text-anchor="middle">{}</text>"#, (px(self.x_range.0) + px(self.x_range.1)) / 2.0, H - 12.0, esc(&self.x_label))?;
        writeln!(
            out,
            r#"<text x="{0}" y="{1}" font-size="12" text-anchor="middle" transform="rotate(-90 {0} {1})">{2}</text>"#,
            ox + 18.0,
            (py(self.y_range.0) + py(self.y_range.1)) / 2.0,
            esc(&self.y_label)
        )?;
        let mut row = 0;
        for s in &self.series {
            // Series without a label are drawn but get no legend row.
            let ly = if s.label.is_empty() { -100.0 } else { T + 14.0 + 18.0 * row as f64 };
            if !s.label.is_empty() {
                row += 1;
            }
            let lx = ox + W - R + 12.0;
            match s.style {
                Style::Solid | Style::Dashed => {
                    let pts: Vec<String> =
                        s.points.iter().map(|(x, y)| format!("{:.2},{:.2}", px(*x), py(y.clamp(self.y_range.0, self.y_range.1)))).collect();
                    let dash = if s.style == Style::Dashed { r#" stroke-dasharray="6 4""# } else { "" };
                    writeln!(out, r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="1.8"{dash}/>"#, pts.join(" "), s.color)?;
                    writeln!(out, r#"<line x1="{lx}" y1="{ly}" x2="{}" y2="{ly}" stroke="{}" stroke-width="1.8"{dash}/>"#, lx + 26.0, s.color)?;
                }
                Style::Dots | Style::Rings => {
                    let fill = if s.style == Style::Dots { s.color } else { "white" };
                    for (x, y) in &s.points {
                        writeln!(
                            out,
                            r#"<circle cx="{:.2}" cy="{:.2}" r="2.6" fill="{fill}" stroke="{}" stroke-width="1.2"/>"#,
                            px(*x),
                            py(y.clamp(self.y_range.0, self.y_range.1)),
                            s.color
                        )?;
                    }
                    writeln!(out, r#"<circle cx="{}" cy="{ly}" r="2.6" fill="{fill}" stroke="{}" stroke-width="1.2"/>"#, lx + 13.0, s.color)?;
                }
            }
            writeln!(out, r#"<text x="{}" y="{}" font-size="11">{}</text>"#, lx + 32.0, ly + 4.0, esc(&s.label))?;
        }
        Ok(())
    }
}

fn fmt(v: f64) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Writes charts side by side to `<prefix>.svg` and `<prefix>.png`.
pub fn save(charts: &[Chart], prefix: &str) -> Result<()> {
    let width = W * charts.len() as f64;
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{H}" font-family="Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="white"/>"#
    );
    for (i, c) in charts.iter().enumerate() {
        c.svg(i as f64 * W, &mut svg)?;
    }
    svg.push_str("</svg>");
    std::fs::write(format!("{prefix}.svg"), &svg)?;
    let mut opt = resvg::usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();
    let tree = resvg::usvg::Tree::from_str(&svg, &opt)?;
    let scale = 2.0;
    let mut pixmap = resvg::tiny_skia::Pixmap::new((width * scale) as u32, (H * scale) as u32).context("pixmap")?;
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32), &mut pixmap.as_mut());
    pixmap.save_png(format!("{prefix}.png"))?;
    Ok(())
}

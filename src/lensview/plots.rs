//! Charts of the lens analyses: Seidel contributions, pupil footprints and
//! transmission losses. They only draw precomputed data.

use crate::footprint::{FIELD_FRACTIONS, Footprints, LIGHT_LOSS_RAYS, LightLoss};
use crate::seidel::{Seidel, TERMS};
use eframe::egui::{self, Color32, Rect, Stroke, pos2, vec2};

const TERM_COLORS: [Color32; 7] = [
    Color32::from_rgb(31, 119, 180),
    Color32::from_rgb(255, 127, 14),
    Color32::from_rgb(44, 160, 44),
    Color32::from_rgb(214, 39, 40),
    Color32::from_rgb(148, 103, 189),
    Color32::from_rgb(140, 86, 75),
    Color32::from_rgb(227, 119, 194),
];

const SURFACE_COLORS: [Color32; 8] = [
    Color32::from_rgb(230, 85, 13),
    Color32::from_rgb(49, 130, 189),
    Color32::from_rgb(49, 163, 84),
    Color32::from_rgb(158, 154, 200),
    Color32::from_rgb(253, 174, 107),
    Color32::from_rgb(107, 174, 214),
    Color32::from_rgb(161, 217, 155),
    Color32::from_rgb(188, 189, 34),
];

const IRIS_COLOR: Color32 = Color32::from_gray(40);
const AXIS_COLOR: Color32 = Color32::from_gray(150);
const TEXT_COLOR: Color32 = Color32::from_gray(60);

fn font() -> egui::FontId {
    egui::FontId::proportional(11.0)
}

fn fmt(v: f64) -> String {
    if v == 0.0 || (0.01..1000.0).contains(&v.abs()) { format!("{v:.3}") } else { format!("{v:.2e}") }
}

fn swatch(ui: &mut egui::Ui, color: Color32) {
    let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().rect_filled(r, 2.0, color);
}

/// Grouped bars of the Seidel coefficients per surface (stop marked) with
/// the sums at the right. The legend toggles terms via `shown`; hovering a
/// bar shows its value.
pub fn seidel_chart(ui: &mut egui::Ui, data: &Seidel, shown: &mut [bool; 7]) {
    ui.horizontal_wrapped(|ui| {
        for (k, name) in TERMS.iter().enumerate() {
            swatch(ui, TERM_COLORS[k]);
            ui.checkbox(&mut shown[k], *name);
        }
    });
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::WHITE);
    let terms: Vec<usize> = (0..7).filter(|&k| shown[k]).collect();
    let n = data.surfaces.len();
    if terms.is_empty() || n == 0 {
        return;
    }
    let plot = Rect::from_min_max(rect.min + vec2(64.0, 10.0), rect.max - vec2(8.0, 30.0));
    let vmax = data.surfaces.iter().chain(std::iter::once(&data.sum)).flat_map(|v| terms.iter().map(move |&k| v[k].abs())).fold(1e-12, f64::max);
    let mid = plot.center().y;
    let yscale = plot.height() as f64 / (2.0 * vmax);
    let at_y = |v: f64| mid - (v * yscale) as f32;
    for f in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let y = at_y(f * vmax);
        let color = if f == 0.0 { AXIS_COLOR } else { Color32::from_gray(225) };
        painter.line_segment([pos2(plot.left(), y), pos2(plot.right(), y)], Stroke::new(1.0, color));
        painter.text(pos2(plot.left() - 4.0, y), egui::Align2::RIGHT_CENTER, fmt(f * vmax), font(), TEXT_COLOR);
    }
    painter.text(rect.left_top() + vec2(4.0, 2.0), egui::Align2::LEFT_TOP, "waves", font(), TEXT_COLOR);

    // One group per surface, a gap, then the sums.
    let gw = plot.width() / (n + 2) as f32;
    let bw = 0.8 * gw / terms.len() as f32;
    let pointer = response.hover_pos();
    let mut hovered = None;
    for g in (0..n).chain(std::iter::once(n + 1)) {
        let (values, label) = if g < n { (&data.surfaces[g], format!("{}", g + 1)) } else { (&data.sum, "Σ".to_string()) };
        let x0 = plot.left() + gw * g as f32 + 0.1 * gw;
        for (j, &k) in terms.iter().enumerate() {
            let (xa, xb) = (x0 + j as f32 * bw, x0 + (j + 1) as f32 * bw);
            painter.rect_filled(Rect::from_two_pos(pos2(xa, mid), pos2(xb, at_y(values[k]))), 0.0, TERM_COLORS[k]);
            if pointer.is_some_and(|p| p.x >= xa && p.x < xb && plot.y_range().contains(p.y)) {
                let place = if g < n { format!("surface {label}") } else { "sum".to_string() };
                hovered = Some(format!("{place}\n{}: {} waves", TERMS[k], fmt(values[k])));
            }
        }
        let cx = plot.left() + gw * (g as f32 + 0.5);
        let stop = g == data.stop_index;
        painter.text(pos2(cx, plot.bottom() + 4.0), egui::Align2::CENTER_TOP, label, font(), if stop { IRIS_COLOR } else { TEXT_COLOR });
        if stop {
            painter.text(pos2(cx, plot.bottom() + 16.0), egui::Align2::CENTER_TOP, "stop", font(), IRIS_COLOR);
        }
    }
    let sx = plot.left() + gw * (n as f32 + 0.5);
    painter.line_segment([pos2(sx, plot.top()), pos2(sx, plot.bottom())], Stroke::new(1.0, Color32::from_gray(200)));
    if let Some(text) = hovered {
        response.on_hover_text_at_pointer(text);
    }
}

fn surface_color(s: usize, stop_index: usize) -> Color32 {
    if s == stop_index { IRIS_COLOR } else { SURFACE_COLORS[s % SURFACE_COLORS.len()] }
}

fn surface_name(s: usize, stop_index: usize) -> String {
    if s == stop_index { "iris".to_string() } else { format!("surface {}", s + 1) }
}

/// One square per field showing which rays through the entrance pupil pass
/// (light) or which surface blocks them, with the relative illumination
/// below and a legend of the blocking surfaces. The circle is the paraxial
/// entrance pupil.
pub fn footprint_plots(ui: &mut egui::Ui, data: &Footprints) {
    let pass = Color32::from_rgb(255, 244, 200);
    let count = FIELD_FRACTIONS.len() as f32;
    let spacing = ui.spacing().item_spacing.x;
    let size = ((ui.available_width() - spacing * (count - 1.0)) / count).clamp(60.0, 240.0);
    ui.horizontal(|ui| {
        for f in &data.fields {
            ui.vertical(|ui| {
                let (rect, _) = ui.allocate_exact_size(vec2(size, size), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                let cs = size / f.grid as f32;
                let mut mesh = egui::Mesh::default();
                for (k, c) in f.cells.iter().enumerate() {
                    let min = rect.min + vec2((k % f.grid) as f32 * cs, (k / f.grid) as f32 * cs);
                    let color = c.map_or(pass, |s| surface_color(s, data.stop_index));
                    mesh.add_colored_rect(Rect::from_min_size(min, vec2(cs, cs)), color);
                }
                painter.add(mesh);
                let r = (data.pupil_radius / (f.cell * f.grid as f64)) as f32 * size;
                painter.circle_stroke(rect.center(), r, Stroke::new(1.0, Color32::from_rgb(150, 0, 150)));
                painter.rect_stroke(rect, 0.0, Stroke::new(1.0, AXIS_COLOR), egui::StrokeKind::Inside);
                ui.label(format!("{:.0} % field, {:.1} mm", f.fraction * 100.0, f.height));
                ui.strong(format!("RI {:.0} %", f.relative_area * 100.0));
            });
        }
    });
    let mut blockers: Vec<usize> = data.fields.iter().flat_map(|f| f.cells.iter().flatten().copied()).collect();
    blockers.sort_unstable();
    blockers.dedup();
    ui.horizontal_wrapped(|ui| {
        swatch(ui, pass);
        ui.label("passes");
        for s in blockers {
            swatch(ui, surface_color(s, data.stop_index));
            ui.label(surface_name(s, data.stop_index));
        }
    });
}

const RAY_COLORS: [Color32; 2] = [Color32::from_rgb(31, 119, 180), Color32::from_rgb(230, 85, 13)];

/// Cumulative transmission against surface number for both rays, coated
/// (solid) and uncoated (dashed), with totals; below, the reflectance of
/// each surface for the on-axis ray, coated and uncoated.
pub fn light_loss_chart(ui: &mut egui::Ui, data: &LightLoss) {
    ui.horizontal_wrapped(|ui| {
        for (k, name) in LIGHT_LOSS_RAYS.iter().enumerate() {
            swatch(ui, RAY_COLORS[k]);
            ui.label(format!("{name}: coated {:.1} %, uncoated {:.1} %", data.coated[k].total() * 100.0, data.uncoated[k].total() * 100.0));
        }
    });
    ui.label("solid: coated, dashed: uncoated");
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::WHITE);
    let n = data.coated.iter().chain(&data.uncoated).map(|t| t.cumulative.len()).max().unwrap_or(0);
    if n == 0 {
        return;
    }
    let split = rect.top() + 0.65 * rect.height();
    let top = Rect::from_min_max(rect.min + vec2(48.0, 10.0), pos2(rect.right() - 10.0, split - 20.0));
    let bottom = Rect::from_min_max(pos2(top.left(), split + 6.0), rect.max - vec2(10.0, 20.0));
    let at_x = |i: f64| top.left() + (i / (n as f64 + 0.5)) as f32 * top.width();

    // Transmission: x = 0 before the lens, i after surface i.
    let tmin = data.coated.iter().chain(&data.uncoated).flat_map(|t| t.cumulative.iter().copied()).fold(1.0, f64::min);
    let lo = ((tmin * 10.0).floor() / 10.0).clamp(0.0, 0.9);
    let at_t = |t: f64| top.bottom() - ((t - lo) / (1.0 - lo)) as f32 * top.height();
    let mut tick = lo;
    while tick <= 1.0 + 1e-9 {
        let y = at_t(tick);
        painter.line_segment([pos2(top.left(), y), pos2(top.right(), y)], Stroke::new(1.0, Color32::from_gray(225)));
        painter.text(pos2(top.left() - 4.0, y), egui::Align2::RIGHT_CENTER, format!("{:.0} %", tick * 100.0), font(), TEXT_COLOR);
        tick += if 1.0 - lo > 0.3 { 0.1 } else { 0.05 };
    }
    for i in 0..=n {
        let color = if i == data.stop_index + 1 { IRIS_COLOR } else { TEXT_COLOR };
        painter.text(pos2(at_x(i as f64), top.bottom() + 3.0), egui::Align2::CENTER_TOP, format!("{i}"), font(), color);
    }
    for (k, color) in RAY_COLORS.iter().enumerate() {
        for (t, dashed) in [(&data.coated[k], false), (&data.uncoated[k], true)] {
            let pts: Vec<egui::Pos2> = std::iter::once(1.0).chain(t.cumulative.iter().copied()).enumerate().map(|(i, v)| pos2(at_x(i as f64), at_t(v))).collect();
            if dashed {
                painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1.5, *color), 6.0, 4.0));
            } else {
                painter.line(pts.clone(), Stroke::new(2.0, *color));
            }
            for p in pts {
                painter.circle_filled(p, 2.0, *color);
            }
        }
    }

    // Reflectance per surface of the on-axis ray: coated (full) and uncoated (pale).
    let (rc, ru) = (&data.coated[0].reflectance, &data.uncoated[0].reflectance);
    let rmax = rc.iter().chain(ru).copied().fold(1e-6, f64::max);
    painter.text(pos2(bottom.left() - 4.0, bottom.top()), egui::Align2::RIGHT_TOP, format!("{:.1} %", rmax * 100.0), font(), TEXT_COLOR);
    painter.text(pos2(bottom.left() - 4.0, bottom.bottom()), egui::Align2::RIGHT_BOTTOM, "R 0", font(), TEXT_COLOR);
    painter.line_segment([bottom.left_bottom(), bottom.right_bottom()], Stroke::new(1.0, AXIS_COLOR));
    let bw = 0.35 * top.width() / (n as f32 + 0.5);
    for i in 0..n {
        let cx = at_x(i as f64 + 1.0);
        for (r, x0, color) in [(rc.get(i), cx - bw, RAY_COLORS[0]), (ru.get(i), cx, RAY_COLORS[0].gamma_multiply(0.4))] {
            let Some(&r) = r else { continue };
            let y = bottom.bottom() - (r / rmax) as f32 * bottom.height();
            painter.rect_filled(Rect::from_two_pos(pos2(x0, bottom.bottom()), pos2(x0 + bw, y)), 0.0, color);
        }
        let color = if i == data.stop_index { IRIS_COLOR } else { TEXT_COLOR };
        painter.text(pos2(cx, bottom.bottom() + 3.0), egui::Align2::CENTER_TOP, format!("{}", i + 1), font(), color);
    }
}

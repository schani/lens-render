//! Glass map (Abbe diagram): refractive index n_d over Abbe number V_d, for
//! the catalogue and the lens' glasses. Drag the selected element's glass,
//! or click a catalogue glass to use it.

use crate::glass::Glass;
use crate::lens::Lens;
use eframe::egui::{self, Color32, Pos2, Stroke};

const VD: (f64, f64) = (95.0, 15.0);
const ND: (f64, f64) = (1.40, 2.05);

#[derive(Default)]
pub struct GlassMap {
    dragging: bool,
}

/// The surface whose following medium is the glass of the selection: the
/// selected surface's own glass, else the glass in front of it.
pub fn glass_row(lens: &Lens, selected: Option<usize>) -> Option<usize> {
    let i = selected.filter(|&i| i < lens.surfaces.len())?;
    if lens.surfaces[i].material.is_some() {
        Some(i)
    } else if i > 0 && lens.surfaces[i - 1].material.is_some() {
        Some(i - 1)
    } else {
        None
    }
}

impl GlassMap {
    /// Returns true when the draft changed.
    pub fn ui(&mut self, ui: &mut egui::Ui, draft: &mut Lens, selected: Option<usize>, undo: &mut Vec<Lens>) -> bool {
        let row = glass_row(draft, selected);
        ui.label(match row {
            Some(r) => format!("Glass after surface {}: drag it, or click a catalogue glass.", r + 1),
            None => "Select an element in the cross-section to change its glass.".into(),
        });
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::WHITE);
        let plot = egui::Rect::from_min_max(rect.min + egui::vec2(36.0, 20.0), rect.max - egui::vec2(16.0, 34.0));
        let at = |vd: f64, nd: f64| {
            egui::pos2(
                plot.left() + ((vd - VD.0) / (VD.1 - VD.0)) as f32 * plot.width(),
                plot.bottom() - ((nd - ND.0) / (ND.1 - ND.0)) as f32 * plot.height(),
            )
        };
        let back = |p: Pos2| {
            let vd = VD.0 + (p.x - plot.left()) as f64 / plot.width() as f64 * (VD.1 - VD.0);
            let nd = ND.0 + (plot.bottom() - p.y) as f64 / plot.height() as f64 * (ND.1 - ND.0);
            (vd.clamp(VD.1, VD.0), nd.clamp(ND.0, ND.1))
        };
        let font = egui::FontId::proportional(11.0);
        let grid = Stroke::new(1.0, Color32::from_gray(225));
        for vd in (20..=90).step_by(10) {
            painter.line_segment([at(vd as f64, ND.0), at(vd as f64, ND.1)], grid);
            painter.text(at(vd as f64, ND.0) + egui::vec2(0.0, 3.0), egui::Align2::CENTER_TOP, vd.to_string(), font.clone(), Color32::GRAY);
        }
        for k in 0..=6 {
            let nd = 1.4 + 0.1 * k as f64;
            painter.line_segment([at(VD.0, nd), at(VD.1, nd)], grid);
            painter.text(at(VD.0, nd) - egui::vec2(4.0, 0.0), egui::Align2::RIGHT_CENTER, format!("{nd:.1}"), font.clone(), Color32::GRAY);
        }
        // Crowns (V_d > 50) disperse little, flints a lot.
        painter.line_segment([at(50.0, ND.0), at(50.0, ND.1)], Stroke::new(1.0, Color32::from_gray(180)));
        painter.text(at(72.0, ND.1), egui::Align2::CENTER_TOP, "crowns", font.clone(), Color32::GRAY);
        painter.text(at(32.0, ND.1), egui::Align2::CENTER_TOP, "flints", font.clone(), Color32::GRAY);
        painter.text(plot.center_bottom() + egui::vec2(0.0, 12.0), egui::Align2::CENTER_TOP, "Abbe number V_d (less dispersion to the left)", font.clone(), Color32::DARK_GRAY);

        let catalogue: Vec<(String, Pos2)> = Glass::catalog().filter_map(|n| Glass::parse(n).ok()).map(|g| (g.name().to_string(), at(g.abbe(), g.nd()))).collect();
        for (name, p) in &catalogue {
            painter.circle_filled(*p, 3.5, Color32::from_gray(120));
            painter.text(*p + egui::vec2(5.0, -2.0), egui::Align2::LEFT_BOTTOM, name, font.clone(), Color32::from_gray(110));
        }
        for (i, s) in draft.surfaces.iter().enumerate() {
            let Some(g) = &s.material else { continue };
            let p = at(g.abbe(), g.nd());
            let mine = row == Some(i);
            painter.circle(p, if mine { 7.0 } else { 5.0 }, if mine { Color32::from_rgb(255, 120, 0) } else { Color32::from_rgb(70, 120, 210) }, Stroke::new(1.0, Color32::BLACK));
            painter.text(p + egui::vec2(-7.0, 2.0), egui::Align2::RIGHT_TOP, (i + 1).to_string(), font.clone(), Color32::BLACK);
        }

        let Some(row) = row else { return false };
        let mut changed = false;
        let mine = draft.surfaces[row].material.as_ref().map(|g| at(g.abbe(), g.nd()));
        if response.drag_started() && response.interact_pointer_pos().zip(mine).is_some_and(|(p, m)| (p - m).length() < 10.0) {
            undo.push(draft.clone());
            self.dragging = true;
        }
        if self.dragging && response.dragged() {
            if let Some(p) = response.interact_pointer_pos() {
                let (vd, nd) = back(p);
                draft.surfaces[row].material = Some(Glass::from_abbe(&format!("{nd:.5}/{vd:.2}"), nd, vd));
                changed = true;
            }
        }
        if response.drag_stopped() {
            self.dragging = false;
        }
        if response.clicked() {
            let p = response.interact_pointer_pos().unwrap_or_default();
            if let Some((name, _)) = catalogue.iter().find(|(_, c)| (*c - p).length() < 8.0) {
                undo.push(draft.clone());
                draft.surfaces[row].material = Glass::parse(name).ok();
                changed = true;
            }
        }
        changed
    }
}

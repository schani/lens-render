//! The prescription as an editable table.

use crate::glass::Glass;
use crate::lens::{Lens, Surface};
use eframe::egui;

/// Edits a draft of the prescription; `changed` reports edits that parse.
pub struct LensEditor {
    pub draft: Lens,
    pub error: String,
}

impl LensEditor {
    pub fn new(lens: &Lens) -> LensEditor {
        LensEditor { draft: lens.clone(), error: String::new() }
    }

    /// Shows the table; returns the edited lens if an edit produced a valid one.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<Lens> {
        let mut edited = false;
        let mut insert = None;
        let mut remove = None;
        let mut stop = None;
        ui.horizontal(|ui| {
            ui.label("name");
            edited |= ui.text_edit_singleline(&mut self.draft.name).lost_focus();
        });
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("prescription").striped(true).show(ui, |ui| {
                for h in ["", "radius", "thickness", "material", "semi-diam.", "stop", ""] {
                    ui.strong(h);
                }
                ui.end_row();
                let n = self.draft.surfaces.len();
                for (i, s) in self.draft.surfaces.iter_mut().enumerate() {
                    ui.label(format!("{}", i + 1));
                    if s.is_stop {
                        ui.label("stop");
                    } else {
                        let r = egui::DragValue::new(&mut s.radius)
                            .speed(0.1)
                            .custom_formatter(|v, _| if v == 0.0 { "∞".into() } else { format!("{v:.3}") });
                        edited |= ui.add(r).on_hover_text("0 = flat").changed();
                    }
                    edited |= ui.add(egui::DragValue::new(&mut s.thickness).speed(0.02).range(0.0..=1000.0).max_decimals(3)).changed();
                    if s.is_stop {
                        ui.label("air");
                    } else {
                        edited |= material_ui(ui, i, &mut s.material);
                    }
                    edited |= ui.add(egui::DragValue::new(&mut s.semi_diameter).speed(0.05).range(0.1..=200.0).max_decimals(3)).changed();
                    if ui.radio(s.is_stop, "").clicked() && !s.is_stop {
                        stop = Some(i);
                    }
                    ui.horizontal(|ui| {
                        if s.is_aspheric() {
                            ui.label("asph").on_hover_text(format!("k = {}, a4… = {:?}", s.conic, s.aspheric));
                        }
                        if ui.small_button("+").on_hover_text("insert an air gap after").clicked() {
                            insert = Some(i + 1);
                        }
                        if n > 2 && !s.is_stop && ui.small_button("−").on_hover_text("remove").clicked() {
                            remove = Some(i);
                        }
                    });
                    ui.end_row();
                }
            });
        });
        if let Some(i) = stop {
            for (k, s) in self.draft.surfaces.iter_mut().enumerate() {
                s.is_stop = k == i;
            }
            let s = &mut self.draft.surfaces[i];
            (s.radius, s.material, s.conic) = (0.0, None, 0.0);
            s.aspheric.clear();
            edited = true;
        }
        if let Some(i) = insert {
            let sd = self.draft.surfaces[i - 1].semi_diameter;
            let s = Surface { radius: 0.0, thickness: 1.0, material: None, semi_diameter: sd, is_stop: false, z: 0.0, decenter: (0.0, 0.0), conic: 0.0, aspheric: Vec::new() };
            self.draft.insert_surface(i, s);
            edited = true;
        }
        if let Some(i) = remove {
            self.draft.remove_surface(i);
            edited = true;
        }
        if !edited {
            return None;
        }
        match Lens::parse(&self.draft.to_text()) {
            Ok(l) => {
                self.error.clear();
                Some(l)
            }
            Err(e) => {
                self.error = format!("{e:#}");
                None
            }
        }
    }
}

/// Air, a catalogue glass, or nd/vd.
fn material_ui(ui: &mut egui::Ui, row: usize, m: &mut Option<Glass>) -> bool {
    let mut changed = false;
    let current = match m {
        None => "air".to_string(),
        Some(Glass::Cauchy { .. }) => "nd/vd".to_string(),
        Some(g) => g.name().to_string(),
    };
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(("material", row)).selected_text(&current).width(100.0).show_ui(ui, |ui| {
            if ui.selectable_label(m.is_none(), "air").clicked() {
                *m = None;
                changed = true;
            }
            for name in Glass::catalog() {
                if ui.selectable_label(current == name, name).clicked() {
                    *m = Glass::parse(name).ok();
                    changed = true;
                }
            }
            if ui.selectable_label(false, "nd/vd").clicked() {
                let (nd, vd) = m.as_ref().map_or((1.5168, 64.17), |g| (g.nd(), g.abbe()));
                *m = Some(Glass::from_abbe(&format!("{nd:.5}/{vd:.2}"), nd, vd));
                changed = true;
            }
        });
        if let Some(Glass::Cauchy { .. }) = m {
            let g = m.as_ref().unwrap();
            let (mut nd, mut vd) = (g.nd(), g.abbe());
            let a = ui.add(egui::DragValue::new(&mut nd).speed(0.001).range(1.3..=2.2).max_decimals(5)).on_hover_text("n_d").changed();
            let b = ui.add(egui::DragValue::new(&mut vd).speed(0.1).range(15.0..=100.0).max_decimals(2)).on_hover_text("V_d").changed();
            if a || b {
                *m = Some(Glass::from_abbe(&format!("{nd:.5}/{vd:.2}"), nd, vd));
                changed = true;
            }
        }
    });
    changed
}


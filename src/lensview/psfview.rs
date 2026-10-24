//! Wavefront maps and white-light PSFs at three image heights.

use crate::fieldpsf::FieldPsf;
use crate::spectrum::srgb_encode;
use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};

/// Textures of the shown results, rebuilt when `key` changes.
#[derive(Default)]
pub struct PsfView {
    key: Option<u64>,
    textures: Vec<(TextureHandle, TextureHandle)>,
}

fn opd_image(f: &FieldPsf) -> ColorImage {
    let scale = f.opd.iter().flatten().fold(0f32, |a, &b| a.max(b.abs())).max(0.1);
    let px = f
        .opd
        .iter()
        .map(|w| match w {
            None => Color32::from_gray(235),
            Some(w) => {
                // Blue behind, white on, red ahead of the reference sphere.
                let t = (w / scale).clamp(-1.0, 1.0);
                let (a, b) = ((255.0 * (1.0 - t.abs())) as u8, 255);
                if t >= 0.0 { Color32::from_rgb(b, a, a) } else { Color32::from_rgb(a, a, b) }
            }
        })
        .collect();
    ColorImage { size: [f.map, f.map], pixels: px, ..Default::default() }
}

fn psf_image(f: &FieldPsf) -> ColorImage {
    let px = f
        .rgb
        .iter()
        .map(|c| {
            // Log tone curve on the brightest channel, so faint structure shows.
            let l = c[0].max(c[1]).max(c[2]);
            let k = if l > 0.0 { (1.0 + 999.0 * l).log10() / 3.0 / l } else { 0.0 };
            let e = |v: f32| (255.0 * srgb_encode((v * k).clamp(0.0, 1.0) as f64)) as u8;
            Color32::from_rgb(e(c[0]), e(c[1]), e(c[2]))
        })
        .collect();
    ColorImage { size: [f.size, f.size], pixels: px, ..Default::default() }
}

impl PsfView {
    pub fn ui(&mut self, ui: &mut egui::Ui, data: Option<(u64, &[FieldPsf])>) {
        let Some((key, fields)) = data else {
            ui.label("computing…");
            return;
        };
        if self.key != Some(key) {
            self.textures = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    (
                        ui.ctx().load_texture(format!("opd{i}"), opd_image(f), TextureOptions::LINEAR),
                        ui.ctx().load_texture(format!("psf{i}"), psf_image(f), TextureOptions::NEAREST),
                    )
                })
                .collect();
            self.key = Some(key);
        }
        ui.label("Wavefront error at 550 nm (red ahead of, blue behind the reference sphere on the sensor) and the white-light PSF, log brightness.");
        let side = ((ui.available_height() - 60.0) / 1.0).min((ui.available_width() - 20.0) / (2.0 * fields.len() as f32 + 0.5)).max(40.0);
        ui.horizontal_top(|ui| {
            for (f, (opd, psf)) in fields.iter().zip(&self.textures) {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::Image::new(opd).fit_to_exact_size(egui::vec2(side, side)));
                        ui.add(egui::Image::new(psf).fit_to_exact_size(egui::vec2(side, side)));
                    });
                    ui.label(format!("image height {:.1} mm", f.height));
                    ui.label(format!("RMS {:.3} λ, P-V {:.2} λ", f.rms, f.pv));
                    ui.label(format!("PSF box ±{:.0} µm", f.half * 1000.0));
                });
                ui.add_space(8.0);
            }
        });
    }
}

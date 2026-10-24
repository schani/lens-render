//! Interactive viewer: progressive GPU rendering with live lens controls.

use crate::gpu::{self, GpuContext, GpuRenderer, Tonemap, wgpu};
use crate::lens::{Coating, Lens};
use crate::math::{Ray, Vec3};
use crate::scene::{Scene, SceneFile};
use anyhow::Result;
use eframe::egui;
use std::time::Instant;

/// Frames to accumulate before diffraction is switched on (it costs a full
/// FFT convolution per displayed frame, and kernels are rebuilt on lens changes).
const DIFFRACTION_AFTER: u32 = 8;
const MAX_FRAMES: u32 = 1024;

#[derive(Clone, PartialEq)]
struct Controls {
    width: usize,
    fstop: f64,
    focus_m: f64,
    focus_inf: bool,
    blades: u32,
    curvature: f64,
    rotation: f64,
    coated: bool,
    ghosts: bool,
}

pub struct Viewer {
    renderer: GpuRenderer,
    render_state: egui_wgpu::RenderState,
    texture_id: Option<egui::TextureId>,
    texture_size: (usize, usize),
    base_lens: Lens,
    lens: Lens,
    scene: Scene,
    file: SceneFile,
    controls: Controls,
    applied: Option<Controls>,
    exposure_ev: f64,
    filmic: bool,
    diffraction: bool,
    max_aperture: f64,
    status: String,
    frame_ms: f64,
    last_frame: Instant,
    /// Self-check: change controls along the way, then after this many frames
    /// screenshot the window to LENS_VIEWER_SHOT and quit.
    shot: Option<(u32, String)>,
    updates: u32,
}

use eframe::egui_wgpu;

pub fn run(file: SceneFile, scene: Scene, lens: Lens, fstop: f64, focus_mm: Option<f64>) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 900.0]).with_title("lens-render"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: std::sync::Arc::new(gpu::device_descriptor),
                ..egui_wgpu::WgpuSetupCreateNew::without_display_handle()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "lens-render",
        options,
        Box::new(move |cc| {
            let rs = cc.wgpu_render_state.clone().ok_or("wgpu renderer unavailable")?;
            Ok(Box::new(Viewer::new(rs, file, scene, lens, fstop, focus_mm)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

impl Viewer {
    fn new(rs: egui_wgpu::RenderState, file: SceneFile, scene: Scene, lens: Lens, fstop: f64, focus_mm: Option<f64>) -> Viewer {
        let mut renderer = GpuRenderer::new(GpuContext { device: rs.device.clone(), queue: rs.queue.clone() });
        renderer.set_scene(&scene);
        let c = &file.camera;
        let controls = Controls {
            width: c.width as usize,
            fstop,
            focus_m: focus_mm.map_or(10.0, |f| f / 1000.0),
            focus_inf: focus_mm.is_none(),
            blades: c.iris_blades,
            curvature: c.iris_curvature,
            rotation: c.iris_rotation_deg,
            coated: c.coating != "none",
            ghosts: c.ghosts,
        };
        let max_aperture = lens.max_aperture();
        Viewer {
            renderer,
            render_state: rs,
            texture_id: None,
            texture_size: (0, 0),
            base_lens: lens.clone(),
            lens,
            exposure_ev: c.exposure_ev,
            filmic: c.tonemap == "filmic",
            diffraction: c.diffraction,
            scene,
            file,
            controls,
            applied: None,
            max_aperture,
            status: String::new(),
            frame_ms: 0.0,
            last_frame: Instant::now(),
            shot: std::env::var("LENS_VIEWER_SHOT").ok().map(|p| (32, p)),
            updates: 0,
        }
    }

    fn height(&self) -> usize {
        let c = &self.file.camera;
        (self.controls.width as f64 * c.sensor_height_mm / c.sensor_width_mm).round() as usize
    }

    /// Rebuilds the lens for the controls and restarts accumulation.
    fn apply(&mut self) -> Result<()> {
        let k = self.controls.clone();
        let mut lens = self.base_lens.clone();
        lens.coating = if k.coated { Coating::Single { n: 1.38, lambda0: 550.0 } } else { Coating::None };
        lens.set_aperture(k.fstop, k.blades, k.rotation, k.curvature)?;
        let focus = if k.focus_inf { None } else { Some(k.focus_m * 1000.0) };
        lens.sensor_z = lens.best_focus(focus)? + self.file.camera.focus_offset_mm;
        let c = &self.file.camera;
        let settings = gpu::Settings {
            w: k.width,
            h: self.height(),
            bins: c.spectral_bins,
            sensor_w: c.sensor_width_mm,
            norm: 4.0 * k.fstop * k.fstop / std::f64::consts::PI,
            spp_per_frame: 1,
            ghosts: k.ghosts,
            ghost_threshold: c.ghost_threshold,
            light_grid: c.ghost_grid.div_ceil(16) as u32 * 16,
            ghost_wavelengths: 4,
            tonemap: Tonemap::Clip,
        };
        self.renderer.configure(&lens, &settings)?;
        self.lens = lens;
        self.applied = Some(k);
        Ok(())
    }

    fn sync_texture(&mut self) {
        let size = (self.controls.width, self.height());
        let Some(tex) = self.renderer.texture() else { return };
        let view = tex.create_view(&Default::default());
        let mut r = self.render_state.renderer.write();
        match self.texture_id {
            Some(id) => r.update_egui_texture_from_wgpu_texture(&self.render_state.device, &view, wgpu::FilterMode::Linear, id),
            None => self.texture_id = Some(r.register_native_texture(&self.render_state.device, &view, wgpu::FilterMode::Linear)),
        }
        self.texture_size = size;
    }

    /// Scene distance (m from the front vertex) seen at a normalised image position.
    fn depth_at(&self, u: f32, v: f32) -> Option<f64> {
        let c = &self.file.camera;
        let (w, h) = (c.sensor_width_mm, c.sensor_width_mm * self.height() as f64 / self.controls.width as f64);
        // Output is upright, the sensor image inverted.
        let o = Vec3::new(-(u as f64 - 0.5) * w, (v as f64 - 0.5) * h, self.lens.sensor_z);
        let sd = self.lens.surfaces.last().unwrap().semi_diameter;
        let m = self.lens.media(crate::glass::LAMBDA_D);
        for k in 0..64 {
            let (px, py) = crate::math::concentric_disk((k % 8) as f64 / 8.0 + 0.0625, (k / 8) as f64 / 8.0 + 0.0625);
            let d = (Vec3::new(px * sd * 0.5, py * sd * 0.5, self.lens.last_z()) - o).normalize();
            if let Some((ray, _)) = self.lens.trace_backward(Ray { o, d }, crate::glass::LAMBDA_D, &m) {
                let t = self.scene.hit_distance(&ray)?;
                return Some(-(ray.o + ray.d * t).z / 1000.0);
            }
        }
        None
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        let k = &mut self.controls;
        ui.heading(&self.lens.name);
        ui.label(format!("f = {:.1} mm", self.lens.focal_length()));
        ui.separator();
        ui.add(egui::Slider::new(&mut k.fstop, self.max_aperture..=32.0).logarithmic(true).text("f-number"));
        ui.horizontal(|ui| {
            ui.add_enabled(!k.focus_inf, egui::Slider::new(&mut k.focus_m, 0.3..=100.0).logarithmic(true).text("focus (m)"));
            ui.checkbox(&mut k.focus_inf, "∞");
        });
        ui.label("Click the image to focus there.");
        ui.separator();
        ui.add(egui::Slider::new(&mut k.blades, 0..=12).text("iris blades"));
        ui.add(egui::Slider::new(&mut k.curvature, 0.0..=1.0).text("blade curvature"));
        ui.add(egui::Slider::new(&mut k.rotation, 0.0..=60.0).text("iris rotation"));
        ui.checkbox(&mut k.coated, "anti-reflection coating");
        ui.checkbox(&mut k.ghosts, "ghosts and veiling glare");
        ui.checkbox(&mut self.diffraction, "diffraction");
        ui.separator();
        ui.add(egui::Slider::new(&mut self.exposure_ev, -8.0..=8.0).text("exposure (EV)"));
        ui.checkbox(&mut self.filmic, "filmic tone curve");
        egui::ComboBox::from_label("width")
            .selected_text(format!("{} px", k.width))
            .show_ui(ui, |ui| {
                for wv in [480, 720, 960, 1200, 1600, 2400] {
                    ui.selectable_value(&mut k.width, wv, format!("{wv} px"));
                }
            });
        ui.separator();
        let frames = self.renderer.frames();
        let spp = frames as usize * self.file.camera.spectral_bins;
        ui.label(format!("{frames} frames, {spp} samples/pixel"));
        ui.label(format!("{:.1} ms per frame ({:.0} fps)", self.frame_ms, 1000.0 / self.frame_ms.max(0.001)));
        if ui.button("Save PNG").clicked() {
            self.status = match self.save("view.png") {
                Ok(()) => "wrote view.png".into(),
                Err(e) => format!("save failed: {e}"),
            };
        }
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
    }

    fn save(&mut self, path: &str) -> Result<()> {
        self.renderer.resolve(self.diffraction)?;
        let film = self.renderer.read_film()?;
        let cs = crate::spectrum::ColorSystem::new();
        let scale = 2f32.powf(self.exposure_ev as f32);
        let rgb: Vec<[f32; 3]> = film.to_rgb(&cs).into_iter().map(|p| p.map(|v| v * scale)).collect();
        crate::write_outputs(&rgb, film.w, film.h, path, if self.filmic { "filmic" } else { "clip" })
    }
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.updates += 1;
        if self.shot.is_some() {
            // Exercise lens and resolution changes before the screenshot.
            match self.updates {
                5 => self.controls.fstop = 8.0,
                10 => self.controls.blades = 0,
                15 => self.controls.width = 720,
                20 => self.controls.focus_inf = true,
                25 => {
                    self.controls.width = 1200;
                    self.controls.blades = 5;
                    self.controls.fstop = 11.0;
                    self.controls.focus_inf = false;
                }
                _ => {}
            }
        }
        egui::Panel::left("controls").exact_size(300.0).show(ui, |ui| self.controls_ui(ui));
        if self.applied.as_ref() != Some(&self.controls) {
            if let Err(e) = self.apply() {
                self.status = format!("{e}");
                self.controls = self.applied.clone().unwrap_or(self.controls.clone());
            }
        }
        let frames = self.renderer.frames();
        if frames < MAX_FRAMES {
            // At most two frames in flight: keeps input latency low and the timing honest.
            let _ = self.renderer.throttle(1);
            self.renderer.render_frame();
        }
        self.renderer.set_display(2f64.powf(self.exposure_ev), if self.filmic { Tonemap::Filmic } else { Tonemap::Clip });
        // Diffraction costs an FFT convolution per display update; once the image
        // has settled, refresh the display every 4th frame only.
        let diffract = self.diffraction && frames >= DIFFRACTION_AFTER;
        if !diffract || frames % 4 == 0 || frames >= MAX_FRAMES {
            if let Err(e) = self.renderer.resolve(diffract) {
                self.status = format!("{e}");
            }
        }
        self.sync_texture();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64() * 1000.0;
        self.last_frame = now;
        self.frame_ms = if self.frame_ms == 0.0 { dt } else { 0.9 * self.frame_ms + 0.1 * dt };

        egui::CentralPanel::default().show(ui, |ui| {
            let Some(id) = self.texture_id else { return };
            let avail = ui.available_size();
            let aspect = self.texture_size.0 as f32 / self.texture_size.1 as f32;
            let size = if avail.x / avail.y > aspect { egui::vec2(avail.y * aspect, avail.y) } else { egui::vec2(avail.x, avail.x / aspect) };
            let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
            ui.painter().image(id, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            if let Some(pos) = response.interact_pointer_pos().filter(|_| response.clicked()) {
                let u = (pos.x - rect.min.x) / rect.width();
                let v = (pos.y - rect.min.y) / rect.height();
                if let Some(d) = self.depth_at(u, v) {
                    self.controls.focus_m = d.clamp(0.3, 100.0);
                    self.controls.focus_inf = false;
                }
            }
        });
        if frames < MAX_FRAMES {
            ui.ctx().request_repaint();
        }
        if let Some((n, path)) = self.shot.clone() {
            if frames == n {
                let ctx = ui.ctx().clone();
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
                    let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                    let _ = image::RgbaImage::from_raw(img.size[0] as u32, img.size[1] as u32, rgba).map(|i| i.save(&path));
                    eprintln!("viewer screenshot written to {path}");
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                })));
            }
        }
    }
}

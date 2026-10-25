//! Interactive viewer: progressive GPU rendering with live lens controls.

use crate::gpu::wave::WaveGpu;
use crate::gpu::{self, GpuContext, GpuRenderer, Tonemap, wgpu};
use crate::lens::{Coating, Lens};
use crate::fieldpsf::FieldPsf;
use crate::ghosts::Ghost;
use crate::lensview::{self, Diagram, FocusTravel, GlassMap, LensEditor, PsfView, RayPath};
use crate::math::{Ray, Vec3};
use crate::scene::{Scene, SceneFile};
use crate::psfbank::PsfBank;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use eframe::egui;
use std::time::Instant;

/// Frames after a change during which every frame is displayed; later only
/// every 4th (with diffraction, each display costs an FFT convolution).
const DISPLAY_EVERY_FRAME: u32 = 8;
const MAX_FRAMES: u32 = 1024;
/// A demo frame is captured once the image has this many passes, so that it
/// looks finished rather than freshly restarted after a change.
const DEMO_PASSES: u32 = 32;

#[derive(Clone, PartialEq)]
struct Controls {
    lens: PathBuf,
    width: usize,
    fstop: f64,
    focus_m: f64,
    focus_inf: bool,
    blades: u32,
    curvature: f64,
    rotation: f64,
    coated: bool,
    ghosts: bool,
    wave: bool,
    /// Counts prescription edits.
    revision: u32,
}

impl Controls {
    /// The settings a PSF bank depends on (coating and ghosts do not matter).
    fn bank_key(&self) -> Controls {
        Controls { coated: true, ghosts: true, wave: true, ..self.clone() }
    }
}

/// A PSF bank being built in the background for the given controls.
struct BankJob {
    key: Controls,
    result: std::sync::mpsc::Receiver<PsfBank>,
    progress: Arc<AtomicU32>,
}

pub struct Viewer {
    renderer: GpuRenderer,
    render_state: egui_wgpu::RenderState,
    texture_id: Option<egui::TextureId>,
    texture_size: (usize, usize),
    /// The current lens file and its lens before aperture and focus are set.
    lens_path: PathBuf,
    base_lens: Lens,
    /// The scene's own lens (with its decentering), restored when reselected.
    scene_lens: (PathBuf, Lens),
    lens_files: Vec<(PathBuf, String)>,
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
    bank: Option<(Controls, Arc<PsfBank>)>,
    bank_job: Option<BankJob>,
    /// PSF banks are built on a device of their own: work submitted from
    /// another thread would make the window's surface reconfiguration fail.
    wave: Arc<std::sync::OnceLock<Option<WaveGpu>>>,
    editor: LensEditor,
    /// An edited prescription waiting to be applied.
    pending: Option<Lens>,
    show_lens: bool,
    by_wavelength: bool,
    show_fans: bool,
    show_ghosts: bool,
    show_focus: bool,
    diagram: Diagram,
    glass_map: GlassMap,
    psf_view: PsfView,
    tab: LensTab,
    seidel_shown: [bool; 7],
    /// Analyses of the applied lens (for `applies`), computed in the background.
    analysis: Option<Analysis>,
    analysis_job: Option<(u64, std::sync::mpsc::Receiver<Analysis>)>,
    /// MTF of the applied lens (slow, so only made while its tab is shown).
    mtf: Option<(u64, crate::mtf::Curves)>,
    mtf_job: Option<(u64, std::sync::mpsc::Receiver<crate::mtf::Curves>)>,
    /// Published MTF next to the lens file (`<stem>…mtf.json`), by lens file.
    mtf_reference: Option<(PathBuf, Option<serde_json::Value>)>,
    save_name: String,
    /// Ray fans of the applied lens, for this many applies and colouring.
    fans: Option<(u64, bool, Vec<RayPath>)>,
    applies: u64,
    frame_ms: f64,
    last_frame: Instant,
    /// Self-check: change controls along the way, then after this many frames
    /// screenshot the window to LENS_VIEWER_SHOT and quit.
    shot: Option<(u32, String)>,
    /// Demo recording (LENS_VIEWER_DEMO = its directory), and the start of
    /// the current simulated drag (phase, point).
    demo: Option<crate::demo::Demo>,
    demo_anchor: Option<(u8, egui::Pos2)>,
    image_rect: egui::Rect,
    updates: u32,
}

use eframe::egui_wgpu;

pub fn run(file: SceneFile, scene: Scene, lens_path: PathBuf, lens: Lens, fstop: f64, focus_mm: Option<f64>) -> Result<()> {
    let options = eframe::NativeOptions {
        // 16:9 when recording a demo.
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(if std::env::var_os("LENS_VIEWER_DEMO").is_some() { [1600.0, 900.0] } else { [1500.0, 900.0] })
            .with_title("lens-render"),
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
            Ok(Box::new(Viewer::new(rs, file, scene, lens_path, lens, fstop, focus_mm)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[derive(Clone, Copy, PartialEq)]
enum LensTab {
    Glass,
    Aberrations,
    Pupils,
    Psf,
    Mtf,
    LightLoss,
    Numbers,
}

/// Slower analyses of a lens, made off the UI thread.
struct Analysis {
    key: u64,
    ghosts: Vec<Ghost>,
    psfs: Vec<FieldPsf>,
    focus: FocusTravel,
    seidel: Option<crate::seidel::Seidel>,
    footprints: crate::footprint::Footprints,
    light_loss: crate::footprint::LightLoss,
}

impl Analysis {
    fn compute(key: u64, lens: &Lens, focus: Option<f64>, half_diag: f64) -> Analysis {
        // Ghosts of a distant light at half the image height.
        let ghosts = crate::psf::field_for_height(lens, 0.5 * half_diag, None)
            .map(|th| crate::ghosts::ghosts(lens, &crate::psf::Source::at_field(lens, th, None), 550.0))
            .unwrap_or_default();
        Analysis {
            key,
            ghosts,
            psfs: crate::fieldpsf::field_psfs(lens, focus, half_diag),
            focus: lensview::focus_travel(lens),
            seidel: crate::psf::field_for_height(lens, half_diag, focus).map(|th| crate::seidel::seidel(lens, th, focus)),
            footprints: crate::footprint::footprints(lens, focus, half_diag, 64),
            light_loss: crate::footprint::light_loss(lens, focus, half_diag),
        }
    }
}

/// Published MTF data next to a lens file: `<stem>…mtf.json`.
fn mtf_reference(lens: &Path) -> Option<serde_json::Value> {
    let stem = lens.file_stem()?.to_string_lossy().into_owned();
    let dir = lens.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let file = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| {
        p.file_name().map(|n| n.to_string_lossy().into_owned()).is_some_and(|n| n.starts_with(&stem) && n.ends_with("mtf.json"))
    })?;
    serde_json::from_str(&std::fs::read_to_string(file).ok()?).ok()
}

/// Default file name for saving an edited copy of the lens at `path`.
fn edited_name(path: &Path) -> String {
    format!("{}-edited.lens", path.file_stem().unwrap_or_default().to_string_lossy())
}

/// `*.lens` files next to `path`, sorted, with their `@name` (or file stem).
fn lens_files(path: &Path) -> Vec<(PathBuf, String)> {
    let dir = path.parent().unwrap_or(Path::new(""));
    let entries = std::fs::read_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir });
    let mut files: Vec<PathBuf> = entries.into_iter().flatten().flatten().map(|e| dir.join(e.file_name())).filter(|p| p.extension().is_some_and(|e| e == "lens")).collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            let name = text.lines().find_map(|l| l.trim().strip_prefix("@name")).map(|n| n.trim().to_string());
            let name = name.unwrap_or_else(|| p.file_stem().unwrap_or_default().to_string_lossy().into_owned());
            (p, name)
        })
        .collect()
}

impl Viewer {
    fn new(rs: egui_wgpu::RenderState, file: SceneFile, scene: Scene, lens_path: PathBuf, lens: Lens, fstop: f64, focus_mm: Option<f64>) -> Viewer {
        let mut renderer = GpuRenderer::new(GpuContext { device: rs.device.clone(), queue: rs.queue.clone() });
        // A demo waits for exact kernels; it is not interactive.
        renderer.background_kernels = std::env::var_os("LENS_VIEWER_DEMO").is_none();
        renderer.set_scene(&scene);
        let c = &file.camera;
        let controls = Controls {
            lens: lens_path.clone(),
            width: c.width as usize,
            fstop,
            focus_m: focus_mm.map_or(10.0, |f| f / 1000.0),
            focus_inf: focus_mm.is_none(),
            blades: c.iris_blades.unwrap_or(lens.blades),
            curvature: c.iris_curvature.unwrap_or(lens.blade_curvature),
            rotation: c.iris_rotation_deg,
            coated: c.coating != "none",
            ghosts: c.ghosts,
            wave: c.wave_optics,
            revision: 0,
        };
        let max_aperture = lens.max_aperture();
        Viewer {
            editor: LensEditor::new(&lens),
            save_name: edited_name(&lens_path),
            renderer,
            render_state: rs,
            texture_id: None,
            texture_size: (0, 0),
            lens_files: lens_files(&lens_path),
            scene_lens: (lens_path.clone(), lens.clone()),
            lens_path,
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
            bank: None,
            bank_job: None,
            wave: Default::default(),
            pending: None,
            show_lens: true,
            by_wavelength: false,
            show_fans: true,
            show_ghosts: false,
            show_focus: false,
            diagram: Diagram::default(),
            glass_map: GlassMap::default(),
            psf_view: PsfView::default(),
            tab: LensTab::Glass,
            // Distortion, in waves, dwarfs the rest; it is a click away.
            seidel_shown: [true, true, true, true, false, true, true],
            analysis: None,
            analysis_job: None,
            mtf: None,
            mtf_job: None,
            mtf_reference: None,
            fans: None,
            applies: 0,
            frame_ms: 0.0,
            last_frame: Instant::now(),
            shot: std::env::var("LENS_VIEWER_SHOT").ok().map(|p| (32, p)),
            demo: std::env::var_os("LENS_VIEWER_DEMO").and_then(|d| match crate::demo::Demo::load(d.into()) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("demo: {e:#}");
                    None
                }
            }),
            demo_anchor: None,
            image_rect: egui::Rect::NOTHING,
            updates: 0,
        }
    }

    fn height(&self) -> usize {
        let c = &self.file.camera;
        (self.controls.width as f64 * c.sensor_height_mm / c.sensor_width_mm).round() as usize
    }

    /// Rebuilds the lens for the controls and restarts accumulation.
    fn apply(&mut self) -> Result<()> {
        // An edited prescription, or another lens file (which brings its iris);
        // the f-number is clamped to the new lens' maximum aperture.
        let from_file = self.pending.is_none() && self.controls.lens != self.lens_path;
        let new_base = if let Some(p) = self.pending.take() {
            Some(p)
        } else if !from_file {
            None
        } else if self.controls.lens == self.scene_lens.0 {
            Some(self.scene_lens.1.clone())
        } else {
            Some(Lens::load(self.controls.lens.to_str().unwrap_or_default())?)
        };
        if let Some(b) = &new_base {
            self.controls.fstop = self.controls.fstop.max(b.max_aperture());
            if from_file {
                (self.controls.blades, self.controls.curvature) = (b.blades, b.blade_curvature);
            }
        }
        let k = self.controls.clone();
        let mut lens = new_base.clone().unwrap_or_else(|| self.base_lens.clone());
        lens.coating = if k.coated { Coating::Single { n: 1.38, lambda0: 550.0 } } else { Coating::None };
        let focus = if k.focus_inf { None } else { Some(k.focus_m * 1000.0) };
        lens.prepare(k.fstop, k.blades, k.rotation, k.curvature, focus, self.file.camera.focus_offset_mm)?;
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
        // Wave optics: use a bank built for these settings, or start building
        // one in the background and render geometrically meanwhile.
        let key = k.bank_key();
        let bank = if k.wave { self.bank.as_ref().filter(|(bk, _)| *bk == key).map(|(_, b)| b.clone()) } else { None };
        if k.wave && bank.is_none() && self.bank_job.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            let progress = Arc::new(AtomicU32::new(0));
            let (lens2, prog, wave) = (lens.clone(), progress.clone(), self.wave.clone());
            let (bins, pitch) = (c.spectral_bins, c.sensor_width_mm / k.width as f64);
            let half_diag = 0.5 * c.sensor_width_mm.hypot(c.sensor_height_mm);
            std::thread::spawn(move || {
                let gpu = wave.get_or_init(|| GpuContext::headless().ok().map(|c| WaveGpu::new(&c)));
                let b = PsfBank::build(&lens2, focus, bins, pitch, half_diag, gpu.as_ref(), &|p| prog.store((p * 1000.0) as u32, Ordering::Relaxed));
                let _ = tx.send(b);
            });
            self.bank_job = Some(BankJob { key, result: rx, progress });
        }
        self.renderer.configure(&lens, &settings, bank.as_deref())?;
        if let Some(b) = new_base {
            self.max_aperture = b.max_aperture();
            if from_file {
                self.editor = LensEditor::new(&b);
                self.save_name = edited_name(&k.lens);
            }
            self.base_lens = b;
            self.lens_path = k.lens.clone();
        }
        self.lens = lens;
        self.applied = Some(k);
        self.applies += 1;
        Ok(())
    }

    /// Takes a finished PSF bank; re-applies the controls so it is used (or a
    /// new one is started if the settings changed meanwhile).
    fn poll_bank(&mut self) {
        let Some(job) = &self.bank_job else { return };
        match job.result.try_recv() {
            Ok(b) => {
                let key = job.key.clone();
                self.bank_job = None;
                self.bank = Some((key, Arc::new(b)));
                if self.controls.wave {
                    self.applied = None;
                }
            }
            // The builder panicked: stop waiting, and don't retry.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.bank_job = None;
                self.controls.wave = false;
                self.status = "computing PSFs failed".into();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
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
        egui::ComboBox::from_id_salt("lens")
            .width(ui.available_width())
            .selected_text(egui::RichText::new(&self.lens.name).heading())
            .show_ui(ui, |ui| {
                for (path, name) in &self.lens_files {
                    ui.selectable_value(&mut k.lens, path.clone(), name);
                }
            });
        ui.horizontal(|ui| {
            ui.label(format!("f = {:.1} mm", self.lens.focal_length()));
            ui.checkbox(&mut self.show_lens, "lens panel");
        });
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
        ui.checkbox(&mut k.wave, "wave-optical PSFs");
        if let Some(job) = &self.bank_job {
            ui.label(format!("computing PSFs… {:.0} %", job.progress.load(Ordering::Relaxed) as f64 / 10.0));
        } else if k.wave {
            ui.label("diffraction and aberrations via wave-optical PSFs");
        }
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

    /// The demo's actions at its current time: controls as functions of the
    /// time within each segment, and simulated drags in the lens panel.
    fn demo_actions(&mut self) {
        let Some((seg, t)) = self.demo.as_ref().and_then(|d| d.segment()).map(|(s, t)| (s.clone(), t)) else { return };
        let u = (t / seg.dur).clamp(0.0, 1.0);
        let ease = |x: f64| {
            let x = x.clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let logl = |a: f64, b: f64, x: f64| (a.ln() + (b.ln() - a.ln()) * ease(x)).exp();
        let mut pointer = None;
        let k = &mut self.controls;
        match seg.id.as_str() {
            "intro" => {
                self.show_lens = false;
                (k.fstop, k.focus_m, k.focus_inf) = (3.5, 1.5, false);
            }
            "aperture" => k.fstop = logl(3.5, 11.0, u / 0.8),
            "focus" => {
                k.fstop = 5.6;
                k.focus_m = if u < 0.5 { logl(1.5, 6.0, u / 0.45) } else { logl(6.0, 0.7, (u - 0.5) / 0.45) };
            }
            "iris" => {
                k.focus_m = 1.5;
                k.blades = if u < 0.25 { 6 } else if u < 0.55 { 3 } else { 9 };
                k.curvature = if u < 0.75 { 0.0 } else { ease((u - 0.75) / 0.2) };
            }
            "coating" => (k.blades, k.curvature, k.coated) = (6, 0.0, false),
            "summilux" => {
                if let Some((p, _)) = self.lens_files.iter().find(|(p, _)| p.to_string_lossy().contains("summilux")) {
                    k.lens = p.clone();
                }
                (k.coated, k.fstop) = (true, 1.4);
            }
            "wave" => k.wave = true,
            "editor" => {
                self.show_lens = true;
                self.tab = LensTab::Glass;
                // Drag an element, bend the front surface by its rim, then move its glass.
                let phases: [(u8, f64, f64, egui::Vec2); 3] = [(1, 0.05, 0.32, egui::vec2(45.0, 0.0)), (2, 0.38, 0.62, egui::vec2(-22.0, -14.0)), (3, 0.68, 0.92, egui::vec2(-90.0, 45.0))];
                for (phase, a, b, delta) in phases {
                    if u < a - 0.03 || u > b + 0.02 {
                        continue;
                    }
                    let lens = &self.lens;
                    let anchor = match self.demo_anchor.filter(|(p, _)| *p == phase) {
                        Some((_, at)) => Some(at),
                        None => {
                            let first_element = (1..lens.surfaces.len()).find(|&i| lens.surfaces[i].material.is_some() && lens.surfaces[i - 1].material.is_none()).unwrap_or(0);
                            let at = match phase {
                                1 => self.diagram.point(lens, first_element, 0.0),
                                2 => self.diagram.point(lens, 0, lens.surfaces[0].semi_diameter),
                                _ => self.editor.draft.surfaces[0].material.as_ref().and_then(|g| self.glass_map.position(g)),
                            };
                            if let Some(at) = at {
                                self.demo_anchor = Some((phase, at));
                            }
                            at
                        }
                    };
                    if let Some(at) = anchor {
                        let down = u >= a && u <= b;
                        pointer = Some((at + delta * ease((u - a) / (b - a - 0.02)) as f32, down));
                    }
                }
                if u > 0.96 && !self.diagram.undo.is_empty() {
                    self.undo();
                }
            }
            "overlays" => {
                (self.show_fans, self.show_ghosts, self.show_focus) = (true, true, true);
                k.fstop = 1.4;
            }
            "aberrations" => self.tab = LensTab::Aberrations,
            "pupils" => self.tab = LensTab::Pupils,
            "psf" => self.tab = LensTab::Psf,
            "mtf" => self.tab = LensTab::Mtf,
            "light" => self.tab = LensTab::LightLoss,
            "outro" => {
                self.show_lens = false;
                (self.show_ghosts, self.show_focus) = (false, false);
            }
            _ => {}
        }
        if let Some(d) = &mut self.demo {
            d.pointer = pointer;
        }
    }

    /// Applies an edited draft (or reports why it is invalid).
    fn edited(&mut self) {
        match Lens::parse(&self.editor.draft.to_text()) {
            Ok(l) => {
                self.editor.error.clear();
                self.pending = Some(l);
                self.controls.revision += 1;
            }
            Err(e) => self.editor.error = format!("{e:#}"),
        }
    }

    fn undo(&mut self) {
        if let Some(l) = self.diagram.undo.pop() {
            self.editor.draft = l;
            self.edited();
        }
    }

    /// The lens panel: interactive cross-section, analyses and the prescription.
    fn lens_ui(&mut self, ui: &mut egui::Ui) {
        let c = &self.file.camera;
        let half_diag = 0.5 * c.sensor_width_mm.hypot(c.sensor_height_mm);
        let k = self.applied.clone().unwrap_or(self.controls.clone());
        let focus = if k.focus_inf { None } else { Some(k.focus_m * 1000.0) };
        let key = (self.applies, self.by_wavelength);
        if self.fans.as_ref().map(|f| (f.0, f.1)) != Some(key) {
            self.fans = Some((key.0, key.1, lensview::trace_fans(&self.lens, focus, half_diag, self.by_wavelength)));
        }
        if let Some((_, rx)) = &self.analysis_job {
            if let Ok(a) = rx.try_recv() {
                self.analysis = Some(a);
                self.analysis_job = None;
            }
        }
        if self.analysis_job.is_none() && self.analysis.as_ref().map(|a| a.key) != Some(self.applies) {
            let (tx, rx) = std::sync::mpsc::channel();
            let (lens, applies) = (self.lens.clone(), self.applies);
            std::thread::spawn(move || {
                let _ = tx.send(Analysis::compute(applies, &lens, focus, half_diag));
            });
            self.analysis_job = Some((applies, rx));
        }
        // The demo needs its MTF ready when the tab comes up; editing is over by then.
        let demo_mtf = self.demo.as_ref().and_then(|d| d.segment()).is_some_and(|(s, _)| ["aberrations", "pupils", "psf", "mtf"].contains(&s.id.as_str()));
        if self.tab == LensTab::Mtf || demo_mtf {
            if let Some((_, rx)) = &self.mtf_job {
                if let Ok(c) = rx.try_recv() {
                    self.mtf = Some((self.mtf_job.take().unwrap().0, c));
                }
            }
            if self.mtf_job.is_none() && self.mtf.as_ref().map(|m| m.0) != Some(self.applies) {
                let (tx, rx) = std::sync::mpsc::channel();
                let lens = self.lens.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(crate::mtf::curves(&lens, focus, half_diag));
                });
                self.mtf_job = Some((self.applies, rx));
            }
            if self.mtf_reference.as_ref().map(|r| &r.0) != Some(&self.lens_path) {
                self.mtf_reference = Some((self.lens_path.clone(), mtf_reference(&self.lens_path)));
            }
        }
        if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z)) {
            self.undo();
        }

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_fans, "rays").on_hover_text("ray fans for image heights 0, 0.7 and 1 (green, red, blue); × where a ray is blocked");
            ui.checkbox(&mut self.by_wavelength, "by wavelength");
            ui.checkbox(&mut self.show_ghosts, "ghosts").on_hover_text("the three strongest ghosts of a distant light at half the image height");
            ui.checkbox(&mut self.show_focus, "focus travel").on_hover_text("the lens focused at ∞ and at its closest distance, aligned on the sensor");
            ui.separator();
            if ui.add_enabled(!self.diagram.undo.is_empty(), egui::Button::new("Undo")).clicked() {
                self.undo();
            }
            if ui.button("Revert").on_hover_text("reload the lens file").clicked() {
                match Lens::load(self.lens_path.to_str().unwrap_or_default()) {
                    Ok(l) => {
                        self.diagram.undo.push(self.editor.draft.clone());
                        self.editor.draft = l;
                        self.edited();
                    }
                    Err(e) => self.status = format!("{e:#}"),
                }
            }
            ui.add(egui::TextEdit::singleline(&mut self.save_name).desired_width(150.0));
            if ui.button("Save as").clicked() {
                let path = self.lens_path.with_file_name(&self.save_name);
                match std::fs::write(&path, self.editor.draft.to_text()) {
                    Ok(()) => {
                        self.status = format!("wrote {}", path.display());
                        self.lens_files = lens_files(&path);
                        self.lens_path = path.clone();
                        self.controls.lens = path;
                    }
                    Err(e) => self.status = format!("save failed: {e}"),
                }
            }
            if !self.editor.error.is_empty() {
                ui.colored_label(egui::Color32::RED, &self.editor.error);
            }
        });
        let h = ui.available_height();
        let w = ui.available_width();
        let taken = self.analysis.take();
        let analysis = taken.as_ref().filter(|a| a.key == self.applies);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(w * 0.55, h), egui::Layout::top_down(egui::Align::LEFT), |ui| {
                let overlays = lensview::Overlays {
                    fans: self.show_fans.then(|| self.fans.as_ref().unwrap().2.as_slice()),
                    ghosts: analysis.filter(|_| self.show_ghosts).map(|a| &a.ghosts[..a.ghosts.len().min(3)]),
                    focus: analysis.filter(|_| self.show_focus).map(|a| &a.focus),
                };
                if self.diagram.ui(ui, &self.lens, &mut self.editor.draft, &overlays, half_diag) {
                    self.edited();
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    let tabs = [
                        (LensTab::Glass, "Glass"),
                        (LensTab::Aberrations, "Aberrations"),
                        (LensTab::Pupils, "Pupils"),
                        (LensTab::Psf, "PSF"),
                        (LensTab::Mtf, "MTF"),
                        (LensTab::LightLoss, "Light loss"),
                        (LensTab::Numbers, "Numbers"),
                    ];
                    for (t, label) in tabs {
                        ui.selectable_value(&mut self.tab, t, label);
                    }
                });
                ui.separator();
                match self.tab {
                    LensTab::Glass => {
                        if self.glass_map.ui(ui, &mut self.editor.draft, self.diagram.selected, &mut self.diagram.undo) {
                            self.edited();
                        }
                    }
                    LensTab::Psf => self.psf_view.ui(ui, analysis.map(|a| (a.key, a.psfs.as_slice()))),
                    LensTab::Mtf => {
                        let published = self.mtf_reference.as_ref().and_then(|r| r.1.as_ref()).and_then(|j| lensview::published(j, k.fstop));
                        match self.mtf.as_ref().filter(|m| m.0 == self.applies) {
                            Some((_, c)) => lensview::mtf_ui(ui, c, published.as_ref()),
                            None => drop(ui.label("computing MTF…")),
                        }
                    }
                    LensTab::Aberrations => match analysis.map(|a| a.seidel.as_ref()) {
                        Some(Some(sd)) => lensview::seidel_chart(ui, sd, &mut self.seidel_shown),
                        Some(None) => drop(ui.label("no chief ray for the full field")),
                        None => drop(ui.label("computing…")),
                    },
                    LensTab::Pupils => match analysis {
                        Some(a) => lensview::footprint_plots(ui, &a.footprints),
                        None => drop(ui.label("computing…")),
                    },
                    LensTab::LightLoss => match analysis {
                        Some(a) => lensview::light_loss_chart(ui, &a.light_loss),
                        None => drop(ui.label("computing…")),
                    },
                    LensTab::Numbers => {
                        let before = self.editor.draft.clone();
                        if let Some(l) = self.editor.ui(ui) {
                            self.diagram.undo.push(before);
                            self.pending = Some(l);
                            self.controls.revision += 1;
                        }
                    }
                }
            });
        });
        self.analysis = taken;
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
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        if let Some(d) = &mut self.demo {
            d.input(raw);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.updates += 1;
        self.poll_bank();
        self.demo_actions();
        let shot_tab = std::env::var("LENS_VIEWER_TAB").ok();
        if self.shot.is_some() && self.updates == 5 && shot_tab.is_some() {
            // Just the given tab, with overlays and the first element selected.
            (self.show_ghosts, self.show_focus) = (true, true);
            self.diagram.selected = Some(0);
            self.tab = match shot_tab.as_deref() {
                Some("glass") => LensTab::Glass,
                Some("aberrations") => LensTab::Aberrations,
                Some("pupils") => LensTab::Pupils,
                Some("light") => LensTab::LightLoss,
                Some("mtf") => LensTab::Mtf,
                Some("numbers") => LensTab::Numbers,
                _ => LensTab::Psf,
            };
        }
        if self.shot.is_some() && !self.controls.wave && shot_tab.is_none() {
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
                // Another lens wide open, back to the scene's (f-number clamped), and again.
                30 | 45 => {
                    if let Some((p, _)) = self.lens_files.iter().rev().find(|(p, _)| *p != self.scene_lens.0) {
                        self.controls.lens = p.clone();
                    }
                }
                35 => self.controls.fstop = self.max_aperture,
                40 => self.controls.lens = self.scene_lens.0.clone(),
                // A prescription edit: a weaker first surface.
                50 => {
                    self.editor.draft.surfaces[0].radius *= 1.2;
                    self.pending = Lens::parse(&self.editor.draft.to_text()).ok();
                    self.controls.revision += 1;
                }
                _ => {}
            }
        }
        egui::Panel::left("controls").exact_size(300.0).show(ui, |ui| self.controls_ui(ui));
        if self.show_lens && self.applied.is_some() {
            egui::Panel::bottom("lens").resizable(true).default_size(340.0).show(ui, |ui| self.lens_ui(ui));
        }
        if self.applied.as_ref() != Some(&self.controls) {
            if let Err(e) = self.apply() {
                self.status = format!("{e:#}");
                self.controls = self.applied.clone().unwrap_or(self.controls.clone());
            }
        }
        if self.applied.is_none() {
            // The scene's settings could not be applied; nothing to render.
            egui::CentralPanel::default().show(ui, |ui| ui.label(&self.status));
            return;
        }
        let frames = self.renderer.frames();
        // A demo renders several passes per update to reach DEMO_PASSES sooner,
        // and no more: every captured frame then has the same passes (and
        // sample patterns), whenever its screenshot is taken.
        let limit = if self.demo.is_some() { DEMO_PASSES } else { MAX_FRAMES };
        for _ in 0..if self.demo.is_some() { 4 } else { 1 } {
            if self.renderer.frames() < limit {
                // At most two frames in flight: keeps input latency low and the timing honest.
                let _ = self.renderer.throttle(1);
                self.renderer.render_frame();
            }
        }
        self.renderer.set_display(2f64.powf(self.exposure_ev), if self.filmic { Tonemap::Filmic } else { Tonemap::Clip });
        let diffract = self.diffraction;
        if !diffract || frames < DISPLAY_EVERY_FRAME || frames % 4 == 0 || frames >= MAX_FRAMES || self.demo.is_some() {
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
            self.image_rect = rect;
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
        if frames < MAX_FRAMES || self.bank_job.is_some() {
            ui.ctx().request_repaint();
        }
        if let Some(d) = &self.demo {
            d.overlay(ui.ctx(), self.image_rect);
            if d.capture(ui.ctx(), self.renderer.frames() >= DEMO_PASSES) {
                eprintln!("demo frames written");
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ui.ctx().request_repaint();
        }
        if let Some((n, path)) = self.shot.clone() {
            // With wave optics, wait until the PSF bank is in use.
            let bank_ready = (!self.controls.wave || (self.bank_job.is_none() && self.bank.is_some()))
                && (!self.show_lens || self.analysis.as_ref().is_some_and(|a| a.key == self.applies))
                && (self.tab != LensTab::Mtf || self.mtf.as_ref().is_some_and(|m| m.0 == self.applies))
                && self.updates > 60;
            if frames >= n && bank_ready && self.shot.is_some() {
                self.shot = None;
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

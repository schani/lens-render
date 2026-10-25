//! Explainer video: a narrated lesson in lens design, recorded offscreen
//! from a timeline (`timeline.json`, written by scripts/explainer_video.py
//! from explainer/script.toml). Each chapter sets and animates the lens, the
//! scene and what is shown around the render; see `draw` for the layout.
//!
//! Every captured frame whose render changed has exactly `PASSES` passes;
//! frames where it didn't reuse the finished image. A second finished image
//! is kept for crossfades and split screens.

mod draw;
mod optics;

use crate::demo::Demo;
use crate::gpu::{self, GpuContext, GpuRenderer, Tonemap, wgpu};
use crate::lens::Lens;
use crate::offscreen::Offscreen;
use crate::scene::{Scene, SceneFile};
use anyhow::{Context, Result};
use eframe::egui;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PASSES: u32 = 32;
/// Crossfade between different lenses or scenes (s).
const CROSSFADE: f64 = 0.6;
/// Fade of views, tags and annotations (s).
const FADE: f64 = 0.35;

#[derive(Deserialize)]
struct LensEntry {
    path: PathBuf,
    label: String,
}

#[derive(Deserialize)]
struct RecapEntry {
    lens: String,
    label: String,
    #[serde(default)]
    tool: String,
}

#[derive(Deserialize)]
struct Timeline {
    width: usize,
    scenes: HashMap<String, PathBuf>,
    lenses: HashMap<String, LensEntry>,
    bending: Option<PathBuf>,
    #[serde(default)]
    recap: Vec<RecapEntry>,
    chapters: Vec<Chapter>,
}

#[derive(Deserialize, Clone, Copy, Default)]
#[serde(rename_all = "lowercase")]
enum Ease {
    #[default]
    Inout,
    In,
    Out,
    Linear,
}

impl Ease {
    fn at(self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Ease::Inout => x * x * (3.0 - 2.0 * x),
            Ease::In => x * x,
            Ease::Out => 1.0 - (1.0 - x) * (1.0 - x),
            Ease::Linear => x,
        }
    }
}

/// Property changes at chapter time `t`, animated over `dur` seconds.
#[derive(Deserialize)]
struct Key {
    t: f64,
    #[serde(default)]
    dur: f64,
    #[serde(default)]
    ease: Ease,
    /// A lens change morphs the prescription instead of crossfading.
    #[serde(default)]
    morph: bool,
    set: Map<String, Value>,
}

/// An annotation: circle, arrow, box or label, on `on` (render, centre,
/// corner, diagram, view, scoreboard, screen).
#[derive(Deserialize, Clone)]
pub struct Note {
    t: f64,
    until: Option<f64>,
    on: String,
    shape: String,
    #[serde(default)]
    pos: [f64; 2],
    to: Option<[f64; 2]>,
    r: Option<[f64; 2]>,
    text: Option<String>,
    text_at: Option<[f64; 2]>,
    color: Option<String>,
    /// On the scoreboard: box these read-outs or groups (comma-separated names).
    pub row: Option<String>,
}

/// A PROBLEM / FIX caption.
#[derive(Deserialize, Clone)]
pub struct Tag {
    t: f64,
    until: Option<f64>,
    kind: String,
    text: String,
}

#[derive(Deserialize)]
struct Tool {
    t: f64,
    text: String,
}

#[derive(Deserialize)]
struct Chapter {
    id: String,
    title: String,
    #[serde(default)]
    number: u32,
    start: f64,
    dur: f64,
    #[serde(default)]
    set: Map<String, Value>,
    #[serde(default)]
    keys: Vec<Key>,
    #[serde(default)]
    notes: Vec<Note>,
    #[serde(default)]
    tags: Vec<Tag>,
    toolbox: Option<Tool>,
    /// A part's title card.
    part: Option<Tool>,
}

/// A boolean that fades: on or off since an absolute video time.
#[derive(Clone, Copy)]
pub struct Toggle {
    on: bool,
    since: f64,
}

impl Default for Toggle {
    fn default() -> Toggle {
        Toggle { on: false, since: f64::NEG_INFINITY }
    }
}

impl Toggle {
    fn set(&mut self, on: bool, at: f64) {
        if on != self.on {
            *self = Toggle { on, since: at };
        }
    }

    /// Visibility at video time `now`, fading over `fade` s.
    pub fn alpha(&self, now: f64, fade: f64) -> f32 {
        let x = ((now - self.since) / fade).clamp(0.0, 1.0) as f32;
        let x = x * x * (3.0 - 2.0 * x);
        if self.on { x } else { 1.0 - x }
    }
}

/// What determines a rendered image.
#[derive(Clone, PartialEq, Debug)]
struct RenderKey {
    scene: String,
    lens: String,
    /// Prescription morphed towards this lens by this much.
    morph: Option<(String, f64)>,
    fstop: f64,
    /// Focus distance (m).
    focus: f64,
    /// Sensor shift from best focus (mm).
    offset: f64,
}

impl RenderKey {
    fn id(&self) -> String {
        format!("{self:?}")
    }
}

/// Everything shown at one moment.
#[derive(Clone)]
pub struct State {
    scene: String,
    lens: String,
    morph: Option<(String, f64)>,
    /// Crossfade from this image, started at this time.
    fade: Option<(f64, RenderKey)>,
    fstop: f64,
    focus: f64,
    offset: f64,
    ev: f64,
    /// 1: the image darkens with the f-number past `dim_ref` (shutter fixed);
    /// 0: exposure compensated by the shutter.
    dim: f64,
    dim_ref: f64,
    /// Split screen: the left part (up to `split` of the width) through this lens.
    split_lens: Option<String>,
    split_fstop: Option<f64>,
    split: f64,
    /// 1: the render fills the screen.
    full: f64,
    /// 1: attention on the centre (corners dimmed).
    spotlight: f64,
    /// Inset crop centres (normalised image coordinates) and magnification.
    insets: [[f64; 2]; 2],
    inset_zoom: f64,
    /// Image heights (fractions of the half diagonal) of the ray fans.
    fields: Vec<f64>,
    rays: usize,
    by_wavelength: bool,
    /// Glass pieces (0 = front) shown highlighted.
    highlight: Vec<usize>,
    /// Focus zoom ranges (mm: half length along, half height across); 0 = auto.
    zoom_len: f64,
    zoom_height: f64,
    /// Field-curve plot range (mm); 0 = auto.
    field_range: f64,
    /// Magnification of the cross-section (about the sensor).
    diagram_zoom: f64,
    flags: HashMap<String, Toggle>,
}

/// Flags that switch parts of the picture on and off.
const FLAGS: [&str; 11] = ["focus_zoom", "field_curves", "bending", "ideal_grid", "scoreboard", "hud", "insets", "diagram", "beam", "recap", "toolbox"];

impl Default for State {
    fn default() -> State {
        State {
            scene: String::new(),
            lens: String::new(),
            morph: None,
            fade: None,
            fstop: 4.0,
            focus: 2.5,
            offset: 0.0,
            ev: 0.0,
            dim: 0.0,
            dim_ref: 4.0,
            split_lens: None,
            split_fstop: None,
            split: 0.0,
            full: 0.0,
            spotlight: 0.0,
            insets: [[0.531, 0.5], [0.067, 0.92]],
            inset_zoom: 3.0,
            fields: vec![0.0],
            rays: 11,
            by_wavelength: false,
            highlight: Vec::new(),
            zoom_len: 0.0,
            zoom_height: 0.0,
            field_range: 0.0,
            diagram_zoom: 1.0,
            flags: FLAGS.iter().map(|f| (f.to_string(), Toggle::default())).collect(),
        }
    }
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn pair(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
}

impl State {
    pub fn flag(&self, name: &str) -> Toggle {
        self.flags.get(name).copied().unwrap_or_default()
    }

    fn key(&self) -> RenderKey {
        match &self.morph {
            Some((to, e)) if *e >= 1.0 => RenderKey { scene: self.scene.clone(), lens: to.clone(), morph: None, fstop: self.fstop, focus: self.focus, offset: self.offset },
            _ => RenderKey { scene: self.scene.clone(), lens: self.lens.clone(), morph: self.morph.clone(), fstop: self.fstop, focus: self.focus, offset: self.offset },
        }
    }

    /// The left image of a split screen.
    fn split_key(&self) -> Option<RenderKey> {
        let lens = self.split_lens.clone().filter(|_| self.split > 0.0)?;
        Some(RenderKey { lens, morph: None, fstop: self.split_fstop.unwrap_or(self.fstop), offset: 0.0, ..self.key() })
    }

    /// Applies `set` by `e` (0…1 of the way) at video time `at`; `corner`
    /// gives the sensor shift that focuses the corners of a lens state.
    fn apply(&mut self, set: &Map<String, Value>, e: f64, at: f64, morph: bool, corner: &mut dyn FnMut(&RenderKey) -> f64) {
        let before = self.key();
        let lin = |a: f64, b: f64| if e >= 1.0 { b } else { a + (b - a) * e };
        let log = |a: f64, b: f64| if e >= 1.0 { b } else { (a.ln() + (b.ln() - a.ln()) * e).exp() };
        for (name, v) in set {
            match name.as_str() {
                "fstop" => self.fstop = num(v).map_or(self.fstop, |b| log(self.fstop, b)),
                "focus" => self.focus = num(v).map_or(self.focus, |b| log(self.focus, b)),
                "offset" => {
                    let b = if v.as_str() == Some("corner") { corner(&self.key()) } else { num(v).unwrap_or(0.0) };
                    self.offset = lin(self.offset, b);
                }
                "ev" => self.ev = num(v).map_or(self.ev, |b| lin(self.ev, b)),
                "dim" => self.dim = num(v).map_or(self.dim, |b| lin(self.dim, b)),
                "dim_ref" => self.dim_ref = num(v).unwrap_or(self.dim_ref),
                "split" => self.split = num(v).map_or(self.split, |b| lin(self.split, b)),
                "full" => self.full = num(v).map_or(self.full, |b| lin(self.full, b)),
                "spotlight" => self.spotlight = num(v).map_or(self.spotlight, |b| lin(self.spotlight, b)),
                "split_lens" => self.split_lens = v.as_str().map(String::from),
                "split_fstop" => self.split_fstop = num(v),
                "inset_zoom" => self.inset_zoom = num(v).map_or(self.inset_zoom, |b| lin(self.inset_zoom, b)),
                "center" | "corner" => {
                    let i = (name == "corner") as usize;
                    if let Some(p) = pair(v) {
                        self.insets[i] = [lin(self.insets[i][0], p[0]), lin(self.insets[i][1], p[1])];
                    }
                }
                "zoom_len" => self.zoom_len = num(v).unwrap_or(0.0),
                "zoom_height" => self.zoom_height = num(v).unwrap_or(0.0),
                "field_range" => self.field_range = num(v).unwrap_or(0.0),
                "diagram_zoom" => self.diagram_zoom = num(v).map_or(self.diagram_zoom, |b| log(self.diagram_zoom, b)),
                "fields" => self.fields = v.as_array().map(|a| a.iter().filter_map(num).collect()).unwrap_or_default(),
                "rays" => self.rays = num(v).map_or(self.rays, |n| n as usize),
                "by_wavelength" => self.by_wavelength = v.as_bool().unwrap_or(false),
                "highlight" => {
                    let all: Vec<f64> = v.as_array().map_or_else(|| num(v).into_iter().collect(), |a| a.iter().filter_map(num).collect());
                    self.highlight = all.into_iter().filter(|n| *n >= 0.0).map(|n| n as usize).collect();
                }
                "scene" | "lens" => {
                    let Some(to) = v.as_str().map(String::from) else { continue };
                    if name == "lens" && morph {
                        if e >= 1.0 {
                            (self.lens, self.morph) = (to, None);
                        } else {
                            self.morph = Some((to, e));
                        }
                        continue;
                    }
                    let current = if name == "scene" { &self.scene } else { &self.lens };
                    if *current != to {
                        // One crossfade from the image before this change.
                        if !before.lens.is_empty() && !before.scene.is_empty() && self.fade.as_ref().is_none_or(|f| f.0 != at) {
                            self.fade = Some((at, before.clone()));
                        }
                        if name == "scene" { self.scene = to } else { self.lens = to }
                    }
                }
                f if FLAGS.contains(&f) => {
                    if let Some(on) = v.as_bool() {
                        self.flags.entry(f.to_string()).or_default().set(on, at);
                    }
                }
                other => eprintln!("explainer: unknown property {other}"),
            }
        }
    }
}

/// Lens state for a render key, with everything shown about it.
pub struct Optics {
    lens: Lens,
    label: String,
    fstop: f64,
    /// Ray fans per field; the axial fan with more rays for the focus zoom.
    fans: Vec<(f64, Vec<optics::RayPath>)>,
    axial: Vec<optics::RayPath>,
    /// The scoreboard (`metrics::scoreboard`).
    score: crate::metrics::Scoreboard,
    curves: Vec<[f64; 4]>,
    shape: Option<f64>,
    grid: Vec<[[f32; 2]; 2]>,
    elements: usize,
}

/// A finished image kept for crossfades and split screens.
struct Held {
    key: RenderKey,
    texture: wgpu::Texture,
    id: egui::TextureId,
}

pub struct Explainer {
    timeline: Timeline,
    /// State at the start of each chapter.
    base: Vec<State>,
    scenes: HashMap<String, (SceneFile, Scene)>,
    lenses: HashMap<String, Lens>,
    renderer: GpuRenderer,
    rs: eframe::egui_wgpu::RenderState,
    /// The scene and image the renderer holds, and its texture in egui.
    scene: Option<String>,
    live: Option<RenderKey>,
    live_id: Option<egui::TextureId>,
    exposure: f64,
    held: Option<Held>,
    optics: HashMap<String, Arc<Optics>>,
    corner_offsets: HashMap<String, f64>,
    bending: Vec<[f64; 2]>,
    /// Smoothed read-outs: 7 aberrations, centre and corner blur (log µm).
    shown: Option<[f64; 9]>,
    /// Smoothed focus-zoom and field-curve ranges.
    zoom: Option<(f64, f64)>,
    field_range: Option<f64>,
    /// Where the render is drawn this frame.
    rr: egui::Rect,
    /// Where the cross-section is drawn this frame.
    dr: egui::Rect,
    frame: u64,
    dt: f64,
}

/// Records the explainer whose timeline is in `dir` into `dir/frames`.
pub fn record(dir: &Path) -> Result<()> {
    let mut demo = Demo::load(dir.to_path_buf())?;
    let timeline: Timeline = serde_json::from_str(&std::fs::read_to_string(dir.join("timeline.json"))?)?;
    let fps = demo.fps();
    let mut off = Offscreen::new(egui::vec2(draw::W, draw::H), 2.0)?;
    draw::fonts(&off.ctx);
    let mut ex = Explainer::new(timeline, off.rs.clone(), 1.0 / fps)?;
    let t0 = std::time::Instant::now();
    let mut renders = 0;
    while !demo.done() {
        if !demo.wants(true) {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        let Some((ci, t)) = demo.segment().and_then(|(s, t)| Some((ex.timeline.chapters.iter().position(|c| c.id == s.id)?, t))) else { break };
        let state = ex.state(ci, t);
        renders += ex.prepare(&state, demo.time())?;
        let ctx = off.ctx.clone();
        let out = ctx.run_ui(off.input(), |ui| ex.draw(ui, ci, t, &state, demo.time()));
        if let Some(img) = off.paint(out, true)? {
            demo.deliver(img);
        }
        ex.frame += 1;
        if ex.frame % 30 == 0 {
            eprint!("\r{:.1} s of video, {renders} renders, {:.0} s", demo.time(), t0.elapsed().as_secs_f64());
        }
    }
    eprintln!("\nexplainer frames written: {} frames, {renders} renders in {:.0} s", ex.frame, t0.elapsed().as_secs_f64());
    Ok(())
}

impl Explainer {
    fn new(timeline: Timeline, rs: eframe::egui_wgpu::RenderState, dt: f64) -> Result<Explainer> {
        let mut lenses = HashMap::new();
        for (id, e) in &timeline.lenses {
            lenses.insert(id.clone(), Lens::load(e.path.to_str().unwrap_or_default()).with_context(|| format!("lens {id}: {}", e.path.display()))?);
        }
        let mut scenes = HashMap::new();
        let cs = crate::spectrum::ColorSystem::new();
        for (id, path) in &timeline.scenes {
            let file: SceneFile = toml::from_str(&std::fs::read_to_string(path).with_context(|| format!("scene {}", path.display()))?)?;
            let scene = Scene::build(&file, path.parent().unwrap_or(Path::new(".")), &cs)?;
            scenes.insert(id.clone(), (file, scene));
        }
        let bending = match &timeline.bending {
            Some(p) => draw::bending_curve(&serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("{}", p.display()))?)?),
            None => Vec::new(),
        };
        let mut renderer = GpuRenderer::new(GpuContext { device: rs.device.clone(), queue: rs.queue.clone() });
        renderer.background_kernels = false;
        let mut ex = Explainer {
            timeline,
            base: Vec::new(),
            scenes,
            lenses,
            renderer,
            rs,
            scene: None,
            live: None,
            live_id: None,
            exposure: 1.0,
            held: None,
            optics: HashMap::new(),
            corner_offsets: HashMap::new(),
            bending,
            shown: None,
            zoom: None,
            field_range: None,
            rr: egui::Rect::NOTHING,
            dr: egui::Rect::NOTHING,
            frame: 0,
            dt,
        };
        let mut s = State::default();
        for ci in 0..ex.timeline.chapters.len() {
            let (set, start) = (ex.timeline.chapters[ci].set.clone(), ex.timeline.chapters[ci].start);
            s.apply(&set, 1.0, start, false, &mut |k| ex.corner_offset(k));
            ex.base.push(s.clone());
            let dur = ex.timeline.chapters[ci].dur;
            s = ex.state(ci, dur + 1.0);
        }
        Ok(ex)
    }

    /// The state at time `t` of chapter `ci`.
    fn state(&mut self, ci: usize, t: f64) -> State {
        let mut s = self.base[ci].clone();
        let start = self.timeline.chapters[ci].start;
        let keys: Vec<(f64, f64, Ease, bool, Map<String, Value>)> = self.timeline.chapters[ci].keys.iter().filter(|k| k.t <= t).map(|k| (k.t, k.dur, k.ease, k.morph, k.set.clone())).collect();
        for (kt, dur, ease, morph, set) in keys {
            let e = if dur > 0.0 { ease.at((t - kt) / dur) } else { 1.0 };
            s.apply(&set, e, start + kt, morph, &mut |k| self.corner_offset(k));
        }
        s
    }

    fn camera(&self, scene: &str) -> &crate::scene::CameraCfg {
        &self.scenes[scene].0.camera
    }

    /// The lens of a render key, set up and focused.
    fn lens(&self, k: &RenderKey) -> Result<(Lens, f64)> {
        let raw = self.lenses.get(&k.lens).with_context(|| format!("unknown lens {}", k.lens))?;
        let mut l = match &k.morph {
            Some((to, e)) => optics::morph(raw, self.lenses.get(to).with_context(|| format!("unknown lens {to}"))?, *e),
            None => raw.clone(),
        };
        let c = self.camera(&k.scene);
        l.sensor_half = (c.sensor_width_mm / 2.0, c.sensor_height_mm / 2.0);
        let fstop = k.fstop.max(l.max_aperture());
        let focus = k.focus.is_finite().then_some(k.focus * 1000.0);
        l.prepare(fstop, c.iris_blades.unwrap_or(l.blades), c.iris_rotation_deg, c.iris_curvature.unwrap_or(l.blade_curvature), focus, k.offset)?;
        Ok((l, fstop))
    }

    fn half_diag(&self, scene: &str) -> f64 {
        let c = self.camera(scene);
        0.5 * c.sensor_width_mm.hypot(c.sensor_height_mm)
    }

    /// The chart's distance (mm): the scene's focus distance.
    fn object(&self, scene: &str) -> Option<f64> {
        let d = self.camera(scene).focus_distance;
        d.is_finite().then_some(d * 1000.0)
    }

    fn corner_offset(&mut self, k: &RenderKey) -> f64 {
        let k = RenderKey { offset: 0.0, ..k.clone() };
        let id = k.id();
        if let Some(o) = self.corner_offsets.get(&id) {
            return *o;
        }
        let o = match self.lens(&k) {
            Ok((l, _)) => optics::best_offset(&l, self.object(&k.scene), 0.9 * self.half_diag(&k.scene)),
            Err(_) => 0.0,
        };
        self.corner_offsets.insert(id, o);
        o
    }

    fn optics(&mut self, k: &RenderKey) -> Option<Arc<Optics>> {
        let id = k.id();
        if let Some(o) = self.optics.get(&id) {
            return Some(o.clone());
        }
        let (lens, fstop) = self.lens(k).map_err(|e| eprintln!("explainer: {e:#}")).ok()?;
        let (object, hd) = (self.object(&k.scene), self.half_diag(&k.scene));
        let z_start = lens.sensor_z - draw::DIAGRAM_DEPTH;
        let wl = |by: bool| if by { optics::RGB_WAVELENGTHS.to_vec() } else { vec![crate::glass::LAMBDA_D] };
        let mut fans = Vec::new();
        for f in [0.0, 0.7, 1.0] {
            fans.push((f, optics::fan(&lens, object, f, hd, 11, &wl(false), z_start)));
        }
        for f in [0.0, 0.7, 1.0] {
            fans.push((-1.0 - f, optics::fan(&lens, object, f, hd, 11, &wl(true), z_start)));
        }
        let c = self.camera(&k.scene);
        let o = Optics {
            axial: optics::fan(&lens, object, 0.0, hd, 25, &optics::RGB_WAVELENGTHS, z_start),
            fans,
            score: crate::metrics::scoreboard(&lens, object.unwrap_or(f64::INFINITY)),
            curves: optics::field_curves(&lens, object, hd),
            shape: optics::shape_factor(&lens),
            grid: optics::ideal_grid(&lens, &self.scenes[&k.scene].1, (c.sensor_width_mm, c.sensor_height_mm)),
            elements: lens.surfaces.iter().filter(|s| s.material.is_some()).count(),
            label: self.timeline.lenses.get(&k.morph.as_ref().filter(|m| m.1 > 0.5).map_or(&k.lens, |m| &m.0).clone()).map_or(lens.name.clone(), |e| e.label.clone()),
            fstop,
            lens,
        };
        if self.optics.len() > 64 {
            self.optics.clear();
        }
        let o = Arc::new(o);
        self.optics.insert(id, o.clone());
        Some(o)
    }

    /// Renders the images `state` shows (at most two), and sets the
    /// exposure. Returns how many images were rendered.
    fn prepare(&mut self, state: &State, now: f64) -> Result<usize> {
        let mut needs = vec![state.key()];
        needs.extend(state.split_key());
        if let Some((since, k)) = &state.fade {
            if now < since + CROSSFADE && needs.len() < 2 {
                needs.push(k.clone());
            }
        }
        let mut renders = 0;
        for k in &needs {
            if self.live.as_ref() == Some(k) || self.held.as_ref().is_some_and(|h| h.key == *k) {
                continue;
            }
            if let Some(l) = self.live.clone().filter(|l| needs.contains(l)) {
                self.hold(l)?;
            }
            self.render(k, state)?;
            renders += 1;
        }
        let n = self.optics(&state.key()).map_or(state.fstop, |o| o.fstop);
        let exposure = 2f64.powf(self.camera(&state.scene).exposure_ev + state.ev - state.dim * 2.0 * (n / state.dim_ref).log2().max(0.0));
        if (exposure - self.exposure).abs() > 1e-9 {
            self.exposure = exposure;
            self.renderer.set_display(exposure, Tonemap::Clip);
            self.renderer.resolve(true)?;
        }
        Ok(renders)
    }

    /// Renders `k` into the live image.
    fn render(&mut self, k: &RenderKey, state: &State) -> Result<()> {
        if self.scene.as_deref() != Some(&k.scene) {
            self.renderer.set_scene(&self.scenes.get(&k.scene).with_context(|| format!("unknown scene {}", k.scene))?.1);
            self.scene = Some(k.scene.clone());
        }
        let (lens, fstop) = self.lens(k)?;
        let c = self.camera(&k.scene);
        let (w, h) = self.size();
        let settings = gpu::Settings {
            w,
            h,
            bins: c.spectral_bins,
            sensor_w: c.sensor_width_mm,
            norm: 4.0 * fstop * fstop / std::f64::consts::PI,
            spp_per_frame: 1,
            ghosts: c.ghosts,
            ghost_threshold: c.ghost_threshold,
            light_grid: gpu::ghost_grid(&lens, c.ghost_grid),
            ghost_wavelengths: 4,
            tonemap: Tonemap::Clip,
        };
        self.renderer.configure(&lens, &settings, None)?;
        for _ in 0..PASSES {
            self.renderer.throttle(2)?;
            self.renderer.render_frame();
        }
        let n = self.optics(k).map_or(fstop, |o| o.fstop);
        self.exposure = 2f64.powf(self.camera(&k.scene).exposure_ev + state.ev - state.dim * 2.0 * (n / state.dim_ref).log2().max(0.0));
        self.renderer.set_display(self.exposure, Tonemap::Clip);
        self.renderer.resolve(true)?;
        self.renderer.wait()?;
        let view = self.renderer.texture().context("no image")?.create_view(&Default::default());
        let mut r = self.rs.renderer.write();
        match self.live_id {
            Some(id) => r.update_egui_texture_from_wgpu_texture(&self.rs.device, &view, wgpu::FilterMode::Linear, id),
            None => self.live_id = Some(r.register_native_texture(&self.rs.device, &view, wgpu::FilterMode::Linear)),
        }
        self.live = Some(k.clone());
        Ok(())
    }

    /// Copies the live image (of `key`) into the held one.
    fn hold(&mut self, key: RenderKey) -> Result<()> {
        let (w, h) = self.size();
        let extent = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
        if self.held.is_none() {
            let texture = self.rs.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("held image"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let id = self.rs.renderer.write().register_native_texture(&self.rs.device, &texture.create_view(&Default::default()), wgpu::FilterMode::Linear);
            self.held = Some(Held { key: key.clone(), texture, id });
        }
        let held = self.held.as_mut().unwrap();
        let mut enc = self.rs.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_texture(self.renderer.texture().context("no image")?.as_image_copy(), held.texture.as_image_copy(), extent);
        self.rs.queue.submit([enc.finish()]);
        held.key = key;
        Ok(())
    }

    fn size(&self) -> (usize, usize) {
        let w = self.timeline.width;
        (w, (w as f64 * 2.0 / 3.0).round() as usize)
    }

    /// The texture showing the image of `k`.
    fn texture(&self, k: &RenderKey) -> Option<egui::TextureId> {
        if self.live.as_ref() == Some(k) {
            return self.live_id;
        }
        self.held.as_ref().filter(|h| h.key == *k).map(|h| h.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(json: &str) -> Map<String, Value> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn keys_tween_and_switch() {
        let mut s = State { lens: "a".into(), scene: "lesson".into(), ..State::default() };
        let mut corner = |_: &RenderKey| 0.0;
        // Halfway from f/4 to f/16 in log space is f/8.
        let mut half = s.clone();
        half.apply(&set(r#"{"fstop": 16}"#), 0.5, 1.0, false, &mut corner);
        assert!((half.fstop - 8.0).abs() < 1e-9, "{}", half.fstop);
        // A lens and f-number change at once crossfades from the image before both.
        s.apply(&set(r#"{"fstop": 2, "lens": "b", "scoreboard": true}"#), 1.0, 3.0, false, &mut corner);
        let (since, before) = s.fade.clone().unwrap();
        assert_eq!((since, before.lens.as_str(), before.fstop), (3.0, "a", 4.0));
        assert_eq!((s.lens.as_str(), s.fstop), ("b", 2.0));
        assert_eq!(s.flag("scoreboard").alpha(3.0, FADE), 0.0);
        assert_eq!(s.flag("scoreboard").alpha(3.0 + FADE, FADE), 1.0);
        // A morph ends on the target lens itself.
        s.apply(&set(r#"{"lens": "c"}"#), 0.4, 5.0, true, &mut corner);
        assert_eq!(s.key().morph, Some(("c".to_string(), 0.4)));
        s.apply(&set(r#"{"lens": "c"}"#), 1.0, 5.0, true, &mut corner);
        assert_eq!((s.key().lens.as_str(), s.key().morph), ("c", None));
    }

    #[test]
    fn morph_ends_on_both_lenses() {
        let a = Lens::parse("stop 0.5 air 6\n50 5 N-BK7 9\n-50 48 air 9\n").unwrap();
        let b = Lens::parse("stop 2.5 air 6\n30 5 N-BK7 9\n-180 48 air 9\n").unwrap();
        let start = optics::morph(&a, &b, 0.0);
        let end = optics::morph(&a, &b, 1.0);
        for i in 0..3 {
            assert_eq!(start.surfaces[i].radius, a.surfaces[i].radius);
            assert_eq!(end.surfaces[i].radius, b.surfaces[i].radius);
        }
        let mid = optics::morph(&a, &b, 0.5);
        assert!((mid.surfaces[1].curvature() - 0.5 * (1.0 / 50.0 + 1.0 / 30.0)).abs() < 1e-12);
        assert!((mid.surfaces[2].z - 1.5 - 5.0).abs() < 1e-12, "stop gap and positions follow");
    }
}

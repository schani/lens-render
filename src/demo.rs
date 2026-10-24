//! Demo recording for the viewer: follows a timeline of narrated segments
//! (`timeline.json` in the demo directory, written by scripts/demo_video.py),
//! and captures one screenshot per video frame into `frames/`.

use eframe::egui;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

#[derive(Deserialize, Clone)]
pub struct Segment {
    pub id: String,
    pub title: String,
    pub start: f64,
    pub dur: f64,
}

#[derive(Deserialize)]
struct Timeline {
    fps: f64,
    segments: Vec<Segment>,
}

pub struct Demo {
    fps: f64,
    pub segments: Vec<Segment>,
    /// Frames captured so far.
    frame: Arc<AtomicUsize>,
    capturing: Arc<AtomicBool>,
    /// When the screenshot on its way was requested.
    requested: Mutex<Option<Instant>>,
    writing: Arc<AtomicUsize>,
    /// Frames to write, for writer threads of our own (the shared pool may
    /// be busy with the viewer's analyses).
    frames: mpsc::Sender<(usize, Arc<egui::ColorImage>)>,
    /// Simulated pointer: position and whether the button is down.
    pub pointer: Option<(egui::Pos2, bool)>,
    sent: Option<(egui::Pos2, bool)>,
}

impl Demo {
    pub fn load(dir: PathBuf) -> anyhow::Result<Demo> {
        let t: Timeline = serde_json::from_str(&std::fs::read_to_string(dir.join("timeline.json"))?)?;
        std::fs::create_dir_all(dir.join("frames"))?;
        let (tx, rx) = mpsc::channel::<(usize, Arc<egui::ColorImage>)>();
        let rx = Arc::new(Mutex::new(rx));
        let writing: Arc<AtomicUsize> = Default::default();
        for _ in 0..4 {
            let (rx, writing, dir) = (rx.clone(), writing.clone(), dir.clone());
            std::thread::spawn(move || {
                loop {
                    let Ok((n, img)) = rx.lock().unwrap().recv() else { return };
                    let rgb: Vec<u8> = img.pixels.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect();
                    if let Some(i) = image::RgbImage::from_raw(img.size[0] as u32, img.size[1] as u32, rgb) {
                        let _ = i.save(dir.join(format!("frames/{n:05}.jpg")));
                    }
                    writing.fetch_sub(1, Ordering::Relaxed);
                }
            });
        }
        Ok(Demo {
            fps: t.fps,
            segments: t.segments,
            frame: Default::default(),
            capturing: Default::default(),
            requested: Mutex::new(None),
            writing,
            frames: tx,
            pointer: None,
            sent: None,
        })
    }

    /// Video time of the frame being made.
    pub fn time(&self) -> f64 {
        self.frame.load(Ordering::Relaxed) as f64 / self.fps
    }

    /// The current segment and the time within it; None when done.
    pub fn segment(&self) -> Option<(&Segment, f64)> {
        let t = self.time();
        self.segments.iter().find(|s| t >= s.start && t < s.start + s.dur).map(|s| (s, t - s.start))
    }

    /// Screenshots the next frame unless one is still on its way (or too
    /// many are still being written); returns true when the demo is over.
    pub fn capture(&self, ctx: &egui::Context) -> bool {
        let end = self.segments.last().map_or(0.0, |s| s.start + s.dur);
        if self.time() >= end {
            return self.writing.load(Ordering::Relaxed) == 0;
        }
        // A screenshot that never arrived is asked for again.
        let stale = self.requested.lock().unwrap().is_some_and(|t| t.elapsed().as_secs_f64() > 3.0);
        if (self.capturing.load(Ordering::Relaxed) && !stale) || self.writing.load(Ordering::Relaxed) > 16 {
            return false;
        }
        self.capturing.store(true, Ordering::Relaxed);
        *self.requested.lock().unwrap() = Some(Instant::now());
        let (frame, capturing, writing, frames) = (self.frame.clone(), self.capturing.clone(), self.writing.clone(), self.frames.clone());
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
            if !capturing.swap(false, Ordering::Relaxed) {
                return;
            }
            let n = frame.fetch_add(1, Ordering::Relaxed);
            writing.fetch_add(1, Ordering::Relaxed);
            let _ = frames.send((n, img.clone()));
        })));
        false
    }

    /// Pointer events for the simulated pointer since the last frame.
    pub fn input(&mut self, raw: &mut egui::RawInput) {
        let Some((p, down)) = self.pointer else { return };
        if self.sent.map(|s| s.0) != Some(p) {
            raw.events.push(egui::Event::PointerMoved(p));
        }
        if self.sent.map(|s| s.1) != Some(down) {
            raw.events.push(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: down, modifiers: Default::default() });
        }
        self.sent = Some((p, down));
    }

    /// The segment title and the simulated pointer, over everything else.
    pub fn overlay(&self, ctx: &egui::Context, image: egui::Rect) {
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("demo")));
        if let Some((s, t)) = self.segment() {
            let alpha = (t / 0.25).min(1.0) as f32;
            let font = egui::FontId::proportional(30.0);
            let galley = painter.layout_no_wrap(s.title.clone(), font, egui::Color32::WHITE.gamma_multiply(alpha));
            let pos = image.left_top() + egui::vec2(24.0, 20.0);
            painter.rect_filled(egui::Rect::from_min_size(pos - egui::vec2(12.0, 8.0), galley.size() + egui::vec2(24.0, 16.0)), 8.0, egui::Color32::from_black_alpha((170.0 * alpha) as u8));
            painter.galley(pos, galley, egui::Color32::WHITE);
        }
        if let Some((p, down)) = self.pointer {
            // An arrow cursor: stem, then head.
            let (white, edge) = (egui::Color32::WHITE, egui::Stroke::new(1.5, egui::Color32::BLACK));
            let stem = [p + egui::vec2(5.0, 14.0), p + egui::vec2(9.0, 13.0), p + egui::vec2(14.0, 24.0), p + egui::vec2(10.0, 25.0)];
            painter.add(egui::Shape::convex_polygon(stem.to_vec(), white, edge));
            painter.add(egui::Shape::convex_polygon(vec![p, p + egui::vec2(0.0, 21.0), p + egui::vec2(15.0, 15.0)], white, edge));
            if down {
                painter.circle_stroke(p, 14.0, egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 120, 0)));
            }
        }
    }
}

//! The explainer's picture, 1600 × 900 points: the render (3:2) top left;
//! on the right two magnified insets, the scoreboard and the HUD; below, the
//! lens cross-section, a special view (focus zoom, field curves, bending
//! curve) and the toolbox. Titles, PROBLEM / FIX tags and annotations go on
//! top.

use super::{Explainer, FADE, Note, Optics, State, Tag, optics};
use eframe::egui::{self, Align2, Color32, CornerRadius, FontFamily, FontId, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, pos2, vec2};
use serde_json::Value;
use std::f32::consts::PI;

pub const W: f32 = 1600.0;
pub const H: f32 = 900.0;
/// How far in front of the sensor (mm) the cross-section reaches.
pub const DIAGRAM_DEPTH: f64 = 340.0;

const BG: Color32 = Color32::from_rgb(13, 16, 21);
const PANEL: Color32 = Color32::from_rgb(22, 27, 34);
const EDGE: Color32 = Color32::from_rgb(40, 47, 58);
const TEXT: Color32 = Color32::from_rgb(232, 234, 237);
const DIM: Color32 = Color32::from_rgb(140, 150, 166);
const GREEN: Color32 = Color32::from_rgb(60, 207, 116);
const AMBER: Color32 = Color32::from_rgb(245, 184, 61);
const RED: Color32 = Color32::from_rgb(240, 84, 84);
const ACCENT: Color32 = Color32::from_rgb(255, 209, 102);
const GLASS: Color32 = Color32::from_rgba_premultiplied(60, 95, 140, 110);
const GLASS_EDGE: Color32 = Color32::from_rgb(150, 190, 240);
const FIELD_COLORS: [Color32; 3] = [Color32::from_rgb(255, 209, 102), Color32::from_rgb(255, 138, 101), Color32::from_rgb(77, 208, 225)];
const RGB: [Color32; 3] = [Color32::from_rgb(90, 140, 255), Color32::from_rgb(90, 215, 120), Color32::from_rgb(255, 90, 90)];
const T_COLOR: Color32 = Color32::from_rgb(255, 138, 101);
const S_COLOR: Color32 = Color32::from_rgb(77, 208, 225);
const P_COLOR: Color32 = Color32::from_rgb(190, 170, 255);
/// Scoreboard thresholds (µm): green up to the first, amber up to the second;
/// distortion (%) has its own.
const GOOD: f64 = 20.0;
const FAIR: f64 = 60.0;
const GOOD_PCT: f64 = 1.0;
const FAIR_PCT: f64 = 2.5;
/// Special views and their titles.
const VIEWS: [(&str, &str); 3] = [("focus_zoom", "Near the focus"), ("field_curves", "Where the image is sharp"), ("bending", "Spherical aberration vs. shape")];
const ROWS: [&str; 7] = ["spherical", "coma", "astigmatism", "field curvature", "distortion", "axial color", "lateral color"];
/// Shutter speeds in third stops (s).
const SPEEDS: [(f64, &str); 31] = [
    (1.0 / 1000.0, "1/1000"),
    (1.0 / 800.0, "1/800"),
    (1.0 / 640.0, "1/640"),
    (1.0 / 500.0, "1/500"),
    (1.0 / 400.0, "1/400"),
    (1.0 / 320.0, "1/320"),
    (1.0 / 250.0, "1/250"),
    (1.0 / 200.0, "1/200"),
    (1.0 / 160.0, "1/160"),
    (1.0 / 125.0, "1/125"),
    (1.0 / 100.0, "1/100"),
    (1.0 / 80.0, "1/80"),
    (1.0 / 60.0, "1/60"),
    (1.0 / 50.0, "1/50"),
    (1.0 / 40.0, "1/40"),
    (1.0 / 30.0, "1/30"),
    (1.0 / 25.0, "1/25"),
    (1.0 / 20.0, "1/20"),
    (1.0 / 15.0, "1/15"),
    (1.0 / 13.0, "1/13"),
    (1.0 / 10.0, "1/10"),
    (1.0 / 8.0, "1/8"),
    (1.0 / 6.0, "1/6"),
    (1.0 / 5.0, "1/5"),
    (1.0 / 4.0, "1/4"),
    (0.3, "0.3"),
    (0.4, "0.4"),
    (0.5, "1/2"),
    (0.6, "0.6"),
    (0.8, "0.8"),
    (1.0, "1"),
];

/// The render's place; `full` 1 fills the screen.
fn render_rect(full: f64) -> Rect {
    let normal = Rect::from_min_size(pos2(24.0, 24.0), vec2(888.0, 592.0));
    let screen = Rect::from_center_size(pos2(W / 2.0, H / 2.0), vec2(1314.0, 876.0));
    let t = smooth(full);
    Rect::from_min_max(normal.min.lerp(screen.min, t), normal.max.lerp(screen.max, t))
}
fn inset_rect(i: usize) -> Rect {
    Rect::from_min_size(pos2(936.0 + i as f32 * 328.0, 24.0), vec2(312.0, 208.0))
}
fn score_rect() -> Rect {
    Rect::from_min_max(pos2(936.0, 248.0), pos2(1576.0, 556.0))
}

/// Scoreboard groups: header and read-outs (indices: the seven bars, then
/// centre and corner blur), in display order.
const GROUPS: [(&str, &[usize]); 2] = [("center: light straight on", &[0, 5, 7]), ("corners: light at an angle", &[1, 2, 3, 4, 6, 8])];
const ROW_H: f32 = 22.0;
const GROUP_HEADER: f32 = 19.0;

/// The header and rows of scoreboard group `g`.
fn group_rect(g: usize) -> Rect {
    let r = score_rect();
    let top = r.top() + 66.0 + if g == 1 { GROUP_HEADER + 3.0 * ROW_H + 8.0 } else { 0.0 };
    Rect::from_min_size(pos2(r.left() + 8.0, top), vec2(r.width() - 16.0, GROUP_HEADER + GROUPS[g].1.len() as f32 * ROW_H))
}

/// Row of read-out `i` on the scoreboard.
fn row_rect(i: usize) -> Rect {
    let (g, k) = GROUPS.iter().enumerate().find_map(|(g, (_, rows))| rows.iter().position(|&x| x == i).map(|k| (g, k))).unwrap_or((0, 0));
    let h = group_rect(g);
    Rect::from_min_size(pos2(h.left(), h.top() + GROUP_HEADER + k as f32 * ROW_H), vec2(h.width(), ROW_H))
}

/// Scoreboard region by name: a read-out (its label) or a group ("center", "corners").
fn score_region(name: &str) -> Option<Rect> {
    match name {
        "center" => Some(group_rect(0)),
        "corners" => Some(group_rect(1)),
        "center blur" => Some(row_rect(7)),
        "corner blur" => Some(row_rect(8)),
        n => ROWS.iter().position(|r| *r == n).map(row_rect),
    }
}
fn hud_rect() -> Rect {
    Rect::from_min_max(pos2(936.0, 568.0), pos2(1576.0, 616.0))
}
/// The cross-section's place: it makes room for a special view (`views` 1).
fn diagram_rect(views: f32) -> Rect {
    Rect::from_min_max(pos2(24.0, 640.0), pos2(1320.0 - 408.0 * views, 876.0))
}
fn view_rect() -> Rect {
    Rect::from_min_max(pos2(936.0, 640.0), pos2(1320.0, 876.0))
}
fn toolbox_rect() -> Rect {
    Rect::from_min_max(pos2(1336.0, 640.0), pos2(1576.0, 876.0))
}

/// Avenir Next (macOS) for text, egui's fonts as fallback.
pub fn fonts(ctx: &egui::Context) {
    let mut defs = egui::FontDefinitions::default();
    let fallback = defs.families[&FontFamily::Proportional].clone();
    if let Ok(bytes) = std::fs::read("/System/Library/Fonts/Avenir Next.ttc") {
        let blob: std::sync::Arc<dyn AsRef<[u8]> + Send + Sync> = std::sync::Arc::new(bytes);
        for (name, index) in [("regular", 7), ("medium", 5), ("bold", 2)] {
            defs.font_data.insert(name.into(), std::sync::Arc::new(egui::FontData::from_blob(blob.clone(), index)));
        }
        let family = |name: &str| std::iter::once(name.to_string()).chain(fallback.iter().cloned()).collect::<Vec<_>>();
        defs.families.insert(FontFamily::Proportional, family("regular"));
        defs.families.insert(FontFamily::Name("medium".into()), family("medium"));
        defs.families.insert(FontFamily::Name("bold".into()), family("bold"));
    } else {
        for name in ["medium", "bold"] {
            defs.families.insert(FontFamily::Name(name.into()), fallback.clone());
        }
    }
    ctx.set_fonts(defs);
}

fn regular(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}
fn medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("medium".into()))
}
fn bold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("bold".into()))
}

fn smooth(x: f64) -> f32 {
    let x = x.clamp(0.0, 1.0) as f32;
    x * x * (3.0 - 2.0 * x)
}

fn panel(p: &Painter, r: Rect) {
    p.rect(r, CornerRadius::same(10), PANEL, Stroke::new(1.0, EDGE), StrokeKind::Inside);
}

/// Text on a dark pill; returns the pill.
fn pill(p: &Painter, pos: Pos2, anchor: Align2, text: &str, font: FontId, color: Color32, fill: Color32) -> Rect {
    let g = p.layout_no_wrap(text.to_string(), font, color);
    let pad = vec2(10.0, 5.0);
    let r = anchor.anchor_size(pos, g.size() + 2.0 * pad);
    p.rect_filled(r, CornerRadius::same(7), fill);
    p.galley(r.min + pad, g, color);
    r
}

/// Colour of a value on the scoreboard with thresholds `good` and `fair`.
fn grade(v: f64, good: f64, fair: f64) -> Color32 {
    let mix = |a: Color32, b: Color32, t: f64| {
        let t = t.clamp(0.0, 1.0) as f32;
        Color32::from_rgb(
            (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
            (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
            (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
        )
    };
    // Blend over ±10 % around the thresholds so animated bars don't flash.
    let l = v.max(1e-3).ln();
    let t1 = (l - good.ln()) / 0.2 + 0.5;
    let t2 = (l - fair.ln()) / 0.2 + 0.5;
    if t2 > 0.0 { mix(AMBER, RED, t2) } else { mix(GREEN, AMBER, t1) }
}

fn fnumber(n: f64) -> String {
    if n >= 10.0 || (n - n.round()).abs() < 0.05 { format!("f/{:.0}", n) } else { format!("f/{:.1}", n) }
}

fn shutter(t: f64) -> String {
    let (_, s) = SPEEDS.iter().min_by(|a, b| (a.0.ln() - t.ln()).abs().total_cmp(&(b.0.ln() - t.ln()).abs())).unwrap();
    format!("{s} s")
}

/// The bending curve (lenses/explainer/bending.json, from `lens-render
/// bending`): shape factor against the 546 nm RMS blur from spherical
/// aberration alone (W040 if absent).
pub fn bending_curve(v: &Value) -> Vec<[f64; 2]> {
    let pts = v.get("points").and_then(|p| p.as_array()).cloned().unwrap_or_default();
    pts.iter().filter_map(|p| Some([p.get("shape")?.as_f64()?, p.get("rms_mono_um").or_else(|| p.get("w040_waves"))?.as_f64()?.abs()])).collect()
}

/// Lens z and y (mm) to points in the cross-section.
#[derive(Clone, Copy)]
struct DiagramView {
    sensor: Pos2,
    sensor_z: f64,
    scale: f64,
}

impl DiagramView {
    /// The view in `r`, magnified `zoom` times about the sensor.
    fn new(r: Rect, sensor_z: f64, zoom: f64) -> DiagramView {
        DiagramView { sensor: pos2(r.right() - 40.0, r.center().y + 6.0), sensor_z, scale: ((r.height() - 24.0) / 48.0) as f64 * zoom }
    }
    fn at(&self, z: f64, y: f64) -> Pos2 {
        self.sensor + vec2(((z - self.sensor_z) * self.scale) as f32, -(y * self.scale) as f32)
    }
}

impl Explainer {
    pub fn draw(&mut self, ui: &mut egui::Ui, ci: usize, t: f64, s: &State, now: f64) {
        let p = ui.painter().clone();
        p.rect_filled(Rect::from_min_size(Pos2::ZERO, vec2(W, H)), 0.0, BG);
        let Some(main) = self.optics(&s.key()) else { return };
        let fade = s.fade.as_ref().filter(|(since, _)| now < since + super::CROSSFADE).and_then(|(since, k)| Some((smooth((now - since) / super::CROSSFADE), self.optics(k)?, k.clone())));
        self.update_readouts(&main);

        // Render and insets.
        let images = Images {
            main: self.texture(&s.key()),
            prev: fade.as_ref().and_then(|(a, _, k)| Some((self.texture(k)?, *a))),
            split: s.split_key().and_then(|k| Some((self.texture(&k)?, s.split as f32))),
        };
        let rr = render_rect(s.full);
        self.rr = rr;
        images.draw(&p, rr, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), CornerRadius::same(8));
        let grid_alpha = s.flag("ideal_grid").alpha(now, FADE);
        if grid_alpha > 0.0 {
            draw_grid(&p.with_clip_rect(rr), rr, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), &main.grid, grid_alpha);
        }
        if s.spotlight > 0.0 {
            spotlight(&p, rr, smooth(s.spotlight));
        }
        if s.split > 0.0 {
            self.draw_split(&p, s);
        }
        let insets_alpha = s.flag("insets").alpha(now, FADE);
        if insets_alpha > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(insets_alpha);
            for i in 0..2 {
                let r = inset_rect(i);
                let uv = self.crop(s, i);
                images.draw(&q, r, uv, CornerRadius::same(8));
                if grid_alpha > 0.0 {
                    draw_grid(&q.with_clip_rect(r), r, uv, &main.grid, grid_alpha);
                }
                let later = if i == 1 { smooth(s.spotlight) } else { 0.0 };
                if later > 0.0 {
                    q.rect_filled(r, CornerRadius::same(8), Color32::from_rgba_unmultiplied(24, 27, 32, (165.0 * later) as u8));
                    let mut l = q.clone();
                    l.multiply_opacity(later);
                    pill(&l, r.center(), Align2::CENTER_CENTER, "corners: part 2", medium(17.0), TEXT, Color32::from_black_alpha(190));
                }
                q.rect_stroke(r, CornerRadius::same(8), Stroke::new(1.0, EDGE), StrokeKind::Inside);
                let label = format!("{} ×{:.0}", if i == 0 { "center" } else { "corner" }, s.inset_zoom);
                pill(&q, r.left_top() + vec2(8.0, 8.0), Align2::LEFT_TOP, &label, medium(14.0), TEXT, Color32::from_black_alpha(170));
                // Where the inset comes from.
                let mut q = q.clone();
                q.multiply_opacity(1.0 - 0.65 * later);
                let m = Rect::from_min_max(rr.lerp_inside(uv.min.to_vec2()), rr.lerp_inside(uv.max.to_vec2()));
                q.rect_stroke(m.expand(1.0), CornerRadius::same(2), Stroke::new(3.0, Color32::from_black_alpha(120)), StrokeKind::Middle);
                q.rect_stroke(m, CornerRadius::same(2), Stroke::new(1.5, Color32::WHITE), StrokeKind::Middle);
            }
        }
        let a = s.flag("recap").alpha(now, FADE);
        if a > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(a);
            self.draw_recap(&q, now - s.flag("recap").since);
        }
        self.draw_title(&p, ci, t);
        self.draw_tags(&p, ci, t);
        self.draw_part(&p, ci, t);

        let a = s.flag("scoreboard").alpha(now, FADE);
        if a > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(a);
            self.draw_scoreboard(&q, s, now, &main);
        }
        let a = s.flag("hud").alpha(now, FADE);
        if a > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(a);
            draw_hud(&q, s, &main);
        }
        let views = VIEWS.iter().map(|(f, _)| s.flag(f).alpha(now, 2.0 * FADE)).fold(0.0, f32::max);
        let dr = diagram_rect(smooth(2.0 * views as f64));
        self.dr = dr;
        let a = s.flag("diagram").alpha(now, FADE);
        if a > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(a);
            panel(&q, dr);
            let clip = q.with_clip_rect(dr.shrink(1.0));
            let view = DiagramView::new(dr, main.lens.sensor_z, s.diagram_zoom);
            if let Some((fa, prev, _)) = &fade {
                let mut old = clip.clone();
                old.multiply_opacity(1.0 - fa);
                draw_lens(&old, &DiagramView::new(dr, prev.lens.sensor_z, s.diagram_zoom), prev, s, now, dr);
                let mut new = clip.clone();
                new.multiply_opacity(*fa);
                draw_lens(&new, &view, &main, s, now, dr);
                // One name at a time: the old one fades out, then the new one in.
                draw_label(&clip, prev, dr, smooth(1.0 - 2.0 * *fa as f64));
                draw_label(&clip, &main, dr, smooth(2.0 * *fa as f64 - 1.0));
            } else {
                draw_lens(&clip, &view, &main, s, now, dr);
                draw_label(&clip, &main, dr, 1.0);
            }
        }
        self.draw_view(&p, s, &main, now);
        let a = s.flag("toolbox").alpha(now, FADE);
        if a > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(a);
            self.draw_toolbox(&q, ci, t);
        }
        let notes = self.timeline.chapters[ci].notes.clone();
        for n in &notes {
            draw_note(&p, n, t, &main, rr, dr, s.diagram_zoom);
        }
    }

    /// Smooths the scoreboard's read-outs towards the lens shown.
    fn update_readouts(&mut self, o: &Optics) {
        // Distortion (%) is drawn where 20 µm per 1 % would be.
        let target: [f64; 9] = std::array::from_fn(|i| match i {
            4 => (o.score.bars[4].abs() * GOOD).max(0.05).ln(),
            i if i < 7 => o.score.bars[i].max(0.05).ln(),
            i => o.score.blur[i - 7].max(0.05).ln(),
        });
        let k = 1.0 - (-self.dt / 0.18).exp();
        let shown = self.shown.get_or_insert(target);
        for (v, t) in shown.iter_mut().zip(target) {
            *v = if t.is_finite() { *v + (t - *v) * k } else { *v };
        }
    }

    /// Normalised image region of inset `i`.
    fn crop(&self, s: &State, i: usize) -> Rect {
        let w = (inset_rect(i).width() / render_rect(0.0).width()) as f64 / s.inset_zoom.max(0.1);
        let c = s.insets[i];
        let (cx, cy) = (c[0].clamp(w / 2.0, 1.0 - w / 2.0), c[1].clamp(w / 2.0, 1.0 - w / 2.0));
        Rect::from_min_max(pos2((cx - w / 2.0) as f32, (cy - w / 2.0) as f32), pos2((cx + w / 2.0) as f32, (cy + w / 2.0) as f32))
    }

    /// Divider and labels of a split screen; each label sits in its part's
    /// outer corner and fades as the part closes.
    fn draw_split(&self, p: &Painter, s: &State) {
        let rr = self.rr;
        let x = rr.left() + rr.width() * s.split as f32;
        let count = |lens: &str| self.lenses.get(lens).map_or(0, |l| l.surfaces.iter().filter(|s| s.material.is_some()).count());
        let label = |n: usize| if n == 1 { "1 element".to_string() } else { format!("{n} elements") };
        let (left, right) = (smooth(s.split / 0.2), smooth((1.0 - s.split) / 0.2));
        let mut q = p.clone();
        q.multiply_opacity(left.min(right));
        q.line_segment([pos2(x, rr.top()), pos2(x, rr.bottom())], Stroke::new(3.0, Color32::WHITE));
        if let Some(l) = &s.split_lens {
            let mut q = p.clone();
            q.multiply_opacity(left);
            pill(&q, rr.left_bottom() + vec2(14.0, -14.0), Align2::LEFT_BOTTOM, &label(count(l)), bold(22.0), TEXT, Color32::from_black_alpha(170));
        }
        let mut q = p.clone();
        q.multiply_opacity(right);
        pill(&q, rr.right_bottom() + vec2(-14.0, -14.0), Align2::RIGHT_BOTTOM, &label(count(&s.key().lens)), bold(22.0), TEXT, Color32::from_black_alpha(170));
    }

    fn draw_title(&self, p: &Painter, ci: usize, t: f64) {
        let c = &self.timeline.chapters[ci];
        if c.title.is_empty() {
            return;
        }
        let mut q = p.clone();
        q.multiply_opacity(smooth(t / 0.5));
        let rr = self.rr;
        let g = q.layout_no_wrap(c.title.clone(), bold(28.0), TEXT);
        let badge = if c.number > 0 { 40.0 } else { 0.0 };
        let r = Rect::from_min_size(rr.left_top() + vec2(14.0, 14.0), vec2(g.size().x + 28.0 + badge, 48.0));
        q.add(egui::epaint::RectShape::filled(r.translate(vec2(0.0, 3.0)), CornerRadius::same(10), Color32::from_black_alpha(90)).with_blur_width(12.0));
        q.rect_filled(r, CornerRadius::same(10), Color32::from_rgba_premultiplied(13, 16, 21, 225));
        if c.number > 0 {
            let cpos = r.left_center() + vec2(26.0, 0.0);
            q.circle_filled(cpos, 15.0, ACCENT);
            q.text(cpos + vec2(0.0, 1.0), Align2::CENTER_CENTER, c.number.to_string(), bold(17.0), BG);
        }
        q.galley(pos2(r.left() + 14.0 + badge, r.center().y - g.size().y / 2.0 + 1.0), g, TEXT);
    }

    /// A part's title card, centred on the render for 1.6 s.
    fn draw_part(&self, p: &Painter, ci: usize, t: f64) {
        let Some(part) = &self.timeline.chapters[ci].part else { return };
        let age = t - part.t;
        if !(0.0..1.6).contains(&age) {
            return;
        }
        let a = smooth(age / 0.3) * (1.0 - smooth((age - 1.25) / 0.35));
        let mut q = p.clone();
        q.multiply_opacity(a);
        let g = q.layout_no_wrap(part.text.clone(), bold(36.0), TEXT);
        let r = Rect::from_center_size(self.rr.center() + vec2(0.0, (1.0 - a) * 10.0), g.size() + vec2(56.0, 30.0));
        q.add(egui::epaint::RectShape::filled(r.translate(vec2(0.0, 4.0)), CornerRadius::same(14), Color32::from_black_alpha(110)).with_blur_width(18.0));
        q.rect(r, CornerRadius::same(14), Color32::from_rgba_premultiplied(13, 16, 21, 235), Stroke::new(2.0, ACCENT), StrokeKind::Inside);
        q.galley(r.center() - g.size() / 2.0 + vec2(0.0, 1.0), g, TEXT);
    }

    fn draw_tags(&self, p: &Painter, ci: usize, t: f64) {
        let rr = self.rr;
        let mut y = rr.top() + 74.0;
        let tags: Vec<&Tag> = self.timeline.chapters[ci].tags.iter().filter(|g| t >= g.t && g.until.is_none_or(|u| t < u + FADE)).collect();
        for g in tags {
            let a = smooth((t - g.t) / FADE) * g.until.map_or(1.0, |u| 1.0 - smooth((t - u) / FADE));
            let mut q = p.clone();
            q.multiply_opacity(a);
            let lift = (1.0 - a) * -12.0;
            let (label, color) = if g.kind == "fix" { ("FIX", GREEN) } else { ("PROBLEM", RED) };
            let badge = q.layout_no_wrap(label.into(), bold(17.0), BG);
            let text = q.layout_no_wrap(g.text.clone(), medium(22.0), TEXT);
            let h = 44.0;
            let r = Rect::from_min_size(pos2(rr.left() + 14.0, y + lift), vec2(badge.size().x + text.size().x + 48.0, h));
            q.rect_filled(r, CornerRadius::same(10), Color32::from_rgba_premultiplied(13, 16, 21, 230));
            let b = Rect::from_min_size(r.min + vec2(7.0, 7.0), vec2(badge.size().x + 18.0, h - 14.0));
            q.rect_filled(b, CornerRadius::same(6), color);
            q.galley(b.center() - badge.size() / 2.0 + vec2(0.0, 1.0), badge, BG);
            q.galley(pos2(b.right() + 12.0, r.center().y - text.size().y / 2.0 + 1.0), text, TEXT);
            y += h + 8.0;
        }
    }

    fn draw_scoreboard(&self, p: &Painter, s: &State, now: f64, o: &Optics) {
        let r = score_rect();
        panel(p, r);
        p.text(r.left_top() + vec2(18.0, 12.0), Align2::LEFT_TOP, "Aberrations", bold(20.0), TEXT);
        p.text(r.right_top() + vec2(-18.0, 18.0), Align2::RIGHT_TOP, "measured by ray tracing, on the sensor", regular(13.0), DIM);
        let shown = self.shown.unwrap_or([0.0; 9]);
        let (x0, x1) = (r.left() + 168.0, r.right() - 84.0);
        let bar_x = |um: f64| x0 + (x1 - x0) * ((um.max(1.0).log10() / 3.0).min(1.0) as f32);
        let (top, bottom) = (row_rect(GROUPS[0].1[0]).top(), row_rect(8).bottom());
        // Decades of µm.
        for (um, label) in [(1.0, "1 µm"), (10.0, "10"), (100.0, "100"), (1000.0, "1 mm")] {
            let x = bar_x(um);
            p.line_segment([pos2(x, top - 18.0), pos2(x, bottom)], Stroke::new(1.0, EDGE));
            p.text(pos2(x, top - 20.0), Align2::CENTER_BOTTOM, label, regular(12.0), DIM);
        }
        let since = s.flag("scoreboard").since;
        // The corner group steps back while the lesson is about the centre.
        let dim = 1.0 - 0.65 * smooth(s.spotlight);
        for (g, (header, rows)) in GROUPS.iter().enumerate() {
            let mut q = p.clone();
            q.multiply_opacity(if g == 1 { dim } else { 1.0 });
            let h = group_rect(g);
            q.text(pos2(r.left() + 18.0, h.top() + 1.0), Align2::LEFT_TOP, *header, medium(13.0), ACCENT);
            for (k, &i) in rows.iter().enumerate() {
                let y = row_rect(i).top();
                let order = g * 3 + k;
                let a = smooth((now - since - 0.12 * order as f64) / 0.3).max(if s.flag("scoreboard").on { 0.0 } else { 1.0 });
                let mut q = q.clone();
                q.multiply_opacity(a);
                let um = shown[i].exp();
                let name = if i < 7 { ROWS[i] } else if i == 7 { "center blur" } else { "corner blur" };
                q.text(pos2(r.left() + 18.0, y + ROW_H / 2.0), Align2::LEFT_CENTER, name, if i < 7 { regular(15.5) } else { medium(15.5) }, TEXT);
                let bar = Rect::from_min_max(pos2(x0, y + 5.0), pos2(bar_x(um).max(x0 + 3.0), y + ROW_H - 5.0));
                let c = if i == 4 { grade(um / GOOD, GOOD_PCT, FAIR_PCT) } else { grade(um, GOOD, FAIR) };
                if i < 7 {
                    q.rect_filled(bar, CornerRadius::same(3), c);
                } else {
                    q.rect(bar, CornerRadius::same(3), c.gamma_multiply(0.35), Stroke::new(1.5, c), StrokeKind::Inside);
                }
                let value = if i == 4 {
                    format!("{:+.1} %", o.score.bars[4].signum() * um / GOOD).replace('-', "−")
                } else if um < 9.95 {
                    format!("{um:.1} µm")
                } else {
                    format!("{um:.0} µm")
                };
                q.text(pos2(r.right() - 18.0, y + ROW_H / 2.0), Align2::RIGHT_CENTER, value, medium(15.0), c);
            }
        }
    }

    /// The special view, once the cross-section has made room for it.
    fn draw_view(&mut self, p: &Painter, s: &State, o: &Optics, now: f64) {
        let shown = |f: &str| smooth(2.0 * s.flag(f).alpha(now, 2.0 * FADE) as f64 - 1.0);
        let all = VIEWS.iter().map(|(f, _)| shown(f)).fold(0.0, f32::max);
        if all > 0.0 {
            let mut q = p.clone();
            q.multiply_opacity(all);
            panel(&q, view_rect());
        }
        for (flag, title) in VIEWS {
            let a = shown(flag);
            if a <= 0.0 {
                if flag == "focus_zoom" {
                    self.zoom = None;
                }
                if flag == "field_curves" {
                    self.field_range = None;
                }
                continue;
            }
            let mut q = p.clone();
            q.multiply_opacity(a);
            let r = view_rect();
            q.text(r.left_top() + vec2(14.0, 10.0), Align2::LEFT_TOP, title, bold(17.0), TEXT);
            let area = Rect::from_min_max(r.min + vec2(14.0, 40.0), r.max - vec2(14.0, 30.0));
            match flag {
                "focus_zoom" => self.draw_focus_zoom(&q, area, s, o),
                "field_curves" => self.draw_field_curves(&q, area, s, o),
                _ => self.draw_bending(&q, area, o),
            }
        }
    }

    fn draw_focus_zoom(&mut self, p: &Painter, area: Rect, s: &State, o: &Optics) {
        let sz = o.lens.sensor_z;
        let rays: Vec<&optics::RayPath> = o.axial.iter().filter(|r| r.passed && r.pts.len() >= 2 && (s.by_wavelength || r.lambda == optics::RGB_WAVELENGTHS[1])).collect();
        let line = |r: &optics::RayPath| {
            let (a, b) = (r.pts[r.pts.len() - 2], r.pts[r.pts.len() - 1]);
            let k = (b.y - a.y) / (b.z - a.z);
            move |z: f64| a.y + (z - a.z) * k
        };
        let crossing = rays.iter().map(|r| {
            let (a, b) = (r.pts[r.pts.len() - 2], r.pts[r.pts.len() - 1]);
            if (b.y - a.y).abs() < 1e-12 { sz } else { a.z - a.y * (b.z - a.z) / (b.y - a.y) }
        });
        let len = crossing.map(|z| (z - sz).abs()).fold(0.0, f64::max).max(0.03) * 1.35;
        let len = if s.zoom_len > 0.0 { s.zoom_len } else { len };
        let height = rays.iter().map(|r| line(r)(sz - len).abs().max(line(r)(sz + len).abs())).fold(0.0, f64::max).max(1e-3) * 1.05;
        let height = if s.zoom_height > 0.0 { s.zoom_height } else { height };
        let k = 1.0 - (-self.dt / 0.35).exp();
        let (l, h) = match self.zoom {
            Some((l, h)) => (l + (len - l) * k, h + (height - h) * k),
            None => (len, height),
        };
        self.zoom = Some((l, h));
        let at = |z: f64, y: f64| pos2(area.center().x + ((z - sz) / l) as f32 * area.width() / 2.0, area.center().y - (y / h) as f32 * area.height() / 2.0);
        let clip = p.with_clip_rect(area);
        clip.line_segment([at(sz - l, 0.0), at(sz + l, 0.0)], Stroke::new(1.0, EDGE));
        for r in &rays {
            let c = if s.by_wavelength {
                RGB[optics::RGB_WAVELENGTHS.iter().position(|x| *x == r.lambda).unwrap_or(1)]
            } else {
                lerp_color(Color32::from_rgb(140, 210, 255), Color32::from_rgb(255, 120, 80), r.pupil.abs() as f32)
            };
            let f = line(r);
            clip.line_segment([at(sz - l, f(sz - l)), at(sz + l, f(sz + l))], Stroke::new(1.4, c));
        }
        let top = at(sz, h);
        clip.extend(Shape::dashed_line(&[pos2(top.x, area.top()), pos2(top.x, area.bottom())], Stroke::new(2.0, Color32::WHITE), 6.0, 4.0));
        p.text(pos2(top.x, area.bottom() + 4.0), Align2::CENTER_TOP, "sensor", medium(13.0), TEXT);
        let dscale = DiagramView::new(self.dr, sz, s.diagram_zoom).scale;
        let (mz, my) = (area.width() as f64 / 2.0 / l / dscale, area.height() as f64 / 2.0 / h / dscale);
        p.text(area.left_bottom() + vec2(0.0, 4.0), Align2::LEFT_TOP, format!("×{} along, ×{} across", round2(mz), round2(my)), regular(12.0), DIM);
        if !s.by_wavelength {
            p.text(area.right_bottom() + vec2(0.0, 4.0), Align2::RIGHT_TOP, "edge rays", medium(12.0), Color32::from_rgb(255, 120, 80));
            p.text(area.right_bottom() + vec2(-66.0, 4.0), Align2::RIGHT_TOP, "center rays", medium(12.0), Color32::from_rgb(140, 210, 255));
        } else {
            for (i, name) in ["blue", "green", "red"].iter().enumerate() {
                p.text(area.right_bottom() + vec2(-46.0 * (2 - i) as f32, 4.0), Align2::RIGHT_TOP, *name, medium(12.0), RGB[i]);
            }
        }
    }

    fn draw_field_curves(&mut self, p: &Painter, area: Rect, s: &State, o: &Optics) {
        let max = o.curves.iter().flat_map(|c| [c[1].abs(), c[2].abs(), c[3].abs()]).fold(0.0, f64::max).max(0.1) * 1.15;
        let target = if s.field_range > 0.0 { s.field_range } else { max };
        let k = 1.0 - (-self.dt / 0.35).exp();
        let range = self.field_range.map_or(target, |r| r + (target - r) * k);
        self.field_range = Some(range);
        let hmax = o.curves.last().map_or(21.6, |c| c[0]);
        let plot = Rect::from_min_max(area.min + vec2(46.0, 16.0), area.max - vec2(0.0, 14.0));
        let at = |dz: f64, h: f64| pos2(plot.center().x + (dz / range) as f32 * plot.width() / 2.0, plot.bottom() - (h / hmax) as f32 * plot.height());
        let clip = p.with_clip_rect(plot.expand(2.0));
        clip.extend(Shape::dashed_line(&[pos2(plot.center().x, plot.top()), pos2(plot.center().x, plot.bottom())], Stroke::new(2.0, Color32::WHITE), 6.0, 4.0));
        p.line_segment([plot.left_bottom(), plot.right_bottom()], Stroke::new(1.0, EDGE));
        for (i, (color, label)) in [(T_COLOR, "T"), (S_COLOR, "S"), (P_COLOR, "Petzval")].into_iter().enumerate() {
            let pts: Vec<Pos2> = o.curves.iter().map(|c| at(c[i + 1], c[0])).collect();
            if i == 2 {
                clip.extend(Shape::dashed_line(&pts, Stroke::new(2.0, color), 5.0, 4.0));
            } else {
                clip.add(Shape::line(pts.clone(), Stroke::new(2.5, color)));
            }
            if let Some(end) = pts.last() {
                let pos = pos2(end.x.clamp(plot.left() + 10.0, plot.right() - 10.0), end.y - 2.0);
                p.text(pos, Align2::CENTER_BOTTOM, label, medium(13.0), color);
            }
        }
        p.text(plot.left_bottom() + vec2(-6.0, 0.0), Align2::RIGHT_BOTTOM, "center", regular(12.0), DIM);
        p.text(plot.left_top() + vec2(-6.0, 0.0), Align2::RIGHT_TOP, "corner", regular(12.0), DIM);
        p.text(plot.left_bottom() + vec2(0.0, 4.0), Align2::LEFT_TOP, format!("−{}", mm(range)), regular(12.0), DIM);
        p.text(plot.right_bottom() + vec2(0.0, 4.0), Align2::RIGHT_TOP, format!("+{}", mm(range)), regular(12.0), DIM);
        p.text(pos2(plot.center().x, plot.bottom() + 4.0), Align2::CENTER_TOP, "sensor", medium(13.0), TEXT);
    }

    fn draw_bending(&self, p: &Painter, area: Rect, o: &Optics) {
        let c = &self.bending;
        if c.len() < 2 {
            p.text(area.center(), Align2::CENTER_CENTER, "no bending curve", regular(14.0), DIM);
            return;
        }
        let (x0, x1) = (c.first().unwrap()[0], c.last().unwrap()[0]);
        let ymax = c.iter().map(|q| q[1]).fold(0.0, f64::max) * 1.1;
        let plot = Rect::from_min_max(area.min + vec2(30.0, 18.0), area.max - vec2(30.0, 16.0));
        let at = |x: f64, y: f64| pos2(plot.left() + ((x - x0) / (x1 - x0)) as f32 * plot.width(), plot.bottom() - (y / ymax) as f32 * plot.height());
        p.line_segment([plot.left_bottom(), plot.right_bottom()], Stroke::new(1.0, EDGE));
        p.add(Shape::line(c.iter().map(|q| at(q[0], q[1])).collect(), Stroke::new(2.5, TEXT)));
        for (x, label) in [(-1.0, "flat front"), (0.0, "equiconvex"), (1.0, "flat back")] {
            if x >= x0 && x <= x1 {
                let b = at(x, 0.0);
                p.line_segment([b, b - vec2(0.0, 5.0)], Stroke::new(1.0, DIM));
                p.text(b + vec2(0.0, 3.0), Align2::CENTER_TOP, label, regular(12.0), DIM);
            }
        }
        let best = c.iter().min_by(|a, b| a[1].total_cmp(&b[1])).unwrap();
        p.text(at(best[0], best[1]) - vec2(0.0, 14.0), Align2::CENTER_BOTTOM, "best form", medium(13.0), GREEN);
        let w = p.text(plot.left_top() + vec2(12.0, -16.0), Align2::LEFT_TOP, "blur from spherical aberration", regular(12.0), DIM);
        p.arrow(pos2(plot.left() + 4.0, w.bottom()), vec2(0.0, -w.height()), Stroke::new(1.0, DIM));
        if let Some(x) = o.shape.filter(|x| x.is_finite()) {
            let x = x.clamp(x0, x1);
            let i = c.windows(2).position(|w| x <= w[1][0]).unwrap_or(c.len() - 2);
            let (a, b) = (c[i], c[i + 1]);
            let y = a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]);
            let q = at(x, y);
            p.circle_filled(q, 9.0, ACCENT.gamma_multiply(0.3));
            p.circle_filled(q, 6.0, ACCENT);
        }
    }

    fn draw_toolbox(&self, p: &Painter, ci: usize, t: f64) {
        let r = toolbox_rect();
        panel(p, r);
        p.text(r.left_top() + vec2(16.0, 10.0), Align2::LEFT_TOP, "Toolbox", bold(17.0), TEXT);
        let mut y = r.top() + 40.0;
        let y0 = y;
        for (k, c) in self.timeline.chapters.iter().enumerate().take(ci + 1) {
            let Some(tool) = &c.toolbox else { continue };
            let age = if k == ci { t - tool.t } else { f64::INFINITY };
            if age < 0.0 {
                continue;
            }
            let pop = if age.is_finite() { smooth(age / 0.25) * (1.0 - smooth((age - 1.2) / 1.5)) } else { 0.0 };
            let mut q = p.clone();
            q.multiply_opacity(smooth(age / 0.25));
            let row = Rect::from_min_size(pos2(r.left() + 8.0, y - 1.0), vec2(r.width() - 16.0, 21.0));
            q.rect_filled(row, CornerRadius::same(5), ACCENT.gamma_multiply(0.22 * pop));
            q.circle_filled(pos2(r.left() + 20.0, y + 9.5), 3.0, ACCENT);
            let galley = q.layout(tool.text.clone(), regular(14.0), lerp_color(TEXT, ACCENT, pop), r.width() - 44.0);
            let h = galley.size().y;
            q.galley(pos2(r.left() + 32.0, y), galley, TEXT);
            y += h + 1.5;
        }
        if y == y0 {
            p.text(pos2(r.left() + 16.0, y), Align2::LEFT_TOP, "Each idea that fixes\na problem goes here.", regular(14.0), DIM);
        }
    }

    fn draw_recap(&self, p: &Painter, age: f64) {
        let rr = self.rr;
        p.rect_filled(rr, CornerRadius::same(8), BG);
        let n = self.timeline.recap.len();
        if n == 0 {
            return;
        }
        let cols = if n <= 9 { 3 } else { 4 };
        let top = 76.0;
        let cell = vec2(rr.width() / cols as f32, (rr.height() - top) / n.div_ceil(cols) as f32);
        let lenses: Vec<_> = self.timeline.recap.iter().map(|e| self.lenses.get(&e.lens)).collect();
        for (i, (e, lens)) in self.timeline.recap.iter().zip(&lenses).enumerate() {
            let Some(lens) = lens else { continue };
            // Schematic: each design as tall as its cell allows.
            let len = (lens.last_z() - lens.surfaces[0].z).max(1.0);
            let sd = lens.surfaces.iter().map(|s| s.semi_diameter).fold(1.0, f64::max);
            let scale = ((cell.x - 80.0) as f64 / len).min((cell.y - 76.0) as f64 / (2.0 * sd));
            let mut q = p.clone();
            q.multiply_opacity(smooth((age - 0.45 * i as f64) / 0.4));
            let c = rr.min + vec2(cell.x * (i % cols) as f32, top + cell.y * (i / cols) as f32);
            let mid = c + vec2(cell.x / 2.0, (cell.y - 48.0) / 2.0);
            let z_mid = 0.5 * (lens.surfaces[0].z + lens.last_z());
            let view = DiagramView { sensor: mid + vec2(((lens.sensor_z - z_mid) * scale) as f32, 0.0), sensor_z: lens.sensor_z, scale };
            draw_elements(&q, &view, lens, &[]);
            q.text(c + vec2(cell.x / 2.0, cell.y - 46.0), Align2::CENTER_TOP, &e.label, regular(13.0), DIM);
            q.text(c + vec2(cell.x / 2.0, cell.y - 28.0), Align2::CENTER_TOP, &e.tool, medium(16.0), ACCENT);
        }
    }
}

fn round2(x: f64) -> String {
    let e = 10f64.powf(x.log10().floor() - 1.0).max(1.0);
    format!("{:.0}", (x / e).round() * e)
}

fn mm(x: f64) -> String {
    if x < 1.0 { format!("{x:.2} mm") } else { format!("{x:.1} mm") }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgba_premultiplied(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()), m(a.a(), b.a()))
}

/// The images shown: the main one, the one it crossfades from, and the left
/// image of a split screen up to a fraction of the width.
struct Images {
    main: Option<egui::TextureId>,
    prev: Option<(egui::TextureId, f32)>,
    split: Option<(egui::TextureId, f32)>,
}

impl Images {
    /// Draws image region `uv` into `r`.
    fn draw(&self, p: &Painter, r: Rect, uv: Rect, round: CornerRadius) {
        let blit = |id: egui::TextureId, r: Rect, uv: Rect, alpha: f32, round: CornerRadius| {
            p.add(egui::epaint::RectShape::filled(r, round, Color32::WHITE.gamma_multiply(alpha)).with_texture(id, uv));
        };
        p.rect_filled(r, round, Color32::BLACK);
        if let Some((id, a)) = self.prev {
            blit(id, r, uv, 1.0, round);
            if let Some(m) = self.main {
                blit(m, r, uv, a, round);
            }
        } else if let Some(m) = self.main {
            blit(m, r, uv, 1.0, round);
        }
        if let Some((id, s)) = self.split.filter(|(_, s)| *s > uv.min.x) {
            let f = ((s - uv.min.x) / uv.width()).min(1.0);
            let sub = Rect::from_min_max(r.min, pos2(r.left() + r.width() * f, r.bottom()));
            let sub_uv = Rect::from_min_max(uv.min, pos2(uv.min.x + uv.width() * f, uv.max.y));
            let round = if f < 1.0 { CornerRadius { ne: 0, se: 0, ..round } } else { round };
            blit(id, sub, sub_uv, 1.0, round);
        }
    }
}

/// Dims and greys the render outside a feathered central ellipse by `a`.
fn spotlight(p: &Painter, r: Rect, a: f32) {
    let (nx, ny) = (48, 32);
    let mut mesh = egui::Mesh::default();
    for j in 0..=ny {
        for i in 0..=nx {
            let (u, v) = (i as f32 / nx as f32, j as f32 / ny as f32);
            let d = (((u - 0.5) / 0.2).powi(2) + ((v - 0.5) / 0.24).powi(2)).sqrt();
            let t = ((d - 0.75) / 0.6).clamp(0.0, 1.0);
            let alpha = a * 0.65 * t * t * (3.0 - 2.0 * t);
            mesh.colored_vertex(r.lerp_inside(vec2(u, v)), Color32::from_rgba_unmultiplied(26, 28, 32, (alpha * 255.0) as u8));
            if i > 0 && j > 0 {
                let k = (j * (nx + 1) + i) as u32;
                let w = (nx + 1) as u32;
                mesh.add_triangle(k - w - 1, k - w, k);
                mesh.add_triangle(k - w - 1, k, k - 1);
            }
        }
    }
    p.add(mesh);
}

/// Undistorted chart grid lines (normalised image coordinates) over region `uv` shown in `r`.
fn draw_grid(p: &Painter, r: Rect, uv: Rect, lines: &[[[f32; 2]; 2]], alpha: f32) {
    let map = |q: [f32; 2]| r.min + vec2((q[0] - uv.min.x) / uv.width() * r.width(), (q[1] - uv.min.y) / uv.height() * r.height());
    for l in lines {
        p.line_segment([map(l[0]), map(l[1])], Stroke::new(1.2, Color32::from_rgb(0, 200, 255).gamma_multiply(0.85 * alpha)));
    }
}

fn draw_hud(p: &Painter, s: &State, o: &Optics) {
    let r = hud_rect();
    panel(p, r);
    let n = o.fstop;
    // Shutter time for equal exposure: 1/500 s at f/2; fixed while dimming.
    let stops = 2.0 * (n / s.dim_ref).log2().max(0.0) * s.dim;
    let t = (n / 2.0).powi(2) / 500.0 / 2f64.powf(stops);
    let pieces = o.elements;
    let cells = [
        ("focal length", format!("{:.0} mm", o.lens.focal_length())),
        ("aperture", fnumber(n)),
        ("shutter", shutter(t)),
        ("glass", if pieces == 1 { "1 element".into() } else { format!("{pieces} elements") }),
    ];
    let w = r.width() / cells.len() as f32;
    for (i, (label, value)) in cells.iter().enumerate() {
        let x = r.left() + w * i as f32 + 16.0;
        p.text(pos2(x, r.top() + 6.0), Align2::LEFT_TOP, *label, regular(12.0), DIM);
        p.text(pos2(x, r.bottom() - 4.0), Align2::LEFT_BOTTOM, value, bold(20.0), TEXT);
        if i > 0 {
            p.line_segment([pos2(r.left() + w * i as f32, r.top() + 10.0), pos2(r.left() + w * i as f32, r.bottom() - 10.0)], Stroke::new(1.0, EDGE));
        }
    }
}

/// Glass, stop and (optionally) a highlighted piece of glass (0 = front) of `lens`.
fn draw_elements(p: &Painter, v: &DiagramView, lens: &crate::lens::Lens, highlight: &[usize]) {
    let n = 40;
    let surf = &lens.surfaces;
    let mut piece = 0;
    for (i, s) in surf.iter().enumerate() {
        if s.material.is_none() || i + 1 >= surf.len() {
            continue;
        }
        let t = &surf[i + 1];
        let lit = highlight.contains(&piece);
        piece += 1;
        let fill = if lit { ACCENT.gamma_multiply(0.55) } else { GLASS };
        let mut mesh = egui::Mesh::default();
        for k in 0..=n {
            let u = 2.0 * k as f64 / n as f64 - 1.0;
            let (ya, yb) = (u * s.semi_diameter, u * t.semi_diameter);
            mesh.colored_vertex(v.at(s.z + s.sag(ya), ya), fill);
            mesh.colored_vertex(v.at(t.z + t.sag(yb), yb), fill);
            if k > 0 {
                let b = 2 * k as u32;
                mesh.add_triangle(b - 2, b - 1, b);
                mesh.add_triangle(b - 1, b + 1, b);
            }
        }
        p.add(mesh);
        for sign in [-1.0, 1.0] {
            let (ya, yb) = (sign * s.semi_diameter, sign * t.semi_diameter);
            p.line_segment([v.at(s.z + s.sag(ya), ya), v.at(t.z + t.sag(yb), yb)], Stroke::new(1.2, GLASS_EDGE));
        }
    }
    for s in surf {
        if s.is_stop {
            for sign in [-1.0, 1.0] {
                p.line_segment([v.at(s.z, sign * lens.iris.max_radius), v.at(s.z, sign * (s.semi_diameter + 3.0))], Stroke::new(3.0, TEXT));
            }
            continue;
        }
        let pts = (0..=n).map(|k| s.semi_diameter * (2.0 * k as f64 / n as f64 - 1.0)).map(|y| v.at(s.z + s.sag(y), y)).collect();
        p.add(Shape::line(pts, Stroke::new(1.5, GLASS_EDGE)));
    }
}

/// The cross-section: lens, rays, sensor and overlays.
fn draw_lens(p: &Painter, v: &DiagramView, o: &Optics, s: &State, now: f64, r: Rect) {
    let lens = &o.lens;
    p.line_segment([pos2(r.left(), v.sensor.y), pos2(r.right(), v.sensor.y)], Stroke::new(1.0, EDGE));
    // The axial beam: where it is narrow.
    let beam = s.flag("beam").alpha(now, FADE);
    if beam > 0.0 {
        if let (Some(top), Some(bottom)) = (o.fans[0].1.last().filter(|r| r.passed), o.fans[0].1.first().filter(|r| r.passed)) {
            let mut mesh = egui::Mesh::default();
            let fill = ACCENT.gamma_multiply(0.5 * beam);
            for (k, (a, b)) in top.pts.iter().zip(&bottom.pts).enumerate() {
                mesh.colored_vertex(v.at(a.z, a.y), fill);
                mesh.colored_vertex(v.at(b.z, b.y), fill);
                if k > 0 {
                    let i = 2 * k as u32;
                    mesh.add_triangle(i - 2, i - 1, i);
                    mesh.add_triangle(i - 1, i + 1, i);
                }
            }
            p.add(mesh);
        }
    }
    draw_elements(p, v, lens, &s.highlight);
    let fans = o.fans.iter().filter(|(f, _)| if s.by_wavelength { *f <= -1.0 } else { *f >= 0.0 });
    for (f, rays) in fans {
        let field = if *f <= -1.0 { -1.0 - f } else { *f };
        if !s.fields.iter().any(|x| (x - field).abs() < 1e-6) {
            continue;
        }
        let fi = [0.0, 0.7, 1.0].iter().position(|x| (x - field).abs() < 1e-6).unwrap_or(0);
        let step = (11 / s.rays.max(2)).max(1);
        for (k, ray) in rays.iter().enumerate() {
            let idx = if s.by_wavelength { k / 3 } else { k };
            if idx % step != 0 && idx != 10 {
                continue;
            }
            let color = if s.by_wavelength { RGB[optics::RGB_WAVELENGTHS.iter().position(|x| *x == ray.lambda).unwrap_or(1)].gamma_multiply(0.75) } else { FIELD_COLORS[fi] };
            let pts: Vec<Pos2> = ray.pts.iter().map(|q| v.at(q.z, q.y)).collect();
            let edge = ray.pupil.abs() > 0.99;
            let width = if beam > 0.0 && field == 0.0 && edge { 1.2 + 1.6 * beam } else { 1.2 };
            p.add(Shape::line(pts.clone(), Stroke::new(width, if ray.passed { color } else { color.gamma_multiply(0.35) })));
            if !ray.passed {
                let q = pts[pts.len() - 1];
                let d = 3.5;
                p.line_segment([q + vec2(-d, -d), q + vec2(d, d)], Stroke::new(1.5, color));
                p.line_segment([q + vec2(-d, d), q + vec2(d, -d)], Stroke::new(1.5, color));
            }
        }
    }
    // The sharp image surface at true scale (medial between T and S).
    let fc = s.flag("field_curves").alpha(now, FADE);
    if fc > 0.0 && o.curves.len() > 1 {
        let pts: Vec<Pos2> = o.curves.iter().rev().map(|c| v.at(lens.sensor_z + 0.5 * (c[1] + c[2]), -c[0])).chain(o.curves.iter().skip(1).map(|c| v.at(lens.sensor_z + 0.5 * (c[1] + c[2]), c[0]))).collect();
        p.add(Shape::line(pts, Stroke::new(2.5, T_COLOR.gamma_multiply(fc))));
    }
    let hd = lens.sensor_half.0.hypot(lens.sensor_half.1);
    p.line_segment([v.at(lens.sensor_z, -hd), v.at(lens.sensor_z, hd)], Stroke::new(4.0, TEXT));
    p.text(v.at(lens.sensor_z, hd) - vec2(0.0, 4.0), Align2::CENTER_BOTTOM, "sensor", medium(13.0), DIM);
}

/// The cross-section's lens name.
fn draw_label(p: &Painter, o: &Optics, r: Rect, alpha: f32) {
    p.text(r.left_top() + vec2(14.0, 10.0), Align2::LEFT_TOP, &o.label, medium(15.0), DIM.gamma_multiply(alpha));
}

/// Points and lengths (per axis) of an annotation's coordinate space.
fn note_space(on: &str, sensor_z: f64, rr: Rect, dr: Rect, zoom: f64) -> Option<(Box<dyn Fn([f64; 2]) -> Pos2>, egui::Vec2)> {
    let rect = |r: Rect| -> (Box<dyn Fn([f64; 2]) -> Pos2>, egui::Vec2) { (Box::new(move |q: [f64; 2]| r.min + vec2(q[0] as f32 * r.width(), q[1] as f32 * r.height())), r.size()) };
    Some(match on {
        "render" => rect(rr),
        "center" => rect(inset_rect(0)),
        "corner" => rect(inset_rect(1)),
        "view" => rect(view_rect()),
        "scoreboard" => rect(score_rect()),
        "hud" => rect(hud_rect()),
        "toolbox" => rect(toolbox_rect()),
        "screen" => rect(Rect::from_min_size(Pos2::ZERO, vec2(W, H))),
        "diagram" => {
            let v = DiagramView::new(dr, sensor_z, zoom);
            (Box::new(move |q: [f64; 2]| v.at(sensor_z + q[0], q[1])), egui::Vec2::splat(v.scale as f32))
        }
        _ => return None,
    })
}

fn named_color(c: Option<&str>) -> Color32 {
    match c.unwrap_or("yellow") {
        "red" => Color32::from_rgb(255, 107, 107),
        "green" => Color32::from_rgb(76, 217, 123),
        "blue" => Color32::from_rgb(108, 182, 255),
        "cyan" => Color32::from_rgb(77, 208, 225),
        "orange" => Color32::from_rgb(255, 159, 67),
        "white" => Color32::WHITE,
        _ => ACCENT,
    }
}

/// An annotation, drawn on and faded out.
fn draw_note(p: &Painter, n: &Note, t: f64, o: &Optics, rr: Rect, dr: Rect, zoom: f64) {
    if t < n.t || n.until.is_some_and(|u| t > u + FADE) {
        return;
    }
    if let Some(rows) = &n.row {
        let a = smooth((t - n.t) / 0.15) * n.until.map_or(1.0, |u| 1.0 - smooth((t - u) / FADE));
        let mut q = p.clone();
        q.multiply_opacity(a);
        let color = named_color(n.color.as_deref());
        for r in rows.split(',').filter_map(|r| score_region(r.trim())) {
            let r = r.expand2(vec2(2.0, 1.0));
            q.rect_stroke(r, CornerRadius::same(5), Stroke::new(6.0, Color32::from_black_alpha(110)), StrokeKind::Middle);
            q.rect_stroke(r, CornerRadius::same(5), Stroke::new(3.0, color), StrokeKind::Middle);
        }
        return;
    }
    let Some((at, unit)) = note_space(&n.on, o.lens.sensor_z, rr, dr, zoom) else { return };
    let grow = smooth((t - n.t) / 0.45);
    let alpha = smooth((t - n.t) / 0.15) * n.until.map_or(1.0, |u| 1.0 - smooth((t - u) / FADE));
    let mut q = p.clone();
    q.multiply_opacity(alpha);
    let color = named_color(n.color.as_deref());
    let c = at(n.pos);
    let shadow = Stroke::new(6.0, Color32::from_black_alpha(110));
    let stroke = Stroke::new(3.0, color);
    let mut label_at = (c, Align2::CENTER_CENTER);
    match n.shape.as_str() {
        "circle" | "box" => {
            let r = n.r.unwrap_or([0.05, 0.05]);
            let (rx, ry) = (r[0] as f32 * unit.x, r[1] as f32 * unit.y);
            if n.shape == "circle" {
                let steps = (72.0 * grow).ceil() as usize;
                let pts: Vec<Pos2> = (0..=steps).map(|k| {
                    let a = -PI / 2.0 + 2.0 * PI * grow * k as f32 / steps.max(1) as f32;
                    c + vec2(rx * a.cos(), ry * a.sin())
                }).collect();
                if pts.len() > 1 {
                    q.add(Shape::line(pts.clone(), shadow));
                    q.add(Shape::line(pts, stroke));
                }
            } else {
                let rect = Rect::from_center_size(c, vec2(2.0 * rx, 2.0 * ry));
                q.rect_stroke(rect, CornerRadius::same(4), shadow, StrokeKind::Middle);
                q.rect_stroke(rect, CornerRadius::same(4), stroke, StrokeKind::Middle);
            }
            label_at = (c + vec2(rx * 0.72 + 6.0, -ry * 0.72 - 6.0), Align2::LEFT_BOTTOM);
        }
        "arrow" => {
            let to = n.to.map_or(c, &at);
            let tip = c + (to - c) * grow;
            let d = (to - c).normalized();
            let head = [tip, tip - d * 14.0 + d.rot90() * 7.0, tip - d * 14.0 - d.rot90() * 7.0];
            q.line_segment([c, tip - d * 8.0], shadow);
            q.line_segment([c, tip - d * 8.0], stroke);
            q.add(Shape::convex_polygon(head.to_vec(), color, Stroke::new(1.5, Color32::from_black_alpha(110))));
            label_at = (c - d * 8.0, if d.x >= 0.0 { Align2::RIGHT_CENTER } else { Align2::LEFT_CENTER });
        }
        _ => {}
    }
    if let Some(text) = &n.text {
        let (pos, anchor) = match n.text_at {
            Some(tp) => (at(tp), Align2::CENTER_CENTER),
            None => label_at,
        };
        pill(&q, pos, anchor, text, medium(18.0), color, Color32::from_rgba_premultiplied(13, 16, 21, 225));
    }
}

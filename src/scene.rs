//! Scene description (TOML), textures and ray-scene intersection.
//!
//! Scene files use metres with x right, y up and z pointing from the camera
//! into the scene, measured from the lens' front vertex. Internally everything
//! is converted to lens coordinates (mm, light travelling towards +z).

use crate::math::{Ray, Vec3};
use crate::spectrum::{ColorSystem, blackbody, srgb_decode};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

fn d_one() -> f64 {
    1.0
}
fn d_true() -> bool {
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneFile {
    pub camera: CameraCfg,
    #[serde(default)]
    pub background: Option<[f64; 3]>,
    #[serde(default, rename = "object")]
    pub objects: Vec<ObjectCfg>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct CameraCfg {
    pub lens: String,
    #[serde(default = "d_36")]
    pub sensor_width_mm: f64,
    #[serde(default = "d_24")]
    pub sensor_height_mm: f64,
    #[serde(default = "d_width")]
    pub width: u32,
    /// Infinity f-number; wide open if absent.
    pub fstop: Option<f64>,
    /// Metres from the front vertex; `inf` for infinity.
    #[serde(default = "d_inf")]
    pub focus_distance: f64,
    #[serde(default)]
    pub focus_offset_mm: f64,
    /// "lens" (the lens file's coating), "none", "single" (MgF2 quarter wave)
    /// or "multi" (broadband multi-layer).
    #[serde(default = "d_coating")]
    pub coating: String,
    /// Overrides the lens' iris.
    #[serde(default)]
    pub iris_blades: Option<u32>,
    #[serde(default)]
    pub iris_rotation_deg: f64,
    #[serde(default)]
    pub iris_curvature: Option<f64>,
    /// Manufacturing error: lateral shift in mm of glass elements, `[[element, dx, dy], ...]`
    /// with elements numbered from 0 at the front.
    #[serde(default)]
    pub decenter: Vec<[f64; 3]>,
    #[serde(default = "d_spp")]
    pub spp: u32,
    #[serde(default)]
    pub exposure_ev: f64,
    #[serde(default = "d_true")]
    pub diffraction: bool,
    /// Image formation through wave-optical PSFs (diffraction and aberrations
    /// together) instead of geometric rays plus an iris diffraction pattern.
    #[serde(default)]
    pub wave_optics: bool,
    #[serde(default = "d_true")]
    pub ghosts: bool,
    /// Forward samples per light for the direct image.
    #[serde(default = "d_light_samples")]
    pub light_samples: u64,
    /// Ray grid resolution (per side) for every light and ghost path.
    #[serde(default = "d_ghost_grid")]
    pub ghost_grid: usize,
    /// Ghosts are traced for lights with at least this fraction of the brightest light's power.
    #[serde(default = "d_ghost_threshold")]
    pub ghost_threshold: f64,
    #[serde(default = "d_bins")]
    pub spectral_bins: usize,
    /// "clip" or "filmic".
    #[serde(default = "d_tonemap")]
    pub tonemap: String,
}

fn d_36() -> f64 {
    36.0
}
fn d_24() -> f64 {
    24.0
}
fn d_width() -> u32 {
    1200
}
fn d_inf() -> f64 {
    f64::INFINITY
}
fn d_coating() -> String {
    "lens".into()
}
fn d_spp() -> u32 {
    64
}
fn d_light_samples() -> u64 {
    4_000_000
}
fn d_ghost_grid() -> usize {
    128
}
fn d_ghost_threshold() -> f64 {
    0.01
}
fn d_bins() -> usize {
    16
}
fn d_tonemap() -> String {
    "clip".into()
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObjectCfg {
    /// Emissive textured rectangle (a lit surface).
    Plane {
        center: [f64; 3],
        size: [f64; 2],
        #[serde(default)]
        rotate: [f64; 3],
        texture: TextureCfg,
        #[serde(default = "d_one")]
        brightness: f64,
    },
    /// Small bright spherical light source.
    Light {
        center: [f64; 3],
        radius: f64,
        temperature: Option<f64>,
        color: Option<[f64; 3]>,
        radiance: f64,
    },
    /// `count` lights evenly spaced from `from` to `to`.
    LightString {
        from: [f64; 3],
        to: [f64; 3],
        count: u32,
        radius: f64,
        temperature: Option<f64>,
        color: Option<[f64; 3]>,
        radiance: f64,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TextureCfg {
    Color { rgb: [f64; 3] },
    Image { path: String },
    Checker {
        n: u32,
        #[serde(default = "d_black")]
        a: [f64; 3],
        #[serde(default = "d_white")]
        b: [f64; 3],
    },
    Grid {
        lines: u32,
        #[serde(default = "d_line_width")]
        width: f64,
    },
    Chart,
    Noise {
        scale: f64,
        palette: Vec<[f64; 3]>,
        #[serde(default)]
        seed: u32,
    },
}

fn d_black() -> [f64; 3] {
    [0.03, 0.03, 0.03]
}
fn d_white() -> [f64; 3] {
    [0.85, 0.85, 0.85]
}
fn d_line_width() -> f64 {
    0.08
}

pub enum Texture {
    Color([f32; 3]),
    Image { w: usize, h: usize, data: Vec<[f32; 3]> },
    Checker { n: f64, a: [f32; 3], b: [f32; 3] },
    Grid { lines: f64, width: f64 },
    Chart,
    Noise { scale: f64, palette: Vec<[f32; 3]>, seed: u32 },
}

fn f3(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6b343) ^ (y as u32).wrapping_mul(0xd8163841) ^ seed.wrapping_mul(0xcb1ab31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 15;
    (h & 0xffffff) as f32 / 16777216.0
}

fn value_noise(x: f64, y: f64, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let tx = (x - x.floor()) as f32;
    let ty = (y - y.floor()) as f32;
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let a = hash(xi, yi, seed);
    let b = hash(xi + 1, yi, seed);
    let c = hash(xi, yi + 1, seed);
    let d = hash(xi + 1, yi + 1, seed);
    (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy
}

impl Texture {
    /// Linear RGB at (u, v) ∈ [0,1]², v pointing up; `aspect` = width / height.
    pub fn eval(&self, u: f64, v: f64, aspect: f64) -> [f32; 3] {
        match self {
            Texture::Color(c) => *c,
            Texture::Image { w, h, data } => {
                let x = (u * *w as f64 - 0.5).clamp(0.0, *w as f64 - 1.0);
                let y = ((1.0 - v) * *h as f64 - 0.5).clamp(0.0, *h as f64 - 1.0);
                let x0 = x.floor() as usize;
                let y0 = y.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);
                let tx = (x - x0 as f64) as f32;
                let ty = (y - y0 as f64) as f32;
                let top = lerp3(data[y0 * w + x0], data[y0 * w + x1], tx);
                let bot = lerp3(data[y1 * w + x0], data[y1 * w + x1], tx);
                lerp3(top, bot, ty)
            }
            Texture::Checker { n, a, b } => {
                let i = (u * n).floor() as i64 + (v * n / aspect).floor() as i64;
                if i.rem_euclid(2) == 0 { *a } else { *b }
            }
            Texture::Grid { lines, width } => {
                let cu = u * lines;
                let cv = v * lines / aspect;
                let du = (cu - cu.round()).abs();
                let dv = (cv - cv.round()).abs();
                if du < width / 2.0 || dv < width / 2.0 { [0.02, 0.02, 0.02] } else { [0.85, 0.85, 0.85] }
            }
            Texture::Chart => chart(u, v, aspect),
            Texture::Noise { scale, palette, seed } => {
                let mut amp = 0.5;
                let mut f = 1.0;
                let mut sum = 0.0;
                let mut norm = 0.0;
                for o in 0..5 {
                    sum += amp * value_noise(u * scale * f, v * scale * f / aspect, seed + o);
                    norm += amp;
                    amp *= 0.5;
                    f *= 2.03;
                }
                let t = (sum / norm).clamp(0.0, 0.9999) * (palette.len() - 1) as f32;
                let i = t.floor() as usize;
                lerp3(palette[i], palette[(i + 1).min(palette.len() - 1)], t - i as f32)
            }
        }
    }
}

/// Test chart: line grid, Siemens stars and a slanted black square.
fn chart(u: f64, v: f64, aspect: f64) -> [f32; 3] {
    const WHITE: [f32; 3] = [0.85, 0.85, 0.85];
    const BLACK: [f32; 3] = [0.02, 0.02, 0.02];
    let x = u * aspect; // square units, height = 1
    let y = v;
    // Siemens stars.
    let r_star = 0.075;
    for sx in [0.08 * aspect.max(1.0), aspect * 0.5, aspect - 0.08 * aspect.max(1.0)] {
        for sy in [0.12, 0.5, 0.88] {
            let dx = x - sx;
            let dy = y - sy;
            let r = dx.hypot(dy);
            if r < r_star * 1.12 {
                if r > r_star || r < r_star * 0.04 {
                    return WHITE;
                }
                let a = dy.atan2(dx);
                return if (a * 18.0).sin() > 0.0 { BLACK } else { WHITE };
            }
        }
    }
    // Slanted square near the centre-left.
    let (cx, cy) = (aspect * 0.3, 0.5);
    let ang = 5f64.to_radians();
    let lx = (x - cx) * ang.cos() + (y - cy) * ang.sin();
    let ly = -(x - cx) * ang.sin() + (y - cy) * ang.cos();
    if lx.abs() < 0.06 && ly.abs() < 0.06 {
        return BLACK;
    }
    // Colour patches on the right.
    let patches: [[f32; 3]; 6] =
        [[0.6, 0.05, 0.04], [0.05, 0.4, 0.06], [0.04, 0.07, 0.5], [0.7, 0.6, 0.03], [0.6, 0.06, 0.4], [0.05, 0.45, 0.55]];
    let (px, py) = (aspect * 0.7 - 0.06, 0.5 - 0.09);
    let (ix, iy) = (((x - px) / 0.06).floor(), ((y - py) / 0.06).floor());
    if (0.0..2.0).contains(&ix) && (0.0..3.0).contains(&iy) {
        return patches[(iy as usize) * 2 + ix as usize];
    }
    // Grid.
    let cells = 16.0;
    let gx = x * cells;
    let gy = y * cells;
    if (gx - gx.round()).abs() < 0.035 || (gy - gy.round()).abs() < 0.035 {
        return BLACK;
    }
    WHITE
}

pub enum LightSpectrum {
    Blackbody { t: f64, scale: f64 },
    Rgb { rgb: [f32; 3], scale: f64 },
}

impl LightSpectrum {
    pub fn eval(&self, cs: &ColorSystem, l: f64) -> f64 {
        match self {
            LightSpectrum::Blackbody { t, scale } => scale * blackbody(l, *t),
            LightSpectrum::Rgb { rgb, scale } => scale * cs.upsample(*rgb, l),
        }
    }
}

pub struct Rect {
    pub center: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub n: Vec3,
    pub w: f64,
    pub h: f64,
    pub texture: Texture,
    pub brightness: f64,
}

pub struct Light {
    pub center: Vec3,
    pub radius: f64,
    pub spectrum: LightSpectrum,
}

pub struct Scene {
    pub rects: Vec<Rect>,
    pub lights: Vec<Light>,
    pub background: [f32; 3],
}

pub enum Hit<'a> {
    Surface { rect: &'a Rect, u: f64, v: f64 },
    Light,
    Background,
}

fn to_lens(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] * 1000.0, p[1] * 1000.0, -p[2] * 1000.0)
}

fn rotate(v: Vec3, r: [f64; 3]) -> Vec3 {
    let (sx, cx) = r[0].to_radians().sin_cos();
    let (sy, cy) = r[1].to_radians().sin_cos();
    let (sz, cz) = r[2].to_radians().sin_cos();
    let v = Vec3::new(v.x, v.y * cx - v.z * sx, v.y * sx + v.z * cx);
    let v = Vec3::new(v.x * cy + v.z * sy, v.y, -v.x * sy + v.z * cy);
    Vec3::new(v.x * cz - v.y * sz, v.x * sz + v.y * cz, v.z)
}

fn light_spectrum(cs: &ColorSystem, temperature: Option<f64>, color: Option<[f64; 3]>, radiance: f64) -> LightSpectrum {
    match (temperature, color) {
        (Some(t), _) => {
            let lum = cs.spectrum_luminance(|l| blackbody(l, t));
            LightSpectrum::Blackbody { t, scale: radiance / lum }
        }
        (None, Some(c)) => {
            let rgb = f3(c);
            let lum = cs.spectrum_luminance(|l| cs.upsample(rgb, l));
            LightSpectrum::Rgb { rgb, scale: radiance / lum }
        }
        (None, None) => LightSpectrum::Rgb { rgb: [1.0; 3], scale: radiance },
    }
}

impl Scene {
    pub fn build(file: &SceneFile, base: &Path, cs: &ColorSystem) -> Result<Scene> {
        let mut rects = Vec::new();
        let mut lights = Vec::new();
        for o in &file.objects {
            match o {
                ObjectCfg::Plane { center, size, rotate: rot, texture, brightness } => {
                    let flip = |v: Vec3| Vec3::new(v.x, v.y, -v.z);
                    let u = flip(rotate(Vec3::new(1.0, 0.0, 0.0), *rot));
                    let v = flip(rotate(Vec3::new(0.0, 1.0, 0.0), *rot));
                    let texture = match texture {
                        TextureCfg::Color { rgb } => Texture::Color(f3(*rgb)),
                        TextureCfg::Checker { n, a, b } => Texture::Checker { n: *n as f64, a: f3(*a), b: f3(*b) },
                        TextureCfg::Grid { lines, width } => Texture::Grid { lines: *lines as f64, width: *width },
                        TextureCfg::Chart => Texture::Chart,
                        TextureCfg::Noise { scale, palette, seed } => {
                            Texture::Noise { scale: *scale, palette: palette.iter().map(|c| f3(*c)).collect(), seed: *seed }
                        }
                        TextureCfg::Image { path } => {
                            let p = base.join(path);
                            let img = image::open(&p).with_context(|| format!("loading {}", p.display()))?.to_rgb8();
                            let data = img
                                .pixels()
                                .map(|px| {
                                    [
                                        srgb_decode(px[0] as f64 / 255.0) as f32,
                                        srgb_decode(px[1] as f64 / 255.0) as f32,
                                        srgb_decode(px[2] as f64 / 255.0) as f32,
                                    ]
                                })
                                .collect();
                            Texture::Image { w: img.width() as usize, h: img.height() as usize, data }
                        }
                    };
                    rects.push(Rect {
                        center: to_lens(*center),
                        u,
                        v,
                        n: u.cross(v).normalize(),
                        w: size[0] * 1000.0,
                        h: size[1] * 1000.0,
                        texture,
                        brightness: *brightness,
                    });
                }
                ObjectCfg::Light { center, radius, temperature, color, radiance } => lights.push(Light {
                    center: to_lens(*center),
                    radius: radius * 1000.0,
                    spectrum: light_spectrum(cs, *temperature, *color, *radiance),
                }),
                ObjectCfg::LightString { from, to, count, radius, temperature, color, radiance } => {
                    for i in 0..*count {
                        let t = if *count > 1 { i as f64 / (*count - 1) as f64 } else { 0.0 };
                        let p = [0, 1, 2].map(|k| from[k] + (to[k] - from[k]) * t);
                        lights.push(Light {
                            center: to_lens(p),
                            radius: radius * 1000.0,
                            spectrum: light_spectrum(cs, *temperature, *color, *radiance),
                        });
                    }
                }
            }
        }
        Ok(Scene { rects, lights, background: f3(file.background.unwrap_or([0.0; 3])) })
    }

    fn rect_hit(r: &Rect, ray: &Ray) -> Option<(f64, f64, f64)> {
        let dn = ray.d.dot(r.n);
        if dn.abs() < 1e-12 {
            return None;
        }
        let t = (r.center - ray.o).dot(r.n) / dn;
        if t <= 1e-6 {
            return None;
        }
        let p = ray.o + ray.d * t;
        let q = p - r.center;
        let u = q.dot(r.u) / r.w + 0.5;
        let v = q.dot(r.v) / r.h + 0.5;
        if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) { Some((t, u, v)) } else { None }
    }

    fn sphere_hit(l: &Light, ray: &Ray) -> Option<f64> {
        let oc = ray.o - l.center;
        let b = oc.dot(ray.d);
        let c = oc.dot(oc) - l.radius * l.radius;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let t = -b - disc.sqrt();
        if t > 1e-6 { Some(t) } else { None }
    }

    pub fn intersect(&self, ray: &Ray) -> Hit<'_> {
        let mut best = f64::INFINITY;
        let mut hit = Hit::Background;
        for r in &self.rects {
            if let Some((t, u, v)) = Self::rect_hit(r, ray) {
                if t < best {
                    best = t;
                    hit = Hit::Surface { rect: r, u, v };
                }
            }
        }
        for l in &self.lights {
            if let Some(t) = Self::sphere_hit(l, ray) {
                if t < best {
                    best = t;
                    hit = Hit::Light;
                }
            }
        }
        hit
    }

    /// Distance to the nearest surface or light along `ray`.
    pub fn hit_distance(&self, ray: &Ray) -> Option<f64> {
        let rects = self.rects.iter().filter_map(|r| Self::rect_hit(r, ray).map(|h| h.0));
        let lights = self.lights.iter().filter_map(|l| Self::sphere_hit(l, ray));
        rects.chain(lights).min_by(|a, b| a.total_cmp(b))
    }

    /// True if a surface blocks the segment from `ray.o` to distance `dist`.
    pub fn occluded(&self, ray: &Ray, dist: f64) -> bool {
        self.rects.iter().any(|r| Self::rect_hit(r, ray).is_some_and(|(t, _, _)| t < dist))
    }

    /// Spectral radiance seen along `ray` (lights are rendered by light tracing and appear black here).
    pub fn radiance(&self, cs: &ColorSystem, ray: &Ray, l: f64) -> f64 {
        match self.intersect(ray) {
            Hit::Surface { rect, u, v } => rect.brightness * cs.upsample(rect.texture.eval(u, v, rect.w / rect.h), l),
            Hit::Light => 0.0,
            Hit::Background => cs.upsample(self.background, l),
        }
    }
}

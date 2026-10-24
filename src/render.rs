//! Spectral Monte Carlo rendering through the lens.
//!
//! * Scene surfaces are rendered by tracing rays backwards from the sensor
//!   through the real lens into the scene.
//! * Small bright lights are rendered by tracing rays forwards from the light
//!   through the lens onto the sensor, both along the direct path and along
//!   all two-reflection ghost paths (lens flare).
//!
//! The film stores, per pixel, the sensor irradiance integrated over a number
//! of wavelength bins.

use crate::lens::{Event, Lens};
use crate::math::{R5, Ray, Rng, Vec3, concentric_disk};
use crate::scene::Scene;
use crate::spectrum::{ColorSystem, LAMBDA_MIN, LAMBDA_RANGE, cie_xyz};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Film {
    pub w: usize,
    pub h: usize,
    pub bins: usize,
    pub data: Vec<f32>,
}

impl Film {
    pub fn new(w: usize, h: usize, bins: usize) -> Film {
        Film { w, h, bins, data: vec![0.0; w * h * bins] }
    }

    pub fn bin_center(&self, b: usize) -> f64 {
        LAMBDA_MIN + LAMBDA_RANGE * (b as f64 + 0.5) / self.bins as f64
    }

    /// Converts to white-balanced linear sRGB.
    pub fn to_rgb(&self, cs: &ColorSystem) -> Vec<[f32; 3]> {
        let width = LAMBDA_RANGE / self.bins as f64;
        let cmf: Vec<[f64; 3]> = (0..self.bins)
            .map(|b| {
                let mut s = [0.0; 3];
                let n = 64;
                for i in 0..n {
                    let l = LAMBDA_MIN + width * (b as f64 + (i as f64 + 0.5) / n as f64);
                    let c = cie_xyz(l);
                    for k in 0..3 {
                        s[k] += c[k] / n as f64;
                    }
                }
                s
            })
            .collect();
        self.data
            .par_chunks(self.bins)
            .map(|px| {
                let mut xyz = [0.0; 3];
                for (v, c) in px.iter().zip(&cmf) {
                    for k in 0..3 {
                        xyz[k] += *v as f64 * c[k];
                    }
                }
                let rgb = cs.xyz_integral_to_rgb(xyz);
                [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]
            })
            .collect()
    }
}

pub struct Renderer<'a> {
    pub lens: &'a Lens,
    pub scene: &'a Scene,
    pub cs: &'a ColorSystem,
    pub w: usize,
    pub h: usize,
    pub bins: usize,
    /// Pixel pitch in mm.
    pub pitch: f64,
    /// Irradiance scale: 4N²/π · 2^EV, so a uniform radiance of 1 maps to ≈1.
    pub norm: f64,
    bounds: Vec<Option<[f64; 4]>>,
    r_max: f64,
}

const BOUNDS_N: usize = 64;

impl<'a> Renderer<'a> {
    pub fn new(lens: &'a Lens, scene: &'a Scene, cs: &'a ColorSystem, w: usize, h: usize, sensor_w: f64, bins: usize, norm: f64) -> Self {
        let pitch = sensor_w / w as f64;
        let r_max = 0.5 * (w as f64).hypot(h as f64) * pitch * 1.01;
        let mut r = Renderer { lens, scene, cs, w, h, bins, pitch, norm, bounds: Vec::new(), r_max };
        r.bounds = r.compute_pupil_bounds();
        r
    }

    fn rear_z(&self) -> f64 {
        self.lens.last_z()
    }

    /// For sensor points (r, 0) at several radii, the bounding box of points on
    /// the rear vertex plane through which rays make it through the lens
    /// (after pbrt's RealisticCamera). Uses a circular stop of the iris'
    /// largest radius so the box is valid for every rotation.
    fn compute_pupil_bounds(&self) -> Vec<Option<[f64; 4]>> {
        let mut lens = self.lens.clone();
        let iris = &self.lens.iris;
        lens.iris = crate::lens::Iris::new(0, 0.0, 0.0, iris.max_radius, true);
        let sd = lens.surfaces.last().unwrap().semi_diameter;
        let zr = self.rear_z();
        let g = 160;
        let cell = 2.0 * sd / g as f64;
        (0..BOUNDS_N)
            .into_par_iter()
            .map(|k| {
                let r = self.r_max * k as f64 / (BOUNDS_N - 1) as f64;
                let o = Vec3::new(r, 0.0, lens.sensor_z);
                let mut bb = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for l in [380.0, 480.0, 580.0, 680.0, 780.0] {
                    let m = lens.media(l);
                    for iy in 0..g {
                        for ix in 0..g {
                            let x = -sd + (ix as f64 + 0.5) * cell;
                            let y = -sd + (iy as f64 + 0.5) * cell;
                            let d = (Vec3::new(x, y, zr) - o).normalize();
                            if lens.trace_backward(Ray { o, d }, l, &m).is_some() {
                                bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
                            }
                        }
                    }
                }
                if bb[0] > bb[2] {
                    None
                } else {
                    Some([bb[0] - cell, bb[1] - cell, bb[2] + cell, bb[3] + cell])
                }
            })
            .collect()
    }

    fn bounds_at(&self, r: f64) -> Option<[f64; 4]> {
        let f = r / self.r_max * (BOUNDS_N - 1) as f64;
        let k0 = (f.floor() as usize).min(BOUNDS_N - 1);
        let k1 = (k0 + 1).min(BOUNDS_N - 1);
        match (self.bounds[k0], self.bounds[k1]) {
            (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
            (a, b) => a.or(b),
        }
    }

    /// Sensor position (lens coordinates) of the centre of output pixel (x, y)
    /// offset by (dx, dy). The image on the sensor is inverted, the output is not.
    #[inline]
    fn sensor_point(&self, x: f64, y: f64) -> (f64, f64) {
        let xo = (x - self.w as f64 / 2.0) * self.pitch;
        let yo = (self.h as f64 / 2.0 - y) * self.pitch;
        (-xo, -yo)
    }

    #[inline]
    fn pixel_of(&self, xs: f64, ys: f64) -> Option<usize> {
        let x = -xs / self.pitch + self.w as f64 / 2.0;
        let y = self.h as f64 / 2.0 + ys / self.pitch;
        if x < 0.0 || y < 0.0 || x >= self.w as f64 || y >= self.h as f64 {
            return None;
        }
        Some(y as usize * self.w + x as usize)
    }

    /// Backward pass: scene surfaces and background.
    /// `spp` is rounded up to a multiple of the number of wavelength bins.
    pub fn render_surfaces(&self, film: &mut Film, spp: u32, progress: bool) {
        let spp = spp.div_ceil(self.bins as u32) * self.bins as u32;
        let r5 = R5::new();
        let zr = self.rear_z();
        let zs = self.lens.sensor_z;
        let dz = zs - zr;
        let bins = self.bins;
        let scale = self.norm * LAMBDA_RANGE / spp as f64;
        let done = std::sync::atomic::AtomicUsize::new(0);
        let w = self.w;
        film.data.par_chunks_mut(w * bins).enumerate().for_each(|(py, row)| {
            let mut rng = Rng::new(0x5eed ^ py as u64, py as u64);
            let mut shifts = vec![[0.0; 5]; bins];
            for px in 0..w {
                // Each wavelength bin gets its own randomly shifted low-discrepancy
                // sample set, so every bin is integrated with QMC quality.
                for sh in shifts.iter_mut() {
                    *sh = [0; 5].map(|_| rng.next_f64());
                }
                let out = &mut row[px * bins..(px + 1) * bins];
                for i in 0..spp {
                    let bin = i as usize % bins;
                    let u = r5.sample((i as usize / bins) as u64, &shifts[bin]);
                    let (xs, ys) = self.sensor_point(px as f64 + u[0], py as f64 + u[1]);
                    let r = xs.hypot(ys);
                    let Some(bb) = self.bounds_at(r) else { continue };
                    let bx = bb[0] + (bb[2] - bb[0]) * u[2];
                    let by = bb[1] + (bb[3] - bb[1]) * u[3];
                    let (s, c) = if r > 0.0 { (ys / r, xs / r) } else { (0.0, 1.0) };
                    let o = Vec3::new(xs, ys, zs);
                    let target = Vec3::new(bx * c - by * s, bx * s + by * c, zr);
                    let d = (target - o).normalize();
                    let l = LAMBDA_MIN + LAMBDA_RANGE * (bin as f64 + u[4]) / bins as f64;
                    let m = self.lens.media(l);
                    let Some((ray, t)) = self.lens.trace_backward(Ray { o, d }, l, &m) else { continue };
                    let cos = -d.z;
                    let weight = (bb[2] - bb[0]) * (bb[3] - bb[1]) * cos * cos * cos * cos / (dz * dz);
                    let radiance = self.scene.radiance(self.cs, &ray, l);
                    out[bin] += (radiance * t * weight * scale) as f32;
                }
            }
            if progress {
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 64 == 0 || n == self.h {
                    eprint!("\r  surfaces: {:5.1}%", 100.0 * n as f64 / self.h as f64);
                }
            }
        });
        if progress {
            eprintln!();
        }
    }

    /// Forward pass for one light along one event path with `n` samples,
    /// sampling directions through a disk of radius `r_plane` on the plane `z_plane`.
    fn trace_light_path(&self, li: usize, path: &[Event], n: u64, z_plane: f64, r_plane: f64, seed: u64, film: &[AtomicU32]) {
        let light = &self.scene.lights[li];
        let n_e = (-light.center).normalize();
        let helper = if n_e.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        let t1 = n_e.cross(helper).normalize();
        let t2 = n_e.cross(t1);
        let a_e = std::f64::consts::PI * light.radius * light.radius;
        let a_p = std::f64::consts::PI * r_plane * r_plane;
        let zs = self.lens.sensor_z;
        let scale = self.norm * LAMBDA_RANGE / (n as f64 * self.pitch * self.pitch);
        let chunk = 1 << 15;
        let chunks = n.div_ceil(chunk);
        (0..chunks).into_par_iter().for_each(|ci| {
            let mut rng = Rng::new(seed, ci);
            let count = chunk.min(n - ci * chunk);
            for _ in 0..count {
                let ul = rng.next_f64();
                let l = LAMBDA_MIN + LAMBDA_RANGE * ul;
                let bin = ((ul * self.bins as f64) as usize).min(self.bins - 1);
                let (ex, ey) = concentric_disk(rng.next_f64(), rng.next_f64());
                let e = light.center + t1 * (ex * light.radius) + t2 * (ey * light.radius);
                let (qx, qy) = concentric_disk(rng.next_f64(), rng.next_f64());
                let q = Vec3::new(qx * r_plane, qy * r_plane, z_plane);
                let dv = q - e;
                let dist2 = dv.dot(dv);
                let d = dv * (1.0 / dist2.sqrt());
                if d.z <= 0.0 {
                    continue;
                }
                let cos_e = d.dot(n_e).max(0.0);
                let ray = Ray { o: e, d };
                let m = self.lens.media(l);
                let Some((out, t)) = self.lens.trace_path(ray, l, &m, path) else { continue };
                if out.d.z <= 0.0 {
                    continue;
                }
                let s = (zs - out.o.z) / out.d.z;
                let p = out.o + out.d * s;
                let Some(pix) = self.pixel_of(p.x, p.y) else { continue };
                if self.scene.occluded(&ray, -e.z / d.z) {
                    continue;
                }
                let flux = light.spectrum.eval(self.cs, l) * cos_e * d.z / dist2 * a_e * a_p * t;
                atomic_add(&film[pix * self.bins + bin], (flux * scale) as f32);
            }
        });
    }

    /// Ghost image of a point light along one two-reflection path, by tracing
    /// a regular grid of rays over the front element and rasterising every
    /// ray-tube quad with its flux spread over its footprint on the sensor.
    /// Noise-free, and caustics come out right (after Hullin et al. 2011).
    fn render_ghost(&self, li: usize, path: &[Event], grid: usize, bin: usize, film: &[AtomicU32]) {
        let light = &self.scene.lights[li];
        if self.scene.occluded(&Ray { o: light.center, d: (-light.center).normalize() }, light.center.length()) {
            return;
        }
        let r_front = self.lens.surfaces[0].semi_diameter;
        let cell = 2.0 * r_front / grid as f64;
        let intensity_area = std::f64::consts::PI * light.radius * light.radius; // intensity / radiance
        let zs = self.lens.sensor_z;
        let bin_width = LAMBDA_RANGE / self.bins as f64;
        let sub = 2;
        let refine = 8;
        let n1 = grid + 1;
        let mut verts: Vec<Option<(f64, f64, f64)>> = vec![None; n1 * n1];
        for k in 0..sub {
            let l = LAMBDA_MIN + bin_width * (bin as f64 + (k as f64 + 0.5) / sub as f64);
            let m = self.lens.media(l);
            let radiance = light.spectrum.eval(self.cs, l);
            // Sensor position (pixel units) and flux density on the entrance plane.
            let trace = |x: f64, y: f64| -> Option<(f64, f64, f64)> {
                let dv = Vec3::new(x, y, 0.0) - light.center;
                let dist2 = dv.dot(dv);
                let d = dv * (1.0 / dist2.sqrt());
                let (out, t) = self.lens.trace_path(Ray { o: light.center, d }, l, &m, path)?;
                if out.d.z <= 0.0 {
                    return None;
                }
                let p = out.o + out.d * ((zs - out.o.z) / out.d.z);
                let f = radiance * intensity_area * d.z / dist2 * t;
                Some((-p.x / self.pitch + self.w as f64 / 2.0, self.h as f64 / 2.0 + p.y / self.pitch, f))
            };
            for j in 0..n1 {
                for i in 0..n1 {
                    verts[j * n1 + i] = trace(-r_front + i as f64 * cell, -r_front + j as f64 * cell);
                }
            }
            let scale = self.norm * bin_width / sub as f64 / (self.pitch * self.pitch);
            let quad = |a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64), d: (f64, f64, f64), area: f64| {
                let flux = 0.25 * (a.2 + b.2 + c.2 + d.2) * scale * area;
                self.raster_triangle(a, b, c, 0.5 * flux, bin, film);
                self.raster_triangle(a, c, d, 0.5 * flux, bin, film);
            };
            let mut fine: Vec<Option<(f64, f64, f64)>> = vec![None; (refine + 1) * (refine + 1)];
            for j in 0..grid {
                for i in 0..grid {
                    let c4 = [verts[j * n1 + i], verts[j * n1 + i + 1], verts[(j + 1) * n1 + i + 1], verts[(j + 1) * n1 + i]];
                    let valid = c4.iter().filter(|v| v.is_some()).count();
                    if valid == 4 {
                        quad(c4[0].unwrap(), c4[1].unwrap(), c4[2].unwrap(), c4[3].unwrap(), cell * cell);
                    } else if valid > 0 {
                        // Partially clipped by an aperture: refine for a clean ghost edge.
                        let fc = cell / refine as f64;
                        let (x0, y0) = (-r_front + i as f64 * cell, -r_front + j as f64 * cell);
                        let r1 = refine + 1;
                        for fj in 0..r1 {
                            for fi in 0..r1 {
                                fine[fj * r1 + fi] = trace(x0 + fi as f64 * fc, y0 + fj as f64 * fc);
                            }
                        }
                        for fj in 0..refine {
                            for fi in 0..refine {
                                if let (Some(a), Some(b), Some(c), Some(d)) =
                                    (fine[fj * r1 + fi], fine[fj * r1 + fi + 1], fine[(fj + 1) * r1 + fi + 1], fine[(fj + 1) * r1 + fi])
                                {
                                    quad(a, b, c, d, fc * fc);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn raster_triangle(&self, a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64), flux: f64, bin: usize, film: &[AtomicU32]) {
        let area = 0.5 * ((b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1));
        let (w, h) = (self.w as f64, self.h as f64);
        if area.abs() < 1.0 {
            // Smaller than a pixel: bilinear splat at the centroid (avoids moiré in caustics).
            let x = (a.0 + b.0 + c.0) / 3.0 - 0.5;
            let y = (a.1 + b.1 + c.1) / 3.0 - 0.5;
            let (xi, yi) = (x.floor(), y.floor());
            let (tx, ty) = (x - xi, y - yi);
            for (dx, dy, wgt) in [(0.0, 0.0, (1.0 - tx) * (1.0 - ty)), (1.0, 0.0, tx * (1.0 - ty)), (0.0, 1.0, (1.0 - tx) * ty), (1.0, 1.0, tx * ty)] {
                let (px, py) = (xi + dx, yi + dy);
                if px >= 0.0 && py >= 0.0 && px < w && py < h {
                    atomic_add(&film[(py as usize * self.w + px as usize) * self.bins + bin], (flux * wgt) as f32);
                }
            }
            return;
        }
        let x0 = a.0.min(b.0).min(c.0).floor().max(0.0);
        let x1 = a.0.max(b.0).max(c.0).ceil().min(w);
        let y0 = a.1.min(b.1).min(c.1).floor().max(0.0);
        let y1 = a.1.max(b.1).max(c.1).ceil().min(h);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let v = (flux / area.abs()) as f32;
        let inv = 1.0 / (2.0 * area);
        for py in y0 as usize..y1 as usize {
            let y = py as f64 + 0.5;
            for px in x0 as usize..x1 as usize {
                let x = px as f64 + 0.5;
                let w0 = ((b.0 - x) * (c.1 - y) - (c.0 - x) * (b.1 - y)) * inv;
                let w1 = ((c.0 - x) * (a.1 - y) - (a.0 - x) * (c.1 - y)) * inv;
                let w2 = 1.0 - w0 - w1;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    atomic_add(&film[(py * self.w + px) * self.bins + bin], v);
                }
            }
        }
    }

    /// Forward pass: direct images of the lights and, optionally, ghosts.
    pub fn render_lights(&self, film: &mut Film, light_samples: u64, ghost_grid: usize, ghosts: bool, progress: bool) {
        if self.scene.lights.is_empty() {
            return;
        }
        let acc: Vec<AtomicU32> = (0..film.data.len()).map(|_| AtomicU32::new(0)).collect();
        let p = self.lens.paraxial_data(crate::glass::LAMBDA_D);
        let r_ep = 1.5 * self.lens.iris.max_radius / p.entrance_pupil_mag;
        let ghost_paths = if ghosts { self.lens.ghost_paths() } else { Vec::new() };
        let nl = self.scene.lights.len();
        for li in 0..nl {
            if progress {
                eprint!("\r  lights: {}/{}", li + 1, nl);
            }
            self.trace_light_path(li, &self.lens.forward_path, light_samples, p.entrance_pupil_z, r_ep, li as u64 * 1000, &acc);
            let jobs: Vec<(usize, usize)> = (0..ghost_paths.len()).flat_map(|g| (0..self.bins).map(move |b| (g, b))).collect();
            jobs.par_iter().for_each(|&(g, b)| self.render_ghost(li, &ghost_paths[g].2, ghost_grid, b, &acc));
        }
        if progress {
            eprintln!();
        }
        film.data.par_iter_mut().zip(acc.par_iter()).for_each(|(v, a)| *v += f32::from_bits(a.load(Ordering::Relaxed)));
    }
}

fn atomic_add(a: &AtomicU32, v: f32) {
    let mut cur = a.load(Ordering::Relaxed);
    loop {
        let new = (f32::from_bits(cur) + v).to_bits();
        match a.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(c) => cur = c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Light, LightSpectrum, Rect, Texture};

    fn lens() -> Lens {
        let mut l = Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap();
        l.set_aperture(5.6, 6, 0.0, 0.0).unwrap();
        l
    }

    fn rect(center: Vec3, size: f64, rgb: [f32; 3]) -> Rect {
        Rect {
            center,
            u: Vec3::new(1.0, 0.0, 0.0),
            v: Vec3::new(0.0, 1.0, 0.0),
            n: Vec3::new(0.0, 0.0, 1.0),
            w: size,
            h: size,
            texture: Texture::Color(rgb),
            brightness: 1.0,
        }
    }

    fn total(film: &Film) -> f64 {
        film.data.iter().map(|v| *v as f64).sum()
    }

    /// A uniform white scene images to ≈ lens transmission at the centre.
    #[test]
    fn uniform_radiance_normalisation() {
        let mut lens = lens();
        lens.sensor_z = lens.best_focus(None).unwrap();
        let cs = ColorSystem::new();
        let scene = Scene { rects: vec![rect(Vec3::new(0.0, 0.0, -1e5), 1e6, [1.0; 3])], lights: vec![], background: [0.0; 3] };
        let norm = 4.0 * 5.6 * 5.6 / std::f64::consts::PI;
        let r = Renderer::new(&lens, &scene, &cs, 60, 40, 36.0, 8, norm);
        let mut film = Film::new(60, 40, 8);
        r.render_surfaces(&mut film, 4096, false);
        let rgb = film.to_rgb(&cs);
        let c = rgb[20 * 60 + 30];
        // Fresnel transmission of an axial ray; the real entrance pupil is ~1.3%
        // smaller than the paraxial one at f/5.6.
        let m = lens.media(550.0);
        let (_, t) = lens.trace_forward(Ray { o: Vec3::new(0.0, 0.1, -1.0), d: Vec3::new(0.0, 0.0, 1.0) }, 550.0, &m).unwrap();
        let expect = t * 0.987;
        for k in 0..3 {
            assert!((c[k] as f64 - expect).abs() < 0.03, "centre {c:?}, expected {expect}");
        }
        // Natural + mechanical vignetting darkens the corners.
        assert!(rgb[0][1] < 0.8 * c[1]);
    }

    /// Light tracing (forward) and camera tracing (backward) agree on the flux
    /// arriving from a small source.
    #[test]
    fn forward_backward_consistency() {
        for (fstop, c) in [(5.6, Vec3::new(0.0, 0.0, -3000.0)), (5.6, Vec3::new(300.0, -200.0, -3000.0)), (3.5, Vec3::new(1050.0, 600.0, -3000.0)), (11.0, Vec3::new(-900.0, 500.0, -3000.0))] {
            let mut lens = lens();
            lens.set_aperture(fstop, 7, 10.0, 0.3).unwrap();
            lens.sensor_z = lens.best_focus(Some(3000.0)).unwrap() + 0.4;
            let cs = ColorSystem::new();
            let side = 200.0;
            let radius = side / std::f64::consts::PI.sqrt();
            let (w, h, bins) = (120, 80, 4);
            let back_scene = Scene { rects: vec![rect(c, side, [1.0; 3])], lights: vec![], background: [0.0; 3] };
            let r = Renderer::new(&lens, &back_scene, &cs, w, h, 36.0, bins, 1.0);
            let mut fb = Film::new(w, h, bins);
            r.render_surfaces(&mut fb, 512, false);
            let fwd_scene = Scene {
                rects: vec![],
                lights: vec![Light { center: c, radius, spectrum: LightSpectrum::Rgb { rgb: [1.0; 3], scale: 1.0 } }],
                background: [0.0; 3],
            };
            let r = Renderer::new(&lens, &fwd_scene, &cs, w, h, 36.0, bins, 1.0);
            let mut ff = Film::new(w, h, bins);
            r.render_lights(&mut ff, 4_000_000, 0, false, false);
            // The square faces the axis, the light disk faces the lens.
            let cos = c.z.abs() / c.length();
            let (a, b) = (total(&fb), total(&ff) * cos);
            println!("f/{fstop}: backward {a} forward {b}");
            assert!(a > 0.0 && (a - b).abs() / a < 0.01, "f/{fstop}: backward {a} forward {b}");
        }
    }

}

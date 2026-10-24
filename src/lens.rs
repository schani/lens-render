//! Lens prescription, paraxial optics and exact (real) ray tracing.
//!
//! Lens coordinates: millimetres, optical axis along +z, light travels from
//! object space (z < 0) towards the sensor. The front vertex is at z = 0.

use crate::glass::{Glass, LAMBDA_D};
use crate::math::{Ray, Vec3};
use anyhow::{Context, Result, bail};
use std::f64::consts::PI;

pub const MAX_SURFACES: usize = 32;

#[derive(Clone, Debug)]
pub struct Surface {
    /// Radius of curvature; 0 means flat. Positive: centre of curvature towards the sensor.
    pub radius: f64,
    /// Distance to the next surface (or to the nominal image plane for the last one).
    pub thickness: f64,
    /// Medium after the surface; `None` is air.
    pub material: Option<Glass>,
    pub semi_diameter: f64,
    pub is_stop: bool,
    /// Vertex position on the axis.
    pub z: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coating {
    None,
    /// Single quarter-wave layer (MgF₂-like) of index `n` tuned to `lambda0` nm.
    Single { n: f64, lambda0: f64 },
}

/// Aperture iris made of straight or curved blades.
#[derive(Clone, Debug)]
pub struct Iris {
    pub blades: u32,
    pub rotation: f64,
    /// 0 = straight blades (polygon), 1 = blades follow the circumscribed circle.
    pub curvature: f64,
    pub circular: bool,
    /// Nominal radius: the opening has the area of a circle of this radius.
    pub radius: f64,
    /// Apothem of the polygon, scaled so the opening has the area of a circle of the nominal radius.
    apothem: f64,
    /// Largest distance of the opening edge from the axis.
    pub max_radius: f64,
}

impl Iris {
    pub fn new(blades: u32, rotation_deg: f64, curvature: f64, radius: f64, circular: bool) -> Iris {
        let mut iris = Iris {
            blades,
            rotation: rotation_deg.to_radians(),
            curvature: curvature.clamp(0.0, 1.0),
            circular: circular || blades < 3,
            radius,
            apothem: 1.0,
            max_radius: radius,
        };
        if !iris.circular {
            // Scale so the area matches π r².
            let n = 4096;
            let area: f64 = (0..n)
                .map(|i| {
                    let r = iris.edge_radius_unit((i as f64 + 0.5) / n as f64 * 2.0 * PI);
                    0.5 * r * r * 2.0 * PI / n as f64
                })
                .sum();
            iris.apothem = radius * (PI / area).sqrt();
            let half = PI / blades as f64;
            iris.max_radius = iris.apothem / half.cos();
        }
        iris
    }

    /// Edge radius at polar angle `theta` for apothem 1.
    fn edge_radius_unit(&self, theta: f64) -> f64 {
        let sector = 2.0 * PI / self.blades as f64;
        let t = theta - self.rotation;
        let local = t - sector * (t / sector).round();
        let poly = 1.0 / local.cos();
        let circ = 1.0 / (PI / self.blades as f64).cos();
        poly + self.curvature * (circ - poly)
    }

    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let r2 = x * x + y * y;
        if self.circular {
            return r2 <= self.max_radius * self.max_radius;
        }
        let a = self.apothem;
        if r2 <= a * a {
            return true;
        }
        if r2 > self.max_radius * self.max_radius {
            return false;
        }
        let r = self.apothem * self.edge_radius_unit(y.atan2(x));
        r2 <= r * r
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub surface: u16,
    pub forward: bool,
    pub reflect: bool,
}

/// Refractive indices of all media for one wavelength. Medium 0 is object
/// space, medium i+1 follows surface i.
pub struct Media {
    pub n: [f64; MAX_SURFACES + 1],
}

#[derive(Clone, Debug)]
pub struct Lens {
    pub name: String,
    pub surfaces: Vec<Surface>,
    pub coating: Coating,
    pub iris: Iris,
    pub stop_index: usize,
    /// Sensor plane position.
    pub sensor_z: f64,
    pub forward_path: Vec<Event>,
    pub backward_path: Vec<Event>,
}

pub struct Paraxial {
    pub efl: f64,
    pub bfd: f64,
    pub entrance_pupil_z: f64,
    pub entrance_pupil_mag: f64,
    pub exit_pupil_z: f64,
    pub exit_pupil_mag: f64,
}

impl Lens {
    /// Parses a prescription. Rows: `radius thickness material semi-diameter`;
    /// radius `inf`/`0` is flat, `stop` marks the aperture stop, material `air` or a glass.
    pub fn parse(text: &str) -> Result<Lens> {
        let mut name = String::from("unnamed lens");
        let mut surfaces = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("@name") {
                name = rest.trim().to_string();
                continue;
            }
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 4 {
                bail!("line {}: expected 4 columns", lineno + 1);
            }
            let ctx = || format!("line {}", lineno + 1);
            let is_stop = cols[0].eq_ignore_ascii_case("stop");
            let radius = if is_stop || cols[0].eq_ignore_ascii_case("inf") { 0.0 } else { cols[0].parse().with_context(ctx)? };
            let thickness: f64 = cols[1].parse().with_context(ctx)?;
            let material = if cols[2].eq_ignore_ascii_case("air") { None } else { Some(Glass::parse(cols[2]).with_context(ctx)?) };
            let semi_diameter: f64 = cols[3].parse().with_context(ctx)?;
            if is_stop && material.is_some() {
                bail!("line {}: the stop must be in air", lineno + 1);
            }
            surfaces.push(Surface { radius, thickness, material, semi_diameter, is_stop, z: 0.0 });
        }
        if surfaces.is_empty() || surfaces.len() > MAX_SURFACES {
            bail!("lens must have 1..={MAX_SURFACES} surfaces");
        }
        let stop_index = surfaces.iter().position(|s| s.is_stop).context("lens has no stop")?;
        let mut z = 0.0;
        for s in surfaces.iter_mut() {
            s.z = z;
            z += s.thickness;
        }
        let n = surfaces.len();
        let forward_path = (0..n).map(|i| Event { surface: i as u16, forward: true, reflect: false }).collect();
        let backward_path = (0..n).rev().map(|i| Event { surface: i as u16, forward: false, reflect: false }).collect();
        let stop_r = surfaces[stop_index].semi_diameter;
        Ok(Lens {
            name,
            surfaces,
            coating: Coating::Single { n: 1.38, lambda0: 550.0 },
            iris: Iris::new(0, 0.0, 0.0, stop_r, true),
            stop_index,
            sensor_z: z,
            forward_path,
            backward_path,
        })
    }

    pub fn load(path: &str) -> Result<Lens> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading lens {path}"))?;
        Lens::parse(&text)
    }

    pub fn media(&self, l: f64) -> Media {
        let mut n = [1.0; MAX_SURFACES + 1];
        for (i, s) in self.surfaces.iter().enumerate() {
            n[i + 1] = s.material.as_ref().map_or(1.0, |g| g.index(l));
        }
        Media { n }
    }

    pub fn last_z(&self) -> f64 {
        self.surfaces.last().unwrap().z
    }

    // ---------------------------------------------------------------- paraxial

    /// Paraxial y-nu trace from the vertex plane of surface `start` to the vertex
    /// plane of surface `end` (or, for `end == len`, just after the last surface).
    fn paraxial(&self, mut y: f64, mut u: f64, start: usize, end: usize, l: f64) -> (f64, f64) {
        let m = self.media(l);
        for i in start..end {
            let s = &self.surfaces[i];
            let (n1, n2) = (m.n[i], m.n[i + 1]);
            let c = if s.radius == 0.0 { 0.0 } else { 1.0 / s.radius };
            u = (n1 * u - y * (n2 - n1) * c) / n2;
            if i + 1 < end {
                y += u * s.thickness;
            }
        }
        (y, u)
    }

    pub fn paraxial_data(&self, l: f64) -> Paraxial {
        let n = self.surfaces.len();
        let (y, u) = self.paraxial(1.0, 0.0, 0, n, l);
        let efl = -1.0 / u;
        let bfd = -y / u;
        let (ya, _) = self.paraxial_to_stop(1.0, 0.0, l);
        let (yb, _) = self.paraxial_to_stop(0.0, 1.0, l);
        let s = self.stop_index;
        let (ya2, ua2) = self.paraxial_from_stop(1.0, 0.0, l);
        let (yb2, ub2) = self.paraxial_from_stop(0.0, 1.0, l);
        let dz = -yb2 / ub2;
        let _ = s;
        Paraxial {
            efl,
            bfd,
            entrance_pupil_z: yb / ya,
            entrance_pupil_mag: ya,
            exit_pupil_z: self.last_z() + dz,
            exit_pupil_mag: ya2 + ua2 * dz,
        }
    }

    /// From the front vertex plane to the stop plane.
    fn paraxial_to_stop(&self, y: f64, u: f64, l: f64) -> (f64, f64) {
        let s = self.stop_index;
        let (mut y, u) = self.paraxial(y, u, 0, s, l);
        if s > 0 {
            y += u * self.surfaces[s - 1].thickness;
        }
        (y, u)
    }

    /// From the stop plane to just after the last surface (heights at the last vertex).
    fn paraxial_from_stop(&self, y: f64, u: f64, l: f64) -> (f64, f64) {
        let s = self.stop_index;
        let y = y + u * self.surfaces[s].thickness;
        let n = self.surfaces.len();
        if s + 1 >= n {
            return (y, u);
        }
        self.paraxial(y, u, s + 1, n, l)
    }

    pub fn focal_length(&self) -> f64 {
        self.paraxial_data(LAMBDA_D).efl
    }

    /// Stop radius giving infinity f-number `fstop`.
    pub fn stop_radius_for_fstop(&self, fstop: f64) -> f64 {
        let p = self.paraxial_data(LAMBDA_D);
        p.entrance_pupil_mag * p.efl / (2.0 * fstop)
    }

    /// Widest f-number allowed by the stop's semi-diameter.
    pub fn max_aperture(&self) -> f64 {
        let p = self.paraxial_data(LAMBDA_D);
        p.entrance_pupil_mag * p.efl / (2.0 * self.surfaces[self.stop_index].semi_diameter)
    }

    pub fn set_aperture(&mut self, fstop: f64, blades: u32, rotation_deg: f64, curvature: f64) -> Result<()> {
        let max = self.max_aperture();
        if fstop < max - 1e-6 {
            bail!("f/{fstop} is wider than the lens' maximum aperture f/{max:.2}");
        }
        let r = self.stop_radius_for_fstop(fstop);
        // Wide open the blades are fully retracted and the opening is round.
        let circular = fstop <= max * 1.001;
        self.iris = Iris::new(blades, rotation_deg, curvature, r, circular);
        Ok(())
    }

    // --------------------------------------------------------------- real rays

    /// Intersects surface `i`; returns the hit point and the surface normal.
    #[inline]
    fn intersect(&self, i: usize, ray: &Ray) -> Option<(Vec3, Vec3)> {
        let s = &self.surfaces[i];
        if s.radius == 0.0 {
            if ray.d.z.abs() < 1e-12 {
                return None;
            }
            let t = (s.z - ray.o.z) / ray.d.z;
            if t < 1e-9 {
                return None;
            }
            return Some((ray.o + ray.d * t, Vec3::new(0.0, 0.0, -1.0)));
        }
        let r = s.radius;
        let c = Vec3::new(0.0, 0.0, s.z + r);
        let oc = ray.o - c;
        let b = oc.dot(ray.d);
        let cc = oc.dot(oc) - r * r;
        let disc = b * b - cc;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        for t in [-b - sq, -b + sq] {
            if t > 1e-9 {
                let p = ray.o + ray.d * t;
                // Accept only the cap on the vertex side of the sphere.
                if (p.z - c.z) * r < 0.0 {
                    return Some((p, (p - c) * (1.0 / r)));
                }
            }
        }
        None
    }

    /// Fresnel reflectance (unpolarised) at an interface from index `n1` to `n2`.
    #[inline]
    pub fn reflectance(&self, n1: f64, n2: f64, cos1: f64, l: f64) -> f64 {
        let sin1_n = n1 * (1.0 - cos1 * cos1).max(0.0).sqrt(); // Snell invariant
        let cosm = |n: f64| -> Option<f64> {
            let s = sin1_n / n;
            if s >= 1.0 { None } else { Some((1.0 - s * s).sqrt()) }
        };
        let air_interface = n1 == 1.0 || n2 == 1.0;
        match self.coating {
            Coating::Single { n: nc, lambda0 } if air_interface => {
                let (na, ng) = if n1 == 1.0 { (n1, n2) } else { (n2, n1) };
                let (Some(ca), Some(cc), Some(cg)) = (cosm(na), cosm(nc), cosm(ng)) else {
                    return 1.0;
                };
                let d = lambda0 / (4.0 * nc);
                let phi = 4.0 * PI * nc * d * cc / l;
                let cphi = phi.cos();
                let film = |r01: f64, r12: f64| {
                    let num = r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi;
                    let den = 1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cphi;
                    num / den
                };
                let rs = film((na * ca - nc * cc) / (na * ca + nc * cc), (nc * cc - ng * cg) / (nc * cc + ng * cg));
                let rp = film((nc * ca - na * cc) / (nc * ca + na * cc), (ng * cc - nc * cg) / (ng * cc + nc * cg));
                0.5 * (rs + rp)
            }
            _ => {
                let Some(cos2) = cosm(n2) else { return 1.0 };
                let rs = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2);
                let rp = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2);
                0.5 * (rs * rs + rp * rp)
            }
        }
    }

    /// Traces a ray along a sequence of surface events. Returns the outgoing
    /// ray and its throughput (Fresnel transmission / reflection products).
    #[inline]
    pub fn trace_path(&self, mut ray: Ray, l: f64, m: &Media, path: &[Event]) -> Option<(Ray, f64)> {
        let mut throughput = 1.0;
        for ev in path {
            let i = ev.surface as usize;
            let s = &self.surfaces[i];
            let (p, mut nrm) = self.intersect(i, &ray)?;
            let r2 = p.x * p.x + p.y * p.y;
            if s.is_stop {
                if !self.iris.contains(p.x, p.y) || r2 > s.semi_diameter * s.semi_diameter {
                    return None;
                }
                ray.o = p;
                continue;
            }
            if r2 > s.semi_diameter * s.semi_diameter {
                return None;
            }
            let (n1, n2) = if ev.forward { (m.n[i], m.n[i + 1]) } else { (m.n[i + 1], m.n[i]) };
            let mut cosi = -nrm.dot(ray.d);
            if cosi < 0.0 {
                nrm = -nrm;
                cosi = -cosi;
            }
            let refl = self.reflectance(n1, n2, cosi, l);
            if ev.reflect {
                throughput *= refl;
                ray = Ray { o: p, d: ray.d + nrm * (2.0 * cosi) };
            } else {
                let eta = n1 / n2;
                let k = 1.0 - eta * eta * (1.0 - cosi * cosi);
                if k < 0.0 {
                    return None; // total internal reflection
                }
                throughput *= 1.0 - refl;
                ray = Ray { o: p, d: (ray.d * eta + nrm * (eta * cosi - k.sqrt())).normalize() };
            }
        }
        Some((ray, throughput))
    }

    pub fn trace_forward(&self, ray: Ray, l: f64, m: &Media) -> Option<(Ray, f64)> {
        self.trace_path(ray, l, m, &self.forward_path)
    }

    pub fn trace_backward(&self, ray: Ray, l: f64, m: &Media) -> Option<(Ray, f64)> {
        self.trace_path(ray, l, m, &self.backward_path)
    }

    /// Event sequences for all two-reflection ghost paths: the light reflects
    /// first off surface `a` (travelling backwards) and then off surface `b < a`.
    pub fn ghost_paths(&self) -> Vec<(usize, usize, Vec<Event>)> {
        let n = self.surfaces.len();
        let mut out = Vec::new();
        for a in 0..n {
            for b in 0..a {
                if self.surfaces[a].is_stop || self.surfaces[b].is_stop {
                    continue;
                }
                let mut p = Vec::new();
                for i in 0..a {
                    p.push(Event { surface: i as u16, forward: true, reflect: false });
                }
                p.push(Event { surface: a as u16, forward: true, reflect: true });
                for i in (b + 1..a).rev() {
                    p.push(Event { surface: i as u16, forward: false, reflect: false });
                }
                p.push(Event { surface: b as u16, forward: false, reflect: true });
                for i in b + 1..n {
                    p.push(Event { surface: i as u16, forward: true, reflect: false });
                }
                out.push((a, b, p));
            }
        }
        out
    }

    /// Ghost paths reversed for tracing from the sensor into the scene.
    /// Refractions flip direction; reflections keep the incident side.
    pub fn ghost_paths_backward(&self) -> Vec<Vec<Event>> {
        self.ghost_paths()
            .into_iter()
            .map(|(_, _, p)| p.iter().rev().map(|e| Event { forward: if e.reflect { e.forward } else { !e.forward }, ..*e }).collect())
            .collect()
    }

    /// Rays from an on-axis object point at `distance` mm in front of the front
    /// vertex (infinite if `None`), sampled on a grid over the front element.
    /// Returns the rays leaving the last surface with their luminance weight.
    pub fn axial_bundle(&self, distance: Option<f64>, grid: usize, wavelengths: &[f64]) -> Vec<(Ray, f64)> {
        let sd = self.surfaces[0].semi_diameter;
        let mut out = Vec::new();
        for &l in wavelengths {
            let m = self.media(l);
            let w = crate::spectrum::cie_xyz(l)[1];
            for iy in 0..grid {
                for ix in 0..grid {
                    let x = ((ix as f64 + 0.5) / grid as f64 * 2.0 - 1.0) * sd;
                    let y = ((iy as f64 + 0.5) / grid as f64 * 2.0 - 1.0) * sd;
                    if x * x + y * y > sd * sd {
                        continue;
                    }
                    let ray = match distance {
                        None => Ray { o: Vec3::new(x, y, -1.0), d: Vec3::new(0.0, 0.0, 1.0) },
                        Some(dist) => {
                            let o = Vec3::new(0.0, 0.0, -dist);
                            Ray { o, d: (Vec3::new(x, y, 0.0) - o).normalize() }
                        }
                    };
                    if let Some((r, t)) = self.trace_forward(ray, l, &m) {
                        out.push((r, w * t));
                    }
                }
            }
        }
        out
    }

    /// Sensor position minimising the luminance-weighted RMS blur of an
    /// on-axis point (what contrast autofocus converges to, approximately).
    pub fn best_focus(&self, distance: Option<f64>) -> Result<f64> {
        let wls: Vec<f64> = (0..13).map(|i| 420.0 + 20.0 * i as f64).collect();
        let rays = self.axial_bundle(distance, 96, &wls);
        if rays.is_empty() {
            bail!("no rays pass the lens");
        }
        let (mut num, mut den) = (0.0, 0.0);
        for (r, w) in &rays {
            let a = r.d.x / r.d.z;
            let b = r.d.y / r.d.z;
            num += w * ((r.o.z * a - r.o.x) * a + (r.o.z * b - r.o.y) * b);
            den += w * (a * a + b * b);
        }
        Ok(num / den)
    }

    /// Image-space working f-number for the current iris and sensor position.
    pub fn working_fnumber(&self) -> f64 {
        let p = self.paraxial_data(LAMBDA_D);
        (self.sensor_z - p.exit_pupil_z) / (2.0 * p.exit_pupil_mag.abs() * self.iris.radius)
    }

    /// Paraxial focus for wavelength `l` and object at `distance`.
    pub fn paraxial_focus(&self, distance: Option<f64>, l: f64) -> f64 {
        let n = self.surfaces.len();
        let (y0, u0) = match distance {
            None => (1.0, 0.0),
            Some(d) => (1e-3 * d, 1e-3),
        };
        let (y, u) = self.paraxial(y0, u0, 0, n, l);
        self.last_z() - y / u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIPLET: &str = include_str!("../lenses/cooke-triplet.lens");

    #[test]
    fn triplet_paraxial() {
        let lens = Lens::parse(TRIPLET).unwrap();
        let f = lens.focal_length();
        assert!((f - 50.0).abs() < 0.1, "efl {f}");
    }

    #[test]
    fn real_ray_matches_paraxial() {
        let lens = Lens::parse(TRIPLET).unwrap();
        let l = LAMBDA_D;
        let m = lens.media(l);
        let h = 0.01;
        let (r, _) = lens.trace_forward(Ray { o: Vec3::new(0.0, h, -1.0), d: Vec3::new(0.0, 0.0, 1.0) }, l, &m).unwrap();
        let z = r.o.z - r.o.y / r.d.y * r.d.z;
        let zp = lens.paraxial_focus(None, l);
        assert!((z - zp).abs() < 1e-3, "{z} vs {zp}");
    }

    #[test]
    fn iris_area() {
        let iris = Iris::new(6, 0.0, 0.0, 1.0, false);
        let n = 1000;
        let mut inside = 0;
        for i in 0..n {
            for j in 0..n {
                let x = (i as f64 + 0.5) / n as f64 * 3.0 - 1.5;
                let y = (j as f64 + 0.5) / n as f64 * 3.0 - 1.5;
                if iris.contains(x, y) {
                    inside += 1;
                }
            }
        }
        let area = inside as f64 * 9.0 / (n * n) as f64;
        assert!((area - PI).abs() < 0.01, "{area}");
    }

    #[test]
    fn uncoated_reflectance() {
        let mut lens = Lens::parse(TRIPLET).unwrap();
        lens.coating = Coating::None;
        let r = lens.reflectance(1.0, 1.5, 1.0, 550.0);
        assert!((r - 0.04).abs() < 1e-9);
        let mut coated = lens.clone();
        coated.coating = Coating::Single { n: 1.38, lambda0: 550.0 };
        assert!(coated.reflectance(1.0, 1.5, 1.0, 550.0) < 0.015);
        assert!(coated.reflectance(1.5, 1.0, 1.0, 550.0) < 0.015);
    }
}

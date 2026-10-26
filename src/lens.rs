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
    /// Lateral misalignment (manufacturing error) of the surface.
    pub decenter: (f64, f64),
    /// Conic constant k (0 = sphere).
    pub conic: f64,
    /// Even aspheric coefficients A4, A6, A8, … of z(r) = c r² / (1 + √(1 − (1+k) c² r²)) + Σ A₂ᵢ r²ⁱ.
    pub aspheric: Vec<f64>,
}

impl Surface {
    pub fn curvature(&self) -> f64 {
        if self.radius == 0.0 { 0.0 } else { 1.0 / self.radius }
    }

    pub fn is_aspheric(&self) -> bool {
        self.conic != 0.0 || self.aspheric.iter().any(|a| *a != 0.0)
    }

    /// Sag as a function of u = r²; None outside the conic's domain.
    pub fn sag_u(&self, u: f64) -> Option<f64> {
        let c = self.curvature();
        let q = 1.0 - (1.0 + self.conic) * c * c * u;
        if q < 0.0 {
            return None;
        }
        let mut z = c * u / (1.0 + q.sqrt());
        let mut un = u;
        for a in &self.aspheric {
            un *= u;
            z += a * un;
        }
        Some(z)
    }

    /// d(sag)/d(r²).
    pub fn dsag_du(&self, u: f64) -> f64 {
        let c = self.curvature();
        let q = (1.0 - (1.0 + self.conic) * c * c * u).max(1e-12);
        let mut d = c / (2.0 * q.sqrt());
        let mut un = 1.0;
        for (i, a) in self.aspheric.iter().enumerate() {
            un *= u;
            d += (i + 2) as f64 * a * un;
        }
        d
    }

    /// Sag at radius `r` (clamped to the domain, for drawing).
    pub fn sag(&self, r: f64) -> f64 {
        let mut u = r * r;
        loop {
            if let Some(z) = self.sag_u(u) {
                return z;
            }
            u *= 0.99;
        }
    }
}

/// An air gap that changes with focus distance (floating elements), linear
/// in object vergence between infinity and the close-focus distance.
#[derive(Clone, Debug)]
pub struct FloatingGap {
    pub surface: usize,
    pub at_infinity: f64,
    pub at_close: f64,
    pub close_distance: f64,
}

/// A zoom lens: the focal lengths of the prescription's zoom positions and
/// the air gaps that change between them.
#[derive(Clone, Debug)]
pub struct Zoom {
    pub focal: Vec<f64>,
    /// (row, gap at each zoom position).
    pub gaps: Vec<(usize, Vec<f64>)>,
    /// Widest f-number at each zoom position, if the stop opening follows it.
    pub fnumber: Option<Vec<f64>>,
    /// Current setting, in the focal lengths of `focal`.
    pub at: f64,
}

impl Zoom {
    /// Gap `values` at the current setting: a cubic Hermite curve in log focal
    /// length with the slopes of the parabola through each position and its
    /// neighbours (exactly that parabola for three positions).
    fn gap(&self, values: &[f64]) -> f64 {
        let x: Vec<f64> = self.focal.iter().map(|f| f.ln()).collect();
        let n = x.len();
        if n == 1 {
            return values[0];
        }
        let slope = |i: usize| {
            if n == 2 {
                return (values[1] - values[0]) / (x[1] - x[0]);
            }
            let j = i.clamp(1, n - 2);
            let (x0, x1, x2) = (x[j - 1], x[j], x[j + 1]);
            let (y0, y1, y2) = (values[j - 1], values[j], values[j + 1]);
            let t = x[i];
            y0 * (2.0 * t - x1 - x2) / ((x0 - x1) * (x0 - x2)) + y1 * (2.0 * t - x0 - x2) / ((x1 - x0) * (x1 - x2)) + y2 * (2.0 * t - x0 - x1) / ((x2 - x0) * (x2 - x1))
        };
        let t = self.at.ln().clamp(x[0], x[n - 1]);
        let i = (0..n - 1).find(|&i| t <= x[i + 1]).unwrap_or(n - 2);
        let h = x[i + 1] - x[i];
        let s = (t - x[i]) / h;
        let (h00, h10, h01, h11) = (2.0 * s.powi(3) - 3.0 * s * s + 1.0, s.powi(3) - 2.0 * s * s + s, 3.0 * s * s - 2.0 * s.powi(3), s.powi(3) - s * s);
        (h00 * values[i] + h10 * h * slope(i) + h01 * values[i + 1] + h11 * h * slope(i + 1)).max(0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coating {
    None,
    /// Single quarter-wave layer (MgF₂-like) of index `n` tuned to `lambda0` nm.
    Single { n: f64, lambda0: f64 },
    /// Broadband quarter–half–quarter stack tuned to `lambda0` nm, from the air
    /// side: λ/4 of `n_low`, λ/2 of `n_high`, λ/4 of n_low·√n_glass(λ0), the
    /// glass-specific index that cancels the reflection at λ0.
    Multi { n_low: f64, n_high: f64, lambda0: f64 },
}

impl Coating {
    pub const SINGLE: Coating = Coating::Single { n: 1.38, lambda0: 550.0 };
    /// MgF₂ over ZrO₂-like high index.
    pub const MULTI: Coating = Coating::Multi { n_low: 1.38, n_high: 2.1, lambda0: 550.0 };

    pub fn parse(name: &str) -> Result<Coating> {
        Ok(match name {
            "none" => Coating::None,
            "single" => Coating::SINGLE,
            "multi" => Coating::MULTI,
            _ => bail!("unknown coating {name} (none, single or multi)"),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Coating::None => "none",
            Coating::Single { .. } => "single",
            Coating::Multi { .. } => "multi",
        }
    }

    /// Index of the multi-layer's inner layer on glass of index `ng0` at λ0.
    fn tailored_index(n_low: f64, ng0: f64) -> f64 {
        (n_low * ng0.sqrt()).clamp(1.45, 2.0)
    }
}

/// Unpolarised reflectance between air and glass of index `ng` through thin
/// `layers` (index, optical thickness in waves at `lambda0`) listed from the
/// air side; `s` is the Snell invariant n·sin θ. Characteristic-matrix method:
/// for lossless layers each matrix is [[a, ib], [ic, d]] with a, b, c, d real.
fn film_stack(layers: &[(f64, f64)], ng: f64, s: f64, lambda0: f64, l: f64) -> f64 {
    let cosm = |n: f64| -> Option<f64> {
        let x = s / n;
        if x >= 1.0 { None } else { Some((1.0 - x * x).sqrt()) }
    };
    let Some(cg) = cosm(ng) else { return 1.0 };
    let Some(ca) = cosm(1.0) else { return 1.0 };
    let mut r = 0.0;
    for p in [false, true] {
        // Tilted admittances: n cos θ (s) and n / cos θ (p).
        let eta = |n: f64, c: f64| if p { n / c } else { n * c };
        let (mut a, mut b, mut c, mut d) = (1.0, 0.0, 0.0, 1.0);
        for &(n, waves) in layers {
            let Some(cl) = cosm(n) else { return 1.0 };
            let (e, delta) = (eta(n, cl), 2.0 * PI * waves * lambda0 * cl / l);
            let (cd, sd) = (delta.cos(), delta.sin());
            (a, b, c, d) = (a * cd - b * e * sd, a * sd / e + b * cd, c * cd + d * e * sd, d * cd - c * sd / e);
        }
        let (e0, es) = (eta(1.0, ca), eta(ng, cg));
        r += ((e0 * a - es * d).powi(2) + (e0 * es * b - c).powi(2)) / ((e0 * a + es * d).powi(2) + (e0 * es * b + c).powi(2));
    }
    0.5 * r
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

    pub fn apothem(&self) -> f64 {
        self.apothem
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
    /// The lens' iris design: blade count (0 = round) and blade curvature.
    pub blades: u32,
    pub blade_curvature: f64,
    pub stop_index: usize,
    /// Sensor plane position.
    pub sensor_z: f64,
    /// Specular reflectance of the sensor (cover glass, filters, microlenses),
    /// the same at all wavelengths and angles; > 0 adds the sensor ghosts.
    pub sensor_reflectance: f64,
    /// Half width and height of the sensor (mm): its reflecting area.
    pub sensor_half: (f64, f64),
    pub forward_path: Vec<Event>,
    pub backward_path: Vec<Event>,
    pub floating: Vec<FloatingGap>,
    pub zoom: Option<Zoom>,
    /// Dichroic annulus at the stop (experimental).
    pub spectral_stop: Option<SpectralStop>,
}

/// A dichroic filter in the stop: inside `inner` × the stop's semi-diameter
/// all wavelengths pass; in the annulus outside it only the band `lo`–`hi`
/// nm, with smooth edges `SpectralStop::EDGE` nm wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralStop {
    pub inner: f64,
    pub lo: f64,
    pub hi: f64,
}

impl SpectralStop {
    pub const EDGE: f64 = 10.0;

    /// Transmission of the annulus at wavelength `l` nm.
    pub fn band(&self, l: f64) -> f64 {
        // Cubic smooth step over ±EDGE/2, as WGSL's smoothstep.
        let step = |d: f64| {
            let t = (d / Self::EDGE + 0.5).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        step(l - self.lo) * step(self.hi - l)
    }

    /// Transmission at squared radius `r2` on a stop of semi-diameter `sd`.
    #[inline]
    pub fn transmission(&self, r2: f64, sd: f64, l: f64) -> f64 {
        let a = self.inner * sd;
        if r2 <= a * a { 1.0 } else { self.band(l) }
    }

    /// Luminous (CIE ȳ, equal-energy light) transmission of an evenly lit
    /// iris whose radius is `open` × the stop's semi-diameter.
    pub fn luminous(&self, open: f64) -> f64 {
        let a2 = (self.inner / open).min(1.0).powi(2);
        let (mut num, mut den) = (0.0, 0.0);
        for l in (380..=780).step_by(2) {
            let y = crate::spectrum::cie_xyz(l as f64)[1];
            num += y * (a2 + (1.0 - a2) * self.band(l as f64));
            den += y;
        }
        num / den
    }
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
        let mut floating = Vec::new();
        let (mut zoom_focal, mut zoom_gaps, mut zoom_fnumber) = (None, Vec::new(), None);
        let (mut blades, mut blade_curvature) = (6, 0.0);
        let mut coating = Coating::SINGLE;
        let mut spectral_stop = None;
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("@name") {
                name = rest.trim().to_string();
                continue;
            }
            if let Some(rest) = line.strip_prefix("@iris") {
                // @iris <blades> [curvature]
                let v: Vec<f64> = rest.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>().with_context(|| format!("line {}", lineno + 1))?;
                if v.is_empty() || v.len() > 2 {
                    bail!("line {}: @iris <blades> [curvature]", lineno + 1);
                }
                blades = v[0] as u32;
                blade_curvature = v.get(1).copied().unwrap_or(0.0);
                continue;
            }
            if let Some(rest) = line.strip_prefix("@coating") {
                coating = Coating::parse(rest.trim()).with_context(|| format!("line {}", lineno + 1))?;
                continue;
            }
            if let Some(rest) = line.strip_prefix("@float") {
                // @float <row> <gap at infinity> <gap at close focus> <close distance in m>
                let v: Vec<f64> = rest.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>().with_context(|| format!("line {}", lineno + 1))?;
                if v.len() != 4 || v[0] < 1.0 {
                    bail!("line {}: @float <row> <gap at infinity> <gap at close focus> <close distance m>", lineno + 1);
                }
                floating.push(FloatingGap { surface: v[0] as usize - 1, at_infinity: v[1], at_close: v[2], close_distance: v[3] * 1000.0 });
                continue;
            }
            if let Some(rest) = line.strip_prefix("@zoom-gap") {
                // @zoom-gap <row> <gap at each zoom position>
                let v: Vec<f64> = rest.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>().with_context(|| format!("line {}", lineno + 1))?;
                if v.len() < 2 || v[0] < 1.0 {
                    bail!("line {}: @zoom-gap <row> <gap at each zoom position>", lineno + 1);
                }
                zoom_gaps.push((v[0] as usize - 1, v[1..].to_vec()));
                continue;
            }
            if let Some(rest) = line.strip_prefix("@zoom-fnumber") {
                // @zoom-fnumber <widest f-number at each zoom position>
                let v: Vec<f64> = rest.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>().with_context(|| format!("line {}", lineno + 1))?;
                zoom_fnumber = Some(v);
                continue;
            }
            if let Some(rest) = line.strip_prefix("@zoom") {
                // @zoom <focal length of each zoom position>
                let v: Vec<f64> = rest.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>().with_context(|| format!("line {}", lineno + 1))?;
                if v.is_empty() || v[0] <= 0.0 || v.windows(2).any(|w| w[1] <= w[0]) {
                    bail!("line {}: @zoom <increasing focal lengths>", lineno + 1);
                }
                zoom_focal = Some(v);
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
            // Optional aspheric terms: k=<conic> a4=… a6=… (up to a20); on
            // the stop a dichroic annulus: annulus=<fraction> pass=<lo>-<hi>.
            let mut conic = 0.0;
            let mut aspheric = Vec::new();
            let (mut annulus, mut pass) = (None, None);
            for kv in &cols[4..] {
                let Some((key, val)) = kv.split_once('=') else { bail!("line {}: expected key=value, got {kv}", lineno + 1) };
                if key.eq_ignore_ascii_case("pass") {
                    let (lo, hi) = val.split_once('-').with_context(|| format!("line {}: pass=<lo>-<hi> in nm", lineno + 1))?;
                    pass = Some((lo.parse::<f64>().with_context(ctx)?, hi.parse::<f64>().with_context(ctx)?));
                    continue;
                }
                let val: f64 = val.parse().with_context(ctx)?;
                match key.to_ascii_lowercase().as_str() {
                    "k" => conic = val,
                    "annulus" => annulus = Some(val),
                    a if a.starts_with('a') => {
                        let order: usize = a[1..].parse().with_context(ctx)?;
                        if order < 4 || order % 2 == 1 || order > 20 {
                            bail!("line {}: aspheric terms are a4, a6, …, a20", lineno + 1);
                        }
                        let i = (order - 4) / 2;
                        if aspheric.len() <= i {
                            aspheric.resize(i + 1, 0.0);
                        }
                        aspheric[i] = val;
                    }
                    _ => bail!("line {}: unknown key {key}", lineno + 1),
                }
            }
            if is_stop && (conic != 0.0 || !aspheric.is_empty()) {
                bail!("line {}: the stop is flat", lineno + 1);
            }
            match (annulus, pass) {
                (None, None) => {}
                (Some(inner), Some((lo, hi))) if is_stop && (0.0..=1.0).contains(&inner) && lo < hi => spectral_stop = Some(SpectralStop { inner, lo, hi }),
                _ => bail!("line {}: annulus=<0..1> pass=<lo>-<hi> go together, on the stop", lineno + 1),
            }
            surfaces.push(Surface { radius, thickness, material, semi_diameter, is_stop, z: 0.0, decenter: (0.0, 0.0), conic, aspheric });
        }
        if let Some(f) = floating.iter().find(|f| f.surface >= surfaces.len()) {
            bail!("@float row {} does not exist", f.surface + 1);
        }
        let zoom = match zoom_focal {
            Some(focal) => {
                for (row, v) in &zoom_gaps {
                    if *row >= surfaces.len() || floating.iter().any(|f| f.surface == *row) {
                        bail!("@zoom-gap row {} does not exist or floats", row + 1);
                    }
                    if v.len() != focal.len() {
                        bail!("@zoom-gap row {}: {} gaps for {} zoom positions", row + 1, v.len(), focal.len());
                    }
                }
                if zoom_fnumber.as_ref().is_some_and(|v: &Vec<f64>| v.len() != focal.len() || v.iter().any(|f| *f <= 0.0)) {
                    bail!("@zoom-fnumber needs one f-number per zoom position");
                }
                Some(Zoom { at: focal[0], focal, gaps: zoom_gaps, fnumber: zoom_fnumber })
            }
            None if !zoom_gaps.is_empty() || zoom_fnumber.is_some() => bail!("@zoom-gap or @zoom-fnumber without @zoom"),
            None => None,
        };
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
        let mut lens = Lens {
            name,
            surfaces,
            coating,
            iris: Iris::new(0, 0.0, 0.0, stop_r, true),
            blades,
            blade_curvature,
            stop_index,
            sensor_z: z,
            sensor_reflectance: 0.0,
            sensor_half: (18.0, 12.0),
            forward_path,
            backward_path,
            floating,
            zoom,
            spectral_stop,
        };
        if let Some(f) = lens.zoom.as_ref().map(|z| z.at) {
            lens.set_zoom(f);
        }
        Ok(lens)
    }

    /// Recomputes vertex positions from the thicknesses.
    fn update_positions(&mut self) {
        let mut z = 0.0;
        for s in self.surfaces.iter_mut() {
            s.z = z;
            z += s.thickness;
        }
    }

    /// Sets the floating-element gaps for an object at `distance` mm from the
    /// front vertex (infinity if `None`). Call before focusing.
    pub fn set_floating(&mut self, distance: Option<f64>) {
        if self.floating.is_empty() {
            return;
        }
        for f in &self.floating {
            let t = distance.map_or(0.0, |d| f.close_distance / d);
            self.surfaces[f.surface].thickness = f.at_infinity + (f.at_close - f.at_infinity) * t;
        }
        self.update_positions();
    }

    /// Focal lengths of the zoom range (None for a prime).
    pub fn zoom_range(&self) -> Option<(f64, f64)> {
        self.zoom.as_ref().map(|z| (z.focal[0], *z.focal.last().unwrap()))
    }

    /// Zooms to focal length `f` (clamped to the range; the actual focal
    /// length, `focal_length()`, differs slightly between zoom positions).
    /// Call before setting aperture and focus.
    pub fn set_zoom(&mut self, f: f64) {
        let Some(z) = &mut self.zoom else { return };
        let (lo, hi) = (z.focal[0], *z.focal.last().unwrap());
        z.at = f.clamp(lo, hi);
        for (row, v) in &z.gaps {
            self.surfaces[*row].thickness = z.gap(v);
        }
        let fnumber = z.fnumber.as_ref().map(|v| z.gap(v));
        self.update_positions();
        if let Some(n) = fnumber {
            self.surfaces[self.stop_index].semi_diameter = self.stop_radius_for_fstop(n);
        }
    }

    /// Glass elements as ranges of surface indices (front to back).
    pub fn elements(&self) -> Vec<std::ops::RangeInclusive<usize>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.surfaces.len() {
            if self.surfaces[i].material.is_some() {
                let start = i;
                while i < self.surfaces.len() && self.surfaces[i].material.is_some() {
                    i += 1;
                }
                out.push(start..=i.min(self.surfaces.len() - 1));
            }
            i += 1;
        }
        out
    }

    /// Shifts glass element `e` (0-based, front to back) sideways by (dx, dy) mm.
    pub fn decenter_element(&mut self, e: usize, dx: f64, dy: f64) -> Result<()> {
        let elements = self.elements();
        let Some(range) = elements.get(e) else { bail!("lens has no element {e}") };
        for i in range.clone() {
            self.surfaces[i].decenter = (dx, dy);
        }
        Ok(())
    }

    pub fn is_centered(&self) -> bool {
        self.surfaces.iter().all(|s| s.decenter == (0.0, 0.0))
    }

    /// The prescription as `.lens` text (decentering is not part of it).
    pub fn to_text(&self) -> String {
        let mut t = format!("@name {}\n@iris {} {}\n@coating {}\n", self.name, self.blades, self.blade_curvature, self.coating.name());
        for f in &self.floating {
            t += &format!("@float {} {} {} {}\n", f.surface + 1, f.at_infinity, f.at_close, f.close_distance / 1000.0);
        }
        if let Some(z) = &self.zoom {
            let list = |v: &[f64]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            t += &format!("@zoom {}\n", list(&z.focal));
            for (row, v) in &z.gaps {
                t += &format!("@zoom-gap {} {}\n", row + 1, list(v));
            }
            if let Some(v) = &z.fnumber {
                t += &format!("@zoom-fnumber {}\n", list(v));
            }
        }
        t += "# radius   thickness   material   semi-diameter\n";
        for s in &self.surfaces {
            let radius = if s.is_stop { "stop".to_string() } else if s.radius == 0.0 { "inf".to_string() } else { s.radius.to_string() };
            let material = s.material.as_ref().map_or("air", |g| g.name());
            t += &format!("{radius} {} {material} {}", s.thickness, s.semi_diameter);
            if s.conic != 0.0 {
                t += &format!(" k={}", s.conic);
            }
            for (i, a) in s.aspheric.iter().enumerate().filter(|(_, a)| **a != 0.0) {
                t += &format!(" a{}={a}", 4 + 2 * i);
            }
            if let (true, Some(f)) = (s.is_stop, &self.spectral_stop) {
                t += &format!(" annulus={} pass={}-{}", f.inner, f.lo, f.hi);
            }
            t += "\n";
        }
        t
    }

    /// Inserts a surface before row `at`, keeping floating and zoom gaps on their rows.
    pub fn insert_surface(&mut self, at: usize, s: Surface) {
        self.surfaces.insert(at, s);
        let rows = self.floating.iter_mut().map(|f| &mut f.surface).chain(self.zoom.iter_mut().flat_map(|z| z.gaps.iter_mut().map(|g| &mut g.0)));
        for r in rows.filter(|r| **r >= at) {
            *r += 1;
        }
    }

    /// Removes row `at` (and a floating or zoom gap on it).
    pub fn remove_surface(&mut self, at: usize) {
        self.surfaces.remove(at);
        self.floating.retain(|f| f.surface != at);
        if let Some(z) = &mut self.zoom {
            z.gaps.retain(|g| g.0 != at);
        }
        let rows = self.floating.iter_mut().map(|f| &mut f.surface).chain(self.zoom.iter_mut().flat_map(|z| z.gaps.iter_mut().map(|g| &mut g.0)));
        for r in rows.filter(|r| **r > at) {
            *r -= 1;
        }
    }

    /// Traces `ray` forward surface by surface: the points it reaches, and
    /// the ray leaving the lens if it gets through (if not, the last point is
    /// where it is blocked).
    pub fn trace_points(&self, ray: Ray, l: f64) -> (Vec<Vec3>, Option<Ray>) {
        let (pts, out) = self.trace_path_points(ray, l, &self.forward_path);
        (pts, out.map(|(r, _)| r))
    }

    /// Like `trace_points` along any event path (e.g. a ghost), also
    /// returning the throughput.
    pub fn trace_path_points(&self, ray: Ray, l: f64, path: &[Event]) -> (Vec<Vec3>, Option<(Ray, f64)>) {
        let m = self.media(l);
        let mut pts = vec![ray.o];
        let mut cur = ray;
        let mut throughput = 1.0;
        for ev in path {
            let Some((r, t, _)) = self.trace_path_opl(cur, l, &m, std::slice::from_ref(ev), false) else { return (pts, None) };
            pts.push(r.o);
            let blocked = match self.surfaces.get(ev.surface as usize) {
                Some(s) => {
                    let (x, y) = (r.o.x - s.decenter.0, r.o.y - s.decenter.1);
                    x * x + y * y > s.semi_diameter * s.semi_diameter || (s.is_stop && !self.iris.contains(x, y))
                }
                None => !self.on_sensor(r.o.x, r.o.y),
            };
            if blocked {
                return (pts, None);
            }
            throughput *= t;
            cur = r;
        }
        (pts, Some((cur, throughput)))
    }

    /// Cumulative throughput (Fresnel / coating transmission) of `ray` after
    /// each surface of the forward path, apertures ignored; shorter if the
    /// ray misses a surface.
    pub fn surface_throughput(&self, ray: Ray, l: f64) -> Vec<f64> {
        let m = self.media(l);
        let mut out = Vec::new();
        let (mut cur, mut t) = (ray, 1.0);
        for ev in &self.forward_path {
            let Some((r, ti, _)) = self.trace_path_opl(cur, l, &m, std::slice::from_ref(ev), false) else { break };
            t *= ti;
            out.push(t);
            cur = r;
        }
        out
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
            let c = s.curvature();
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

    /// Sets the aperture, then focuses on `distance` mm (infinity if `None`):
    /// floating elements, and the sensor `offset` mm behind best focus.
    pub fn prepare(&mut self, fstop: f64, blades: u32, rotation_deg: f64, curvature: f64, distance: Option<f64>, offset: f64) -> Result<()> {
        self.set_aperture(fstop, blades, rotation_deg, curvature)?;
        self.set_floating(distance);
        self.sensor_z = self.best_focus(distance)? + offset;
        Ok(())
    }

    // --------------------------------------------------------------- real rays

    /// Intersects surface `i`; returns the hit point and the surface normal.
    #[inline]
    fn intersect(&self, i: usize, ray: &Ray) -> Option<(Vec3, Vec3)> {
        let s = &self.surfaces[i];
        if s.is_aspheric() {
            return Self::intersect_asphere(s, ray);
        }
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
        let c = Vec3::new(s.decenter.0, s.decenter.1, s.z + r);
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

    /// Newton iteration on the sag equation, starting from the base sphere.
    fn intersect_asphere(s: &Surface, ray: &Ray) -> Option<(Vec3, Vec3)> {
        let off = Vec3::new(s.decenter.0, s.decenter.1, s.z);
        let o = ray.o - off;
        let d = ray.d;
        let c = s.curvature();
        // Base sphere c(x²+y²+z²) − 2z = 0, stable vertex-side root (plane if c = 0).
        let b = c * o.dot(d) - d.z;
        let cc = c * o.dot(o) - 2.0 * o.z;
        let disc = b * b - c * cc;
        let mut t = if disc >= 0.0 {
            let q = -b - b.signum() * disc.sqrt();
            if q != 0.0 { cc / q } else { -o.z / d.z }
        } else {
            -o.z / d.z
        };
        for _ in 0..40 {
            let p = o + d * t;
            let u = p.x * p.x + p.y * p.y;
            let f = p.z - s.sag_u(u)?;
            let df = d.z - s.dsag_du(u) * 2.0 * (p.x * d.x + p.y * d.y);
            if df == 0.0 {
                return None;
            }
            let dt = f / df;
            t -= dt;
            if dt.abs() < 1e-12 * (1.0 + t.abs()) {
                if t < 1e-9 {
                    return None;
                }
                let p = o + d * t;
                let g = 2.0 * s.dsag_du(p.x * p.x + p.y * p.y);
                let n = Vec3::new(g * p.x, g * p.y, -1.0).normalize();
                return Some((p + off, n));
            }
        }
        None
    }

    /// Fresnel reflectance (unpolarised) at an interface from index `n1` to `n2`.
    /// Air–glass interfaces are coated; `ng0` is the glass index at the
    /// coating's design wavelength.
    #[inline]
    pub fn reflectance(&self, n1: f64, n2: f64, cos1: f64, l: f64, ng0: f64) -> f64 {
        let sin1_n = n1 * (1.0 - cos1 * cos1).max(0.0).sqrt(); // Snell invariant
        let cosm = |n: f64| -> Option<f64> {
            let s = sin1_n / n;
            if s >= 1.0 { None } else { Some((1.0 - s * s).sqrt()) }
        };
        let air_interface = (n1 == 1.0) != (n2 == 1.0);
        let ng = n1.max(n2);
        match self.coating {
            Coating::Single { n, lambda0 } if air_interface => film_stack(&[(n, 0.25)], ng, sin1_n, lambda0, l),
            Coating::Multi { n_low, n_high, lambda0 } if air_interface => {
                film_stack(&[(n_low, 0.25), (n_high, 0.5), (Coating::tailored_index(n_low, ng0), 0.25)], ng, sin1_n, lambda0, l)
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
    pub fn trace_path(&self, ray: Ray, l: f64, m: &Media, path: &[Event]) -> Option<(Ray, f64)> {
        self.trace_path_opl(ray, l, m, path, true).map(|(r, t, _)| (r, t))
    }

    /// Like `trace_path`, also returning the optical path length from the
    /// ray origin to the last surface. With `clip` false, apertures (element
    /// rims, stop, iris) do not block rays.
    #[inline]
    pub fn trace_path_opl(&self, mut ray: Ray, l: f64, m: &Media, path: &[Event], clip: bool) -> Option<(Ray, f64, f64)> {
        let mut throughput = 1.0;
        let mut opl = 0.0;
        for ev in path {
            let i = ev.surface as usize;
            if i == self.surfaces.len() {
                // The sensor: a flat mirror over its area.
                let t = (self.sensor_z - ray.o.z) / ray.d.z;
                if ray.d.z <= 0.0 || t < 1e-9 {
                    return None;
                }
                let p = ray.o + ray.d * t;
                if clip && !self.on_sensor(p.x, p.y) {
                    return None;
                }
                opl += m.n[i] * t;
                throughput *= self.sensor_reflectance;
                ray = Ray { o: p, d: Vec3::new(ray.d.x, ray.d.y, -ray.d.z) };
                continue;
            }
            let s = &self.surfaces[i];
            let (p, mut nrm) = self.intersect(i, &ray)?;
            let (n1, n2) = if ev.forward { (m.n[i], m.n[i + 1]) } else { (m.n[i + 1], m.n[i]) };
            opl += n1 * (p - ray.o).length();
            let (px, py) = (p.x - s.decenter.0, p.y - s.decenter.1);
            let r2 = px * px + py * py;
            if s.is_stop {
                if clip && (!self.iris.contains(px, py) || r2 > s.semi_diameter * s.semi_diameter) {
                    return None;
                }
                if let Some(f) = &self.spectral_stop {
                    throughput *= f.transmission(r2, s.semi_diameter, l);
                }
                ray.o = p;
                continue;
            }
            if clip && r2 > s.semi_diameter * s.semi_diameter {
                return None;
            }
            let mut cosi = -nrm.dot(ray.d);
            if cosi < 0.0 {
                nrm = -nrm;
                cosi = -cosi;
            }
            let ng0 = match self.coating {
                Coating::Multi { lambda0, .. } => self.surfaces[i].material.as_ref().or(i.checked_sub(1).and_then(|j| self.surfaces[j].material.as_ref())).map_or(1.0, |g| g.index(lambda0)),
                _ => 1.0,
            };
            let refl = self.reflectance(n1, n2, cosi, l, ng0);
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
        Some((ray, throughput, opl))
    }

    /// Transmission of the spectral stop for `ray` entering the lens
    /// (apertures ignored); 1 without one.
    pub fn spectral_weight(&self, ray: Ray, l: f64, m: &Media) -> f64 {
        let Some(f) = &self.spectral_stop else { return 1.0 };
        let k = self.stop_index;
        let Some((r, _, _)) = self.trace_path_opl(ray, l, m, &self.forward_path[..k], false) else { return 0.0 };
        let s = &self.surfaces[k];
        let t = (s.z - r.o.z) / r.d.z;
        let (x, y) = (r.o.x + r.d.x * t - s.decenter.0, r.o.y + r.d.y * t - s.decenter.1);
        f.transmission(x * x + y * y, s.semi_diameter, l)
    }

    fn on_sensor(&self, x: f64, y: f64) -> bool {
        x.abs() <= self.sensor_half.0 && y.abs() <= self.sensor_half.1
    }

    pub fn trace_forward(&self, ray: Ray, l: f64, m: &Media) -> Option<(Ray, f64)> {
        self.trace_path(ray, l, m, &self.forward_path)
    }

    pub fn trace_backward(&self, ray: Ray, l: f64, m: &Media) -> Option<(Ray, f64)> {
        self.trace_path(ray, l, m, &self.backward_path)
    }

    /// Event sequences for all two-reflection ghost paths: the light reflects
    /// first off surface `a` (travelling backwards) and then off surface `b < a`.
    /// With a reflecting sensor, `a` may also be the sensor: `a == surfaces.len()`,
    /// which is also its surface index in events.
    pub fn ghost_paths(&self) -> Vec<(usize, usize, Vec<Event>)> {
        let n = self.surfaces.len();
        let mut out = Vec::new();
        let last = if self.sensor_reflectance > 0.0 { n } else { n - 1 };
        for a in 0..=last {
            for b in 0..a {
                if (a < n && self.surfaces[a].is_stop) || self.surfaces[b].is_stop {
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
        let r = lens.reflectance(1.0, 1.5, 1.0, 550.0, 1.5);
        assert!((r - 0.04).abs() < 1e-9);
        let mut coated = lens.clone();
        coated.coating = Coating::Single { n: 1.38, lambda0: 550.0 };
        assert!(coated.reflectance(1.0, 1.5, 1.0, 550.0, 1.5) < 0.015);
        assert!(coated.reflectance(1.5, 1.0, 1.0, 550.0, 1.5) < 0.015);
    }

    /// Band-averaged (420–680 nm) reflectance of the multi-layer vs the single
    /// layer on glass of index 1.47–1.92, at 0° and 30°.
    #[test]
    fn multi_coating() {
        let lens = |coating| Lens { coating, ..Lens::parse(TRIPLET).unwrap() };
        let (single, multi) = (lens(Coating::SINGLE), lens(Coating::MULTI));
        let band = |l: &Lens, ng: f64, deg: f64| (420..=680).step_by(5).map(|w| l.reflectance(1.0, ng, deg.to_radians().cos(), w as f64, ng)).sum::<f64>() / 53.0;
        let (mut sum_s, mut sum_m) = (0.0, 0.0);
        for ng in [1.47, 1.52, 1.6, 1.7, 1.8, 1.92] {
            let (s0, m0, s30, m30) = (band(&single, ng, 0.0), band(&multi, ng, 0.0), band(&single, ng, 30.0), band(&multi, ng, 30.0));
            println!("n {ng}: 0° single {:.3} % multi {:.3} %, 30° single {:.3} % multi {:.3} %", s0 * 100.0, m0 * 100.0, s30 * 100.0, m30 * 100.0);
            assert!(m0 < 0.005 && m30 < 0.02);
            (sum_s, sum_m) = (sum_s + s0, sum_m + m0);
        }
        assert!(sum_m < sum_s / 3.0);
        // Zero at λ0; the single layer through the stack is the quarter-wave formula.
        assert!(multi.reflectance(1.0, 1.6, 1.0, 550.0, 1.6) < 1e-12);
        assert!((single.reflectance(1.6, 1.0, 1.0, 550.0, 1.6) - ((1.6 - 1.38f64.powi(2)) / (1.6 + 1.38f64.powi(2))).powi(2)).abs() < 1e-12);
        assert_eq!(Lens::parse(include_str!("../lenses/summilux-50-asph.lens")).unwrap().coating, Coating::MULTI);
    }

    /// The Newton asphere intersection reproduces the exact sphere solution.
    #[test]
    fn asphere_path_matches_sphere() {
        let lens = Lens::parse(TRIPLET).unwrap();
        let mut asph = lens.clone();
        for s in asph.surfaces.iter_mut().filter(|s| !s.is_stop) {
            s.aspheric = vec![1e-30];
        }
        let m = lens.media(550.0);
        for (y, a) in [(0.5, 0.0), (3.0, 0.1), (-4.0, -0.2), (6.0, 0.3)] {
            let ray = Ray { o: Vec3::new(0.2, y, -20.0), d: Vec3::new(0.0, a, 1.0).normalize() };
            let p = lens.trace_forward(ray, 550.0, &m);
            let q = asph.trace_forward(ray, 550.0, &m);
            match (p, q) {
                (Some((p, tp)), Some((q, tq))) => {
                    assert!((p.o - q.o).length() < 1e-8 && (p.d - q.d).length() < 1e-10 && (tp - tq).abs() < 1e-12);
                }
                (None, None) => {}
                _ => panic!("sphere and asphere paths disagree for {y} {a}"),
            }
        }
    }

    /// A conic with k = −1 is a paraboloid: z = c r² / 2.
    #[test]
    fn conic_sag() {
        let s = Surface {
            radius: 20.0,
            thickness: 1.0,
            material: None,
            semi_diameter: 10.0,
            is_stop: false,
            z: 0.0,
            decenter: (0.0, 0.0),
            conic: -1.0,
            aspheric: vec![1e-5],
        };
        let r: f64 = 8.0;
        assert!((s.sag(r) - (r * r / 40.0 + 1e-5 * r.powi(4))).abs() < 1e-12);
    }

    #[test]
    fn text_round_trip() {
        for text in [include_str!("../lenses/cooke-triplet.lens"), include_str!("../lenses/summilux-50-asph.lens"), include_str!("../lenses/double-gauss.lens"), ZOOM] {
            let a = Lens::parse(text).unwrap();
            let b = Lens::parse(&a.to_text()).unwrap();
            assert_eq!(a.to_text(), b.to_text());
            assert_eq!((a.name.as_str(), a.blades, a.surfaces.len(), a.floating.len(), a.coating), (b.name.as_str(), b.blades, b.surfaces.len(), b.floating.len(), b.coating));
            assert!((a.focal_length() - b.focal_length()).abs() < 1e-12);
            for (s, t) in a.surfaces.iter().zip(&b.surfaces) {
                assert_eq!((s.radius, s.thickness, s.semi_diameter, s.conic, &s.aspheric), (t.radius, t.thickness, t.semi_diameter, t.conic, &t.aspheric));
            }
            let zoom = |l: &Lens| l.zoom.as_ref().map(|z| (z.focal.clone(), z.gaps.clone(), z.fnumber.clone()));
            assert_eq!(zoom(&a), zoom(&b));
        }
    }

    const ZOOM: &str = include_str!("../lenses/canon-24-70-f2.8l-ii.lens");

    /// The spectral stop parses, round-trips and weights rays crossing its annulus.
    #[test]
    fn spectral_stop() {
        let text = TRIPLET.replace("air         5.57", "air         5.57  annulus=0.5 pass=480-600");
        let a = Lens::parse(&text).unwrap();
        let f = a.spectral_stop.unwrap();
        assert_eq!(f, SpectralStop { inner: 0.5, lo: 480.0, hi: 600.0 });
        assert_eq!(Lens::parse(&a.to_text()).unwrap().spectral_stop, Some(f));
        assert!(Lens::parse(TRIPLET).unwrap().spectral_stop.is_none());
        for bad in ["annulus=0.5", "pass=480-600", "annulus=1.5 pass=480-600", "annulus=0.5 pass=600-480"] {
            assert!(Lens::parse(&TRIPLET.replace("air         5.57", &format!("air 5.57 {bad}"))).is_err(), "{bad}");
        }
        assert!(Lens::parse(&TRIPLET.replace("N-SK16      8.2", "N-SK16 8.2 annulus=0.5 pass=480-600")).is_err());
        assert_eq!((f.band(540.0), f.band(470.0), f.band(480.0), f.band(610.0)), (1.0, 0.0, 0.5, 0.0));
        // A ray through the annulus is filtered out of band, one near the axis is not.
        let m = a.media(430.0);
        let outer = Ray { o: Vec3::new(0.0, 5.0, -10.0), d: Vec3::new(0.0, 0.0, 1.0) };
        let inner = Ray { o: Vec3::new(0.0, 1.0, -10.0), d: Vec3::new(0.0, 0.0, 1.0) };
        let plain = Lens::parse(TRIPLET).unwrap();
        assert_eq!(a.trace_forward(outer, 430.0, &m).unwrap().1, 0.0);
        assert_eq!(a.trace_forward(inner, 430.0, &m).unwrap().1, plain.trace_forward(inner, 430.0, &m).unwrap().1);
        assert_eq!((a.spectral_weight(outer, 430.0, &m), a.spectral_weight(outer, 540.0, &a.media(540.0)), a.spectral_weight(inner, 430.0, &m)), (0.0, 1.0, 1.0));
    }

    /// The zoom at the patent's positions (US 8,736,971 example 3): focal
    /// length, back focus and F-number; sharp on axis and plausible at the
    /// edge, wide open. Between them, the gaps follow smoothly and f/2.91 stays.
    #[test]
    fn zoom_matches_patent() {
        let mut lens = Lens::parse(ZOOM).unwrap();
        assert_eq!(lens.zoom_range(), Some((24.70, 67.88)));
        for (f, bf) in [(24.70, 38.08), (34.93, 46.62), (67.88, 63.32)] {
            lens.set_zoom(f);
            let p = lens.paraxial_data(LAMBDA_D);
            let max = lens.max_aperture();
            lens.prepare(2.91, 9, 0.0, 0.8, None, 0.0).unwrap();
            let best = lens.sensor_z - lens.last_z();
            let mtf = |h: f64| crate::mtf::mtf_at(&lens, h, None, &[10.0, 30.0]).unwrap();
            let ((s0, t0), (s1, t1)) = (mtf(0.0), mtf(15.0));
            println!("f {f}: EFL {:.3} BFD {:.3} best focus {best:.3} f/{max:.3}; MTF 10/30 lp/mm centre {:.2}/{:.2}, 15 mm S {:.2}/{:.2} T {:.2}/{:.2}", p.efl, p.bfd, s0[0], s0[1], s1[0], s1[1], t1[0], t1[1]);
            assert!((p.efl - f).abs() < 0.005 * f && (p.bfd - bf).abs() < 0.005 * bf && (best - bf).abs() < 0.1);
            assert!((max - 2.91).abs() < 1e-9);
            assert!(s0[1] > 0.85 && t0[1] > 0.85 && s1[0] > 0.8 && t1[0] > 0.8 && s1[1] > 0.4 && t1[1] > 0.4);
        }
        let mut last = Lens::parse(ZOOM).unwrap();
        for i in 1..=40 {
            lens.set_zoom(24.70 * (67.88f64 / 24.70).powf(i as f64 / 40.0));
            assert!((lens.max_aperture() - 2.91).abs() < 1e-9);
            assert!(lens.focal_length() > last.focal_length());
            // Gaps change by a fraction of a millimetre per step.
            assert!(lens.surfaces.iter().zip(&last.surfaces).all(|(a, b)| (a.thickness - b.thickness).abs() < 1.0));
            last = lens.clone();
        }
        lens.set_zoom(200.0);
        assert!((lens.surfaces[4].thickness - 30.30).abs() < 1e-12);
    }

    #[test]
    fn iris_directive() {
        let l = Lens::parse(&format!("{TRIPLET}\n@iris 11 0.5\n")).unwrap();
        assert_eq!((l.blades, l.blade_curvature), (11, 0.5));
        assert_eq!(Lens::parse(TRIPLET).unwrap().blades, 6);
    }

    /// Three zoom positions: the gaps lie on the parabola in log focal length.
    #[test]
    fn zoom_gap_interpolation() {
        let mut lens = Lens::parse(&format!("{TRIPLET}\n@zoom 40 50 80\n@zoom-gap 2 6 7 9\n")).unwrap();
        assert_eq!(lens.surfaces[1].thickness, 6.0);
        let x = [40f64.ln(), 50f64.ln(), 80f64.ln()];
        let y = [6.0, 7.0, 9.0];
        for f in [40.0, 45.0, 50.0, 63.0, 80.0] {
            lens.set_zoom(f);
            let t = f64::ln(f);
            let q: f64 = (0..3).map(|i| y[i] * (0..3).filter(|&j| j != i).map(|j| (t - x[j]) / (x[i] - x[j])).product::<f64>()).sum();
            assert!((lens.surfaces[1].thickness - q).abs() < 1e-12, "{f}");
            assert!((lens.surfaces[2].z - lens.surfaces[1].z - q).abs() < 1e-12);
        }
        assert!(Lens::parse(&format!("{TRIPLET}\n@zoom-gap 2 6 7 9\n")).is_err());
        assert!(Lens::parse(&format!("{TRIPLET}\n@zoom 40 50 80\n@zoom-gap 2 6 7\n")).is_err());
    }

    /// Floating gaps interpolate in object vergence.
    #[test]
    fn floating_gap() {
        let mut lens = Lens::parse(&format!("{TRIPLET}\n@float 2 6.0 7.0 0.5\n")).unwrap();
        lens.set_floating(Some(1000.0));
        assert!((lens.surfaces[1].thickness - 6.5).abs() < 1e-12);
        assert!((lens.surfaces[2].z - lens.surfaces[1].z - 6.5).abs() < 1e-12);
        lens.set_floating(None);
        assert!((lens.surfaces[1].thickness - 6.0).abs() < 1e-12);
    }
}

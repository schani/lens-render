//! Optical analysis: paraxial data, aberrations versus field, spot diagrams
//! and a lens layout drawing.

use crate::canvas::{Canvas, wavelength_color};
use crate::glass::{LAMBDA_C, LAMBDA_D, LAMBDA_F};
use crate::lens::Lens;
use crate::math::{Ray, Vec3};
use anyhow::Result;

const START_Z: f64 = -5.0;

/// Height at the stop plane of a collimated ray (field angle `theta`) entering at height `y0`.
fn stop_height(lens: &Lens, theta: f64, y0: f64, l: f64) -> Option<f64> {
    let m = lens.media(l);
    let ray = Ray { o: Vec3::new(0.0, y0, START_Z), d: Vec3::new(0.0, theta.sin(), theta.cos()) };
    let s = lens.stop_index;
    let r = if s == 0 { ray } else { lens.trace_path(ray, l, &m, &lens.forward_path[..s])?.0 };
    let zs = lens.surfaces[s].z;
    Some(r.o.y + (zs - r.o.z) * r.d.y / r.d.z)
}

/// Entry height (at z = START_Z) of the real chief ray for field angle `theta`.
pub fn chief_ray_y0(lens: &Lens, theta: f64, l: f64) -> f64 {
    let mut a = 0.0;
    let mut b = 0.5;
    let mut fa = stop_height(lens, theta, a, l).unwrap_or(0.0);
    let mut fb = stop_height(lens, theta, b, l).unwrap_or(0.0);
    for _ in 0..40 {
        if (fb - fa).abs() < 1e-14 {
            break;
        }
        let c = b - fb * (b - a) / (fb - fa);
        a = b;
        fa = fb;
        b = c;
        fb = match stop_height(lens, theta, b, l) {
            Some(v) => v,
            None => break,
        };
        if fb.abs() < 1e-10 {
            break;
        }
    }
    b
}

pub struct Bundle {
    /// Rays leaving the last surface, with throughput.
    pub rays: Vec<(Ray, f64, f64)>, // (ray, throughput, wavelength)
}

/// Collimated bundle at field angle `theta` sampled on a square grid around the chief ray.
pub fn field_bundle(lens: &Lens, theta: f64, wls: &[f64], grid: usize, fan: Option<(bool, f64)>) -> Bundle {
    let p = lens.paraxial_data(LAMBDA_D);
    let r_ep = lens.iris.max_radius / p.entrance_pupil_mag;
    let half = r_ep * 2.0;
    let mut rays = Vec::new();
    for &l in wls {
        let y0 = chief_ray_y0(lens, theta, LAMBDA_D);
        let m = lens.media(l);
        let d = Vec3::new(0.0, theta.sin(), theta.cos());
        let mut push = |x: f64, y: f64| {
            let ray = Ray { o: Vec3::new(x, y0 + y, START_Z), d };
            if let Some((r, t)) = lens.trace_forward(ray, l, &m) {
                rays.push((r, t, l));
            }
        };
        match fan {
            None => {
                for iy in 0..grid {
                    for ix in 0..grid {
                        let x = ((ix as f64 + 0.5) / grid as f64 * 2.0 - 1.0) * half;
                        let y = ((iy as f64 + 0.5) / grid as f64 * 2.0 - 1.0) * half;
                        push(x, y);
                    }
                }
            }
            Some((tangential, frac)) => {
                for i in 0..grid {
                    let s = ((i as f64 + 0.5) / grid as f64 * 2.0 - 1.0) * r_ep * frac;
                    if tangential { push(0.0, s) } else { push(s, 0.0) }
                }
            }
        }
    }
    Bundle { rays }
}

fn at_z(r: &Ray, z: f64) -> (f64, f64) {
    let t = (z - r.o.z) / r.d.z;
    (r.o.x + r.d.x * t, r.o.y + r.d.y * t)
}

/// Plane minimising the variance of one coordinate of a set of rays.
fn variance_focus(rays: &[(Ray, f64, f64)], use_y: bool) -> f64 {
    let n = rays.len() as f64;
    let ab: Vec<(f64, f64)> = rays
        .iter()
        .map(|(r, _, _)| {
            let (dir, pos) = if use_y { (r.d.y, r.o.y) } else { (r.d.x, r.o.x) };
            let b = dir / r.d.z;
            (pos - b * r.o.z, b)
        })
        .collect();
    let am = ab.iter().map(|v| v.0).sum::<f64>() / n;
    let bm = ab.iter().map(|v| v.1).sum::<f64>() / n;
    let num: f64 = ab.iter().map(|(a, b)| (a - am) * (b - bm)).sum();
    let den: f64 = ab.iter().map(|(_, b)| (b - bm) * (b - bm)).sum();
    -num / den
}

fn spot_rms(rays: &[(Ray, f64, f64)], z: f64) -> (f64, f64, f64) {
    let mut sw = 0.0;
    let (mut cx, mut cy) = (0.0, 0.0);
    let pts: Vec<(f64, f64, f64)> = rays
        .iter()
        .map(|(r, t, l)| {
            let (x, y) = at_z(r, z);
            let w = t * crate::spectrum::cie_xyz(*l)[1];
            (x, y, w)
        })
        .collect();
    for (x, y, w) in &pts {
        sw += w;
        cx += x * w;
        cy += y * w;
    }
    cx /= sw;
    cy /= sw;
    let var = pts.iter().map(|(x, y, w)| w * ((x - cx).powi(2) + (y - cy).powi(2))).sum::<f64>() / sw;
    (var.sqrt(), cx, cy)
}

pub fn chief_height_at(lens: &Lens, theta: f64, l: f64, z: f64) -> Option<f64> {
    let y0 = chief_ray_y0(lens, theta, l);
    let m = lens.media(l);
    let ray = Ray { o: Vec3::new(0.0, y0, START_Z), d: Vec3::new(0.0, theta.sin(), theta.cos()) };
    // The chief ray passes the stop centre, so the iris never blocks it; vignetting at other surfaces may.
    let (r, _) = lens.trace_forward(ray, l, &m)?;
    Some(at_z(&r, z).1)
}

pub fn report(lens: &Lens, half_diag: f64, out_prefix: Option<&str>) -> Result<String> {
    use std::fmt::Write;
    let mut s = String::new();
    let p = lens.paraxial_data(LAMBDA_D);
    writeln!(s, "{}", lens.name)?;
    writeln!(s, "\nSurfaces (n at d-line):")?;
    for (i, srf) in lens.surfaces.iter().enumerate() {
        let mat = srf.material.as_ref().map_or("air".to_string(), |g| format!("{} (nd {:.5}, vd {:.2})", g.name(), g.nd(), g.abbe()));
        let rad = if srf.is_stop { "STOP".to_string() } else if srf.radius == 0.0 { "flat".into() } else { format!("{:.4}", srf.radius) };
        writeln!(s, "  {i:2}  R {rad:>10}  t {:8.4}  sd {:6.3}  {mat}", srf.thickness, srf.semi_diameter)?;
    }
    writeln!(s, "\nParaxial (d-line):")?;
    writeln!(s, "  EFL                 {:.3} mm", p.efl)?;
    writeln!(s, "  BFD                 {:.3} mm", p.bfd)?;
    writeln!(s, "  Entrance pupil      z = {:.3} mm, magnification {:.4}", p.entrance_pupil_z, 1.0 / p.entrance_pupil_mag)?;
    writeln!(s, "  Exit pupil          z = {:.3} mm, magnification {:.4}", p.exit_pupil_z, p.exit_pupil_mag)?;
    writeln!(s, "  Max aperture        f/{:.2}", lens.max_aperture())?;
    writeln!(s, "  Current iris        stop radius {:.3} mm", lens.iris.max_radius)?;

    writeln!(s, "\nLongitudinal colour (paraxial focus relative to d-line, infinity):")?;
    let zd = lens.paraxial_focus(None, LAMBDA_D);
    for l in [420.0, 450.0, LAMBDA_F, 520.0, 550.0, LAMBDA_D, 620.0, LAMBDA_C, 700.0] {
        writeln!(s, "  {l:6.1} nm  {:+8.4} mm", lens.paraxial_focus(None, l) - zd)?;
    }

    let z = lens.sensor_z;
    let max_theta = (half_diag / p.efl).atan();
    writeln!(s, "\nSensor plane z = {z:.4} mm (paraxial d focus {zd:.4}); max field {:.2}°", max_theta.to_degrees())?;
    writeln!(s, "\n  field   img h   distortion  lat.colour  RMS spot  T-focus   S-focus   vignetting  rel.illum")?;
    writeln!(s, "  (deg)   (mm)    (%)         F-C (µm)    (µm)      (mm)      (mm)      (pass)      (incl. cos⁴)")?;
    let wls: Vec<f64> = (0..11).map(|i| 450.0 + 20.0 * i as f64).collect();
    let on_axis = field_bundle(lens, 0.0, &[LAMBDA_D], 64, None);
    let t0: f64 = on_axis.rays.iter().map(|r| r.1).sum();
    let fracs = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let h_of = |th: f64| chief_height_at(lens, th, LAMBDA_D, z);
    let mut spots = Vec::new();
    for &fr in &fracs {
        let th = max_theta * fr;
        let h = h_of(th);
        let hs = h.map_or("   --  ".into(), |h| format!("{:7.3}", h));
        let dist = match h {
            Some(h) if fr > 0.0 => format!("{:+8.3}", (h - p.efl * th.tan()) / (p.efl * th.tan()) * 100.0),
            _ => "    --  ".into(),
        };
        let lc = match (chief_height_at(lens, th, LAMBDA_F, z), chief_height_at(lens, th, LAMBDA_C, z)) {
            (Some(f), Some(c)) => format!("{:+8.2}", (f - c) * 1000.0),
            _ => "   --   ".into(),
        };
        let b = field_bundle(lens, th, &wls, 48, None);
        let (rms, _, _) = if b.rays.is_empty() { (f64::NAN, 0.0, 0.0) } else { spot_rms(&b.rays, z) };
        let tf = field_bundle(lens, th, &[LAMBDA_D], 64, Some((true, 0.3)));
        let sf = field_bundle(lens, th, &[LAMBDA_D], 64, Some((false, 0.3)));
        let tz = if tf.rays.len() > 4 { variance_focus(&tf.rays, true) - z } else { f64::NAN };
        let sz = if sf.rays.len() > 4 { variance_focus(&sf.rays, false) - z } else { f64::NAN };
        let bd = field_bundle(lens, th, &[LAMBDA_D], 64, None);
        let td: f64 = bd.rays.iter().map(|r| r.1).sum();
        let vig = td * th.cos() / t0;
        // Relative illumination: flux per solid angle over image area per solid angle.
        let ri = if fr == 0.0 {
            1.0
        } else {
            let dth = 1e-4;
            match (h_of(th - dth), h_of(th + dth), h) {
                (Some(h1), Some(h2), Some(h)) => {
                    let dh = (h2 - h1).abs() / (2.0 * dth);
                    vig * th.sin() / (h.abs() * dh) * p.efl * p.efl
                }
                _ => f64::NAN,
            }
        };
        writeln!(
            s,
            "  {:5.2}  {hs}  {dist}    {lc}    {:7.1}   {:+7.4}   {:+7.4}   {:6.3}      {:6.3}",
            th.to_degrees(),
            rms * 1000.0,
            tz,
            sz,
            vig,
            ri
        )?;
        spots.push((th, b));
    }

    if let Some(prefix) = out_prefix {
        draw_spots(&spots, z, &format!("{prefix}-spots.png"))?;
        draw_layout(lens, max_theta, &format!("{prefix}-layout.png"))?;
        writeln!(s, "\nWrote {prefix}-spots.png and {prefix}-layout.png")?;
    }
    Ok(s)
}

fn draw_spots(spots: &[(f64, Bundle)], z: f64, path: &str) -> Result<()> {
    let cell = 220;
    let box_um = 200.0;
    let mut c = Canvas::new(cell * spots.len(), cell, [0.0, 0.0, 0.0]);
    for (k, (_, b)) in spots.iter().enumerate() {
        let x0 = (k * cell) as f64;
        for i in 0..cell {
            c.blend(k as i64 * cell as i64, i as i64, [0.3, 0.3, 0.3], 1.0);
        }
        if b.rays.is_empty() {
            continue;
        }
        let (_, cx, cy) = spot_rms(&b.rays, z);
        let scale = cell as f64 / (box_um / 1000.0);
        for (r, t, l) in &b.rays {
            let (x, y) = at_z(r, z);
            let col = wavelength_color(*l);
            c.dot(x0 + cell as f64 / 2.0 + (x - cx) * scale, cell as f64 / 2.0 - (y - cy) * scale, col, (0.5 * t) as f32);
        }
    }
    c.save(path)
}

fn draw_layout(lens: &Lens, max_theta: f64, path: &str) -> Result<()> {
    let zmin = -8.0;
    let zmax = lens.sensor_z + 3.0;
    let ymax = lens.surfaces.iter().map(|s| s.semi_diameter).fold(0.0, f64::max).max(22.0) + 2.0;
    let scale = 1400.0 / (zmax - zmin);
    let w = 1400;
    let h = (2.0 * ymax * scale) as usize;
    let mut c = Canvas::new(w, h, [1.0, 1.0, 1.0]);
    let tx = |z: f64| (z - zmin) * scale;
    let ty = |y: f64| h as f64 / 2.0 - y * scale;
    c.line(tx(zmin), ty(0.0), tx(zmax), ty(0.0), [0.6, 0.6, 0.6], 1.0);
    let sag = |s: &crate::lens::Surface, y: f64| s.sag(y);
    let surf = &lens.surfaces;
    for (i, s) in surf.iter().enumerate() {
        if s.is_stop {
            let r = lens.iris.max_radius;
            c.line(tx(s.z), ty(r), tx(s.z), ty(s.semi_diameter + 2.0), [0.0, 0.0, 0.0], 1.0);
            c.line(tx(s.z), ty(-r), tx(s.z), ty(-s.semi_diameter - 2.0), [0.0, 0.0, 0.0], 1.0);
            continue;
        }
        let n = 100;
        for k in 0..n {
            let y1 = -s.semi_diameter + 2.0 * s.semi_diameter * k as f64 / n as f64;
            let y2 = -s.semi_diameter + 2.0 * s.semi_diameter * (k + 1) as f64 / n as f64;
            c.line(tx(s.z + sag(s, y1)), ty(y1), tx(s.z + sag(s, y2)), ty(y2), [0.0, 0.0, 0.4], 1.0);
        }
        // Element edges.
        if s.material.is_some() && i + 1 < surf.len() {
            let t = &surf[i + 1];
            let sd = s.semi_diameter.max(t.semi_diameter);
            for sign in [-1.0, 1.0] {
                let za = s.z + sag(s, s.semi_diameter);
                let zb = t.z + sag(t, t.semi_diameter);
                c.line(tx(za), ty(sign * s.semi_diameter), tx(za), ty(sign * sd), [0.0, 0.0, 0.4], 1.0);
                c.line(tx(za), ty(sign * sd), tx(zb), ty(sign * sd), [0.0, 0.0, 0.4], 1.0);
                c.line(tx(zb), ty(sign * sd), tx(zb), ty(sign * t.semi_diameter), [0.0, 0.0, 0.4], 1.0);
            }
        }
    }
    c.line(tx(lens.sensor_z), ty(-ymax + 1.0), tx(lens.sensor_z), ty(ymax - 1.0), [0.0, 0.0, 0.0], 1.0);
    let p = lens.paraxial_data(LAMBDA_D);
    let r_ep = lens.iris.max_radius / p.entrance_pupil_mag;
    for (fi, fr) in [0.0, 0.7, 1.0].iter().enumerate() {
        let th = max_theta * fr;
        let col = [[0.0, 0.5, 0.0], [0.8, 0.0, 0.0], [0.0, 0.3, 0.9]][fi];
        let y0 = chief_ray_y0(lens, th, LAMBDA_D);
        let m = lens.media(LAMBDA_D);
        for k in 0..=10 {
            let dy = (k as f64 / 10.0 * 2.0 - 1.0) * r_ep * 1.3;
            let mut ray = Ray { o: Vec3::new(0.0, y0 + dy, START_Z), d: Vec3::new(0.0, th.sin(), th.cos()) };
            ray.o = ray.o + ray.d * ((zmin - START_Z) / ray.d.z);
            let mut pts = vec![ray.o];
            let mut cur = ray;
            let mut ok = true;
            for ev in &lens.forward_path {
                match lens.trace_path(cur, LAMBDA_D, &m, std::slice::from_ref(ev)) {
                    Some((r, _)) => {
                        pts.push(r.o);
                        cur = r;
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let t = (lens.sensor_z - cur.o.z) / cur.d.z;
            pts.push(cur.o + cur.d * t);
            for w in pts.windows(2) {
                c.line(tx(w[0].z), ty(w[0].y), tx(w[1].z), ty(w[1].y), col, 0.8);
            }
        }
    }
    c.save(path)
}


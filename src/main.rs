mod analysis;
mod canvas;
mod diffraction;
mod glass;
mod lens;
mod math;
mod render;
mod scene;
mod spectrum;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Physically based spectral simulation of photographic lenses")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render a scene through the lens.
    Render {
        scene: String,
        /// Output PNG; a linear EXR is written next to it.
        #[arg(short, long, default_value = "out.png")]
        out: String,
        /// Override settings, e.g. `--set fstop=8 --set spp=256`.
        #[arg(long = "set")]
        overrides: Vec<String>,
    },
    /// Print optical analysis of a lens prescription and draw diagnostics.
    Analyze {
        lens: String,
        #[arg(long, default_value_t = 0.0)]
        fstop: f64,
        /// Half-diagonal of the image format in mm (36x24: 21.63).
        #[arg(long, default_value_t = 21.63)]
        half_diag: f64,
        /// Write <prefix>-spots.png and <prefix>-layout.png.
        #[arg(long)]
        out: Option<String>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Analyze { lens, fstop, half_diag, out } => {
            let mut lens = lens::Lens::load(&lens)?;
            let f = if fstop > 0.0 { fstop } else { lens.max_aperture() };
            lens.set_aperture(f, 0, 0.0, 0.0)?;
            lens.sensor_z = lens.best_focus(None)?;
            print!("{}", analysis::report(&lens, half_diag, out.as_deref())?);
        }
        Cmd::Render { scene, out, overrides } => render(&scene, &out, &overrides)?,
    }
    Ok(())
}

fn render(scene_path: &str, out: &str, overrides: &[String]) -> Result<()> {
    let text = std::fs::read_to_string(scene_path)?;
    let mut doc: toml::Table = toml::from_str(&text)?;
    let cam = doc.get_mut("camera").and_then(|c| c.as_table_mut()).ok_or_else(|| anyhow::anyhow!("missing [camera]"))?;
    for o in overrides {
        let Some((k, v)) = o.split_once('=') else { bail!("override must be key=value: {o}") };
        let value: toml::Value = toml::from_str::<toml::Table>(&format!("v = {v}"))
            .map(|t| t["v"].clone())
            .unwrap_or_else(|_| toml::Value::String(v.to_string()));
        cam.insert(k.to_string(), value);
    }
    let file: scene::SceneFile = doc.try_into()?;
    let c = &file.camera;
    let base = Path::new(scene_path).parent().unwrap_or(Path::new("."));
    let cs = spectrum::ColorSystem::new();
    let scene = scene::Scene::build(&file, base, &cs)?;

    let mut lens = lens::Lens::load(base.join(&c.lens).to_str().unwrap())?;
    lens.coating = match c.coating.as_str() {
        "none" => lens::Coating::None,
        "single" => lens::Coating::Single { n: 1.38, lambda0: 550.0 },
        other => bail!("unknown coating {other}"),
    };
    for d in &c.decenter {
        lens.decenter_element(d[0] as usize, d[1], d[2])?;
    }
    let fstop = c.fstop.unwrap_or_else(|| lens.max_aperture());
    lens.set_aperture(fstop, c.iris_blades, c.iris_rotation_deg, c.iris_curvature)?;
    let focus = if c.focus_distance.is_finite() { Some(c.focus_distance * 1000.0) } else { None };
    lens.sensor_z = lens.best_focus(focus)? + c.focus_offset_mm;

    let w = c.width as usize;
    let h = (c.width as f64 * c.sensor_height_mm / c.sensor_width_mm).round() as usize;
    let norm = 4.0 * fstop * fstop / std::f64::consts::PI * 2f64.powf(c.exposure_ev);
    eprintln!(
        "{}: f = {:.2} mm, f/{:.2} (working f/{:.2}), focus {} (sensor at {:.3} mm), {}x{} px, {} spp",
        lens.name,
        lens.focal_length(),
        fstop,
        lens.working_fnumber(),
        focus.map_or("∞".to_string(), |d| format!("{:.3} m", d / 1000.0)),
        lens.sensor_z,
        w,
        h,
        c.spp
    );
    let t0 = Instant::now();
    let r = render::Renderer::new(&lens, &scene, &cs, w, h, c.sensor_width_mm, c.spectral_bins, norm);
    let mut film = render::Film::new(w, h, c.spectral_bins);
    r.render_surfaces(&mut film, c.spp, c.ghosts, true);
    eprintln!("  surfaces done in {:.1}s", t0.elapsed().as_secs_f64());
    let t1 = Instant::now();
    r.render_lights(&mut film, c.light_samples, c.ghost_grid, if c.ghosts { c.ghost_threshold } else { f64::INFINITY }, true);
    if !scene.lights.is_empty() {
        eprintln!("  lights done in {:.1}s", t1.elapsed().as_secs_f64());
    }
    let bad = film.data.iter().filter(|v| !v.is_finite() || **v < 0.0).count();
    if bad > 0 {
        eprintln!("  warning: {bad} invalid film values before diffraction");
    }
    if c.diffraction {
        let t2 = Instant::now();
        diffraction::apply(&mut film, &lens.iris, lens.surfaces[lens.stop_index].semi_diameter, lens.working_fnumber(), r.pitch);
        eprintln!("  diffraction done in {:.1}s", t2.elapsed().as_secs_f64());
    }
    let rgb = film.to_rgb(&cs);
    write_outputs(&rgb, w, h, out, &c.tonemap)?;
    eprintln!("wrote {out} in {:.1}s total", t0.elapsed().as_secs_f64());
    Ok(())
}

fn write_outputs(rgb: &[[f32; 3]], w: usize, h: usize, out: &str, tonemap: &str) -> Result<()> {
    let tm = |v: f32| -> f64 {
        let v = v.max(0.0) as f64;
        match tonemap {
            // ACES-like filmic curve (Narkowicz), scaled so mid grey stays put.
            "filmic" => {
                let x = v * 0.6;
                (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
            }
            _ => v,
        }
    };
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, p) in rgb.iter().enumerate() {
        // Like a sensor clipping in raw space, highlights far above white turn
        // white instead of keeping out-of-gamut hues.
        let y = (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).max(0.0) as f64;
        let t = ((y - 1.0) / 3.0).clamp(0.0, 1.0);
        let white = t * t * (3.0 - 2.0 * t);
        let px = p.map(|v| {
            let v = tm(v).clamp(0.0, 1.0);
            (spectrum::srgb_encode(v + (1.0 - v) * white) * 255.0 + 0.5) as u8
        });
        img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(px));
    }
    img.save(out)?;
    let exr_path = Path::new(out).with_extension("exr");
    let flat: Vec<f32> = rgb.iter().flat_map(|p| p.map(|v| v.max(0.0))).collect();
    image::Rgb32FImage::from_raw(w as u32, h as u32, flat).unwrap().save(&exr_path)?;
    Ok(())
}

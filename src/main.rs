mod analysis;
mod canvas;
mod demo;
mod diffraction;
mod explainer;
mod fieldpsf;
mod footprint;
mod ghosts;
mod glass;
mod gpu;
mod lens;
mod lensedit;
mod lensview;
mod math;
mod mtf;
mod offscreen;
mod plot;
mod psf;
mod psfbank;
mod render;
mod scene;
mod seidel;
mod spectrum;
mod viewer;

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
        /// Use the CPU reference renderer instead of the GPU.
        #[arg(long)]
        cpu: bool,
    },
    /// Open an interactive, progressively rendered view of a scene.
    View {
        scene: String,
        /// Override settings, e.g. `--set fstop=8`.
        #[arg(long = "set")]
        overrides: Vec<String>,
    },
    /// Record the explainer video's frames from `<dir>/timeline.json` (see
    /// scripts/explainer_video.py) into `<dir>/frames`.
    Explain { dir: String },
    /// Plot MTF curves (wave optics, white light) of a lens.
    Mtf {
        lens: String,
        /// f-numbers to plot (wide open if empty).
        #[arg(long, value_delimiter = ',')]
        fstop: Vec<f64>,
        /// Focus distance in m (infinity if absent).
        #[arg(long)]
        focus: Option<f64>,
        /// Iris blades (0 = round; default: the lens').
        #[arg(long)]
        blades: Option<u32>,
        #[arg(long, default_value_t = 21.63)]
        half_diag: f64,
        /// Focal length in mm of a zoom lens (default: its widest).
        #[arg(long)]
        focal: Option<f64>,
        /// Published MTF data (JSON) to overlay.
        #[arg(long)]
        reference: Option<String>,
        #[arg(long, default_value = "mtf")]
        out: String,
    },
    /// Print optical analysis of a lens prescription and draw diagnostics.
    Analyze {
        lens: String,
        #[arg(long, default_value_t = 0.0)]
        fstop: f64,
        /// Half-diagonal of the image format in mm (36x24: 21.63).
        #[arg(long, default_value_t = 21.63)]
        half_diag: f64,
        /// Focal length in mm of a zoom lens (default: its widest).
        #[arg(long)]
        focal: Option<f64>,
        /// Write <prefix>-spots.png and <prefix>-layout.png.
        #[arg(long)]
        out: Option<String>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Analyze { lens, fstop, half_diag, focal, out } => {
            let mut lens = lens::Lens::load(&lens)?;
            if let Some(f) = focal {
                lens.set_zoom(f);
            }
            let f = if fstop > 0.0 { fstop } else { lens.max_aperture() };
            lens.set_aperture(f, 0, 0.0, 0.0)?;
            lens.sensor_z = lens.best_focus(None)?;
            print!("{}", analysis::report(&lens, half_diag, out.as_deref())?);
        }
        Cmd::Mtf { lens, fstop, focus, blades, half_diag, focal, reference, out } => {
            let mut lens = lens::Lens::load(&lens)?;
            if let Some(f) = focal {
                lens.set_zoom(f);
            }
            let fstops = if fstop.is_empty() { vec![lens.max_aperture()] } else { fstop };
            let reference: Option<serde_json::Value> = match reference {
                Some(p) => Some(serde_json::from_str(&std::fs::read_to_string(&p)?)?),
                None => None,
            };
            print!("{}", mtf::run(&lens, &fstops, blades.unwrap_or(lens.blades), focus.map(|f| f * 1000.0), half_diag, reference.as_ref(), &out)?);
        }
        Cmd::Explain { dir } => explainer::record(Path::new(&dir))?,
        Cmd::Render { scene, out, overrides, cpu } => render(&scene, &out, &overrides, cpu)?,
        Cmd::View { scene, overrides } => {
            let Setup { file, scene, lens_path, raw_lens, fstop, focus, .. } = setup(&scene, &overrides)?;
            if std::env::var_os("LENS_VIEWER_DEMO").is_some() {
                viewer::record_demo(file, scene, lens_path, raw_lens, fstop, focus)?;
            } else {
                viewer::run(file, scene, lens_path, raw_lens, fstop, focus)?;
            }
        }
    }
    Ok(())
}

/// A scene file with its lens set up (aperture, focus) as the camera says.
struct Setup {
    file: scene::SceneFile,
    scene: scene::Scene,
    lens: lens::Lens,
    lens_path: std::path::PathBuf,
    /// The lens before aperture and focus are set.
    raw_lens: lens::Lens,
    cs: spectrum::ColorSystem,
    fstop: f64,
    focus: Option<f64>,
    w: usize,
    h: usize,
    norm: f64,
}

fn setup(scene_path: &str, overrides: &[String]) -> Result<Setup> {
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

    let lens_path = base.join(&c.lens);
    let mut lens = lens::Lens::load(lens_path.to_str().unwrap())?;
    if c.coating != "lens" {
        lens.coating = lens::Coating::parse(&c.coating)?;
    }
    lens.sensor_reflectance = c.sensor_reflectance;
    lens.sensor_half = (c.sensor_width_mm / 2.0, c.sensor_height_mm / 2.0);
    for d in &c.decenter {
        lens.decenter_element(d[0] as usize, d[1], d[2])?;
    }
    if let Some(f) = c.focal_length_mm {
        lens.set_zoom(f);
    }
    let raw_lens = lens.clone();
    let fstop = c.fstop.unwrap_or_else(|| lens.max_aperture());
    let focus = if c.focus_distance.is_finite() { Some(c.focus_distance * 1000.0) } else { None };
    let (blades, curvature) = (c.iris_blades.unwrap_or(lens.blades), c.iris_curvature.unwrap_or(lens.blade_curvature));
    lens.prepare(fstop, blades, c.iris_rotation_deg, curvature, focus, c.focus_offset_mm)?;

    let w = c.width as usize;
    let h = (c.width as f64 * c.sensor_height_mm / c.sensor_width_mm).round() as usize;
    let norm = 4.0 * fstop * fstop / std::f64::consts::PI * 2f64.powf(c.exposure_ev);
    Ok(Setup { file, scene, lens, lens_path, raw_lens, cs, fstop, focus, w, h, norm })
}

fn render(scene_path: &str, out: &str, overrides: &[String], cpu: bool) -> Result<()> {
    let Setup { file, scene, lens, cs, fstop, focus, w, h, norm, .. } = setup(scene_path, overrides)?;
    let c = &file.camera;
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
    let pitch = c.sensor_width_mm / w as f64;
    let gpu = if cpu {
        None
    } else {
        match gpu::GpuContext::headless() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("  no GPU ({e}), using the CPU renderer");
                None
            }
        }
    };
    let mut gpu_diffracted = false;
    let mut film = if let Some(ctx) = gpu {
        let mut g = gpu::GpuRenderer::new(ctx.clone());
        g.set_scene(&scene);
        let settings = gpu_settings(&file, &lens, w, h, norm);
        let bank = if c.wave_optics {
            let tb = Instant::now();
            let half_diag = 0.5 * c.sensor_width_mm.hypot(c.sensor_height_mm);
            let b = psfbank::PsfBank::build(&lens, focus, c.spectral_bins, pitch, half_diag, Some(&gpu::wave::WaveGpu::new(&ctx)), &|_| {});
            eprintln!(
                "  PSF bank: {} PSFs ({}×{}×{}×{} bins), kernels {}² px, built in {:.1} s",
                b.refs.len(),
                b.n_theta,
                b.n_verg,
                b.n_az,
                b.bins,
                2 * b.half + 1,
                tb.elapsed().as_secs_f64()
            );
            Some(b)
        } else {
            None
        };
        g.configure(&lens, &settings, bank.as_ref())?;
        g.wait()?;
        let t1 = Instant::now();
        eprintln!("  GPU setup {:.0} ms", (t1 - t0).as_secs_f64() * 1000.0);
        if std::env::var("LENS_PROFILE").is_ok() {
            let (s_ms, l_ms) = g.profile(10)?;
            eprintln!("  profile: surfaces {s_ms:.1} ms, lights {l_ms:.1} ms per frame");
            g.reset();
        }
        // Lights are sampled at one point of their area and one wavelength per bin
        // each frame; a few frames are needed for them to converge.
        let min_frames = if scene.lights.is_empty() { 1 } else { 16 };
        let frames = c.spp.div_ceil(c.spectral_bins as u32 * settings.spp_per_frame).max(min_frames);
        for _ in 0..frames {
            g.render_frame();
        }
        g.wait()?;
        let ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  {frames} GPU frames in {ms:.0} ms ({:.1} ms/frame)", ms / frames as f64);
        let t2 = Instant::now();
        gpu_diffracted = g.resolve(c.diffraction)?;
        g.wait()?;
        if gpu_diffracted {
            eprintln!("  GPU diffraction done in {:.0} ms", t2.elapsed().as_secs_f64() * 1000.0);
        }
        let film = g.read_film()?;
        if std::env::var("LENS_PROFILE").is_ok() {
            for diffract in [true, false] {
                let t3 = Instant::now();
                g.resolve(diffract)?;
                g.wait()?;
                eprintln!("  profile: resolve (diffraction {diffract}) {:.1} ms", t3.elapsed().as_secs_f64() * 1000.0);
            }
        }
        film
    } else {
        let r = render::Renderer::new(&lens, &scene, &cs, w, h, c.sensor_width_mm, c.spectral_bins, norm);
        let mut film = render::Film::new(w, h, c.spectral_bins);
        r.render_surfaces(&mut film, c.spp, c.ghosts, true);
        eprintln!("  surfaces done in {:.1}s", t0.elapsed().as_secs_f64());
        let t1 = Instant::now();
        r.render_lights(&mut film, c.light_samples, c.ghost_grid, if c.ghosts { c.ghost_threshold } else { f64::INFINITY }, true);
        if !scene.lights.is_empty() {
            eprintln!("  lights done in {:.1}s", t1.elapsed().as_secs_f64());
        }
        film
    };
    // Wave-optical PSFs include diffraction.
    if c.diffraction && !gpu_diffracted && !c.wave_optics {
        let t2 = Instant::now();
        diffraction::apply(&mut film, &lens.iris, lens.surfaces[lens.stop_index].semi_diameter, lens.working_fnumber(), pitch);
        eprintln!("  diffraction done in {:.1}s", t2.elapsed().as_secs_f64());
    }
    let rgb = film.to_rgb(&cs);
    write_outputs(&rgb, w, h, out, &c.tonemap)?;
    eprintln!("wrote {out} in {:.1}s total", t0.elapsed().as_secs_f64());
    Ok(())
}

fn gpu_settings(file: &scene::SceneFile, lens: &lens::Lens, w: usize, h: usize, norm: f64) -> gpu::Settings {
    let c = &file.camera;
    gpu::Settings {
        w,
        h,
        bins: c.spectral_bins,
        sensor_w: c.sensor_width_mm,
        norm,
        spp_per_frame: 1,
        ghosts: c.ghosts,
        ghost_threshold: c.ghost_threshold,
        light_grid: gpu::ghost_grid(lens, c.ghost_grid),
        ghost_wavelengths: 0,
        tonemap: if c.tonemap == "filmic" { gpu::Tonemap::Filmic } else { gpu::Tonemap::Clip },
    }
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

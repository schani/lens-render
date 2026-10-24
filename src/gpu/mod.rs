//! GPU renderer: the same physics as the CPU reference (`render.rs`), as
//! progressive wgpu compute passes. Lenses of any number of surfaces and
//! glasses are uploaded generically (surfaces, per-medium index tables and
//! event paths), so nothing here is specific to one design.

use crate::lens::{Coating, Lens};
use crate::render::Film;
use crate::scene::{Scene, Texture};
use crate::spectrum::{ColorSystem, LAMBDA_MIN, LAMBDA_RANGE, cie_xyz, srgb_encode};
use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
mod diffraction;

pub use eframe::wgpu;
use wgpu::util::DeviceExt;

const BOUNDS_G: u32 = 200;
const LIGHT_BOUNDS_G: u32 = 128;
const LIGHT_BOUNDS_BASE: usize = 2048;
const MAX_BOUNDS_NODES: usize = 512;
const MAX_BINS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    w: u32,
    h: u32,
    bins: u32,
    spp: u32,
    frame: u32,
    n_surf: u32,
    n_rects: u32,
    n_lights: u32,
    bounds_mode: u32,
    bounds_nodes: u32,
    grid_nx: u32,
    grid_ny: u32,
    bounds_g: u32,
    n_ghosts: u32,
    veil: u32,
    coating: u32,
    iris_blades: u32,
    iris_circular: u32,
    light_grid: u32,
    override_iris: u32,
    pitch: f32,
    sensor_z: f32,
    rear_z: f32,
    norm: f32,
    r_max: f32,
    half_w: f32,
    half_h: f32,
    bounds_half: f32,
    iris_rotation: f32,
    iris_curvature: f32,
    iris_apothem: f32,
    iris_max_radius: f32,
    coat_n: f32,
    coat_l0: f32,
    front_z: f32,
    ep_z: f32,
    front_half: f32,
    light_bounds_g: u32,
    frames: f32,
    tonemap: u32,
    a: [u32; 5],
    media_off: u32,
    up_off: u32,
    spectra_off: u32,
    palette_off: u32,
    cmf_off: u32,
    path_stride: u32,
    path_base: u32,
    light_base: u32,
    bg: [f32; 3],
    m: [f32; 9],
    _pad: [u32; 3],
}

/// Device and queue, either our own (headless) or borrowed from the viewer.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Limits we ask for: whatever the adapter offers (large buffers, many bindings).
pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let wanted = wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::TIMESTAMP_QUERY;
    wgpu::DeviceDescriptor {
        label: Some("lens-render"),
        required_features: adapter.features() & wanted,
        required_limits: adapter.limits(),
        ..Default::default()
    }
}

/// Shader source with the film accumulation specialised for the device:
/// native float atomics where available, a compare-and-swap loop otherwise.
fn shader_source(device: &wgpu::Device, body: &str) -> String {
    let native = device.features().contains(wgpu::Features::SHADER_FLOAT32_ATOMIC);
    let (ty, add, load) = if native {
        ("atomic<f32>", "atomicAdd(&film[idx], v);", "atomicLoad(&film[i])")
    } else {
        (
            "atomic<u32>",
            "var old = atomicLoad(&film[idx]);
    loop {
        let r = atomicCompareExchangeWeak(&film[idx], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }",
            "bitcast<f32>(atomicLoad(&film[i]))",
        )
    };
    let low_add = add.replace("film[", "lowres[");
    let low_load = load.replace("film[", "lowres[");
    format!("{}\n{}", include_str!("common.wgsl"), body)
        .replace("FILM_ATOMIC", ty)
        .replace("LOWRES_ADD", &low_add)
        .replace("LOWRES_LOAD", &low_load)
        .replace("FILM_ADD", add)
        .replace("FILM_LOAD", load)
}

impl GpuContext {
    pub fn headless() -> Result<GpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;
        let (device, queue) = pollster::block_on(adapter.request_device(&device_descriptor(&adapter)))?;
        Ok(GpuContext { device, queue })
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tonemap {
    Clip,
    Filmic,
}

/// Per-render settings that change the sampling but not the scene.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub w: usize,
    pub h: usize,
    pub bins: usize,
    pub sensor_w: f64,
    /// Irradiance scale (4N²/π · 2^EV).
    pub norm: f64,
    /// Samples per wavelength bin and pixel in each frame.
    pub spp_per_frame: u32,
    pub ghosts: bool,
    pub ghost_threshold: f64,
    /// Ray grid per light, path and bin (multiple of 16).
    pub light_grid: u32,
    /// Wavelengths per ghost path and frame (stratified, weighted); 0 = every bin.
    pub ghost_wavelengths: usize,
    pub tonemap: Tonemap,
}

struct SceneData {
    rects: Vec<[f32; 4]>,
    n_rects: usize,
    lights: Vec<[f32; 4]>,
    n_lights: usize,
    spectra: Vec<f32>,
    palettes: Vec<f32>,
    images: Vec<u32>,
    background: [f32; 3],
    /// Irradiance at the lens per light (for the ghost threshold).
    power: Vec<f64>,
}

pub struct GpuRenderer {
    ctx: GpuContext,
    surfaces_pl: wgpu::ComputePipeline,
    veil_pl: wgpu::ComputePipeline,
    light_grid_pl: wgpu::ComputePipeline,
    pupil_bounds_pl: wgpu::ComputePipeline,
    light_bounds_pl: wgpu::ComputePipeline,
    normalize_pl: wgpu::ComputePipeline,
    to_rgb_pl: wgpu::ComputePipeline,
    layouts: [wgpu::BindGroupLayout; 3],
    params_buf: wgpu::Buffer,
    bounds_buf: wgpu::Buffer,
    jobs_buf: wgpu::Buffer,
    jobs_cap: usize,
    static_bufs: Option<[wgpu::Buffer; 4]>,
    film_buf: Option<wgpu::Buffer>,
    lowres_buf: Option<wgpu::Buffer>,
    spectral_buf: Option<wgpu::Buffer>,
    texture: Option<wgpu::Texture>,
    g0: Option<wgpu::BindGroup>,
    g1: Option<wgpu::BindGroup>,
    g2: Option<wgpu::BindGroup>,
    scene: Option<SceneData>,
    cs: ColorSystem,
    params: Params,
    settings: Option<Settings>,
    n_ghosts: usize,
    frames: u32,
    /// Iris, stop radius and working f-number of the configured lens.
    optics: Option<(crate::lens::Iris, f64, f64)>,
    diffraction: Option<diffraction::GpuDiffraction>,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Encodes an event list for the lens blob.
fn encode_path(p: &[crate::lens::Event]) -> Vec<u32> {
    let mut v = vec![p.len() as u32];
    v.extend(p.iter().map(|e| e.surface as u32 | (e.forward as u32) << 8 | (e.reflect as u32) << 9));
    v
}

impl GpuRenderer {
    pub fn new(ctx: GpuContext) -> GpuRenderer {
        let device = &ctx.device;
        let trace = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace"),
            source: wgpu::ShaderSource::Wgsl(shader_source(device, include_str!("trace.wgsl")).into()),
        });
        let resolve = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resolve"),
            source: wgpu::ShaderSource::Wgsl(shader_source(device, include_str!("resolve.wgsl")).into()),
        });
        let g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("static"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, true),
            ],
        });
        let g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("film"),
            entries: &[storage_entry(0, false), storage_entry(1, false)],
        });
        let g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("output"),
            entries: &[
                storage_entry(0, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&g0), Some(&g1)],
            immediate_size: 0,
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&g0), Some(&g1), Some(&g2)],
            immediate_size: 0,
        });
        let pipeline = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bounds_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bounds"),
            size: 4 * 4 * (MAX_BOUNDS_NODES as u64 + 4096),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let jobs_cap = 1024;
        let jobs_buf = Self::make_jobs_buf(device, jobs_cap);
        GpuRenderer {
            surfaces_pl: pipeline(&trace_layout, &trace, "surfaces"),
            veil_pl: pipeline(&trace_layout, &trace, "veil"),
            light_grid_pl: pipeline(&trace_layout, &trace, "light_grid"),
            pupil_bounds_pl: pipeline(&trace_layout, &trace, "pupil_bounds"),
            light_bounds_pl: pipeline(&trace_layout, &trace, "light_bounds"),
            normalize_pl: pipeline(&resolve_layout, &resolve, "normalize_film"),
            to_rgb_pl: pipeline(&resolve_layout, &resolve, "to_rgb"),
            layouts: [g0, g1, g2],
            params_buf,
            bounds_buf,
            jobs_buf,
            jobs_cap,
            static_bufs: None,
            film_buf: None,
            lowres_buf: None,
            spectral_buf: None,
            texture: None,
            g0: None,
            g1: None,
            g2: None,
            scene: None,
            cs: ColorSystem::new(),
            params: Params::default(),
            settings: None,
            n_ghosts: 0,
            frames: 0,
            optics: None,
            diffraction: None,
            ctx,
        }
    }

    fn make_jobs_buf(device: &wgpu::Device, cap: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jobs"),
            size: (cap * 48) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn frames(&self) -> u32 {
        self.frames
    }

    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.texture.as_ref()
    }

    /// Uploads the scene (rectangles, textures, lights).
    pub fn set_scene(&mut self, scene: &Scene) {
        let cs = &self.cs;
        let v4 = |v: crate::math::Vec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
        let mut rects = Vec::new();
        let mut palettes: Vec<f32> = Vec::new();
        let mut images: Vec<u32> = Vec::new();
        for r in &scene.rects {
            let (kind, p0, p1, a, b, info) = match &r.texture {
                Texture::Color(c) => (0, 0.0, 0.0, *c, [0.0; 3], [0, 0, 0]),
                Texture::Image { w, h, data } => {
                    let off = images.len() as u32;
                    images.extend(data.iter().map(|p| {
                        let e = |v: f32| (srgb_encode(v as f64) * 255.0).round() as u32;
                        e(p[0]) | e(p[1]) << 8 | e(p[2]) << 16
                    }));
                    (1, 0.0, 0.0, [0.0; 3], [0.0; 3], [off, *w as u32, *h as u32])
                }
                Texture::Checker { n, a, b } => (2, *n, 0.0, *a, *b, [0, 0, 0]),
                Texture::Grid { lines, width } => (3, *lines, *width, [0.0; 3], [0.0; 3], [0, 0, 0]),
                Texture::Chart => (4, 0.0, 0.0, [0.0; 3], [0.0; 3], [0, 0, 0]),
                Texture::Noise { scale, palette, seed } => {
                    let off = (palettes.len() / 3) as u32;
                    palettes.extend(palette.iter().flat_map(|c| c.iter().copied()));
                    (5, *scale, 0.0, [0.0; 3], [0.0; 3], [off, palette.len() as u32, *seed])
                }
            };
            rects.push(v4(r.center, r.w));
            rects.push(v4(r.u, r.h));
            rects.push(v4(r.v, r.brightness));
            rects.push(v4(r.n, p0));
            rects.push([a[0], a[1], a[2], p1 as f32]);
            rects.push([b[0], b[1], b[2], 0.0]);
            rects.push(bytemuck::cast([kind as u32, info[0], info[1], info[2]]));
        }
        let mut lights = Vec::new();
        let mut spectra = Vec::new();
        let mut power = Vec::new();
        for l in &scene.lights {
            let n_e = (-l.center).normalize();
            let helper = if n_e.x.abs() < 0.9 { crate::math::Vec3::new(1.0, 0.0, 0.0) } else { crate::math::Vec3::new(0.0, 1.0, 0.0) };
            let t1 = n_e.cross(helper).normalize();
            let t2 = n_e.cross(t1);
            lights.push(v4(l.center, l.radius));
            lights.push(v4(t1, 0.0));
            lights.push(v4(t2, 0.0));
            spectra.extend((0..=400).map(|i| l.spectrum.eval(cs, LAMBDA_MIN + i as f64) as f32));
            power.push(l.radius * l.radius / l.center.dot(l.center) * l.spectrum.eval(cs, 555.0));
        }
        self.scene = Some(SceneData {
            n_rects: scene.rects.len(),
            n_lights: scene.lights.len(),
            rects,
            lights,
            spectra,
            palettes,
            images,
            background: scene.background,
            power,
        });
        self.settings = None;
    }

    /// Uploads the lens and settings, recomputes pupil bounds and restarts accumulation.
    pub fn configure(&mut self, lens: &Lens, s: &Settings) -> Result<()> {
        if s.bins > MAX_BINS || s.light_grid % 16 != 0 || s.light_grid == 0 {
            bail!("unsupported settings: at most {MAX_BINS} bins, light grid a multiple of 16");
        }
        let scene = self.scene.as_ref().context("set_scene first")?;
        if scene.n_lights > 4096 {
            bail!("at most 4096 lights");
        }
        let device = &self.ctx.device;
        let n = lens.surfaces.len();

        // Lens blob: surfaces, then paths.
        let mut lens_blob: Vec<u32> = Vec::new();
        for srf in &lens.surfaces {
            let c = if srf.radius == 0.0 { 0.0 } else { 1.0 / srf.radius };
            lens_blob.extend([
                (srf.z as f32).to_bits(),
                (c as f32).to_bits(),
                (srf.semi_diameter as f32).to_bits(),
                srf.is_stop as u32,
                (srf.decenter.0 as f32).to_bits(),
                (srf.decenter.1 as f32).to_bits(),
                0,
                0,
            ]);
        }
        let ghosts = lens.ghost_paths();
        let ghosts_back = lens.ghost_paths_backward();
        let mut paths = vec![encode_path(&lens.forward_path), encode_path(&lens.backward_path)];
        paths.extend(ghosts.iter().map(|g| encode_path(&g.2)));
        paths.extend(ghosts_back.iter().map(|g| encode_path(g)));
        let stride = paths.iter().map(|p| p.len()).max().unwrap();
        let path_base = lens_blob.len();
        for p in &paths {
            let mut p = p.clone();
            p.resize(stride, 0);
            lens_blob.extend(p);
        }

        // Tables: media, upsampling, per-bin CMF, light spectra, palettes.
        let mut tables: Vec<f32> = Vec::new();
        let media_off = tables.len();
        for m in 0..=n {
            for i in 0..=400 {
                let media = lens.media(LAMBDA_MIN + i as f64);
                tables.push(media.n[m] as f32);
            }
        }
        let up_off = tables.len();
        for i in 0..=400 {
            tables.extend(self.cs.up_vec(LAMBDA_MIN + i as f64).map(|v| v as f32));
        }
        let cmf_off = tables.len();
        let width = LAMBDA_RANGE / s.bins as f64;
        for b in 0..s.bins {
            let mut acc = [0.0; 3];
            for i in 0..64 {
                let c = cie_xyz(LAMBDA_MIN + width * (b as f64 + (i as f64 + 0.5) / 64.0));
                for k in 0..3 {
                    acc[k] += c[k] / 64.0;
                }
            }
            tables.extend(acc.map(|v| v as f32));
        }
        let spectra_off = tables.len();
        tables.extend(&scene.spectra);
        let palette_off = tables.len();
        tables.extend(&scene.palettes);

        let mut scene_blob = scene.rects.clone();
        let light_base = scene_blob.len();
        scene_blob.extend(&scene.lights);
        let pad4 = |mut v: Vec<u32>| {
            if v.is_empty() {
                v.push(0);
            }
            v
        };
        let mk = |label: &str, data: &[u8]| {
            let mut bytes = data.to_vec();
            bytes.resize(bytes.len().max(16), 0);
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let bufs = [
            mk("lens", bytemuck::cast_slice(&lens_blob)),
            mk("tables", bytemuck::cast_slice(&tables)),
            mk("scene", bytemuck::cast_slice(&scene_blob)),
            mk("images", bytemuck::cast_slice(&pad4(scene.images.clone()))),
        ];

        // Film and output, reallocated on size change.
        let same_size = self.settings.as_ref().is_some_and(|o| o.w == s.w && o.h == s.h && o.bins == s.bins);
        if !same_size || self.film_buf.is_none() {
            let size = (s.w * s.h * s.bins * 4) as u64;
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
            self.film_buf = Some(device.create_buffer(&wgpu::BufferDescriptor { label: Some("film"), size, usage, mapped_at_creation: false }));
            let low = ((s.w.div_ceil(4) * s.h.div_ceil(4) * s.bins * 4) as u64).max(16);
            self.lowres_buf =
                Some(device.create_buffer(&wgpu::BufferDescriptor { label: Some("lowres"), size: low, usage, mapped_at_creation: false }));
            self.spectral_buf =
                Some(device.create_buffer(&wgpu::BufferDescriptor { label: Some("spectral"), size, usage, mapped_at_creation: false }));
            self.texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("image"),
                size: wgpu::Extent3d { width: s.w as u32, height: s.h as u32, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
        }

        // Parameters.
        let pitch = s.sensor_w / s.w as f64;
        let half_w = s.w as f64 * pitch / 2.0;
        let half_h = s.h as f64 * pitch / 2.0;
        let par = lens.paraxial_data(crate::glass::LAMBDA_D);
        let iris = &lens.iris;
        let front_z = lens
            .surfaces
            .iter()
            .map(|srf| {
                let r = srf.semi_diameter;
                let sag = if srf.radius == 0.0 { 0.0 } else { srf.radius - srf.radius.signum() * (srf.radius * srf.radius - r * r).max(0.0).sqrt() };
                srf.z + sag.min(0.0)
            })
            .fold(0.0, f64::min)
            - 1.0;
        let m = self.cs.rgb_matrix();
        let centered = lens.is_centered();
        let (nx, ny) = (25u32, 17u32);
        self.n_ghosts = ghosts.len();
        self.params = Params {
            w: s.w as u32,
            h: s.h as u32,
            bins: s.bins as u32,
            spp: s.spp_per_frame,
            frame: 0,
            n_surf: n as u32,
            n_rects: scene.n_rects as u32,
            n_lights: scene.n_lights as u32,
            bounds_mode: if centered { 0 } else { 1 },
            bounds_nodes: if centered { 64 } else { nx * ny },
            grid_nx: nx,
            grid_ny: ny,
            bounds_g: BOUNDS_G,
            n_ghosts: ghosts.len() as u32,
            veil: (s.ghosts && !ghosts.is_empty()) as u32,
            coating: matches!(lens.coating, Coating::Single { .. }) as u32,
            iris_blades: iris.blades,
            iris_circular: iris.circular as u32,
            light_grid: s.light_grid,
            override_iris: 0,
            pitch: pitch as f32,
            sensor_z: lens.sensor_z as f32,
            rear_z: lens.last_z() as f32,
            norm: s.norm as f32,
            r_max: (half_w.hypot(half_h) * 1.01) as f32,
            half_w: half_w as f32,
            half_h: half_h as f32,
            bounds_half: (1.5 * lens.surfaces.last().unwrap().semi_diameter) as f32,
            iris_rotation: iris.rotation as f32,
            iris_curvature: iris.curvature as f32,
            iris_apothem: iris.apothem() as f32,
            iris_max_radius: iris.max_radius as f32,
            coat_n: match lens.coating {
                Coating::Single { n, .. } => n as f32,
                Coating::None => 1.0,
            },
            coat_l0: match lens.coating {
                Coating::Single { lambda0, .. } => lambda0 as f32,
                Coating::None => 550.0,
            },
            front_z: front_z as f32,
            ep_z: par.entrance_pupil_z as f32,
            front_half: (1.5 * lens.surfaces[0].semi_diameter) as f32,
            light_bounds_g: LIGHT_BOUNDS_G,
            frames: 1.0,
            tonemap: (s.tonemap == Tonemap::Filmic) as u32,
            a: crate::math::R5::new().alphas_fixed(),
            media_off: media_off as u32,
            up_off: up_off as u32,
            spectra_off: spectra_off as u32,
            palette_off: palette_off as u32,
            cmf_off: cmf_off as u32,
            path_stride: stride as u32,
            path_base: path_base as u32,
            light_base: light_base as u32,
            bg: scene.background,
            m: [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]].map(|v| v as f32),
            _pad: [0; 3],
        };

        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layouts[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bufs[3].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bounds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.jobs_buf.as_entire_binding() },
            ],
        });
        self.g0 = Some(g0);
        self.static_bufs = Some(bufs);
        self.g1 = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layouts[1],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.film_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.lowres_buf.as_ref().unwrap().as_entire_binding() },
            ],
        }));
        let view = self.texture.as_ref().unwrap().create_view(&Default::default());
        self.g2 = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layouts[2],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.spectral_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        }));
        self.settings = Some(s.clone());
        self.optics = Some((lens.iris.clone(), lens.surfaces[lens.stop_index].semi_diameter, lens.working_fnumber()));
        self.compute_bounds();
        self.reset();
        Ok(())
    }

    /// Restarts accumulation (keeps lens, scene and bounds).
    pub fn reset(&mut self) {
        self.frames = 0;
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.clear_buffer(self.film_buf.as_ref().unwrap(), 0, None);
        enc.clear_buffer(self.lowres_buf.as_ref().unwrap(), 0, None);
        self.ctx.queue.submit([enc.finish()]);
    }

    fn compute_bounds(&mut self) {
        let mut init = vec![[i32::MAX, i32::MAX, -i32::MAX, -i32::MAX]; MAX_BOUNDS_NODES + 4096];
        init.truncate(MAX_BOUNDS_NODES + 4096);
        self.ctx.queue.write_buffer(&self.bounds_buf, 0, bytemuck::cast_slice(&init));
        let mut p = self.params;
        p.override_iris = 1;
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, self.g0.as_ref(), &[]);
            pass.set_bind_group(1, self.g1.as_ref(), &[]);
            pass.set_pipeline(&self.pupil_bounds_pl);
            pass.dispatch_workgroups((BOUNDS_G * BOUNDS_G).div_ceil(64), p.bounds_nodes, 1);
            if p.n_lights > 0 {
                pass.set_pipeline(&self.light_bounds_pl);
                pass.dispatch_workgroups((LIGHT_BOUNDS_G * LIGHT_BOUNDS_G).div_ceil(64), p.n_lights, 1);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        let _ = LIGHT_BOUNDS_BASE;
    }

    /// Light jobs for one frame: per light the direct image in every
    /// wavelength bin, plus ghosts for lights bright enough, with fresh jitter.
    /// Ghosts may use fewer, stratified wavelengths, weighted to stay unbiased.
    fn jobs(&self) -> Vec<[u32; 12]> {
        let s = self.settings.as_ref().unwrap();
        let scene = self.scene.as_ref().unwrap();
        let max_power = scene.power.iter().cloned().fold(0.0, f64::max);
        let ghost_k = if s.ghost_wavelengths == 0 { s.bins } else { s.ghost_wavelengths.min(s.bins) };
        let mut jobs = Vec::new();
        for li in 0..scene.n_lights {
            let ghosts = if s.ghosts && scene.power[li] >= s.ghost_threshold * max_power { self.n_ghosts } else { 0 };
            for path in 0..=ghosts {
                let samples = if path == 0 { s.bins } else { ghost_k };
                for k in 0..samples {
                    // Jitter depends only on (frame, light, path, sample), so jobs are reproducible.
                    let key = ((self.frames as u64) << 40) ^ ((li as u64) << 20) ^ ((path as u64) << 8) ^ k as u64;
                    let mut rng = crate::math::Rng::new(key, 0x1e5);
                    let mut r = || rng.next_f64() as f32;
                    let u = (k as f32 + r()) / samples as f32 * s.bins as f32;
                    let bin = (u as usize).min(s.bins - 1);
                    let (ex, ey) = crate::math::concentric_disk(r() as f64, r() as f64);
                    let path_index = if path == 0 { 0 } else { 1 + path as u32 };
                    let weight = s.bins as f32 / samples as f32;
                    let f = [u - bin as f32, ex as f32, ey as f32, r(), r(), weight];
                    jobs.push([
                        li as u32,
                        path_index,
                        bin as u32,
                        (path > 0) as u32,
                        f[0].to_bits(),
                        f[1].to_bits(),
                        f[2].to_bits(),
                        f[3].to_bits(),
                        f[4].to_bits(),
                        f[5].to_bits(),
                        0,
                        0,
                    ]);
                }
            }
        }
        jobs
    }

    /// Adds one progressive frame.
    pub fn render_frame(&mut self) {
        self.render_frame_parts(true, true);
    }

    /// Wall time in ms of the surface and light passes, averaged over `n` frames.
    pub fn profile(&mut self, n: u32) -> Result<(f64, f64)> {
        let mut t = [0.0; 2];
        for (k, (surf, lights)) in [(true, false), (false, true)].into_iter().enumerate() {
            self.wait()?;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                self.render_frame_parts(surf, lights);
            }
            self.wait()?;
            t[k] = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        }
        Ok((t[0], t[1]))
    }

    fn render_frame_parts(&mut self, surfaces: bool, lights: bool) {
        let jobs = self.jobs();
        if jobs.len() > self.jobs_cap {
            self.jobs_cap = jobs.len().next_power_of_two();
            self.jobs_buf = Self::make_jobs_buf(&self.ctx.device, self.jobs_cap);
            // Rebind the static group with the new jobs buffer.
            let bufs = self.static_bufs.as_ref().unwrap();
            self.g0 = Some(self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layouts[0],
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bufs[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bufs[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: bufs[2].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: bufs[3].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.bounds_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.jobs_buf.as_entire_binding() },
                ],
            }));
        }
        if !jobs.is_empty() {
            self.ctx.queue.write_buffer(&self.jobs_buf, 0, bytemuck::cast_slice(&jobs));
        }
        let mut p = self.params;
        p.frame = self.frames;
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, self.g0.as_ref(), &[]);
            pass.set_bind_group(1, self.g1.as_ref(), &[]);
            if surfaces {
                pass.set_pipeline(&self.surfaces_pl);
                pass.dispatch_workgroups(p.w.div_ceil(8), p.h.div_ceil(8), 1);
                if p.veil != 0 {
                    pass.set_pipeline(&self.veil_pl);
                    pass.dispatch_workgroups(p.w.div_ceil(4).div_ceil(8), p.h.div_ceil(4).div_ceil(8), 1);
                }
            }
            if lights && !jobs.is_empty() {
                pass.set_pipeline(&self.light_grid_pl);
                let tiles = (p.light_grid / 16) * (p.light_grid / 16);
                // jobs are indexed by workgroup y; split into chunks of the dispatch limit
                let max_y = 65535usize;
                assert!(jobs.len() <= max_y, "too many light jobs");
                pass.dispatch_workgroups(tiles, jobs.len() as u32, 1);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        self.frames += 1;
    }

    /// Builds the diffraction kernels for the current lens if needed; false
    /// if the GPU cannot do this size (use the CPU path then).
    fn prepare_diffraction(&mut self) -> Result<bool> {
        let s = self.settings.as_ref().context("not configured")?;
        let (iris, stop_r, n_w) = self.optics.clone().context("not configured")?;
        let pitch = s.sensor_w / s.w as f64;
        let key = diffraction::KernelKey::new(s.w, s.h, s.bins, &iris, stop_r, n_w, pitch);
        let d = self.diffraction.get_or_insert_with(|| diffraction::GpuDiffraction::new(&self.ctx.device, &self.ctx.queue));
        if d.is_configured(&key) {
            return Ok(true);
        }
        let spec = crate::diffraction::PupilSpectrum::get(&iris, stop_r);
        let width = LAMBDA_RANGE / s.bins as f64;
        let kernels: Vec<(usize, Vec<f64>)> = (0..s.bins)
            .map(|b| {
                let l = LAMBDA_MIN + width * (b as f64 + 0.5);
                let half = crate::diffraction::kernel_half(l, n_w, pitch);
                (half, crate::diffraction::kernel(&spec, l, n_w, pitch, half))
            })
            .collect();
        Ok(d.configure(key, &kernels, self.spectral_buf.as_ref().unwrap()).is_ok())
    }

    /// Normalises the film, optionally applies diffraction, and converts it
    /// to the display texture. Returns whether diffraction was applied.
    pub fn resolve(&mut self, diffraction: bool) -> Result<bool> {
        let diffract = diffraction && self.prepare_diffraction()?;
        let mut p = self.params;
        p.frames = self.frames.max(1) as f32;
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, self.g0.as_ref(), &[]);
            pass.set_bind_group(1, self.g1.as_ref(), &[]);
            pass.set_bind_group(2, self.g2.as_ref(), &[]);
            pass.set_pipeline(&self.normalize_pl);
            let n = p.w * p.h * p.bins;
            let groups = n.div_ceil(256);
            pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
        }
        if diffract {
            self.diffraction.as_ref().unwrap().apply(&mut enc);
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, self.g0.as_ref(), &[]);
            pass.set_bind_group(1, self.g1.as_ref(), &[]);
            pass.set_bind_group(2, self.g2.as_ref(), &[]);
            pass.set_pipeline(&self.to_rgb_pl);
            pass.dispatch_workgroups(p.w.div_ceil(8), p.h.div_ceil(8), 1);
        }
        self.ctx.queue.submit([enc.finish()]);
        Ok(diffract)
    }

    pub fn profile_diffraction(&mut self) -> Result<Vec<(usize, f64)>> {
        self.prepare_diffraction()?;
        Ok(self.diffraction.as_ref().unwrap().profile())
    }

    /// Runs only the diffraction on `data` (spectral layout), for tests.
    #[cfg(test)]
    fn diffract_spectral(&mut self, data: &[f32]) -> Result<Film> {
        assert!(self.prepare_diffraction()?);
        self.ctx.queue.write_buffer(self.spectral_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(data));
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        self.diffraction.as_ref().unwrap().apply(&mut enc);
        self.ctx.queue.submit([enc.finish()]);
        self.read_film()
    }

    /// Reads the normalised spectral film (after `resolve`).
    pub fn read_film(&self) -> Result<Film> {
        let s = self.settings.as_ref().context("not configured")?;
        let src = self.spectral_buf.as_ref().unwrap();
        let size = src.size();
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.ctx.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        rx.recv()??;
        let data: Vec<f32> = {
            let view = staging.get_mapped_range(..).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            bytemuck::cast_slice(&view).to_vec()
        };
        staging.unmap();
        Ok(Film { w: s.w, h: s.h, bins: s.bins, data })
    }

    /// Blocks until the GPU is idle.
    pub fn wait(&self) -> Result<()> {
        self.ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::scene::{Light, LightSpectrum, Rect};

    fn lens(fstop: f64, focus: Option<f64>) -> Lens {
        let mut l = Lens::parse(include_str!("../../lenses/cooke-triplet.lens")).unwrap();
        l.set_aperture(fstop, 6, 10.0, 0.0).unwrap();
        l.sensor_z = l.best_focus(focus).unwrap();
        l
    }

    fn settings(w: usize, h: usize, bins: usize, ghosts: bool) -> Settings {
        Settings {
            w,
            h,
            bins,
            sensor_w: 36.0,
            norm: 1.0,
            spp_per_frame: 1,
            ghosts,
            ghost_threshold: 0.0,
            light_grid: 128,
            ghost_wavelengths: 0,
            tonemap: Tonemap::Clip,
        }
    }

    fn gpu_film(lens: &Lens, scene: &Scene, s: &Settings, frames: u32) -> Option<Film> {
        let ctx = GpuContext::headless().ok()?;
        let mut g = GpuRenderer::new(ctx);
        g.set_scene(scene);
        g.configure(lens, s).unwrap();
        for _ in 0..frames {
            g.render_frame();
        }
        g.resolve(false).unwrap();
        Some(g.read_film().unwrap())
    }

    fn total(f: &Film) -> f64 {
        f.data.iter().map(|v| *v as f64).sum()
    }

    fn chart(center: Vec3, size: f64) -> Rect {
        Rect {
            center,
            u: Vec3::new(1.0, 0.0, 0.0),
            v: Vec3::new(0.0, 1.0, 0.0),
            n: Vec3::new(0.0, 0.0, 1.0),
            w: size * 1.5,
            h: size,
            texture: Texture::Chart,
            brightness: 1.0,
        }
    }

    /// Mean absolute difference of 6×6-pixel block averages, relative to the mean.
    fn block_diff(a: &Film, b: &Film) -> f64 {
        let (w, h) = (a.w, a.h);
        let mut diff = 0.0;
        let mut sum = 0.0;
        for by in 0..h / 6 {
            for bx in 0..w / 6 {
                let (mut sa, mut sb) = (0.0, 0.0);
                for y in by * 6..by * 6 + 6 {
                    for x in bx * 6..bx * 6 + 6 {
                        for k in 0..a.bins {
                            sa += a.data[(y * w + x) * a.bins + k] as f64;
                            sb += b.data[(y * w + x) * a.bins + k] as f64;
                        }
                    }
                }
                diff += (sa - sb).abs();
                sum += sa;
            }
        }
        diff / sum
    }

    #[test]
    fn surfaces_match_cpu() {
        for decenter in [0.0, 0.2] {
            let mut lens = lens(5.6, Some(2000.0));
            lens.decenter_element(1, decenter, 0.0).unwrap();
            lens.coating = Coating::None;
            let scene = Scene { rects: vec![chart(Vec3::new(0.0, 0.0, -2000.0), 1000.0)], lights: vec![], background: [0.05, 0.1, 0.2] };
            let s = settings(120, 80, 8, true);
            let Some(g) = gpu_film(&lens, &scene, &s, 256) else { return };
            let cs = ColorSystem::new();
            let r = crate::render::Renderer::new(&lens, &scene, &cs, 120, 80, 36.0, 8, 1.0);
            let mut c = Film::new(120, 80, 8);
            r.render_surfaces(&mut c, 2048, true, false);
            let (tg, tc) = (total(&g), total(&c));
            let d = block_diff(&c, &g);
            println!("decenter {decenter}: total gpu {tg} cpu {tc}, block diff {d}");
            assert!((tg - tc).abs() / tc < 0.005);
            assert!(d < 0.02);
        }
    }

    #[test]
    fn lights_match_cpu() {
        let mut lens = lens(4.0, Some(3000.0));
        lens.coating = Coating::None;
        let light = |c: Vec3| Light { center: c, radius: 30.0, spectrum: LightSpectrum::Blackbody { t: 4000.0, scale: 1e-6 } };
        let scene = Scene {
            rects: vec![],
            lights: vec![light(Vec3::new(400.0, 250.0, -5000.0)), light(Vec3::new(-1500.0, 900.0, -6000.0))],
            background: [0.0; 3],
        };
        let cs = ColorSystem::new();
        for ghosts in [false, true] {
            let s = settings(120, 80, 8, ghosts);
            let Some(g) = gpu_film(&lens, &scene, &s, 64) else { return };
            let r = crate::render::Renderer::new(&lens, &scene, &cs, 120, 80, 36.0, 8, 1.0);
            let mut c = Film::new(120, 80, 8);
            r.render_lights(&mut c, 4_000_000, 256, if ghosts { 0.0 } else { f64::INFINITY }, false);
            let (tg, tc) = (total(&g), total(&c));
            let d = block_diff(&c, &g);
            println!("ghosts {ghosts}: total gpu {tg} cpu {tc}, block diff {d}");
            assert!((tg - tc).abs() / tc < 0.01);
            assert!(d < 0.05);
        }
    }

    #[test]
    fn ghosts_match_cpu() {
        let mut lens = lens(4.0, Some(3000.0));
        lens.coating = Coating::None;
        let light = |c: Vec3| Light { center: c, radius: 30.0, spectrum: LightSpectrum::Blackbody { t: 4000.0, scale: 1e-6 } };
        let cs = ColorSystem::new();
        for c in [Vec3::new(400.0, 250.0, -5000.0), Vec3::new(-1500.0, 900.0, -6000.0)] {
            let scene = Scene { rects: vec![], lights: vec![light(c)], background: [0.0; 3] };
            if GpuContext::headless().is_err() {
                return;
            }
            let mut ghost = |grid: u32, frames: u32| {
                let mut s = settings(120, 80, 8, true);
                s.light_grid = grid;
                let with = total(&gpu_film(&lens, &scene, &s, frames).unwrap());
                s.ghosts = false;
                with - total(&gpu_film(&lens, &scene, &s, frames).unwrap())
            };
            let g128 = ghost(128, 64);
            let r = crate::render::Renderer::new(&lens, &scene, &cs, 120, 80, 36.0, 8, 1.0);
            let mut a = Film::new(120, 80, 8);
            r.render_lights(&mut a, 4_000_000, 256, 0.0, false);
            let mut b = Film::new(120, 80, 8);
            r.render_lights(&mut b, 4_000_000, 256, f64::INFINITY, false);
            let cpu = total(&a) - total(&b);
            println!("ghost energy: gpu {g128} cpu {cpu}");
            assert!((g128 - cpu).abs() / cpu < 0.01);
        }
    }

    #[test]
    fn veil_matches_cpu() {
        let mut lens = lens(4.0, None);
        lens.coating = Coating::None;
        let bright = Rect { texture: Texture::Color([1.0; 3]), w: 300.0, ..chart(Vec3::new(1300.0, 800.0, -5000.0), 300.0) };
        let scene = Scene { rects: vec![bright], lights: vec![], background: [0.0; 3] };
        let mut s = settings(120, 80, 8, true);
        s.sensor_w = 72.0;
        let Some(with) = gpu_film(&lens, &scene, &s, 1024) else { return };
        s.ghosts = false;
        let without = gpu_film(&lens, &scene, &s, 1024).unwrap();
        let gpu = total(&with) - total(&without);
        let cs = ColorSystem::new();
        let r = crate::render::Renderer::new(&lens, &scene, &cs, 120, 80, 72.0, 8, 1.0);
        let mut a = Film::new(120, 80, 8);
        r.render_surfaces(&mut a, 4096, true, false);
        let mut b = Film::new(120, 80, 8);
        r.render_surfaces(&mut b, 4096, false, false);
        let cpu = total(&a) - total(&b);
        println!("veil energy: gpu {gpu} cpu {cpu}");
        assert!((gpu - cpu).abs() / cpu < 0.04);
    }

    #[test]
    fn diffraction_matches_cpu() {
        let Ok(ctx) = GpuContext::headless() else { return };
        for fstop in [3.5, 11.0] {
            let mut lens = lens(fstop, None);
            lens.set_aperture(fstop, 6, 10.0, 0.0).unwrap();
            let scene = Scene { rects: vec![], lights: vec![], background: [0.0; 3] };
            let (w, h, bins) = (160, 100, 4);
            let s = settings(w, h, bins, false);
            let mut g = GpuRenderer::new(ctx.clone());
            g.set_scene(&scene);
            g.configure(&lens, &s).unwrap();
            // Faint background, a few points of very different brightness.
            let mut film = Film::new(w, h, bins);
            for (i, v) in film.data.iter_mut().enumerate() {
                *v = 0.01 + 0.001 * ((i * 7919) % 13) as f32;
            }
            for (x, y, v) in [(20, 20, 1.0e5f32), (100, 60, 300.0), (150, 95, 30.0), (80, 10, 1.0)] {
                for b in 0..bins {
                    film.data[(y * w + x) * bins + b] += v * (1.0 + b as f32 * 0.3);
                }
            }
            let gpu = g.diffract_spectral(&film.data).unwrap();
            let mut cpu = Film { w, h, bins, data: film.data.clone() };
            crate::diffraction::apply(&mut cpu, &lens.iris, lens.surfaces[lens.stop_index].semi_diameter, lens.working_fnumber(), 36.0 / w as f64);
            // f32 transforms are exact to ~1e-7 of the brightest pixel.
            let floor = 2e-7 * film.data.iter().cloned().fold(0.0f32, f32::max) as f64;
            let mut worst = (0.0f64, 0usize);
            for (i, (a, b)) in gpu.data.iter().zip(&cpu.data).enumerate() {
                let e = (*a as f64 - *b as f64).abs() / ((*b as f64).abs() * 0.02).max(floor);
                if e > worst.0 {
                    worst = (e, i);
                }
            }
            let i = worst.1;
            println!(
                "f/{fstop}: worst error / tolerance {:.2} at pixel ({}, {}) bin {}: gpu {} cpu {}",
                worst.0,
                (i / bins) % w,
                (i / bins) / w,
                i % bins,
                gpu.data[i],
                cpu.data[i]
            );
            assert!(worst.0 < 1.0);
        }
    }
}

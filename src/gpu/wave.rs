//! Wave-optical PSF kernels on the GPU, for the PSF bank. Per entry the GPU
//! traces every pupil sample for its transmission (in f32, which is plenty
//! to decide whether a ray passes), takes the OPD from bicubic interpolation
//! of the CPU's f64 nodes (f32 path lengths would not resolve nanometres),
//! transforms the pupil function, and bins the energy into a sub-pixel core
//! and a pixel wing like `psfbank` does on the CPU.

use super::diffraction::FftParams;
use super::{GpuContext, Params, coating_params, encode_path, front_z, media_table, shader_source, storage_entry, surface_words, wgpu};
use crate::lens::Lens;
use crate::psf::{Source, WavefrontPlan};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Kernels of one entry: core (2 hc + 1)·sub cells per side at pitch/sub,
/// wing 2 hw + 1 pixels per side with the core's square zeroed.
pub struct WaveKernels {
    pub hc: usize,
    pub hw: usize,
    pub sub: usize,
    pub core: Vec<f64>,
    pub wing: Vec<f64>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaveEntry {
    src: [f32; 4],
    grid: [f32; 4],
    lambda: f32,
    wave: f32,
    n: u32,
    step: u32,
    nodes_off: u32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaveBatch {
    nn: u32,
    count: u32,
    pitch: f32,
    front_z: f32,
}

const WING: usize = 129;
const OUT_STRIDE: usize = 129 * 129 + 128 * 128;
/// Per transform buffer.
const MAX_GRID_BYTES: u64 = 256 << 20;
const MAX_BATCH: usize = 256;

/// Pipelines, created once per device.
pub struct WaveGpu {
    ctx: GpuContext,
    layouts: [wgpu::BindGroupLayout; 2],
    pupil: wgpu::ComputePipeline,
    map: wgpu::ComputePipeline,
    wing: wgpu::ComputePipeline,
    sizes: wgpu::ComputePipeline,
    core: wgpu::ComputePipeline,
    transpose: wgpu::ComputePipeline,
    ffts: std::sync::Mutex<Vec<(u32, wgpu::ComputePipeline)>>,
}

impl WaveGpu {
    pub fn new(ctx: &GpuContext) -> WaveGpu {
        let device = &ctx.device;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave lens"),
            entries: &[uniform(0), storage_entry(1, true), storage_entry(2, true)],
        });
        let g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave batch"),
            entries: &[uniform(3), storage_entry(4, true), storage_entry(5, true), storage_entry(6, false), storage_entry(7, false), storage_entry(8, false), storage_entry(9, false), storage_entry(10, false), storage_entry(11, false)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&g0), Some(&g1)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wave"),
            source: wgpu::ShaderSource::Wgsl(shader_source(device, &format!("{}\n{}", include_str!("trace.wgsl"), include_str!("wave.wgsl"))).into()),
        });
        let pipeline = |layout: Option<&wgpu::PipelineLayout>, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let conv = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv"),
            source: wgpu::ShaderSource::Wgsl(include_str!("conv.wgsl").into()),
        });
        WaveGpu {
            pupil: pipeline(Some(&layout), &module, "wave_pupil"),
            map: pipeline(Some(&layout), &module, "wave_map"),
            wing: pipeline(Some(&layout), &module, "wave_wing"),
            sizes: pipeline(Some(&layout), &module, "wave_sizes"),
            core: pipeline(Some(&layout), &module, "wave_core"),
            transpose: pipeline(None, &conv, "transpose"),
            ffts: Default::default(),
            layouts: [g0, g1],
            ctx: ctx.clone(),
        }
    }

    fn fft(&self, n: u32) -> wgpu::ComputePipeline {
        let mut ffts = self.ffts.lock().unwrap();
        if let Some((_, p)) = ffts.iter().find(|(m, _)| *m == n) {
            return p.clone();
        }
        let src = include_str!("fft.wgsl").replace("FFT_LOG2N", &n.trailing_zeros().to_string()).replace("FFT_N", &format!("{n}u"));
        let device = &self.ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fft"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft_rows"),
            layout: None,
            module: &module,
            entry_point: Some("fft_rows"),
            compilation_options: Default::default(),
            cache: None,
        });
        ffts.push((n, p.clone()));
        p
    }

    /// Kernels for every plan, in order (None where the plan is None or no
    /// light passes). `progress` gets the fraction done.
    pub fn kernels(&self, lens: &Lens, plans: &[Option<WavefrontPlan>], pitch: f64, progress: &dyn Fn(f64)) -> Result<Vec<Option<WaveKernels>>> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let max_n = plans.iter().flatten().map(|p| 2 * p.n as u64).max().unwrap_or(0);
        if max_n * 8 > (device.limits().max_compute_workgroup_storage_size as u64) {
            anyhow::bail!("PSF transforms of {max_n} points exceed the GPU's workgroup memory");
        }
        let grid_bytes = MAX_GRID_BYTES.min(device.limits().max_storage_buffer_binding_size as u64).min(device.limits().max_buffer_size);
        if max_n * max_n * 8 > grid_bytes {
            anyhow::bail!("PSF transforms of {max_n}² points exceed the GPU's buffer size");
        }

        // The lens: surfaces and the forward path, media, iris and coating.
        let mut blob = surface_words(lens)?;
        let path_base = blob.len() as u32;
        let path = encode_path(&lens.forward_path);
        blob.extend(&path);
        let iris = &lens.iris;
        let (coating, coat_n, coat_h, coat_l0) = coating_params(lens.coating);
        let params = Params {
            n_surf: lens.surfaces.len() as u32,
            coating,
            coat_n,
            coat_h,
            coat_l0,
            iris_blades: iris.blades,
            iris_circular: iris.circular as u32,
            iris_rotation: iris.rotation as f32,
            iris_curvature: iris.curvature as f32,
            iris_apothem: iris.apothem() as f32,
            iris_max_radius: iris.max_radius as f32,
            media_off: 0,
            path_base,
            path_stride: path.len() as u32,
            ..Default::default()
        };
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            let mut bytes = contents.to_vec();
            bytes.resize(bytes.len().max(16), 0);
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &bytes, usage })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let params_buf = init("wave params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);
        let lens_buf = init("wave lens", bytemuck::cast_slice(&blob), storage);
        let tables_buf = init("wave media", bytemuck::cast_slice(&media_table(lens)), storage);
        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layouts[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lens_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tables_buf.as_entire_binding() },
            ],
        });

        // Batches of entries with the same transform size, as many as fit.
        let mut order: Vec<usize> = (0..plans.len()).filter(|&i| plans[i].is_some()).collect();
        order.sort_by_key(|&i| plans[i].as_ref().unwrap().n);
        let mut batches: Vec<&[usize]> = Vec::new();
        let mut rest = order.as_slice();
        while let Some(&first) = rest.first() {
            let nn = 2 * plans[first].as_ref().unwrap().n;
            let fit = ((grid_bytes / (nn * nn * 8) as u64) as usize).min(65535 / nn).min(MAX_BATCH).max(1);
            let len = rest.iter().take(fit).take_while(|&&i| 2 * plans[i].as_ref().unwrap().n == nn).count();
            batches.push(&rest[..len]);
            rest = &rest[len..];
        }
        let grid_size = batches.iter().map(|b| (b.len() * (2 * plans[b[0]].as_ref().unwrap().n).pow(2) * 8) as u64).max().unwrap_or(16);
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
        };
        let grid_a = buffer("wave grid a", grid_size, storage);
        let grid_b = buffer("wave grid b", grid_size, storage);
        let out_bytes = (MAX_BATCH * OUT_STRIDE * 4) as u64;
        let out = buffer("wave out", out_bytes, storage | wgpu::BufferUsages::COPY_SRC);
        let sizes = buffer("wave sizes", (MAX_BATCH * 16) as u64, storage | wgpu::BufferUsages::COPY_SRC);
        let moments = buffer("wave moments", (MAX_BATCH * 64) as u64, storage | wgpu::BufferUsages::COPY_DST);
        let maps = buffer("wave maps", (MAX_BATCH * 16) as u64, storage);
        let hist = buffer("wave histograms", (MAX_BATCH * 92 * 4) as u64, storage | wgpu::BufferUsages::COPY_DST);
        // Two, so that one batch is read back while the next one runs.
        let staging = [0, 1].map(|_| buffer("wave staging", out_bytes + (MAX_BATCH * 16) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST));
        let fz = front_z(lens) as f32;

        let mut result: Vec<Option<WaveKernels>> = (0..plans.len()).map(|_| None).collect();
        let mut done = 0;
        // Reads back a submitted batch into `result`.
        let mut collect = |batch: &[usize], staging: &wgpu::Buffer, index: wgpu::SubmissionIndex| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::channel();
            staging.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })?;
            rx.recv()?.context("mapping PSF kernels")?;
            {
                let data = staging.get_mapped_range(..).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let vals: &[f32] = bytemuck::cast_slice(&data[..batch.len() * OUT_STRIDE * 4]);
                let szs: &[[u32; 4]] = bytemuck::cast_slice(&data[out_bytes as usize..out_bytes as usize + batch.len() * 16]);
                for (k, &i) in batch.iter().enumerate() {
                    let [hc, hw, sub, _] = szs[k].map(|v| v as usize);
                    let o = &vals[k * OUT_STRIDE..(k + 1) * OUT_STRIDE];
                    let size = (2 * hc + 1) * sub;
                    let core: Vec<f64> = o[WING * WING..WING * WING + size * size].iter().map(|&v| v as f64).collect();
                    if core.iter().sum::<f64>() <= 0.0 {
                        continue;
                    }
                    let c = WING / 2;
                    let mut wing = Vec::with_capacity((2 * hw + 1).pow(2));
                    for y in c - hw..=c + hw {
                        for x in c - hw..=c + hw {
                            let in_core = x.abs_diff(c) <= hc && y.abs_diff(c) <= hc;
                            wing.push(if in_core { 0.0 } else { o[y * WING + x] as f64 });
                        }
                    }
                    result[i] = Some(WaveKernels { hc, hw, sub, core, wing });
                }
            }
            staging.unmap();
            done += batch.len();
            progress(done as f64 / order.len().max(1) as f64);
            Ok(())
        };
        let mut pending: Option<(&[usize], usize, wgpu::SubmissionIndex)> = None;
        for (bi, batch) in batches.into_iter().enumerate() {
            let n = plans[batch[0]].as_ref().unwrap().n as u32;
            let nn = 2 * n;
            let count = batch.len() as u32;
            let mut entries = Vec::with_capacity(batch.len());
            let mut nodes: Vec<f32> = Vec::new();
            for &i in batch {
                let p = plans[i].as_ref().unwrap();
                let src = match p.source {
                    Source::Point(v) => [v.x as f32, v.y as f32, v.z as f32, 1.0],
                    Source::Infinity(d) => [d.x as f32, d.y as f32, d.z as f32, 0.0],
                };
                entries.push(WaveEntry {
                    src,
                    grid: [p.x0 as f32, p.y0 as f32, p.z as f32, p.du as f32],
                    lambda: p.lambda as f32,
                    wave: (p.lambda * 1e-6) as f32,
                    n,
                    step: p.step as u32,
                    nodes_off: nodes.len() as u32,
                    _p: [0; 3],
                });
                nodes.extend(&p.nodes);
            }
            let batch_buf = init("wave batch", bytemuck::bytes_of(&WaveBatch { nn, count, pitch: pitch as f32, front_z: fz }), wgpu::BufferUsages::UNIFORM);
            let entries_buf = init("wave entries", bytemuck::cast_slice(&entries), storage);
            let nodes_buf = init("wave nodes", bytemuck::cast_slice(&nodes), storage);
            let g1 = |grid: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.layouts[1],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 3, resource: batch_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: entries_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: nodes_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: grid.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: sizes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: moments.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: maps.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 11, resource: hist.as_entire_binding() },
                    ],
                })
            };
            let (g1a, g1b) = (g1(&grid_a), g1(&grid_b));
            let fft = self.fft(nn);
            let twiddles: Vec<[f32; 2]> = (0..nn / 2)
                .map(|k| {
                    let a = -2.0 * std::f64::consts::PI * k as f64 / nn as f64;
                    [a.cos() as f32, a.sin() as f32]
                })
                .collect();
            let tw = init("twiddles", bytemuck::cast_slice(&twiddles), storage);
            let fft_group = |data: &wgpu::Buffer, params: FftParams| {
                let ub = init("fft params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &fft.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: tw.as_entire_binding() },
                    ],
                })
            };
            let mut transpose = FftParams::default();
            (transpose.rows, transpose.cols, transpose.src_rows) = (nn, nn, n);
            let t_params = init("transpose params", bytemuck::bytes_of(&transpose), wgpu::BufferUsages::UNIFORM);
            let t_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.transpose.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: t_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: grid_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: grid_b.as_entire_binding() },
                ],
            });
            // Rows of the pupil: the first n of each entry (the rest is zero).
            let mut pupil_rows = FftParams::default();
            pupil_rows.batch_rows = n;

            let mut enc = device.create_command_encoder(&Default::default());
            enc.clear_buffer(&moments, 0, None);
            enc.clear_buffer(&hist, 0, None);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pupil);
                pass.set_bind_group(0, &g0, &[]);
                pass.set_bind_group(1, &g1a, &[]);
                pass.dispatch_workgroups(nn / 64, n / 8, count);
                pass.set_pipeline(&self.map);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
                pass.set_pipeline(&fft);
                pass.set_bind_group(0, &fft_group(&grid_a, pupil_rows), &[]);
                pass.dispatch_workgroups(count * n, 1, 1);
                pass.set_pipeline(&self.transpose);
                pass.set_bind_group(0, &t_group, &[]);
                pass.dispatch_workgroups(nn / 16, nn / 16, count);
                pass.set_pipeline(&fft);
                pass.set_bind_group(0, &fft_group(&grid_b, FftParams::default()), &[]);
                pass.dispatch_workgroups(count * nn, 1, 1);
                pass.set_bind_group(0, &g0, &[]);
                pass.set_bind_group(1, &g1b, &[]);
                pass.set_pipeline(&self.wing);
                pass.dispatch_workgroups((WING as u32).div_ceil(8), (WING as u32).div_ceil(8), count);
                pass.set_pipeline(&self.sizes);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
                pass.set_pipeline(&self.core);
                pass.dispatch_workgroups(16, 16, count);
            }
            let st = &staging[bi % 2];
            enc.copy_buffer_to_buffer(&out, 0, st, 0, (batch.len() * OUT_STRIDE * 4) as u64);
            enc.copy_buffer_to_buffer(&sizes, 0, st, out_bytes, (batch.len() * 16) as u64);
            let index = queue.submit([enc.finish()]);
            if let Some((b, k, i)) = pending.take() {
                collect(b, &staging[k], i)?;
            }
            pending = Some((batch, bi % 2, index));
        }
        if let Some((b, k, i)) = pending {
            collect(b, &staging[k], i)?;
        }
        Ok(result)
    }
}

//! Diffraction on the GPU: the spectral film is convolved with per-bin PSF
//! kernels (built on the CPU, see `crate::diffraction`) by FFT, two real
//! wavelength bins packed into one complex transform.

use super::wgpu;
use crate::lens::Iris;
use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct FftParams {
    w: u32,
    h: u32,
    pw: u32,
    ph: u32,
    bins: u32,
    b0: u32,
    b1: u32,
    inverse: u32,
    rows: u32,
    cols: u32,
    kbin: u32,
    koff: u32,
    khalf: u32,
    _p: [u32; 3],
}

/// What the kernels depend on.
#[derive(Clone, PartialEq)]
pub struct KernelKey {
    pub w: usize,
    pub h: usize,
    pub bins: usize,
    pub iris: (u32, u64, u64, bool, u64),
    pub stop_radius: u64,
    pub n_w: u64,
    pub pitch: u64,
}

impl KernelKey {
    pub fn new(w: usize, h: usize, bins: usize, iris: &Iris, stop_radius: f64, n_w: f64, pitch: f64) -> KernelKey {
        KernelKey {
            w,
            h,
            bins,
            iris: (iris.blades, iris.rotation.to_bits(), iris.curvature.to_bits(), iris.circular, iris.radius.to_bits()),
            stop_radius: stop_radius.to_bits(),
            n_w: n_w.to_bits(),
            pitch: pitch.to_bits(),
        }
    }
}

struct Cmd {
    pipeline: usize,
    bind_group: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

pub struct GpuDiffraction {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    fft_pipelines: Vec<(u32, usize)>,
    key: Option<KernelKey>,
    convolve: Vec<Cmd>,
    // Keeps the buffers referenced by the commands alive.
    _buffers: Vec<wgpu::Buffer>,
}

const TRANSPOSE: usize = 0;
const PACK: usize = 1;
const MULTIPLY: usize = 2;
const UNPACK: usize = 3;
const PLACE: usize = 4;
const STORE: usize = 5;

impl GpuDiffraction {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuDiffraction {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv"),
            source: wgpu::ShaderSource::Wgsl(include_str!("conv.wgsl").into()),
        });
        let pipelines = ["transpose", "pack", "multiply", "unpack", "place_kernel", "store_kernel"]
            .iter()
            .map(|e| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: None,
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        GpuDiffraction {
            device: device.clone(),
            queue: queue.clone(),
            pipelines,
            fft_pipelines: Vec::new(),
            key: None,
            convolve: Vec::new(),
            _buffers: Vec::new(),
        }
    }

    pub fn is_configured(&self, key: &KernelKey) -> bool {
        self.key.as_ref() == Some(key)
    }

    fn fft_pipeline(&mut self, n: u32) -> usize {
        if let Some((_, i)) = self.fft_pipelines.iter().find(|(m, _)| *m == n) {
            return *i;
        }
        let src = include_str!("fft.wgsl").replace("FFT_LOG2N", &n.trailing_zeros().to_string()).replace("FFT_N", &format!("{n}u"));
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fft"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        self.pipelines.push(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft_rows"),
            layout: None,
            module: &module,
            entry_point: Some("fft_rows"),
            compilation_options: Default::default(),
            cache: None,
        }));
        let i = self.pipelines.len() - 1;
        self.fft_pipelines.push((n, i));
        i
    }

    /// Builds the kernel spectra and the convolution command list for
    /// `spectral` (w × h × bins). Fails if the transforms would be too large.
    pub fn configure(&mut self, key: KernelKey, kernels: &[(usize, Vec<f64>)], spectral: &wgpu::Buffer) -> Result<()> {
        let (w, h, bins) = (key.w as u32, key.h as u32, key.bins as u32);
        let hmax = kernels.iter().map(|k| k.0).max().unwrap_or(1) as u32;
        let pw = (w + hmax + 1).max(2 * hmax + 2).next_power_of_two();
        let ph = (h + hmax + 1).max(2 * hmax + 2).next_power_of_two();
        let max_n = self.device.limits().max_compute_workgroup_storage_size / 8;
        if pw > max_n || ph > max_n {
            bail!("diffraction transform {pw}×{ph} exceeds the GPU's workgroup memory");
        }
        let dev = self.device.clone();
        let storage = |label: &str, size: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let size = (pw * ph) as u64;
        let cbuf = storage("fft a", size * 8);
        let tbuf = storage("fft b", size * 8);
        let kspec = storage("kernel spectra", size * 4 * bins as u64);
        let mut kdata: Vec<f32> = Vec::new();
        let mut koffs = Vec::new();
        for (_, k) in kernels {
            koffs.push(kdata.len() as u32);
            kdata.extend(k.iter().map(|v| *v as f32));
        }
        let kimg = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernels"),
            contents: bytemuck::cast_slice(&kdata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let twiddles = |n: u32| {
            let t: Vec<[f32; 2]> = (0..n / 2)
                .map(|k| {
                    let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    [a.cos() as f32, a.sin() as f32]
                })
                .collect();
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("twiddles"),
                contents: bytemuck::cast_slice(&t),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let tw_w = twiddles(pw);
        let tw_h = twiddles(ph);
        let fft_w = self.fft_pipeline(pw);
        let fft_h = self.fft_pipeline(ph);

        let base = FftParams { w, h, pw, ph, bins, ..Default::default() };
        let pipelines = &self.pipelines;
        let cmd = |pipeline: usize, p: FftParams, bufs: &[(u32, &wgpu::Buffer)], groups: (u32, u32, u32)| {
            let ub = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&p),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }];
            entries.extend(bufs.iter().map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() }));
            let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines[pipeline].get_bind_group_layout(0),
                entries: &entries,
            });
            Cmd { pipeline, bind_group, groups }
        };
        let g16 = |x: u32, y: u32| (x.div_ceil(16), y.div_ceil(16), 1);
        let fft = |pl: usize, data: &wgpu::Buffer, tw: &wgpu::Buffer, rows: u32, inverse: bool| {
            cmd(pl, FftParams { rows, inverse: inverse as u32, ..base }, &[(1, data), (2, tw)], (rows, 1, 1))
        };
        let transpose = |src: &wgpu::Buffer, dst: &wgpu::Buffer, rows: u32, cols: u32| {
            cmd(TRANSPOSE, FftParams { rows, cols, ..base }, &[(3, src), (4, dst)], g16(cols, rows))
        };

        // Kernel spectra, computed once.
        let mut setup = Vec::new();
        for (b, (half, _)) in kernels.iter().enumerate() {
            let p = FftParams { kbin: b as u32, koff: koffs[b], khalf: *half as u32, ..base };
            setup.push(cmd(PLACE, p, &[(1, &cbuf), (7, &kimg)], g16(pw, ph)));
            setup.push(fft(fft_w, &cbuf, &tw_w, ph, false));
            setup.push(transpose(&cbuf, &tbuf, ph, pw));
            setup.push(fft(fft_h, &tbuf, &tw_h, pw, false));
            let groups = (size as u32).div_ceil(256);
            setup.push(cmd(STORE, p, &[(3, &tbuf), (6, &kspec)], (groups.min(65535), groups.div_ceil(65535), 1)));
        }
        self.run(&setup);

        // Convolution, two bins per transform.
        let mut conv = Vec::new();
        for b0 in (0..bins).step_by(2) {
            let p = FftParams { b0, b1: b0 + 1, ..base };
            conv.push(cmd(PACK, p, &[(1, &cbuf), (5, spectral)], g16(pw, ph)));
            conv.push(fft(fft_w, &cbuf, &tw_w, ph, false));
            conv.push(transpose(&cbuf, &tbuf, ph, pw));
            conv.push(fft(fft_h, &tbuf, &tw_h, pw, false));
            conv.push(cmd(MULTIPLY, p, &[(3, &tbuf), (4, &cbuf), (6, &kspec)], g16(ph, pw)));
            conv.push(fft(fft_h, &cbuf, &tw_h, pw, true));
            conv.push(transpose(&cbuf, &tbuf, pw, ph));
            conv.push(fft(fft_w, &tbuf, &tw_w, ph, true));
            conv.push(cmd(UNPACK, p, &[(3, &tbuf), (5, spectral)], g16(w, h)));
        }
        self.convolve = conv;
        self._buffers = vec![cbuf, tbuf, kspec, kimg, tw_w, tw_h];
        self.key = Some(key);
        Ok(())
    }

    fn run(&self, cmds: &[Cmd]) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, cmds);
        self.queue.submit([enc.finish()]);
    }

    fn encode(&self, enc: &mut wgpu::CommandEncoder, cmds: &[Cmd]) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for c in cmds {
            pass.set_pipeline(&self.pipelines[c.pipeline]);
            pass.set_bind_group(0, &c.bind_group, &[]);
            pass.dispatch_workgroups(c.groups.0, c.groups.1, c.groups.2);
        }
    }

    /// Convolves the spectral film in place.
    pub fn apply(&self, enc: &mut wgpu::CommandEncoder) {
        self.encode(enc, &self.convolve);
    }
}

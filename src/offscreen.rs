//! Offscreen egui: the UI is drawn into a texture and read back, one image
//! per video frame (demo and explainer recordings). A window would stall
//! whenever macOS stops redrawing it (covered, another desktop).

use crate::gpu::{self, wgpu};
use anyhow::Result;
use eframe::{egui, egui_wgpu};
use std::sync::Arc;
use std::time::Instant;

pub struct Offscreen {
    pub rs: egui_wgpu::RenderState,
    pub ctx: egui::Context,
    size: egui::Vec2,
    ppp: f32,
    px: [u32; 2],
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// Bytes per row of the readback buffer (padded).
    row: u32,
    readback: wgpu::Buffer,
    start: Instant,
}

impl Offscreen {
    /// `size` points at `ppp` pixels per point.
    pub fn new(size: egui::Vec2, ppp: f32) -> Result<Offscreen> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&gpu::device_descriptor(&adapter)))?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let renderer = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        let rs = egui_wgpu::RenderState {
            adapter,
            available_adapters: Vec::new(),
            instance,
            device: device.clone(),
            queue: queue.clone(),
            target_format: format,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        };
        let px = [(size.x * ppp) as u32, (size.y * ppp) as u32];
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen frame"),
            size: wgpu::Extent3d { width: px[0], height: px[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let row = (px[0] * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen readback"),
            size: (row * px[1]) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Offscreen { rs, ctx: egui::Context::default(), size, ppp, px, target, view, row, readback, start: Instant::now() })
    }

    /// Input for the next UI pass.
    pub fn input(&self) -> egui::RawInput {
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, self.size)),
            max_texture_side: Some(self.rs.device.limits().max_texture_dimension_2d as usize),
            time: Some(self.start.elapsed().as_secs_f64()),
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(self.ppp);
        raw
    }

    /// Paints a UI pass's output; with `capture`, reads the frame back.
    pub fn paint(&mut self, out: egui::FullOutput, capture: bool) -> Result<Option<egui::ColorImage>> {
        let (device, queue, px) = (&self.rs.device, &self.rs.queue, self.px);
        let jobs = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: px, pixels_per_point: out.pixels_per_point };
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut r = self.rs.renderer.write();
            for (id, deltas) in &out.textures_delta.set {
                for delta in deltas {
                    r.update_texture(device, queue, *id, delta);
                }
            }
            let extra = r.update_buffers(device, queue, &mut enc, &jobs, &screen);
            queue.submit(extra);
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("offscreen ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            r.render(&mut pass, &jobs, &screen);
            drop(pass);
            for id in &out.textures_delta.free {
                r.free_texture(id);
            }
        }
        if capture {
            enc.copy_texture_to_buffer(
                self.target.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: &self.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.row), rows_per_image: None } },
                wgpu::Extent3d { width: px[0], height: px[1], depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
        if !capture {
            return Ok(None);
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        rx.recv()??;
        let img = {
            let data = self.readback.get_mapped_range(..).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let row = self.row as usize;
            let pixels = (0..px[1] as usize)
                .flat_map(|y| data[y * row..y * row + px[0] as usize * 4].chunks(4).map(|c| egui::Color32::from_rgba_premultiplied(c[0], c[1], c[2], 255)).collect::<Vec<_>>())
                .collect();
            egui::ColorImage { size: [px[0] as usize, px[1] as usize], pixels, ..Default::default() }
        };
        self.readback.unmap();
        Ok(Some(img))
    }
}

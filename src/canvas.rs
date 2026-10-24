//! Minimal raster canvas for diagnostic plots.

use anyhow::Result;

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pix: Vec<[f32; 3]>,
}

impl Canvas {
    pub fn new(w: usize, h: usize, bg: [f32; 3]) -> Canvas {
        Canvas { w, h, pix: vec![bg; w * h] }
    }

    pub fn blend(&mut self, x: i64, y: i64, c: [f32; 3], a: f32) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return;
        }
        let p = &mut self.pix[y as usize * self.w + x as usize];
        for k in 0..3 {
            p[k] = p[k] * (1.0 - a) + c[k] * a;
        }
    }

    /// Anti-aliased line by dense sampling.
    pub fn line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: [f32; 3], a: f32) {
        let len = ((x1 - x0).hypot(y1 - y0) * 2.0).ceil().max(1.0) as usize;
        for i in 0..=len {
            let t = i as f64 / len as f64;
            self.dot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, c, a * 0.5);
        }
    }

    /// Bilinear splat of a point.
    pub fn dot(&mut self, x: f64, y: f64, c: [f32; 3], a: f32) {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let ix = fx.floor() as i64;
        let iy = fy.floor() as i64;
        let tx = (fx - ix as f64) as f32;
        let ty = (fy - iy as f64) as f32;
        self.blend(ix, iy, c, a * (1.0 - tx) * (1.0 - ty));
        self.blend(ix + 1, iy, c, a * tx * (1.0 - ty));
        self.blend(ix, iy + 1, c, a * (1.0 - tx) * ty);
        self.blend(ix + 1, iy + 1, c, a * tx * ty);
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let mut img = image::RgbImage::new(self.w as u32, self.h as u32);
        for (i, p) in self.pix.iter().enumerate() {
            let px = image::Rgb([
                (p[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                (p[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                (p[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            ]);
            img.put_pixel((i % self.w) as u32, (i / self.w) as u32, px);
        }
        img.save(path)?;
        Ok(())
    }
}

/// Approximate display colour of a wavelength for plots.
pub fn wavelength_color(l: f64) -> [f32; 3] {
    let cs = crate::spectrum::cie_xyz(l);
    let rgb = crate::spectrum::xyz_to_linear_srgb(cs);
    let m = rgb.iter().cloned().fold(1e-9, f64::max);
    [
        crate::spectrum::srgb_encode(rgb[0].max(0.0) / m) as f32,
        crate::spectrum::srgb_encode(rgb[1].max(0.0) / m) as f32,
        crate::spectrum::srgb_encode(rgb[2].max(0.0) / m) as f32,
    ]
}

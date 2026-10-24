//! Spectral helpers: CIE 1931 colour matching functions, XYZ -> sRGB,
//! RGB -> spectrum upsampling and black-body emission.

pub const LAMBDA_MIN: f64 = 380.0;
pub const LAMBDA_MAX: f64 = 780.0;
pub const LAMBDA_RANGE: f64 = LAMBDA_MAX - LAMBDA_MIN;

fn g(x: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let s = if x < mu { s1 } else { s2 };
    let t = (x - mu) / s;
    (-0.5 * t * t).exp()
}

/// CIE 1931 2° colour matching functions, multi-lobe fit by
/// Wyman, Sloan & Shirley (2013).
pub fn cie_xyz(l: f64) -> [f64; 3] {
    let x = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7)
        - 0.065 * g(l, 501.1, 20.4, 26.2);
    let y = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1);
    let z = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8);
    [x, y, z]
}

pub fn xyz_to_linear_srgb(c: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = c;
    [
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    ]
}

pub fn srgb_encode(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

pub fn srgb_decode(v: f64) -> f64 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Smooth blue / green / red basis spectra forming a partition of unity, so a
/// flat (white) spectrum is exactly representable.
fn basis(l: f64) -> [f64; 3] {
    let b = 1.0 - logistic((l - 490.0) / 8.0);
    let r = logistic((l - 590.0) / 8.0);
    [r, 1.0 - b - r, b]
}

/// Converts spectra to white-balanced linear sRGB, where a flat spectrum of
/// value 1 maps to (1, 1, 1). Also provides the inverse (RGB -> spectrum).
pub struct ColorSystem {
    /// Per-wavelength (1 nm table) RGB sensitivity for upsampling: S(λ) = rgb · up[λ].
    up: Vec<[f64; 3]>,
    white: [f64; 3],
    y_flat: f64,
}

impl ColorSystem {
    pub fn new() -> Self {
        let n = LAMBDA_RANGE as usize;
        let mut y_flat = 0.0;
        let mut xyz_flat = [0.0; 3];
        let mut xyz_basis = [[0.0; 3]; 3];
        for i in 0..n {
            let l = LAMBDA_MIN + i as f64 + 0.5;
            let c = cie_xyz(l);
            let bs = basis(l);
            y_flat += c[1];
            for k in 0..3 {
                xyz_flat[k] += c[k];
                for j in 0..3 {
                    xyz_basis[j][k] += c[k] * bs[j];
                }
            }
        }
        let norm = |v: [f64; 3]| [v[0] / y_flat, v[1] / y_flat, v[2] / y_flat];
        let white = xyz_to_linear_srgb(norm(xyz_flat));
        let mut cs = ColorSystem { up: Vec::new(), white, y_flat };
        // Matrix M: column j = balanced RGB of basis j.
        let mut m = [[0.0; 3]; 3];
        for j in 0..3 {
            let rgb = cs.balance(xyz_to_linear_srgb(norm(xyz_basis[j])));
            for i in 0..3 {
                m[i][j] = rgb[i];
            }
        }
        let mi = invert3(m);
        // S(λ) = Σ_j w_j B_j(λ), w = M⁻¹ rgb  =>  S = rgb · (M⁻ᵀ B(λ))
        cs.up = (0..=n)
            .map(|i| {
                let bs = basis(LAMBDA_MIN + i as f64);
                let mut s = [0.0; 3];
                for k in 0..3 {
                    for j in 0..3 {
                        s[k] += mi[j][k] * bs[j];
                    }
                }
                s
            })
            .collect();
        cs
    }

    fn balance(&self, rgb: [f64; 3]) -> [f64; 3] {
        [rgb[0] / self.white[0], rgb[1] / self.white[1], rgb[2] / self.white[2]]
    }

    /// Converts an (un-normalised, Σ over 1 nm steps-equivalent) XYZ integral to balanced linear sRGB.
    /// `xyz` must be ∫ S(λ) cmf(λ) dλ in nm units.
    pub fn xyz_integral_to_rgb(&self, xyz: [f64; 3]) -> [f64; 3] {
        let n = [xyz[0] / self.y_flat, xyz[1] / self.y_flat, xyz[2] / self.y_flat];
        self.balance(xyz_to_linear_srgb(n))
    }

    /// Matrix taking an XYZ integral (as in `xyz_integral_to_rgb`) to balanced linear sRGB.
    pub fn rgb_matrix(&self) -> [[f64; 3]; 3] {
        let mut m = [[0.0; 3]; 3];
        for k in 0..3 {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let col = self.xyz_integral_to_rgb(e);
            for i in 0..3 {
                m[i][k] = col[i];
            }
        }
        m
    }

    /// Spectral value at wavelength `l` for linear RGB colour `rgb`.
    #[inline]
    pub fn upsample(&self, rgb: [f32; 3], l: f64) -> f64 {
        let s = self.up_vec(l);
        (rgb[0] as f64 * s[0] + rgb[1] as f64 * s[1] + rgb[2] as f64 * s[2]).max(0.0)
    }

    #[inline]
    pub fn up_vec(&self, l: f64) -> [f64; 3] {
        let i = ((l - LAMBDA_MIN).round() as usize).min(self.up.len() - 1);
        self.up[i]
    }

    /// Normalised luminance-weighted colour of a spectrum (used for black bodies),
    /// returns RGB of the spectrum scaled to luminance 1.
    pub fn spectrum_luminance(&self, f: impl Fn(f64) -> f64) -> f64 {
        let mut y = 0.0;
        for i in 0..LAMBDA_RANGE as usize {
            let l = LAMBDA_MIN + i as f64 + 0.5;
            y += f(l) * cie_xyz(l)[1];
        }
        y / self.y_flat
    }
}

fn invert3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let id = 1.0 / det;
    let mut r = [[0.0; 3]; 3];
    r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id;
    r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id;
    r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id;
    r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id;
    r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id;
    r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id;
    r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    r
}

/// Planck's law (relative spectral radiance), wavelength in nm.
pub fn blackbody(l_nm: f64, t: f64) -> f64 {
    let l = l_nm * 1e-9;
    let c1 = 3.741771852e-16;
    let c2 = 1.438776877e-2;
    c1 / (l.powi(5) * ((c2 / (l * t)).exp() - 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_roundtrip() {
        let cs = ColorSystem::new();
        for rgb in [[1.0f32, 1.0, 1.0], [0.8, 0.2, 0.1], [0.1, 0.5, 0.9], [0.3, 0.7, 0.2]] {
            let mut xyz = [0.0; 3];
            for i in 0..LAMBDA_RANGE as usize {
                let l = LAMBDA_MIN + i as f64 + 0.5;
                let s = cs.upsample(rgb, l);
                let c = cie_xyz(l);
                for k in 0..3 {
                    xyz[k] += s * c[k];
                }
            }
            let out = cs.xyz_integral_to_rgb(xyz);
            for k in 0..3 {
                assert!((out[k] - rgb[k] as f64).abs() < 0.02, "{rgb:?} -> {out:?}");
            }
        }
    }

    #[test]
    fn cmf_peak() {
        let y = cie_xyz(555.0)[1];
        assert!((y - 1.0).abs() < 0.01, "{y}");
    }
}

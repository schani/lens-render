//! Optical glass dispersion models.

use anyhow::{Result, bail};

#[derive(Clone, Debug)]
pub enum Glass {
    /// Sellmeier: n² = 1 + Σ Bᵢ λ² / (λ² − Cᵢ), λ in µm.
    Sellmeier { name: String, b: [f64; 3], c: [f64; 3] },
    /// Two-term Cauchy fitted to n_d and Abbe number V_d.
    Cauchy { name: String, a: f64, b: f64 },
}

const CATALOG: &[(&str, [f64; 3], [f64; 3])] = &[
    ("N-BK7", [1.03961212, 0.231792344, 1.01046945], [0.00600069867, 0.0200179144, 103.560653]),
    ("N-SK16", [1.34317774, 0.241144399, 0.994317969], [0.00704687339, 0.0229005, 92.7508526]),
    ("F2", [1.34533359, 0.209073176, 0.937357162], [0.00997743871, 0.0470450767, 111.886764]),
    ("N-SF5", [1.52481889, 0.187085527, 1.42729015], [0.011254756, 0.0588995392, 129.141675]),
    ("N-SF2", [1.47343127, 0.163681849, 1.36920899], [0.0109019098, 0.0585683687, 127.404933]),
    ("N-SF11", [1.73759695, 0.313747346, 1.89878101], [0.013188707, 0.0623068142, 155.23629]),
    ("N-BAK1", [1.12365662, 0.309276848, 0.881511957], [0.00644742752, 0.0222284402, 107.297751]),
    ("N-LAK9", [1.46231905, 0.344399589, 1.15508372], [0.00724270156, 0.0243353131, 85.4686868]),
    ("FUSED-SILICA", [0.6961663, 0.4079426, 0.8974794], [0.004679148, 0.01351206, 97.934]),
];

const ALIASES: &[(&str, &str)] = &[("BK7", "N-BK7"), ("SK16", "N-SK16"), ("SF5", "N-SF5"), ("SF2", "N-SF2"), ("SF11", "N-SF11"), ("BAK1", "N-BAK1"), ("LAK9", "N-LAK9")];

pub const LAMBDA_D: f64 = 587.5618;
pub const LAMBDA_F: f64 = 486.1327;
pub const LAMBDA_C: f64 = 656.2725;

impl Glass {
    /// Looks up a catalogue glass by name, or parses "nd/vd" (e.g. "1.62041/60.32").
    pub fn parse(spec: &str) -> Result<Glass> {
        if let Some((nd, vd)) = spec.split_once('/') {
            let nd: f64 = nd.parse()?;
            let vd: f64 = vd.parse()?;
            return Ok(Glass::from_abbe(spec, nd, vd));
        }
        let upper = spec.to_ascii_uppercase();
        let name = ALIASES.iter().find(|(a, _)| *a == upper).map(|(_, n)| *n).unwrap_or(&upper);
        match CATALOG.iter().find(|(n, _, _)| *n == name) {
            Some((n, b, c)) => Ok(Glass::Sellmeier { name: n.to_string(), b: *b, c: *c }),
            None => bail!("unknown glass '{spec}' (use a catalogue name or nd/vd)"),
        }
    }

    pub fn from_abbe(name: &str, nd: f64, vd: f64) -> Glass {
        let inv2 = |l: f64| 1.0 / (l * 1e-3).powi(2);
        let b = (nd - 1.0) / vd / (inv2(LAMBDA_F) - inv2(LAMBDA_C));
        let a = nd - b * inv2(LAMBDA_D);
        Glass::Cauchy { name: name.to_string(), a, b }
    }

    pub fn name(&self) -> &str {
        match self {
            Glass::Sellmeier { name, .. } | Glass::Cauchy { name, .. } => name,
        }
    }

    /// Refractive index at wavelength `l` in nm.
    pub fn index(&self, l: f64) -> f64 {
        let um2 = (l * 1e-3).powi(2);
        match self {
            Glass::Sellmeier { b, c, .. } => {
                (1.0 + b[0] * um2 / (um2 - c[0]) + b[1] * um2 / (um2 - c[1]) + b[2] * um2 / (um2 - c[2])).sqrt()
            }
            Glass::Cauchy { a, b, .. } => a + b / um2,
        }
    }

    pub fn nd(&self) -> f64 {
        self.index(LAMBDA_D)
    }

    pub fn abbe(&self) -> f64 {
        (self.nd() - 1.0) / (self.index(LAMBDA_F) - self.index(LAMBDA_C))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_values() {
        // (name, nd, vd) from the Schott / Malitson data sheets.
        for (name, nd, vd) in [
            ("N-BK7", 1.51680, 64.17),
            ("N-SK16", 1.62041, 60.32),
            ("F2", 1.62004, 36.37),
            ("N-SF5", 1.67271, 32.25),
            ("N-SF2", 1.64769, 33.82),
            ("N-SF11", 1.78472, 25.68),
            ("N-BAK1", 1.57250, 57.55),
            ("N-LAK9", 1.69100, 54.71),
            ("FUSED-SILICA", 1.45846, 67.82),
        ] {
            let g = Glass::parse(name).unwrap();
            assert!((g.nd() - nd).abs() < 2e-4, "{name} nd {}", g.nd());
            assert!((g.abbe() - vd).abs() < 0.3, "{name} vd {}", g.abbe());
        }
    }

    #[test]
    fn abbe_model() {
        let g = Glass::parse("1.62/36.4").unwrap();
        assert!((g.nd() - 1.62).abs() < 1e-9);
        assert!((g.abbe() - 36.4).abs() < 1e-6);
    }
}

//! Optical glass dispersion models.

use anyhow::{Result, bail};

#[derive(Clone, Debug)]
pub enum Glass {
    /// Sellmeier: n² = 1 + Σ Bᵢ λ² / (λ² − Cᵢ), λ in µm. A term with Cᵢ = 0 adds the constant Bᵢ.
    Sellmeier { name: String, b: &'static [f64], c: &'static [f64] },
    /// Cauchy n = a + b/λ² + c/λ⁴ (λ in µm) fitted to n_d, V_d and, with c, P_g,F.
    Cauchy { name: String, a: f64, b: f64, c: f64 },
}

const fn sq(x: f64) -> f64 {
    x * x
}

// Klein's ZnS fit n² = 8.393 + Aᵢ/(λ² − Cᵢ), rewritten as Sellmeier terms via A/(λ²−C) = (A/C)·λ²/(λ²−C) − A/C.
const ZNS_A: [f64; 2] = [0.14383, 4430.99];
const ZNS_C: [f64; 2] = [sq(0.2421), sq(36.71)];

const CATALOG: &[(&str, &[f64], &[f64])] = &[
    ("N-BK7", &[1.03961212, 0.231792344, 1.01046945], &[0.00600069867, 0.0200179144, 103.560653]),
    ("N-SK16", &[1.34317774, 0.241144399, 0.994317969], &[0.00704687339, 0.0229005, 92.7508526]),
    ("F2", &[1.34533359, 0.209073176, 0.937357162], &[0.00997743871, 0.0470450767, 111.886764]),
    ("N-SF5", &[1.52481889, 0.187085527, 1.42729015], &[0.011254756, 0.0588995392, 129.141675]),
    ("N-SF2", &[1.47343127, 0.163681849, 1.36920899], &[0.0109019098, 0.0585683687, 127.404933]),
    ("N-SF11", &[1.73759695, 0.313747346, 1.89878101], &[0.013188707, 0.0623068142, 155.23629]),
    ("N-BAK1", &[1.12365662, 0.309276848, 0.881511957], &[0.00644742752, 0.0222284402, 107.297751]),
    ("N-LAK9", &[1.46231905, 0.344399589, 1.15508372], &[0.00724270156, 0.0243353131, 85.4686868]),
    // Malitson 1965, J. Opt. Soc. Am. 55, 1205; 20 °C.
    ("FUSED-SILICA", &[0.6961663, 0.4079426, 0.8974794], &[0.004679148, 0.01351206, 97.934]),
    // Schott and Ohara datasheets (Zemax catalogues schott_2017-01-20b, ohara_2017-11-30).
    ("N-LASF9", &[2.00029547, 0.298926886, 1.80691843], &[0.0121426017, 0.0538736236, 156.530829]),
    ("N-LASF31A", &[1.96485075, 0.475231259, 1.48360109], &[0.00982060155, 0.0344713438, 110.739863]),
    ("N-LASF46B", &[2.17988922, 0.306495184, 1.56882437], &[0.0125805384, 0.0567191367, 105.316538]),
    ("S-LAH58", &[1.78764964, 0.6526356, 1.79914564], &[0.00847378536, 0.0313126408, 132.788001]),
    ("S-LAH79", &[2.32557148, 0.507967133, 2.43087198], &[0.0132895208, 0.0528335449, 161.122408]),
    // The glass map's high-index, low-dispersion edge.
    ("N-LAK34", &[1.26661442, 0.665919318, 1.1249612], &[0.00589278062, 0.0197509041, 78.8894174]),
    ("N-LAK33B", &[1.42288601, 0.593661336, 1.1613526], &[0.00670283452, 0.021941621, 80.7407701]),
    ("S-LAH65V", &[1.81419034, 0.361376301, 1.32729484], &[0.00874935029, 0.0318352836, 91.3406898]),
    ("S-LAH55V", &[1.97025325, 0.30489414, 1.39214665], &[0.00991088134, 0.0383202295, 97.7785249]),
    // Summilux reconstruction (same catalogues).
    ("N-FK51A", &[0.971247817, 0.216901417, 0.904651666], &[0.00472301995, 0.0153575612, 168.68133]),
    ("S-NBH51", &[1.71203689, 0.255989588, 1.81456998], &[0.0107724134, 0.0488593504, 136.359013]),
    ("S-TIH13", &[1.62224674, 0.293844589, 1.99225164], &[0.0118368386, 0.0590208025, 171.959976]),
    ("S-LAH59", &[1.51372967, 0.702462343, 1.33600982], &[0.00705246901, 0.0249488689, 100.085908]),
    ("N-PK52A", &[1.029607, 0.1880506, 0.736488165], &[0.00516800155, 0.0166658798, 138.964129]),
    ("N-KZFS4", &[1.35055424, 0.197575506, 1.09962992], &[0.0087628207, 0.0371767201, 90.3866994]),
    ("N-KZFS5", &[1.47460789, 0.193584488, 1.26589974], &[0.00986143816, 0.0445477583, 106.436258]),
    ("N-KZFS8", &[1.62693651, 0.24369876, 1.62007141], &[0.010880863, 0.0494207753, 131.009163]),
    // Malitson 1963, Appl. Opt. 2, 1103; 24 °C.
    ("CAF2", &[0.5675888, 0.4710914, 3.8484723], &[sq(0.050263605), sq(0.1003909), sq(34.649040)]),
    // Peter 1923, Z. Phys. 15, 358.
    ("DIAMOND", &[0.3306, 4.3356], &[sq(0.1750), sq(0.1060)]),
    // Wood & Nassau 1982, Appl. Opt. 21, 2978; ZrO₂ with 12 mol % Y₂O₃, 25 °C.
    ("CUBIC-ZIRCONIA", &[1.347091, 2.117788, 9.452943], &[sq(0.062543), sq(0.166739), sq(24.320570)]),
    // Cubic ZnS: Klein 1986, Appl. Opt. 25, 1873, fit to Debenham 1984, Appl. Opt. 23, 2238; 20 °C.
    ("ZNS", &[ZNS_A[0] / ZNS_C[0], ZNS_A[1] / ZNS_C[1], 8.393 - 1.0 - ZNS_A[0] / ZNS_C[0] - ZNS_A[1] / ZNS_C[1]], &[ZNS_C[0], ZNS_C[1], 0.0]),
    // Zelmon, Small & Page 1998, Appl. Opt. 37, 4933; undoped YAG (Y₃Al₅O₁₂), cubic, also as transparent ceramic.
    ("YAG", &[2.282, 3.27644], &[0.01185, 282.734]),
    // Hrabovský et al. 2021, Opt. Mater. Express 11, 1218: n² = 2.077 + 1.237 λ²/(λ² − 0.1376²) − 0.0104 λ²; the
    // last term as a pole at 10 µm (error < 10⁻⁵ in the visible). LuAG (Lu₃Al₅O₁₂), cubic.
    ("LUAG", &[1.077, 1.237, 1.04], &[0.0, sq(0.1376), 100.0]),
    // Tropf & Thomas 1991 (Palik, Handbook of Optical Constants II, 881), refit in Handbook of Optics 3rd ed., Vol. 4.
    // MgAl₂O₄ spinel, cubic, also as transparent ceramic.
    ("SPINEL", &[1.8938, 3.0755], &[sq(0.09942), sq(15.826)]),
    // Malitson 1964, J. Opt. Soc. Am. 54, 628; 25 °C.
    ("BAF2", &[0.643356, 0.506762, 3.8261], &[sq(0.057789), sq(0.10968), sq(46.3864)]),
    // Li 1976, J. Phys. Chem. Ref. Data 5, 329; 24 °C.
    ("LIF", &[0.92549, 6.96747], &[sq(0.07376), sq(32.790)]),
    // PMMA: Szczurowski 2013 (refractiveindex.info), 0.40–1.08 µm.
    ("PMMA", &[0.99654, 0.18964, 0.00411], &[0.00787, 0.02191, 3.85727]),
    // Polycarbonate and polystyrene: two-term fits to Zhang et al. 2020, Appl. Opt. 59, 2337 (tabulated, 0.40–1.0 µm;
    // residual 5e-4 and 1.5e-4). Partial dispersion uncertain by ~0.02 (data noise); Sultanova et al. 2009's one-term
    // fits (Acta Phys. Pol. A 116, 585) put it 0.01–0.015 lower still.
    ("POLYCARBONATE", &[1.424723, -14.921249], &[0.022650, 1000.0]),
    ("POLYSTYRENE", &[1.435620, -7.167381], &[0.019881, 1000.0]),
    // Daimon & Masumura 2007, Appl. Opt. 46, 3811; distilled water, 20 °C.
    ("WATER", &[5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1], &[5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1]),
];

const ALIASES: &[(&str, &str)] = &[("BK7", "N-BK7"), ("SK16", "N-SK16"), ("SF5", "N-SF5"), ("SF2", "N-SF2"), ("SF11", "N-SF11"), ("BAK1", "N-BAK1"), ("LAK9", "N-LAK9"), ("LASF9", "N-LASF9"), ("LASF31A", "N-LASF31A"), ("LASF46B", "N-LASF46B"), ("LAH58", "S-LAH58"), ("LAH79", "S-LAH79"), ("PK52A", "N-PK52A"), ("KZFS4", "N-KZFS4"), ("KZFS5", "N-KZFS5"), ("KZFS8", "N-KZFS8"), ("FLUORITE", "CAF2"), ("PC", "POLYCARBONATE"), ("PS", "POLYSTYRENE"), ("CZ", "CUBIC-ZIRCONIA"), ("H2O", "WATER")];

pub const LAMBDA_D: f64 = 587.5618;
pub const LAMBDA_F: f64 = 486.1327;
pub const LAMBDA_C: f64 = 656.2725;

impl Glass {
    /// Looks up a catalogue glass by name, or parses "nd/vd" (e.g. "1.62041/60.32").
    pub fn parse(spec: &str) -> Result<Glass> {
        if let Some((nd, vd)) = spec.split_once('/') {
            let nd: f64 = nd.parse()?;
            // nd/vd/ΔP_g,F: a model glass with a realistic secondary spectrum.
            if let Some((vd, dp)) = vd.split_once('/') {
                return Ok(Glass::model(spec, nd, vd.parse()?, dp.parse()?));
            }
            let vd: f64 = vd.parse()?;
            return Ok(Glass::from_abbe(spec, nd, vd));
        }
        let upper = spec.to_ascii_uppercase();
        let name = ALIASES.iter().find(|(a, _)| *a == upper).map(|(_, n)| *n).unwrap_or(&upper);
        match CATALOG.iter().find(|(n, _, _)| *n == name) {
            Some((n, b, c)) => Ok(Glass::Sellmeier { name: n.to_string(), b, c }),
            None => bail!("unknown glass '{spec}' (use a catalogue name or nd/vd)"),
        }
    }

    /// Names of the catalogue glasses.
    pub fn catalog() -> impl Iterator<Item = &'static str> {
        CATALOG.iter().map(|(n, _, _)| *n)
    }

    pub fn from_abbe(name: &str, nd: f64, vd: f64) -> Glass {
        let inv2 = |l: f64| 1.0 / (l * 1e-3).powi(2);
        let b = (nd - 1.0) / vd / (inv2(LAMBDA_F) - inv2(LAMBDA_C));
        let a = nd - b * inv2(LAMBDA_D);
        Glass::Cauchy { name: name.to_string(), a, b, c: 0.0 }
    }

    /// Model glass: three-term Cauchy through n_d, V_d and a relative partial
    /// dispersion P_g,F = 0.6438 − 0.001682 V_d + `dp` (Schott's normal line
    /// plus the anomaly ΔP_g,F). `from_abbe` has P_g,F ≈ 0.54 whatever V_d.
    pub fn model(name: &str, nd: f64, vd: f64, dp: f64) -> Glass {
        let u = |l: f64| 1.0 / (l * 1e-3).powi(2);
        let (ud, uf, uc, ug) = (u(LAMBDA_D), u(LAMBDA_F), u(LAMBDA_C), u(435.8343));
        let dn = (nd - 1.0) / vd;
        let p = 0.6438 - 0.001682 * vd + dp;
        let (a11, a12, a21, a22) = (uf - uc, uf * uf - uc * uc, ug - uf, ug * ug - uf * uf);
        let det = a11 * a22 - a12 * a21;
        let b = (dn * a22 - a12 * p * dn) / det;
        let c = (a11 * p * dn - a21 * dn) / det;
        Glass::Cauchy { name: name.to_string(), a: nd - b * ud - c * ud * ud, b, c }
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
            Glass::Sellmeier { b, c, .. } => b.iter().zip(c.iter()).fold(1.0, |n2, (b, c)| n2 + b * um2 / (um2 - c)).sqrt(),
            Glass::Cauchy { a, b, c, .. } => a + b / um2 + c / (um2 * um2),
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
        // (name, nd, vd) from the Schott / Ohara / Malitson data sheets.
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
            ("N-LASF9", 1.85025, 32.17),
            ("N-LASF31A", 1.88300, 40.76),
            ("N-LASF46B", 1.90366, 31.32),
            ("S-LAH58", 1.88300, 40.77),
            ("S-LAH79", 2.00330, 28.27),
            ("N-LAK34", 1.72916, 54.50),
            ("N-LAK33B", 1.75500, 52.30),
            ("S-LAH65V", 1.80400, 46.58),
            ("S-LAH55V", 1.83481, 42.73),
            ("N-FK51A", 1.48656, 84.47),
            ("S-NBH51", 1.74950, 35.33),
            ("S-TIH13", 1.74077, 27.79),
            ("S-LAH59", 1.81600, 46.62),
            ("N-PK52A", 1.49700, 81.61),
            ("N-KZFS4", 1.61336, 44.49),
            ("N-KZFS5", 1.65412, 39.70),
            ("N-KZFS8", 1.72047, 34.70),
        ] {
            let g = Glass::parse(name).unwrap();
            assert!((g.nd() - nd).abs() < 2e-4, "{name} nd {}", g.nd());
            assert!((g.abbe() - vd).abs() < 0.3, "{name} vd {}", g.abbe());
        }
    }

    #[test]
    fn crystal_and_liquid_values() {
        // (name, λ in nm, n(λ), vd) from sources independent of the catalogue formula. The tolerance on n is
        // looser than for glasses: different samples, and diamond samples alone spread by 10⁻³ (Edwards & Ochoa).
        for (name, l, n, vd) in [
            ("CAF2", LAMBDA_D, 1.43384, 95.23),                    // Hellma Materials CaF₂ datasheet.
            ("DIAMOND", 589.0, 2.4175, f64::NAN),                  // Edwards & Ochoa 1981, J. Opt. Soc. Am. 71, 607, Table 1.
            ("CUBIC-ZIRCONIA", 589.3, 2.15847, 1.15847 / 0.03455), // Wood & Nassau 1982: N_D and N_F − N_C.
            ("ZNS", LAMBDA_D, 2.36775, 15.41),                     // Li 1984, J. Phys. Chem. Ref. Data 13, 103, Eq. (19), 293 K.
            ("WATER", LAMBDA_D, 1.33340, 55.78),                   // IAPWS R9-97, 20 °C, 0.1 MPa.
        ] {
            let g = Glass::parse(name).unwrap();
            assert!((g.index(l) - n).abs() < 5e-4, "{name} n {}", g.index(l));
            assert!(vd.is_nan() || (g.abbe() - vd).abs() < 0.3, "{name} vd {}", g.abbe());
        }
        // Against sources other than the catalogue's (polymer samples spread by ~0.005).
        for (name, l, n, tol) in [
            ("YAG", 600.0, 1.8347, 4e-3),           // Bond 1965, J. Appl. Phys. 36, 1674.
            ("YAG", 450.0, 1.8532, 4e-3),           // Bond 1965.
            ("LUAG", 632.8, 1.8361, 1e-4),          // Hrabovský et al. 2021, exact formula.
            ("SPINEL", 587.6, 1.7162, 1e-3),        // Regression: no independent value at hand.
            ("BAF2", LAMBDA_D, 1.47456, 2e-4),      // Li 1980, J. Phys. Chem. Ref. Data 9, 161.
            ("LIF", LAMBDA_D, 1.3921, 3e-4),        // Regression: no independent value at hand.
            ("PMMA", LAMBDA_D, 1.49253, 2.5e-3),    // Beadie et al. 2015, Appl. Opt. 54, F139.
            ("POLYCARBONATE", 600.0, 1.5835, 6e-3), // Sultanova et al. 2009.
            ("POLYSTYRENE", 600.0, 1.5904, 6e-3),   // Sultanova et al. 2009.
        ] {
            let g = Glass::parse(name).unwrap();
            assert!((g.index(l) - n).abs() < tol, "{name} n({l}) {}", g.index(l));
        }
        // Abbe numbers of these formulas (regression; Beadie's PMMA: 59.4).
        for (name, vd) in [("YAG", 52.6), ("LUAG", 58.4), ("SPINEL", 60.7), ("BAF2", 81.6), ("LIF", 97.3), ("PMMA", 58.0)] {
            assert!((Glass::parse(name).unwrap().abbe() - vd).abs() < 0.3, "{name} vd");
        }
        // Diamond's gemmological dispersion n(430.8 nm) − n(686.7 nm) is 0.044.
        let d = Glass::parse("DIAMOND").unwrap();
        assert!((d.index(430.8) - d.index(686.7) - 0.044).abs() < 1e-3);
    }

    #[test]
    fn anomalous_partial_dispersion() {
        // ΔP_g,F from the Schott datasheets, relative to the normal line P_g,F = 0.6438 − 0.001682 V_d.
        for (name, dp) in [("N-PK52A", 0.0311), ("N-KZFS4", -0.0100), ("N-FK51A", 0.0342)] {
            let g = Glass::parse(name).unwrap();
            let p = (g.index(435.8343) - g.index(LAMBDA_F)) / (g.index(LAMBDA_F) - g.index(LAMBDA_C));
            let dp_calc = p - (0.6438 - 0.001682 * g.abbe());
            assert!((dp_calc - dp).abs() < 5e-4, "{name} ΔPgF {dp_calc}");
        }
    }

    #[test]
    fn abbe_model() {
        let g = Glass::parse("1.62/36.4").unwrap();
        assert!((g.nd() - 1.62).abs() < 1e-9);
        assert!((g.abbe() - 36.4).abs() < 1e-6);
        // With ΔP_g,F: n_d, V_d and P_g,F as given.
        for (spec, nd, vd, dp) in [("1.9185/37.1/0", 1.9185, 37.1, 0.0), ("1.48656/84.47/0.0342", 1.48656, 84.47, 0.0342)] {
            let g = Glass::parse(spec).unwrap();
            let p = (g.index(435.8343) - g.index(LAMBDA_F)) / (g.index(LAMBDA_F) - g.index(LAMBDA_C));
            assert!((g.nd() - nd).abs() < 1e-9 && (g.abbe() - vd).abs() < 1e-6, "{spec}");
            assert!((p - (0.6438 - 0.001682 * vd + dp)).abs() < 1e-9, "{spec} P_gF {p}");
        }
        // It follows a catalogue glass of the same n_d, V_d and ΔP_g,F across the visible.
        let (cat, model) = (Glass::parse("N-FK51A").unwrap(), Glass::parse("1.48656/84.47/0.0342").unwrap());
        for l in [420.0, 500.0, 600.0, 700.0] {
            assert!((cat.index(l) - model.index(l)).abs() < 2e-4, "{l} nm: {} vs {}", cat.index(l), model.index(l));
        }
    }
}

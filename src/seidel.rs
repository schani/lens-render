//! Third-order (Seidel) aberration contributions of each surface, from
//! paraxial traces of the marginal and chief rays (Welford, Aberrations of
//! Optical Systems, ch. 8). Sign conventions follow Welford: positive S_I is
//! undercorrected spherical aberration.

use crate::glass::{LAMBDA_C, LAMBDA_D, LAMBDA_F};
use crate::lens::Lens;

/// Wavelength (nm) the wavefront coefficients are expressed in.
pub const LAMBDA_W: f64 = 550.0;

/// Names of the coefficients, in the order of the arrays in `Seidel`.
pub const TERMS: [&str; 7] = ["spherical W040", "coma W131", "astigmatism W222", "Petzval W220P", "distortion W311", "axial colour W020", "lateral colour W111"];

pub struct Seidel {
    /// Per surface: W040, W131, W222, W220P, W311, W020, W111 in waves at `LAMBDA_W`.
    pub surfaces: Vec<[f64; 7]>,
    pub sum: [f64; 7],
    pub stop_index: usize,
}

/// Seidel sums S_I…S_V, C_I, C_II (mm) of each surface. The marginal ray
/// fills the entrance pupil from the axial object at `distance` mm (None =
/// infinity); the chief ray has object-space slope tan `theta` and passes the
/// centre of the stop. Colour uses n_F − n_C.
fn seidel_sums(lens: &Lens, theta: f64, distance: Option<f64>) -> Vec<[f64; 7]> {
    let par = lens.paraxial_data(LAMBDA_D);
    let r_ep = lens.iris.max_radius / par.entrance_pupil_mag;
    let z_ep = par.entrance_pupil_z;
    // Slopes in object space, heights at the front vertex.
    let (mut u, mut y) = match distance {
        None => (0.0, r_ep),
        Some(d) => (r_ep / (d + z_ep), r_ep * d / (d + z_ep)),
    };
    let mut ub = theta.tan();
    let mut yb = -ub * z_ep;
    let (m, mf, mc) = (lens.media(LAMBDA_D), lens.media(LAMBDA_F), lens.media(LAMBDA_C));
    let h = m.n[0] * (ub * y - u * yb);
    let disp = |k: usize| (mf.n[k] - mc.n[k]) / m.n[k];
    lens.surfaces
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (n, n2) = (m.n[i], m.n[i + 1]);
            let c = s.curvature();
            let a = n * (u + y * c);
            let ab = n * (ub + yb * c);
            let u2 = (n * u - y * c * (n2 - n)) / n2;
            let ub2 = (n * ub - yb * c * (n2 - n)) / n2;
            let du = u2 / n2 - u / n;
            let d1 = 1.0 / n2 - 1.0 / n;
            let d2 = 1.0 / (n2 * n2) - 1.0 / (n * n);
            let dd = disp(i + 1) - disp(i);
            // Fourth-order departure b r⁴ of the sag from the base sphere.
            let b = s.conic * c.powi(3) / 8.0 + s.aspheric.first().copied().unwrap_or(0.0);
            let e = 8.0 * b * (n2 - n);
            let out = [
                -a * a * y * du + e * y.powi(4),
                -a * ab * y * du + e * y.powi(3) * yb,
                -ab * ab * y * du + e * y * y * yb * yb,
                -h * h * c * d1,
                // (Ā/A)(S_III + S_IV) without the division, which fails where A = 0.
                -ab.powi(3) * y * d2 + ab * yb * c * (2.0 * ab * y - a * yb) * d1 + e * y * yb.powi(3),
                a * y * dd,
                ab * y * dd,
            ];
            y += u2 * s.thickness;
            yb += ub2 * s.thickness;
            (u, ub) = (u2, ub2);
            out
        })
        .collect()
}

/// Per-surface Seidel wavefront coefficients (waves at `LAMBDA_W`) for the
/// full field angle `theta` (radians) and object `distance` mm from the front
/// vertex (None = infinity).
pub fn seidel(lens: &Lens, theta: f64, distance: Option<f64>) -> Seidel {
    const SCALE: [f64; 7] = [1.0 / 8.0, 0.5, 0.5, 0.25, 0.5, 0.5, 1.0];
    let lambda = LAMBDA_W * 1e-6;
    let surfaces: Vec<[f64; 7]> = seidel_sums(lens, theta, distance).into_iter().map(|s| std::array::from_fn(|k| s[k] * SCALE[k] / lambda)).collect();
    let sum = std::array::from_fn(|k| surfaces.iter().map(|s| s[k]).sum());
    Seidel { surfaces, sum, stop_index: lens.stop_index }
}

/// Seidel sums `s.sum` as transverse ray aberrations on the sensor in µm:
/// the image-plane error of the pupil-edge ray, ε = 2N ∂W/∂ρ (N the image-side
/// working f-number), for the ray in the fan that shows it most:
/// spherical 8N·W040, tangential coma 6N·W131, astigmatism 4N·W222,
/// field curvature 4N·W220P, distortion 2N·W311, axial colour 4N·W020,
/// lateral colour 2N·W111 (× λ). Signed; the size is the magnitude.
pub fn transverse_um(s: &Seidel, fnumber: f64) -> [f64; 7] {
    const K: [f64; 7] = [8.0, 6.0, 4.0, 4.0, 2.0, 4.0, 2.0];
    std::array::from_fn(|k| K[k] * fnumber * s.sum[k] * LAMBDA_W * 1e-3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Ray, Vec3};

    fn total(s: &[[f64; 7]], k: usize) -> f64 {
        s.iter().map(|x| x[k]).sum()
    }

    /// Thin lens at the stop, object at infinity, against the closed-form
    /// thin-lens Seidel sums (Welford §8.5).
    #[test]
    fn thin_lens() {
        let (n, v, y, th): (f64, f64, f64, f64) = (1.5, 60.0, 10.0, 0.1);
        let phi: f64 = 0.01;
        let h = th.tan() * y;
        // Shape factor B = (c1 + c2) / (c1 − c2); conjugate factor C = −1 at infinity.
        for (text, b) in [("stop 0 air 10\n50 0 1.5/60 10\ninf 100 air 10\n", 1.0), ("stop 0 air 10\ninf 0 1.5/60 10\n-50 100 air 10\n", -1.0)] {
            let lens = Lens::parse(text).unwrap();
            let s = seidel_sums(&lens, th, None);
            let c: f64 = -1.0;
            let s1 = y.powi(4) * phi.powi(3) / 4.0 * ((n / (n - 1.0)).powi(2) + (n + 2.0) / (n * (n - 1.0).powi(2)) * (b + 2.0 * (n * n - 1.0) / (n + 2.0) * c).powi(2) - n / (n + 2.0) * c * c);
            let s2 = y * y * phi * phi * h / 2.0 * ((n + 1.0) / (n * (n - 1.0)) * b + (2.0 * n + 1.0) / n * c);
            let close = |a: f64, b: f64| (a - b).abs() < 1e-9 * (1.0 + b.abs());
            assert!(close(total(&s, 0), s1), "S_I {} vs {s1}", total(&s, 0));
            assert!(close(total(&s, 1).abs(), s2.abs()), "S_II {} vs {s2}", total(&s, 1));
            assert!(close(total(&s, 2), h * h * phi), "S_III");
            assert!(close(total(&s, 3), h * h * phi / n), "S_IV");
            assert!(close(total(&s, 4), 0.0), "S_V");
            assert!(close(total(&s, 5), y * y * phi / v), "C_I {}", total(&s, 5));
            assert!(close(total(&s, 6), 0.0), "C_II");
        }
    }

    /// An ellipsoid with k = −1/n² images an infinite object into glass free
    /// of spherical aberration.
    #[test]
    fn aspheric_term() {
        let lens = Lens::parse("stop 0 air 10\n50 100 1.5/60 10 k=-0.4444444444444444\n").unwrap();
        let s = seidel_sums(&lens, 0.1, None);
        let sphere = seidel_sums(&Lens::parse("stop 0 air 10\n50 100 1.5/60 10\n").unwrap(), 0.1, None);
        assert!(total(&s, 0).abs() < 1e-9 * total(&sphere, 0).abs(), "{}", total(&s, 0));
    }

    /// Petzval sum H² Σ c (n' − n) / (n n') of the Cooke triplet.
    #[test]
    fn petzval_sum() {
        let lens = Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap();
        let th = 0.35_f64;
        let par = lens.paraxial_data(LAMBDA_D);
        let h = th.tan() * lens.iris.max_radius / par.entrance_pupil_mag;
        let m = lens.media(LAMBDA_D);
        let p: f64 = lens.surfaces.iter().enumerate().map(|(i, s)| s.curvature() * (m.n[i + 1] - m.n[i]) / (m.n[i] * m.n[i + 1])).sum();
        let s4 = total(&seidel_sums(&lens, th, None), 3);
        assert!((s4 - h * h * p).abs() < 1e-12, "{s4} vs {}", h * h * p);
        let w = seidel(&lens, th, None);
        assert!((w.sum[3] - s4 / 4.0 / (LAMBDA_W * 1e-6)).abs() < 1e-9);
    }

    /// Longitudinal spherical aberration −S_I ρ² / (2 n' u'²) of the Cooke
    /// triplet against real rays.
    #[test]
    fn triplet_spherical() {
        let lens = Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap();
        let par = lens.paraxial_data(LAMBDA_D);
        let r_ep = lens.iris.max_radius / par.entrance_pupil_mag;
        let s1 = total(&seidel_sums(&lens, 0.0, None), 0);
        let u = -r_ep / par.efl;
        let m = lens.media(LAMBDA_D);
        let z_par = lens.paraxial_focus(None, LAMBDA_D);
        let lsa = |rho: f64| {
            let ray = Ray { o: Vec3::new(0.0, rho * r_ep, -1.0), d: Vec3::new(0.0, 0.0, 1.0) };
            let (r, _, _) = lens.trace_path_opl(ray, LAMBDA_D, &m, &lens.forward_path, false).unwrap();
            r.o.z - r.o.y / r.d.y * r.d.z - z_par
        };
        // Higher orders vanish as ρ → 0; at full aperture they balance part of it.
        for (rho, tol) in [(0.05, 0.002), (0.1, 0.007)] {
            let seidel = -s1 * rho * rho / (2.0 * u * u);
            assert!((lsa(rho) / seidel - 1.0).abs() < tol, "ρ {rho}: real {} vs Seidel {seidel}", lsa(rho));
        }
        assert!(s1 > 0.0 && lsa(1.0) < 0.0, "undercorrected");
    }

    /// Transverse spherical aberration 8N·W040 against the real marginal
    /// ray of a slow singlet (third order dominates), at the paraxial focus.
    #[test]
    fn transverse_spherical() {
        let lens = Lens::parse("stop 0 air 2\n50 3 N-BK7 3\n-50 50 air 3\n").unwrap();
        let w = seidel(&lens, 0.0, None);
        let n_w = lens.paraxial_data(LAMBDA_D).efl / (2.0 * lens.iris.max_radius);
        let t = transverse_um(&w, n_w);
        // Seidel sums use d-line indices.
        let m = lens.media(LAMBDA_D);
        let z = lens.paraxial_focus(None, LAMBDA_D);
        let (r, _) = lens.trace_forward(Ray { o: Vec3::new(0.0, 2.0, -1.0), d: Vec3::new(0.0, 0.0, 1.0) }, LAMBDA_D, &m).unwrap();
        let real = (r.o.y + (z - r.o.z) * r.d.y / r.d.z) * 1000.0;
        assert!((t[0].abs() / real.abs() - 1.0).abs() < 0.03, "8N·W040 {} vs real {real}", t[0]);
        assert!(t[0] * real < 0.0, "undercorrected: marginal ray crosses before focus");
        assert_eq!(t[4], 0.0);
    }
}

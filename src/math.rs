use std::ops::{Add, AddAssign, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        self * (1.0 / self.length())
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
}

/// PCG32 random number generator.
#[derive(Clone)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Rng { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) * (1.0 / 4294967296.0)
    }
}

/// Roberts' R_d low-discrepancy sequence (generalised golden ratio), 5 dimensions.
pub struct R5 {
    alpha: [f64; 5],
}

impl R5 {
    pub fn new() -> Self {
        // phi_5: unique positive root of x^6 = x + 1
        let mut g = 1.0f64;
        for _ in 0..50 {
            g = (1.0 + g).powf(1.0 / 6.0);
        }
        let mut alpha = [0.0; 5];
        for (j, a) in alpha.iter_mut().enumerate() {
            *a = (1.0 / g.powi(j as i32 + 1)).fract();
        }
        R5 { alpha }
    }

    /// The generators as 0.32 fixed-point numbers.
    pub fn alphas_fixed(&self) -> [u32; 5] {
        self.alpha.map(|a| (a * 4294967296.0) as u64 as u32)
    }

    /// Sample `i`, shifted by a per-pixel offset (Cranley-Patterson rotation).
    pub fn sample(&self, i: u64, shift: &[f64; 5]) -> [f64; 5] {
        let mut out = [0.0; 5];
        for j in 0..5 {
            out[j] = (shift[j] + self.alpha[j] * (i as f64 + 1.0)).fract();
        }
        out
    }
}

/// Uniform sample on the unit disk (concentric mapping).
pub fn concentric_disk(u: f64, v: f64) -> (f64, f64) {
    let a = 2.0 * u - 1.0;
    let b = 2.0 * v - 1.0;
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }
    let (r, theta) = if a.abs() > b.abs() {
        (a, std::f64::consts::FRAC_PI_4 * (b / a))
    } else {
        (b, std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4 * (a / b))
    };
    (r * theta.cos(), r * theta.sin())
}

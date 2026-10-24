// Shared declarations. Field order must match `Params` in src/gpu/mod.rs.
struct Params {
    w: u32, h: u32, bins: u32, spp: u32,
    frame: u32, n_surf: u32, n_rects: u32, n_lights: u32,
    bounds_mode: u32, bounds_nodes: u32, grid_nx: u32, grid_ny: u32,
    bounds_g: u32, n_ghosts: u32, veil: u32, coating: u32,
    iris_blades: u32, iris_circular: u32, light_grid: u32, override_iris: u32,
    pitch: f32, sensor_z: f32, rear_z: f32, norm: f32,
    r_max: f32, half_w: f32, half_h: f32, bounds_half: f32,
    iris_rotation: f32, iris_curvature: f32, iris_apothem: f32, iris_max_radius: f32,
    coat_n: f32, coat_l0: f32, front_z: f32, ep_z: f32,
    front_half: f32, light_bounds_g: u32, frames: f32, tonemap: u32,
    a0: u32, a1: u32, a2: u32, a3: u32,
    a4: u32, media_off: u32, up_off: u32, spectra_off: u32,
    palette_off: u32, cmf_off: u32, path_stride: u32, path_base: u32,
    light_base: u32, bg_r: f32, bg_g: f32, bg_b: f32,
    m0: f32, m1: f32, m2: f32, m3: f32,
    m4: f32, m5: f32, m6: f32, m7: f32,
    m8: f32, exposure: f32, n_tasks: u32, bank_layout_off: u32,
    bank_on: u32, bank_nt: u32, bank_nv: u32, bank_na: u32,
    bank_half: u32, bank_refs_off: u32, bank_cdf_off: u32, psf_margin: u32,
    bank_tan_max: f32, bank_v0: f32, bank_dv: f32, bank_period: f32,
}

const LAMBDA_MIN: f32 = 380.0;
const LAMBDA_RANGE: f32 = 400.0;
const PI: f32 = 3.14159265358979;
const LIGHT_BOUNDS_BASE: u32 = 2048u;
const FIXED: f32 = 1000.0;

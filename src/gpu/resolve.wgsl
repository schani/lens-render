// Converts the accumulated spectral film to an image. Prepended with common.wgsl.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(1) @binding(0) var<storage, read_write> film: array<FILM_ATOMIC>;
@group(1) @binding(1) var<storage, read_write> lowres: array<FILM_ATOMIC>;
@group(2) @binding(0) var<storage, read_write> spectral: array<f32>;
@group(2) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

fn lowres_at(x: i32, y: i32, b: u32) -> f32 {
    let lw = i32((P.w + 3u) / 4u);
    let lh = i32((P.h + 3u) / 4u);
    let i = u32(clamp(y, 0, lh - 1) * lw + clamp(x, 0, lw - 1)) * P.bins + b;
    return LOWRES_LOAD;
}

// spectral = (film + upsampled quarter-resolution ghosts) / frames
@compute @workgroup_size(256)
fn normalize_film(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= P.w * P.h * P.bins) { return; }
    let b = i % P.bins;
    let pix = i / P.bins;
    let fx = (f32(pix % P.w) + 0.5) * 0.25 - 0.5;
    let fy = (f32(pix / P.w) + 0.5) * 0.25 - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let top = mix(lowres_at(x0, y0, b), lowres_at(x0 + 1, y0, b), tx);
    let bot = mix(lowres_at(x0, y0 + 1, b), lowres_at(x0 + 1, y0 + 1, b), tx);
    // A quarter-resolution pixel spreads over 16 full-resolution ones.
    spectral[i] = (FILM_LOAD + mix(top, bot, ty) / 16.0) / P.frames;
}

fn srgb_encode(v: f32) -> f32 {
    let x = clamp(v, 0.0, 1.0);
    return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308);
}

fn tonemap(v: f32) -> f32 {
    let x = max(v, 0.0);
    if (P.tonemap == 1u) {
        let y = x * 0.6;
        return (y * (2.51 * y + 0.03)) / (y * (2.43 * y + 0.59) + 0.14);
    }
    return x;
}

@compute @workgroup_size(8, 8)
fn to_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let pix = gid.y * P.w + gid.x;
    var xyz = vec3<f32>(0.0);
    for (var b = 0u; b < P.bins; b++) {
        let c = P.cmf_off + 3u * b;
        xyz += spectral[pix * P.bins + b] * vec3<f32>(tables[c], tables[c + 1u], tables[c + 2u]);
    }
    let rgb = vec3<f32>(P.m0 * xyz.x + P.m1 * xyz.y + P.m2 * xyz.z,
                        P.m3 * xyz.x + P.m4 * xyz.y + P.m5 * xyz.z,
                        P.m6 * xyz.x + P.m7 * xyz.y + P.m8 * xyz.z);
    // Highlights far above white clip to white, as on a sensor.
    let y = max(0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z, 0.0);
    let t = clamp((y - 1.0) / 3.0, 0.0, 1.0);
    let white = t * t * (3.0 - 2.0 * t);
    var out = vec3<f32>(0.0);
    for (var k = 0; k < 3; k++) {
        let v = clamp(tonemap(rgb[k]), 0.0, 1.0);
        out[k] = srgb_encode(v + (1.0 - v) * white);
    }
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(out, 1.0));
}

// Transposes, packing and kernel multiplication for the FFT convolution.
// Layouts: "rows" = ph rows of pw; "transposed" = pw rows of ph.

struct FftParams {
    w: u32, h: u32, pw: u32, ph: u32,
    bins: u32, b0: u32, b1: u32, inverse: u32,
    rows: u32, cols: u32, kbin: u32, koff: u32,
    khalf: u32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> F: FftParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> spectral: array<f32>;
@group(0) @binding(6) var<storage, read_write> kspec: array<f32>;
@group(0) @binding(7) var<storage, read> kimg: array<f32>;

var<workgroup> tile: array<vec2<f32>, 272>;

// dst (cols × rows) = transpose of src (rows × cols).
@compute @workgroup_size(16, 16)
fn transpose(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let x = wg.x * 16u + lid.x;
    let y = wg.y * 16u + lid.y;
    if (x < F.cols && y < F.rows) {
        tile[lid.y * 17u + lid.x] = src[y * F.cols + x];
    }
    workgroupBarrier();
    let ox = wg.y * 16u + lid.x;
    let oy = wg.x * 16u + lid.y;
    if (ox < F.rows && oy < F.cols) {
        dst[oy * F.rows + ox] = tile[lid.x * 17u + lid.y];
    }
}

// data (rows layout) = bins b0 + i·b1 of the film, zero padded.
@compute @workgroup_size(16, 16)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= F.pw || gid.y >= F.ph) { return; }
    var v = vec2<f32>(0.0);
    if (gid.x < F.w && gid.y < F.h) {
        let p = (gid.y * F.w + gid.x) * F.bins;
        v.x = spectral[p + F.b0];
        if (F.b1 < F.bins) { v.y = spectral[p + F.b1]; }
    }
    data[gid.y * F.pw + gid.x] = v;
}

// Separates the spectra of the two packed real images, multiplies each by
// its kernel spectrum (real: the kernels are even) and recombines.
// src and dst use the transposed layout.
@compute @workgroup_size(16, 16)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ky = gid.x;
    let kx = gid.y;
    if (ky >= F.ph || kx >= F.pw) { return; }
    let i = kx * F.ph + ky;
    let ni = ((F.pw - kx) % F.pw) * F.ph + (F.ph - ky) % F.ph;
    let z = src[i];
    let zn = vec2<f32>(src[ni].x, -src[ni].y);
    let a = 0.5 * (z + zn);
    let d = 0.5 * (z - zn);
    let b = vec2<f32>(d.y, -d.x); // (z − conj z(−k)) / 2i
    let size = F.pw * F.ph;
    let ka = kspec[F.b0 * size + i];
    var kb = 0.0;
    if (F.b1 < F.bins) { kb = kspec[F.b1 * size + i]; }
    // a·ka + i·b·kb
    dst[i] = a * ka + vec2<f32>(-b.y, b.x) * kb;
}

// Film bins b0, b1 = real, imaginary parts of src (rows layout) / (pw·ph).
@compute @workgroup_size(16, 16)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= F.w || gid.y >= F.h) { return; }
    let v = src[gid.y * F.pw + gid.x] / f32(F.pw * F.ph);
    let p = (gid.y * F.w + gid.x) * F.bins;
    spectral[p + F.b0] = max(v.x, 0.0);
    if (F.b1 < F.bins) { spectral[p + F.b1] = max(v.y, 0.0); }
}

// data (rows layout) = kernel `kbin`, centred at the origin with wrap-around.
@compute @workgroup_size(16, 16)
fn place_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= F.pw || gid.y >= F.ph) { return; }
    let size = 2u * F.khalf + 1u;
    // Signed offsets of this cell from the origin.
    let dx = select(i32(gid.x), i32(gid.x) - i32(F.pw), gid.x >= F.pw / 2u);
    let dy = select(i32(gid.y), i32(gid.y) - i32(F.ph), gid.y >= F.ph / 2u);
    var v = 0.0;
    if (abs(dx) <= i32(F.khalf) && abs(dy) <= i32(F.khalf)) {
        v = kimg[F.koff + u32(dy + i32(F.khalf)) * size + u32(dx + i32(F.khalf))];
    }
    data[gid.y * F.pw + gid.x] = vec2<f32>(v, 0.0);
}

// kspec[kbin] = real part of src (transposed layout), normalised by its DC term.
@compute @workgroup_size(256)
fn store_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = F.pw * F.ph;
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= size) { return; }
    kspec[F.kbin * size + i] = src[i].x / src[0].x;
}

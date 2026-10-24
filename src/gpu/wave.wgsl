// Wave-optical PSF kernels for the PSF bank (see wave.rs), appended to
// trace.wgsl for its lens tracer. A batch holds entries of one transform
// size NN = 2n; the FFT itself runs in fft.wgsl.

struct WaveEntry {
    // Point source (w = 1) or direction of collimated light (w = 0).
    src: vec4<f32>,
    // Pupil sample (i, j) lies at (x0 + (i + ½) du, y0 + (j + ½) du, z): x0, y0, z, du.
    grid: vec4<f32>,
    lambda: f32,
    // Wavelength in mm.
    wave: f32,
    n: u32,
    step: u32,
    nodes_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct WaveBatch {
    nn: u32,
    count: u32,
    pitch: f32,
    front_z: f32,
}

@group(1) @binding(3) var<uniform> WB: WaveBatch;
@group(1) @binding(4) var<storage, read> wentries: array<WaveEntry>;
@group(1) @binding(5) var<storage, read> wnodes: array<f32>;
@group(1) @binding(6) var<storage, read_write> wgrid: array<vec2<f32>>;
@group(1) @binding(7) var<storage, read_write> wout: array<f32>;
// Per entry: hc, hw, sub.
@group(1) @binding(8) var<storage, read_write> wsizes: array<vec4<u32>>;
// Per entry, 16 words: sums over passing samples for the affine fit of exit
// direction cosines (L, M) over pupil coordinates (u, v), as f32 bits:
// 1, u, v, uu, uv, vv, L, M, Lu, Lv, Mu, Mv.
@group(1) @binding(9) var<storage, read_write> wmoments: array<atomic<u32>>;
// Per entry: sensor offset (mm) of frequency (kx, ky) is (x kx + y ky, z kx + w ky).
@group(1) @binding(10) var<storage, read_write> wmaps: array<vec4<f32>>;
// Per entry: the wing's energy by pixel-centre radius, rounded up (f32 bits).
@group(1) @binding(11) var<storage, read_write> whist: array<atomic<u32>>;

// Wing: 129² pixels around the reference; core: up to 128² sub-pixel cells.
const WING: u32 = 129u;
const WING_HALF: u32 = 64u;
const CORE: u32 = 128u;
const OUT_STRIDE: u32 = 33025u; // WING² + CORE²

// Cubic through nodes at −1, 0, 1, 2, evaluated at t in [0, 1].
fn cubic(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let u = t - 1.0;
    let v = t - 2.0;
    let w = t + 1.0;
    return (-a * t * u * v + 3.0 * b * w * u * v - 3.0 * c * w * t * v + d * w * t * u) / 6.0;
}

var<workgroup> wsums: array<f32, 768>;

fn moment_add(i: u32, v: f32) {
    var old = atomicLoad(&wmoments[i]);
    loop {
        let r = atomicCompareExchangeWeak(&wmoments[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

const HIST: u32 = 92u;

fn hist_add(i: u32, v: f32) {
    var old = atomicLoad(&whist[i]);
    loop {
        let r = atomicCompareExchangeWeak(&whist[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

// Pupil function √T · exp(2πi OPD/λ) of entry wg.z: transmission traced
// here, OPD interpolated from the CPU's nodes, zero where blocked and in the
// padding columns n…NN (padding rows are left to the transpose). Each thread
// does 8 samples of a row, and the workgroup's passing rays are added to the
// entry's fit sums.
@compute @workgroup_size(8, 8)
fn wave_pupil(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let e = wentries[wg.z];
    var m: array<f32, 12>;
    let y = wg.y * 8u + lid.y;
    let cj = y / e.step;
    let fy = f32(y % e.step) / f32(e.step);
    let nc = e.n / e.step + 3u;
    for (var k = 0u; k < 8u; k++) {
        let x = wg.x * 64u + k * 8u + lid.x;
        let at = wg.z * WB.nn * WB.nn + y * WB.nn + x;
        wgrid[at] = vec2<f32>(0.0);
        if (x >= e.n) { continue; }
        let q = vec3<f32>(e.grid.x + (f32(x) + 0.5) * e.grid.w, e.grid.y + (f32(y) + 0.5) * e.grid.w, e.grid.z);
        var d = e.src.xyz;
        if (e.src.w != 0.0) { d = normalize(q - e.src.xyz); }
        var ray = Ray(q + d * ((WB.front_z - q.z) / d.z), d);
        let t = trace_path(&ray, e.lambda, 0u);
        if (t <= 0.0) { continue; }
        let ci = x / e.step;
        let fx = f32(x % e.step) / f32(e.step);
        var r: array<f32, 4>;
        for (var dj = 0u; dj < 4u; dj++) {
            let b = e.nodes_off + (cj + dj) * nc + ci;
            r[dj] = cubic(wnodes[b], wnodes[b + 1u], wnodes[b + 2u], wnodes[b + 3u], fx);
        }
        let w = cubic(r[0], r[1], r[2], r[3], fy) / e.wave;
        let ph = 2.0 * PI * (w - round(w));
        wgrid[at] = sqrt(t) * vec2<f32>(cos(ph), sin(ph));
        let u = (f32(x) + 0.5 - 0.5 * f32(e.n)) * e.grid.w;
        let v = (f32(y) + 0.5 - 0.5 * f32(e.n)) * e.grid.w;
        let l = ray.d.x;
        let mm = ray.d.y;
        let add = array<f32, 12>(1.0, u, v, u * u, u * v, v * v, l, mm, l * u, l * v, mm * u, mm * v);
        for (var i = 0u; i < 12u; i++) { m[i] += add[i]; }
    }
    for (var i = 0u; i < 12u; i++) { wsums[li * 12u + i] = m[i]; }
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) {
            for (var i = 0u; i < 12u; i++) { wsums[li * 12u + i] += wsums[(li + s) * 12u + i]; }
        }
        workgroupBarrier();
    }
    if (li < 12u) { moment_add(wg.z * 16u + li, wsums[li]); }
}

// The entry's sensor mapping of the 2n-point transform from the affine fit:
// Δ = −λ/(NN du) · A^{-T} (kx, ky), A the fitted d(L, M)/d(u, v).
@compute @workgroup_size(64)
fn wave_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WB.count) { return; }
    var m: array<f32, 12>;
    for (var k = 0u; k < 12u; k++) { m[k] = bitcast<f32>(atomicLoad(&wmoments[gid.x * 16u + k])); }
    let s = max(m[0], 1.0);
    let cuu = m[3] - m[1] * m[1] / s;
    let cuv = m[4] - m[1] * m[2] / s;
    let cvv = m[5] - m[2] * m[2] / s;
    let clu = m[8] - m[6] * m[1] / s;
    let clv = m[9] - m[6] * m[2] / s;
    let cmu = m[10] - m[7] * m[1] / s;
    let cmv = m[11] - m[7] * m[2] / s;
    let dc = cuu * cvv - cuv * cuv;
    // A = [[a, b], [c, d]]: L ≈ a u + b v, M ≈ c u + d v.
    let a = (clu * cvv - clv * cuv) / dc;
    let b = (clv * cuu - clu * cuv) / dc;
    let c = (cmu * cvv - cmv * cuv) / dc;
    let dd = (cmv * cuu - cmu * cuv) / dc;
    let e = wentries[gid.x];
    let scale = -e.wave / (f32(WB.nn) * e.grid.w) / (a * dd - b * c);
    wmaps[gid.x] = scale * vec4<f32>(dd, -c, -b, a);
}

// Interval of k with lo ≤ a k + b < hi (empty if lo > hi on return).
fn k_range(a: f32, b: f32, lo: f32, hi: f32) -> vec2<f32> {
    if (a == 0.0) {
        return select(vec2<f32>(1.0, -1.0), vec2<f32>(-1e9, 1e9), b >= lo && b < hi);
    }
    let k0 = (lo - b) / a;
    let k1 = (hi - b) / a;
    return vec2<f32>(min(k0, k1), max(k0, k1));
}

// Energy of the transformed samples of entry `ei` in cell `c` of a grid of
// `cell`-mm cells whose index is floor(offset / cell + o) per axis. The
// transform is stored transposed: frequency (kx, ky) at kx · NN + ky.
fn gather(ei: u32, c: vec2<u32>, cell: f32, o: f32) -> f32 {
    let m = wmaps[ei];
    let nn = i32(WB.nn);
    let half = nn / 2;
    let lo = (vec2<f32>(c) - o) * cell;
    let hi = lo + cell;
    // Columns kx whose k-space parallelogram meets the cell: bounding box of
    // the corners; then the ky interval, contiguous in memory.
    let det = m.x * m.w - m.y * m.z;
    var kx_lo = 1e9;
    var kx_hi = -1e9;
    for (var k = 0u; k < 4u; k++) {
        let p = vec2<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u));
        let kx = (m.w * p.x - m.y * p.y) / det;
        kx_lo = min(kx_lo, kx);
        kx_hi = max(kx_hi, kx);
    }
    let base = ei * WB.nn * WB.nn;
    var sum = 0.0;
    for (var kx = max(i32(floor(kx_lo)), -half); kx <= min(i32(ceil(kx_hi)), half - 1); kx++) {
        let fx = f32(kx);
        let rx = k_range(m.y, m.x * fx, lo.x, hi.x);
        let ry = k_range(m.w, m.z * fx, lo.y, hi.y);
        let a = max(rx.x, ry.x);
        let b = min(rx.y, ry.y);
        if (a > b) { continue; }
        let row = base + u32((kx + nn) % nn) * WB.nn;
        for (var ky = max(i32(floor(a)) - 1, -half); ky <= min(i32(ceil(b)) + 1, half - 1); ky++) {
            let fy = f32(ky);
            let off = vec2<f32>(m.x * fx + m.y * fy, m.z * fx + m.w * fy);
            let idx = floor(off / cell + o);
            if (any(idx < vec2<f32>(0.0)) || any(vec2<u32>(idx) != c)) { continue; }
            let v = wgrid[row + u32((ky + nn) % nn)];
            sum += dot(v, v);
        }
    }
    return sum;
}

// Wing: pixels within ±64 of the reference point.
@compute @workgroup_size(8, 8)
fn wave_wing(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WING || gid.y >= WING || gid.z >= WB.count) { return; }
    let v = gather(gid.z, gid.xy, WB.pitch, f32(WING_HALF) + 0.5);
    wout[gid.z * OUT_STRIDE + gid.y * WING + gid.x] = v;
    if (v > 0.0) {
        let r = length(vec2<f32>(gid.xy) - f32(WING_HALF));
        hist_add(gid.z * HIST + min(u32(ceil(r)), HIST - 1u), v);
    }
}

// Kernel sizes from the wing's energy by pixel-centre radius: the core holds
// 99 %, the wing 99.9 % (as in psfbank.rs).
@compute @workgroup_size(64)
fn wave_sizes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= WB.count) { return; }
    var total = 0.0;
    for (var i = 0u; i < HIST; i++) { total += bitcast<f32>(atomicLoad(&whist[gid.x * HIST + i])); }
    // Radii (whole pixels) holding 99 % and 99.9 % of the energy.
    var acc = 0.0;
    var r99 = HIST - 1u;
    var r999 = HIST - 1u;
    for (var i = HIST - 1u; i > 0u; i--) {
        acc += bitcast<f32>(atomicLoad(&whist[gid.x * HIST + i]));
        if (acc <= 0.001 * total) { r999 = i - 1u; }
        if (acc <= 0.01 * total) { r99 = i - 1u; }
    }
    let hc = min(r99 + 1u, 24u);
    let hw = clamp(r999 + 1u, hc, 64u);
    let sub = (clamp(128u / (2u * hc + 1u), 1u, 16u) - 1u) | 1u;
    wsizes[gid.x] = vec4<u32>(hc, hw, sub, 0u);
}

// Core: (2 hc + 1) · sub cells per side at pitch / sub.
@compute @workgroup_size(8, 8)
fn wave_core(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.z >= WB.count) { return; }
    let s = wsizes[gid.z];
    let size = (2u * s.x + 1u) * s.z;
    if (gid.x >= size || gid.y >= size) { return; }
    wout[gid.z * OUT_STRIDE + WING * WING + gid.y * size + gid.x] = gather(gid.z, gid.xy, WB.pitch / f32(s.z), (f32(s.x) + 0.5) * f32(s.z));
}


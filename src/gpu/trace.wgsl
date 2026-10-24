// Spectral ray tracing through a sequential lens. Prepended with common.wgsl.
//
// Lens blob (`lens`): 8 words per surface (z, curvature, semi-diameter,
// is_stop, decenter x, decenter y, -, -), then event paths of
// `path_stride` words each: [length, events...]; an event is
// surface | forward << 8 | reflect << 9. Path 0 traces forwards, path 1
// backwards, then the ghost paths forwards, then backwards.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> lens: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(0) @binding(3) var<storage, read> scene: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> images: array<u32>;
@group(0) @binding(5) var<storage, read_write> bounds: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read> jobs: array<vec4<u32>>;
@group(1) @binding(0) var<storage, read_write> film: array<FILM_ATOMIC>;
// Quarter-resolution film for large, smooth ghost footprints.
@group(1) @binding(1) var<storage, read_write> lowres: array<FILM_ATOMIC>;

// ------------------------------------------------------------------ random

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return pcg(a ^ pcg(b ^ pcg(c ^ pcg(d))));
}

fn to_unit(x: u32) -> f32 {
    return f32(x >> 8u) * (1.0 / 16777216.0);
}

struct Rng { state: u32 }

fn rnd(r: ptr<function, Rng>) -> f32 {
    (*r).state = pcg((*r).state);
    return to_unit((*r).state);
}

// Roberts' R5 sequence in 0.32 fixed point (exact for any index).
fn r5(j: u32, d: u32) -> u32 {
    var a = P.a0;
    switch d {
        case 1u: { a = P.a1; }
        case 2u: { a = P.a2; }
        case 3u: { a = P.a3; }
        case 4u: { a = P.a4; }
        default: {}
    }
    return a * (j + 1u);
}

// ---------------------------------------------------------------- tables

fn medium_n(m: u32, l: f32) -> f32 {
    let x = clamp(l - LAMBDA_MIN, 0.0, LAMBDA_RANGE - 0.001);
    let i = u32(x);
    let base = P.media_off + m * 401u + i;
    return mix(tables[base], tables[base + 1u], x - f32(i));
}

fn upsample(rgb: vec3<f32>, l: f32) -> f32 {
    let i = min(u32(round(l - LAMBDA_MIN)), 400u);
    let b = P.up_off + i * 3u;
    return max(dot(rgb, vec3<f32>(tables[b], tables[b + 1u], tables[b + 2u])), 0.0);
}

fn light_spectrum(li: u32, l: f32) -> f32 {
    let x = clamp(l - LAMBDA_MIN, 0.0, LAMBDA_RANGE - 0.001);
    let i = u32(x);
    let base = P.spectra_off + li * 401u + i;
    return mix(tables[base], tables[base + 1u], x - f32(i));
}

// ------------------------------------------------------------------ lens

struct Surf { z: f32, c: f32, sd: f32, stop: bool, dx: f32, dy: f32 }

fn surf(i: u32) -> Surf {
    let b = i * 8u;
    return Surf(bitcast<f32>(lens[b]), bitcast<f32>(lens[b + 1u]), bitcast<f32>(lens[b + 2u]),
                lens[b + 3u] != 0u, bitcast<f32>(lens[b + 4u]), bitcast<f32>(lens[b + 5u]));
}

fn iris_edge_unit(theta: f32) -> f32 {
    let sector = 2.0 * PI / f32(P.iris_blades);
    let t = theta - P.iris_rotation;
    let local = t - sector * round(t / sector);
    let poly = 1.0 / cos(local);
    let circ = 1.0 / cos(PI / f32(P.iris_blades));
    return poly + P.iris_curvature * (circ - poly);
}

fn iris_contains(x: f32, y: f32) -> bool {
    let r2 = x * x + y * y;
    let rm = P.iris_max_radius;
    if (P.iris_circular != 0u || P.override_iris != 0u) {
        return r2 <= rm * rm;
    }
    let a = P.iris_apothem;
    if (r2 <= a * a) { return true; }
    if (r2 > rm * rm) { return false; }
    let r = a * iris_edge_unit(atan2(y, x));
    return r2 <= r * r;
}

fn reflectance(n1: f32, n2: f32, cos1: f32, l: f32) -> f32 {
    let s = n1 * sqrt(max(1.0 - cos1 * cos1, 0.0));
    let air = (n1 == 1.0) != (n2 == 1.0);
    if (P.coating == 1u && air) {
        let na = select(n2, n1, n1 == 1.0);
        let ng = select(n1, n2, n1 == 1.0);
        let nc = P.coat_n;
        let sa = s / na;
        let sc = s / nc;
        let sg = s / ng;
        if (sa >= 1.0 || sc >= 1.0 || sg >= 1.0) { return 1.0; }
        let ca = sqrt(1.0 - sa * sa);
        let cc = sqrt(1.0 - sc * sc);
        let cg = sqrt(1.0 - sg * sg);
        let d = P.coat_l0 / (4.0 * nc);
        let cphi = cos(4.0 * PI * nc * d * cc / l);
        var r01 = (na * ca - nc * cc) / (na * ca + nc * cc);
        var r12 = (nc * cc - ng * cg) / (nc * cc + ng * cg);
        let rs = (r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi) / (1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cphi);
        r01 = (nc * ca - na * cc) / (nc * ca + na * cc);
        r12 = (ng * cc - nc * cg) / (ng * cc + nc * cg);
        let rp = (r01 * r01 + r12 * r12 + 2.0 * r01 * r12 * cphi) / (1.0 + r01 * r01 * r12 * r12 + 2.0 * r01 * r12 * cphi);
        return 0.5 * (rs + rp);
    }
    let s2 = s / n2;
    if (s2 >= 1.0) { return 1.0; }
    let cos2 = sqrt(1.0 - s2 * s2);
    let rs = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2);
    let rp = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2);
    return 0.5 * (rs * rs + rp * rp);
}

struct Ray { o: vec3<f32>, d: vec3<f32> }

// Traces `ray` along event path `path`; returns the throughput, or -1 if blocked.
fn trace_path(ray: ptr<function, Ray>, l: f32, path: u32) -> f32 {
    let base = P.path_base + path * P.path_stride;
    let len = lens[base];
    var t = 1.0;
    for (var k = 0u; k < len; k++) {
        let ev = lens[base + 1u + k];
        let i = ev & 0xffu;
        let fwd = ((ev >> 8u) & 1u) != 0u;
        let refl = ((ev >> 9u) & 1u) != 0u;
        let s = surf(i);
        let d = (*ray).d;
        // Sphere through the vertex, c(x²+y²+z²) − 2z = 0, in vertex coordinates;
        // the numerically stable root on the vertex side (also covers c = 0).
        let p0 = (*ray).o - vec3<f32>(s.dx, s.dy, s.z);
        let b = s.c * dot(p0, d) - d.z;
        let cc = s.c * dot(p0, p0) - 2.0 * p0.z;
        let disc = b * b - s.c * cc;
        if (disc < 0.0) { return -1.0; }
        let den = -b - select(1.0, -1.0, b < 0.0) * sqrt(disc);
        if (den == 0.0) { return -1.0; }
        let tt = cc / den;
        if (tt < 1e-5) { return -1.0; }
        let pl = p0 + d * tt;
        if (pl.x * pl.x + pl.y * pl.y > s.sd * s.sd) { return -1.0; }
        let p = (*ray).o + d * tt;
        if (s.stop) {
            if (!iris_contains(pl.x, pl.y)) { return -1.0; }
            (*ray).o = p;
            continue;
        }
        var nrm = normalize(vec3<f32>(s.c * pl.x, s.c * pl.y, s.c * pl.z - 1.0));
        var n1 = medium_n(i + 1u, l);
        var n2 = medium_n(i, l);
        if (fwd) { let tmp = n1; n1 = n2; n2 = tmp; }
        var cosi = -dot(nrm, d);
        if (cosi < 0.0) {
            nrm = -nrm;
            cosi = -cosi;
        }
        let r = reflectance(n1, n2, cosi, l);
        if (refl) {
            t *= r;
            (*ray).d = d + nrm * (2.0 * cosi);
        } else {
            let eta = n1 / n2;
            let kk = 1.0 - eta * eta * (1.0 - cosi * cosi);
            if (kk < 0.0) { return -1.0; }
            t *= 1.0 - r;
            (*ray).d = normalize(d * eta + nrm * (eta * cosi - sqrt(kk)));
        }
        (*ray).o = p;
    }
    return t;
}

// ----------------------------------------------------------------- scene
// Rect: 7 vec4 — centre(w = width), u(w = height), v(w = brightness),
// n(w = param0), colour a(w = param1), colour b, info(kind, offset, w, h | seed).
// Light: 3 vec4 at light_base — centre(w = radius), t1, t2.

fn rect_hit(k: u32, o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let b = k * 7u;
    let c = scene[b];
    let n = scene[b + 3u].xyz;
    let dn = dot(d, n);
    if (abs(dn) < 1e-12) { return vec3<f32>(-1.0); }
    let t = dot(c.xyz - o, n) / dn;
    if (t <= 1e-3) { return vec3<f32>(-1.0); }
    let q = o + d * t - c.xyz;
    let u = dot(q, scene[b + 1u].xyz) / c.w + 0.5;
    let v = dot(q, scene[b + 2u].xyz) / scene[b + 1u].w + 0.5;
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) { return vec3<f32>(-1.0); }
    return vec3<f32>(t, u, v);
}

fn sphere_hit(li: u32, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let s = scene[P.light_base + li * 3u];
    let oc = o - s.xyz;
    let b = dot(oc, d);
    let c = dot(oc, oc) - s.w * s.w;
    let disc = b * b - c;
    if (disc < 0.0) { return -1.0; }
    let t = -b - sqrt(disc);
    return select(-1.0, t, t > 1e-3);
}

fn occluded(o: vec3<f32>, d: vec3<f32>, dist: f32) -> bool {
    for (var k = 0u; k < P.n_rects; k++) {
        let h = rect_hit(k, o, d);
        if (h.x > 0.0 && h.x < dist) { return true; }
    }
    return false;
}

fn hash_noise(x: i32, y: i32, seed: u32) -> f32 {
    var h = (u32(x) * 0x8da6b343u) ^ (u32(y) * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h ^= h >> 13u;
    h *= 0x5bd1e995u;
    h ^= h >> 15u;
    return f32(h & 0xffffffu) / 16777216.0;
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xf = floor(x);
    let yf = floor(y);
    let xi = i32(xf);
    let yi = i32(yf);
    let tx = x - xf;
    let ty = y - yf;
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let a = hash_noise(xi, yi, seed);
    let b = hash_noise(xi + 1, yi, seed);
    let c = hash_noise(xi, yi + 1, seed);
    let d = hash_noise(xi + 1, yi + 1, seed);
    let top = a + (b - a) * sx;
    return top + ((c + (d - c) * sx) - top) * sy;
}

fn srgb_decode(v: f32) -> f32 {
    return select(pow((v + 0.055) / 1.055, 2.4), v / 12.92, v <= 0.04045);
}

fn texel(off: u32, w: u32, x: u32, y: u32) -> vec3<f32> {
    let p = images[off + y * w + x];
    return vec3<f32>(srgb_decode(f32(p & 0xffu) / 255.0), srgb_decode(f32((p >> 8u) & 0xffu) / 255.0),
                     srgb_decode(f32((p >> 16u) & 0xffu) / 255.0));
}

fn chart(u: f32, v: f32, aspect: f32) -> vec3<f32> {
    let white = vec3<f32>(0.85);
    let black = vec3<f32>(0.02);
    let x = u * aspect;
    let y = v;
    let r_star = 0.075;
    let m = 0.08 * max(aspect, 1.0);
    let sxs = array<f32, 3>(m, aspect * 0.5, aspect - m);
    let sys = array<f32, 3>(0.12, 0.5, 0.88);
    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            let dx = x - sxs[i];
            let dy = y - sys[j];
            let r = sqrt(dx * dx + dy * dy);
            if (r < r_star * 1.12) {
                if (r > r_star || r < r_star * 0.04) { return white; }
                return select(white, black, sin(atan2(dy, dx) * 18.0) > 0.0);
            }
        }
    }
    let cx = aspect * 0.3;
    let ang = radians(5.0);
    let lx = (x - cx) * cos(ang) + (y - 0.5) * sin(ang);
    let ly = -(x - cx) * sin(ang) + (y - 0.5) * cos(ang);
    if (abs(lx) < 0.06 && abs(ly) < 0.06) { return black; }
    let px = aspect * 0.7 - 0.06;
    let py = 0.5 - 0.09;
    let ix = floor((x - px) / 0.06);
    let iy = floor((y - py) / 0.06);
    if (ix >= 0.0 && ix < 2.0 && iy >= 0.0 && iy < 3.0) {
        let patches = array<vec3<f32>, 6>(vec3<f32>(0.6, 0.05, 0.04), vec3<f32>(0.05, 0.4, 0.06), vec3<f32>(0.04, 0.07, 0.5),
                                          vec3<f32>(0.7, 0.6, 0.03), vec3<f32>(0.6, 0.06, 0.4), vec3<f32>(0.05, 0.45, 0.55));
        return patches[u32(iy) * 2u + u32(ix)];
    }
    let gx = x * 16.0;
    let gy = y * 16.0;
    if (abs(gx - round(gx)) < 0.035 || abs(gy - round(gy)) < 0.035) { return black; }
    return white;
}

fn texture_eval(b: u32, u: f32, v: f32) -> vec3<f32> {
    let info = bitcast<vec4<u32>>(scene[b + 6u]);
    let aspect = scene[b].w / scene[b + 1u].w;
    let p0 = scene[b + 3u].w;
    let p1 = scene[b + 4u].w;
    let ca = scene[b + 4u].xyz;
    let cb = scene[b + 5u].xyz;
    switch info.x {
        case 0u: { return ca; }
        case 1u: {
            let w = info.z;
            let h = info.w;
            let x = clamp(u * f32(w) - 0.5, 0.0, f32(w) - 1.0);
            let y = clamp((1.0 - v) * f32(h) - 0.5, 0.0, f32(h) - 1.0);
            let x0 = u32(x);
            let y0 = u32(y);
            let x1 = min(x0 + 1u, w - 1u);
            let y1 = min(y0 + 1u, h - 1u);
            let tx = x - f32(x0);
            let ty = y - f32(y0);
            let top = mix(texel(info.y, w, x0, y0), texel(info.y, w, x1, y0), tx);
            let bot = mix(texel(info.y, w, x0, y1), texel(info.y, w, x1, y1), tx);
            return mix(top, bot, ty);
        }
        case 2u: {
            let i = i32(floor(u * p0)) + i32(floor(v * p0 / aspect));
            return select(cb, ca, (i & 1) == 0);
        }
        case 3u: {
            let cu = u * p0;
            let cv = v * p0 / aspect;
            if (abs(cu - round(cu)) < p1 * 0.5 || abs(cv - round(cv)) < p1 * 0.5) { return vec3<f32>(0.02); }
            return vec3<f32>(0.85);
        }
        case 4u: { return chart(u, v, aspect); }
        default: {
            // Noise: fBm over a palette.
            var amp = 0.5;
            var f = 1.0;
            var sum = 0.0;
            var norm = 0.0;
            for (var o = 0u; o < 5u; o++) {
                sum += amp * value_noise(u * p0 * f, v * p0 * f / aspect, info.w + o);
                norm += amp;
                amp *= 0.5;
                f *= 2.03;
            }
            let n = info.z;
            let t = clamp(sum / norm, 0.0, 0.9999) * f32(n - 1u);
            let i = u32(floor(t));
            let pa = P.palette_off + (info.y + i) * 3u;
            let pb = P.palette_off + (info.y + min(i + 1u, n - 1u)) * 3u;
            return mix(vec3<f32>(tables[pa], tables[pa + 1u], tables[pa + 2u]),
                       vec3<f32>(tables[pb], tables[pb + 1u], tables[pb + 2u]), t - f32(i));
        }
    }
}

// Spectral radiance seen along a ray leaving the lens (lights are black here:
// they are rendered by light tracing).
fn scene_radiance(o: vec3<f32>, d: vec3<f32>, l: f32) -> f32 {
    var best = 3.0e38;
    var hit = -1;
    var uv = vec2<f32>(0.0);
    for (var k = 0u; k < P.n_rects; k++) {
        let h = rect_hit(k, o, d);
        if (h.x > 0.0 && h.x < best) {
            best = h.x;
            hit = i32(k);
            uv = h.yz;
        }
    }
    for (var li = 0u; li < P.n_lights; li++) {
        let t = sphere_hit(li, o, d);
        if (t > 0.0 && t < best) { return 0.0; }
    }
    if (hit < 0) { return upsample(vec3<f32>(P.bg_r, P.bg_g, P.bg_b), l); }
    let b = u32(hit) * 7u;
    return scene[b + 2u].w * upsample(texture_eval(b, uv.x, uv.y), l);
}

// -------------------------------------------------------------- film

fn film_add(idx: u32, v: f32) {
    FILM_ADD
}

fn lowres_add(idx: u32, v: f32) {
    LOWRES_ADD
}

fn read_bounds(k: u32) -> vec4<f32> {
    return vec4<f32>(f32(atomicLoad(&bounds[4u * k])), f32(atomicLoad(&bounds[4u * k + 1u])),
                     f32(atomicLoad(&bounds[4u * k + 2u])), f32(atomicLoad(&bounds[4u * k + 3u]))) / FIXED;
}

fn union_bounds(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(min(a.xy, b.xy), max(a.zw, b.zw));
}

// --------------------------------------------------------- pupil bounds

fn bound_point(k: u32, x: f32, y: f32) {
    let ix = i32(round(x * FIXED));
    let iy = i32(round(y * FIXED));
    atomicMin(&bounds[4u * k], ix);
    atomicMin(&bounds[4u * k + 1u], iy);
    atomicMax(&bounds[4u * k + 2u], ix);
    atomicMax(&bounds[4u * k + 3u], iy);
}

// Rays from sensor node `gid.y` through grid cell `gid.x` of the rear vertex plane.
@compute @workgroup_size(64)
fn pupil_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = P.bounds_g;
    if (gid.x >= g * g || gid.y >= P.bounds_nodes) { return; }
    let k = gid.y;
    var o: vec3<f32>;
    if (P.bounds_mode == 0u) {
        o = vec3<f32>(P.r_max * f32(k) / f32(P.bounds_nodes - 1u), 0.0, P.sensor_z);
    } else {
        let ix = k % P.grid_nx;
        let iy = k / P.grid_nx;
        o = vec3<f32>(-P.half_w + 2.0 * P.half_w * f32(ix) / f32(P.grid_nx - 1u),
                      -P.half_h + 2.0 * P.half_h * f32(iy) / f32(P.grid_ny - 1u), P.sensor_z);
    }
    let cell = 2.0 * P.bounds_half / f32(g);
    let x = -P.bounds_half + (f32(gid.x % g) + 0.5) * cell;
    let y = -P.bounds_half + (f32(gid.x / g) + 0.5) * cell;
    let d = normalize(vec3<f32>(x, y, P.rear_z) - o);
    for (var w = 0u; w < 5u; w++) {
        var ray = Ray(o, d);
        if (trace_path(&ray, 380.0 + 100.0 * f32(w), 1u) >= 0.0) {
            bound_point(k, x, y);
            return;
        }
    }
}

struct LightGeom { c: vec3<f32>, radius: f32, t1: vec3<f32>, t2: vec3<f32> }

fn light_geom(li: u32) -> LightGeom {
    let b = P.light_base + li * 3u;
    return LightGeom(scene[b].xyz, scene[b].w, scene[b + 1u].xyz, scene[b + 2u].xyz);
}

// Square on the entrance-pupil plane that light `li` sees the front element through.
fn light_search(c: vec3<f32>) -> vec3<f32> {
    let k = (P.ep_z - c.z) / -c.z;
    let centre = c * (1.0 - k);
    return vec3<f32>(centre.xy, P.front_half * k);
}

// Rays from light centre `gid.y` through cell `gid.x` of its search square.
@compute @workgroup_size(64)
fn light_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = P.light_bounds_g;
    if (gid.x >= g * g || gid.y >= P.n_lights) { return; }
    let lg = light_geom(gid.y);
    let s = light_search(lg.c);
    let cell = 2.0 * s.z / f32(g);
    let x = s.x - s.z + (f32(gid.x % g) + 0.5) * cell;
    let y = s.y - s.z + (f32(gid.x / g) + 0.5) * cell;
    let q = vec3<f32>(x, y, P.ep_z);
    let d = normalize(q - lg.c);
    let o = q - d * ((q.z - P.front_z) / d.z);
    for (var w = 0u; w < 3u; w++) {
        var ray = Ray(o, d);
        if (trace_path(&ray, 380.0 + 200.0 * f32(w), 0u) >= 0.0) {
            bound_point(LIGHT_BOUNDS_BASE / 4u + gid.y, x, y);
            return;
        }
    }
}

// --------------------------------------------------- surfaces (backward)

@compute @workgroup_size(8, 8)
fn surfaces(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    var acc: array<f32, 32>;
    let pix = gid.y * P.w + gid.x;
    let dz = P.sensor_z - P.rear_z;
    for (var bin = 0u; bin < P.bins; bin++) {
        // Per pixel and bin a fixed random shift of the low-discrepancy sequence.
        var sh: array<u32, 5>;
        for (var k = 0u; k < 5u; k++) { sh[k] = hash4(pix, bin, k, 0x9e37u); }
        for (var s = 0u; s < P.spp; s++) {
            let j = P.frame * P.spp + s;
            let u0 = to_unit(sh[0] + r5(j, 0u));
            let u1 = to_unit(sh[1] + r5(j, 1u));
            let u2 = to_unit(sh[2] + r5(j, 2u));
            let u3 = to_unit(sh[3] + r5(j, 3u));
            let u4 = to_unit(sh[4] + r5(j, 4u));
            let xo = (f32(gid.x) + u0 - f32(P.w) * 0.5) * P.pitch;
            let yo = (f32(P.h) * 0.5 - f32(gid.y) - u1) * P.pitch;
            let xs = -xo;
            let ys = -yo;
            let o = vec3<f32>(xs, ys, P.sensor_z);
            let l = LAMBDA_MIN + LAMBDA_RANGE * (f32(bin) + u4) / f32(P.bins);
            // Main path within the pupil bounds.
            var bb: vec4<f32>;
            var sn = 0.0;
            var cs = 1.0;
            let r = sqrt(xs * xs + ys * ys);
            if (P.bounds_mode == 0u) {
                let f = r / P.r_max * f32(P.bounds_nodes - 1u);
                let k0 = min(u32(f), P.bounds_nodes - 1u);
                let k1 = min(k0 + 1u, P.bounds_nodes - 1u);
                bb = union_bounds(read_bounds(k0), read_bounds(k1));
                if (r > 0.0) {
                    sn = ys / r;
                    cs = xs / r;
                }
            } else {
                let fx = clamp((xs + P.half_w) / (2.0 * P.half_w) * f32(P.grid_nx - 1u), 0.0, f32(P.grid_nx - 1u));
                let fy = clamp((ys + P.half_h) / (2.0 * P.half_h) * f32(P.grid_ny - 1u), 0.0, f32(P.grid_ny - 1u));
                let x0 = min(u32(fx), P.grid_nx - 2u);
                let y0 = min(u32(fy), P.grid_ny - 2u);
                let k = y0 * P.grid_nx + x0;
                bb = union_bounds(union_bounds(read_bounds(k), read_bounds(k + 1u)),
                                  union_bounds(read_bounds(k + P.grid_nx), read_bounds(k + P.grid_nx + 1u)));
            }
            if (bb.x > bb.z) { continue; }
            let cell = 2.0 * P.bounds_half / f32(P.bounds_g);
            bb += vec4<f32>(-cell, -cell, cell, cell);
            let bx = bb.x + (bb.z - bb.x) * u2;
            let by = bb.y + (bb.w - bb.y) * u3;
            let aim = vec3<f32>(bx * cs - by * sn, bx * sn + by * cs, P.rear_z);
            let d = normalize(aim - o);
            var ray = Ray(o, d);
            let t = trace_path(&ray, l, 1u);
            if (t < 0.0) { continue; }
            let c4 = d.z * d.z * d.z * d.z;
            acc[bin] += scene_radiance(ray.o, ray.d, l) * t * (bb.z - bb.x) * (bb.w - bb.y) * c4 / (dz * dz);
        }
    }
    let scale = P.norm * LAMBDA_RANGE / f32(P.bins * P.spp);
    for (var b = 0u; b < P.bins; b++) {
        film_add(pix * P.bins + b, acc[b] * scale);
    }
}

// ------------------------------------------- veiling glare (quarter res)
// Ghost paths sampled from the sensor: light from bright scene surfaces
// reflected twice inside the lens. Low-frequency, so traced per 4×4 block.

@compute @workgroup_size(8, 8)
fn veil(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lw = (P.w + 3u) / 4u;
    let lh = (P.h + 3u) / 4u;
    if (gid.x >= lw || gid.y >= lh || P.n_ghosts == 0u) { return; }
    var acc: array<f32, 32>;
    let pix = gid.y * lw + gid.x;
    let dz = P.sensor_z - P.rear_z;
    let rear_area = PI * P.bounds_half * P.bounds_half;
    var rng = Rng(hash4(pix, P.frame, 0x51u, 7u));
    for (var bin = 0u; bin < P.bins; bin++) {
        for (var s = 0u; s < P.spp; s++) {
            let px = min(f32(gid.x * 4u) + 4.0 * rnd(&rng), f32(P.w) - 0.001);
            let py = min(f32(gid.y * 4u) + 4.0 * rnd(&rng), f32(P.h) - 0.001);
            let o = vec3<f32>((f32(P.w) * 0.5 - px) * P.pitch, (py - f32(P.h) * 0.5) * P.pitch, P.sensor_z);
            let l = LAMBDA_MIN + LAMBDA_RANGE * (f32(bin) + rnd(&rng)) / f32(P.bins);
            let rr = P.bounds_half * sqrt(rnd(&rng));
            let ph = 2.0 * PI * rnd(&rng);
            let g = min(u32(rnd(&rng) * f32(P.n_ghosts)), P.n_ghosts - 1u);
            let d = normalize(vec3<f32>(rr * cos(ph), rr * sin(ph), P.rear_z) - o);
            var ray = Ray(o, d);
            let t = trace_path(&ray, l, 2u + P.n_ghosts + g);
            if (t > 0.0 && ray.d.z < 0.0) {
                let c4 = d.z * d.z * d.z * d.z;
                acc[bin] += scene_radiance(ray.o, ray.d, l) * t * rear_area * c4 / (dz * dz);
            }
        }
    }
    // One of n_ghosts paths per sample; quarter-resolution values are 16× irradiance.
    let scale = 16.0 * f32(P.n_ghosts) * P.norm * LAMBDA_RANGE / f32(P.bins * P.spp);
    for (var b = 0u; b < P.bins; b++) {
        lowres_add(pix * P.bins + b, acc[b] * scale);
    }
}

// ------------------------------------------------- lights (forward grids)
// Job: (light, path, bin, kind), (λ jitter, disk x, disk y, grid jitter x), (grid jitter y, weight, -, -).
// Each workgroup traces a 17×17 tile of grid rays and rasterises the 16×16
// quads between them with their flux (after Hullin et al. 2011).

var<workgroup> verts: array<vec4<f32>, 289>;
var<workgroup> region: vec4<f32>;
var<workgroup> region_ok: u32;

fn splat(x: f32, y: f32, bin: u32, v: f32) {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let xi = floor(fx);
    let yi = floor(fy);
    let tx = fx - xi;
    let ty = fy - yi;
    for (var k = 0u; k < 4u; k++) {
        let px = xi + f32(k & 1u);
        let py = yi + f32(k >> 1u);
        let wgt = select(1.0 - tx, tx, (k & 1u) != 0u) * select(1.0 - ty, ty, (k >> 1u) != 0u);
        if (px >= 0.0 && py >= 0.0 && px < f32(P.w) && py < f32(P.h)) {
            film_add((u32(py) * P.w + u32(px)) * P.bins + bin, v * wgt);
        }
    }
}

// Rasterises a triangle (pixel coordinates) carrying `flux`. Large
// triangles go to the quarter-resolution film: they only occur in widely
// spread, smooth ghosts and would otherwise dominate the cost.
fn raster_triangle(a0: vec3<f32>, b0: vec3<f32>, c0: vec3<f32>, flux: f32, bin: u32) {
    let area0 = 0.5 * ((b0.x - a0.x) * (c0.y - a0.y) - (c0.x - a0.x) * (b0.y - a0.y));
    if (abs(area0) < 1.0) {
        splat((a0.x + b0.x + c0.x) / 3.0, (a0.y + b0.y + c0.y) / 3.0, bin, flux);
        return;
    }
    let low = abs(area0) > 16.0;
    let k = select(1.0, 0.25, low);
    let a = a0 * k;
    let b = b0 * k;
    let c = c0 * k;
    let area = area0 * k * k;
    let fw = select(P.w, (P.w + 3u) / 4u, low);
    let fh = select(P.h, (P.h + 3u) / 4u, low);
    let x0 = max(floor(min(a.x, min(b.x, c.x))), 0.0);
    let x1 = min(ceil(max(a.x, max(b.x, c.x))), f32(fw));
    let y0 = max(floor(min(a.y, min(b.y, c.y))), 0.0);
    let y1 = min(ceil(max(a.y, max(b.y, c.y))), f32(fh));
    if (x0 >= x1 || y0 >= y1) { return; }
    if ((x1 - x0) * (y1 - y0) > 65536.0) {
        splat((a0.x + b0.x + c0.x) / 3.0, (a0.y + b0.y + c0.y) / 3.0, bin, flux);
        return;
    }
    let v = flux / abs(area);
    let inv = 1.0 / (2.0 * area);
    for (var py = u32(y0); py < u32(y1); py++) {
        let y = f32(py) + 0.5;
        for (var px = u32(x0); px < u32(x1); px++) {
            let x = f32(px) + 0.5;
            let w0 = ((b.x - x) * (c.y - y) - (c.x - x) * (b.y - y)) * inv;
            let w1 = ((c.x - x) * (a.y - y) - (a.x - x) * (c.y - y)) * inv;
            if (w0 >= 0.0 && w1 >= 0.0 && 1.0 - w0 - w1 >= 0.0) {
                let idx = (py * fw + px) * P.bins + bin;
                if (low) { lowres_add(idx, v); } else { film_add(idx, v); }
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn light_grid(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(local_invocation_index) li_idx: u32) {
    let job = jobs[wg.y * 3u];
    let jf = bitcast<vec4<f32>>(jobs[wg.y * 3u + 1u]);
    let jf2 = bitcast<vec4<f32>>(jobs[wg.y * 3u + 2u]);
    let li = job.x;
    let lg = light_geom(li);
    let ghost = job.w != 0u;
    let z_plane = select(P.ep_z, 0.0, ghost);
    if (li_idx == 0u) {
        if (ghost) {
            region = vec4<f32>(-P.front_half, -P.front_half, P.front_half, P.front_half);
            region_ok = 1u;
        } else {
            let bb = read_bounds(LIGHT_BOUNDS_BASE / 4u + li);
            let s = light_search(lg.c);
            let m = 2.0 * s.z / f32(P.light_bounds_g) + 2.0 * lg.radius * abs(P.ep_z / lg.c.z);
            region = bb + vec4<f32>(-m, -m, m, m);
            region_ok = select(0u, 1u, bb.x <= bb.z);
        }
    }
    if (workgroupUniformLoad(&region_ok) == 0u) { return; }
    let reg = region;
    let g = P.light_grid;
    let tiles = g / 16u;
    let tx = wg.x % tiles;
    let ty = wg.x / tiles;
    let cell = vec2<f32>((reg.z - reg.x) / f32(g), (reg.w - reg.y) / f32(g));
    let bin_width = LAMBDA_RANGE / f32(P.bins);
    let l = LAMBDA_MIN + bin_width * (f32(job.z) + jf.x);
    let radiance = light_spectrum(li, l);
    let e = lg.c + lg.t1 * (jf.y * lg.radius) + lg.t2 * (jf.z * lg.radius);
    let ne = normalize(-lg.c);
    let area_e = PI * lg.radius * lg.radius;
    for (var v = li_idx; v < 289u; v += 256u) {
        let vi = tx * 16u + v % 17u;
        let vj = ty * 16u + v / 17u;
        let q = vec3<f32>(reg.x + (f32(vi) + jf.w - 0.5) * cell.x, reg.y + (f32(vj) + jf2.x - 0.5) * cell.y, z_plane);
        let dv = q - e;
        let dist2 = dot(dv, dv);
        let d = dv / sqrt(dist2);
        var out = vec4<f32>(0.0);
        if (d.z > 0.0) {
            let o = q - d * ((q.z - P.front_z) / d.z);
            if (!occluded(o, -d, length(o - e))) {
                var ray = Ray(o, d);
                let t = trace_path(&ray, l, job.y);
                if (t > 0.0 && ray.d.z > 0.0) {
                    let p = ray.o + ray.d * ((P.sensor_z - ray.o.z) / ray.d.z);
                    let f = radiance * area_e * max(dot(d, ne), 0.0) * d.z / dist2 * t;
                    out = vec4<f32>(-p.x / P.pitch + f32(P.w) * 0.5, f32(P.h) * 0.5 + p.y / P.pitch, f, 1.0);
                }
            }
        }
        verts[v] = out;
    }
    workgroupBarrier();
    // jf2.y weights jobs that sample only some of the wavelength bins.
    let scale = cell.x * cell.y * P.norm * bin_width * jf2.y / (P.pitch * P.pitch);
    let i00 = lid.y * 17u + lid.x;
    let a = verts[i00];
    let b = verts[i00 + 1u];
    let c = verts[i00 + 18u];
    let d = verts[i00 + 17u];
    let valid = a.w + b.w + c.w + d.w;
    if (valid == 4.0) {
        let flux = 0.25 * (a.z + b.z + c.z + d.z) * scale;
        raster_triangle(a.xyz, b.xyz, c.xyz, 0.5 * flux, job.z);
        raster_triangle(a.xyz, c.xyz, d.xyz, 0.5 * flux, job.z);
    } else if (valid > 0.0) {
        // Partly clipped quad: each surviving corner carries its quarter.
        for (var k = 0u; k < 4u; k++) {
            var cv = a;
            switch k {
                case 1u: { cv = b; }
                case 2u: { cv = c; }
                case 3u: { cv = d; }
                default: {}
            }
            if (cv.w > 0.0) { splat(cv.x, cv.y, job.z, 0.25 * cv.z * scale); }
        }
    }
}

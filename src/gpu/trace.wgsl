// Spectral ray tracing through a sequential lens. Prepended with common.wgsl.
//
// Lens blob (`lens`): 16 words per surface (z, curvature, semi-diameter,
// is_stop, decenter x, decenter y, conic, is_aspheric, A4 … A18), then event paths of
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
// Flux of light jobs imaged through the PSF bank, per job (first task index).
@group(1) @binding(2) var<storage, read_write> job_flux: array<FILM_ATOMIC>;

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

struct Surf { z: f32, c: f32, sd: f32, stop: bool, dx: f32, dy: f32, k: f32, asph: bool, base: u32 }

fn surf(i: u32) -> Surf {
    let b = i * 16u;
    return Surf(bitcast<f32>(lens[b]), bitcast<f32>(lens[b + 1u]), bitcast<f32>(lens[b + 2u]),
                lens[b + 3u] != 0u, bitcast<f32>(lens[b + 4u]), bitcast<f32>(lens[b + 5u]),
                bitcast<f32>(lens[b + 6u]), lens[b + 7u] != 0u, b + 8u);
}

// Sag of an even asphere at u = r², and d(sag)/du; x < 0 outside the conic's domain.
fn asph_sag(s: Surf, u: f32) -> vec2<f32> {
    let q = 1.0 - (1.0 + s.k) * s.c * s.c * u;
    if (q < 0.0) { return vec2<f32>(0.0, -1.0); }
    let sq = sqrt(q);
    var z = s.c * u / (1.0 + sq);
    var dz = s.c / (2.0 * max(sq, 1e-6));
    var un = u;
    var un1 = 1.0;
    for (var j = 0u; j < 8u; j++) {
        let a = bitcast<f32>(lens[s.base + j]);
        un1 = un;
        un *= u;
        z += a * un;
        dz += f32(j + 2u) * a * un1;
    }
    return vec2<f32>(z, dz);
}

// Newton iteration from the base-sphere root `t0`; returns t or −1.
fn asph_intersect(s: Surf, p0: vec3<f32>, d: vec3<f32>, t0: f32) -> f32 {
    var t = t0;
    for (var it = 0; it < 16; it++) {
        let p = p0 + d * t;
        let u = p.x * p.x + p.y * p.y;
        if ((1.0 + s.k) * s.c * s.c * u >= 1.0) { return -1.0; }
        let sd = asph_sag(s, u);
        let f = p.z - sd.x;
        let df = d.z - sd.y * 2.0 * (p.x * d.x + p.y * d.y);
        if (df == 0.0) { return -1.0; }
        let dt = f / df;
        t -= dt;
        if (abs(dt) < 1e-6 * (1.0 + abs(t))) {
            return select(-1.0, t, t >= 1e-5);
        }
    }
    return -1.0;
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

// Radius of (x, y) relative to the iris edge in its direction (≤ 1 inside).
fn iris_ratio(x: f32, y: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    if (P.iris_circular != 0u || P.override_iris != 0u) {
        return r / P.iris_max_radius;
    }
    return r / (P.iris_apothem * iris_edge_unit(atan2(y, x)));
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
    var clip = 0.0;
    return trace_clip(ray, l, path, false, false, &clip);
}

// With `track`, `clip` gets the largest radius relative to the apertures (> 1
// if one blocks the ray), to cut ray grids smoothly; with `soft` as well,
// apertures do not block.
fn trace_clip(ray: ptr<function, Ray>, l: f32, path: u32, track: bool, soft: bool, clip: ptr<function, f32>) -> f32 {
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
        // Sphere through the vertex, c(x²+y²+z²) − 2z = 0, in vertex coordinates.
        // Both roots in numerically stable form (cc/q also covers c = 0); the
        // nearest one ahead on the vertex-side cap (c·z < 1) is the surface.
        let p0 = (*ray).o - vec3<f32>(s.dx, s.dy, s.z);
        let b = s.c * dot(p0, d) - d.z;
        let cc = s.c * dot(p0, p0) - 2.0 * p0.z;
        let disc = b * b - s.c * cc;
        let q = -b - select(1.0, -1.0, b < 0.0) * sqrt(max(disc, 0.0));
        var tt = -1.0;
        if (disc >= 0.0 && q != 0.0) {
            tt = cc / q;
            let ok1 = tt >= 1e-5 && s.c * (p0.z + d.z * tt) < 1.0;
            if (s.c != 0.0) {
                let t2 = q / s.c;
                let ok2 = t2 >= 1e-5 && s.c * (p0.z + d.z * t2) < 1.0;
                tt = select(select(-1.0, t2, ok2), select(tt, min(tt, t2), ok2), ok1);
            } else if (!ok1) {
                tt = -1.0;
            }
        }
        if (s.asph) {
            // Start from the base sphere's root (or the vertex plane).
            let t0 = select(-p0.z / d.z, tt, tt >= 0.0);
            tt = asph_intersect(s, p0, d, t0);
        }
        if (tt < 0.0) { return -1.0; }
        let pl = p0 + d * tt;
        let r2 = pl.x * pl.x + pl.y * pl.y;
        if (track) {
            *clip = max(*clip, sqrt(r2) / s.sd);
            if (*clip > 1.0 && !soft) { return -1.0; }
        } else if (r2 > s.sd * s.sd) {
            return -1.0;
        }
        let p = (*ray).o + d * tt;
        if (s.stop) {
            if (track) {
                *clip = max(*clip, iris_ratio(pl.x, pl.y));
                if (*clip > 1.0 && !soft) { return -1.0; }
            } else if (!iris_contains(pl.x, pl.y)) {
                return -1.0;
            }
            (*ray).o = p;
            continue;
        }
        var nrm = normalize(vec3<f32>(s.c * pl.x, s.c * pl.y, s.c * pl.z - 1.0));
        if (s.asph) {
            let g = 2.0 * asph_sag(s, pl.x * pl.x + pl.y * pl.y).y;
            nrm = normalize(vec3<f32>(g * pl.x, g * pl.y, -1.0));
        }
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
    // r² − (distance of the line from the centre)², stable for distant spheres.
    let perp = oc - d * b;
    let disc = s.w * s.w - dot(perp, perp);
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

// What a ray leaving the lens sees: spectral radiance and the object point
// (or, for the background, the direction; lights are black here: they are
// rendered by light tracing).
struct SceneHit { radiance: f32, p: vec3<f32>, at_inf: bool }

fn scene_hit(o: vec3<f32>, d: vec3<f32>, l: f32) -> SceneHit {
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
        if (t > 0.0 && t < best) { return SceneHit(0.0, o + d * t, false); }
    }
    if (hit < 0) { return SceneHit(upsample(vec3<f32>(P.bg_r, P.bg_g, P.bg_b), l), d, true); }
    let b = u32(hit) * 7u;
    return SceneHit(scene[b + 2u].w * upsample(texture_eval(b, uv.x, uv.y), l), o + d * best, false);
}

fn scene_radiance(o: vec3<f32>, d: vec3<f32>, l: f32) -> f32 {
    return scene_hit(o, d, l).radiance;
}

// ------------------------------------------------------- wave-optical PSFs
// Bank entries: field (uniform in tan θ) × vergence × iris azimuth × bin,
// each a CDF over (2·half + 1)² sensor pixels around the chief-ray image.

struct BankPos { ok: bool, ft: f32, fv: f32, alpha: f32 }

// Bank coordinates of an object point (or direction at infinity).
fn bank_pos(p: vec3<f32>, at_inf: bool) -> BankPos {
    var lateral: vec2<f32>;
    var verg = 0.0;
    if (at_inf) {
        lateral = p.xy / max(-p.z, 1e-6);
    } else {
        let dist = P.ep_z - p.z;
        if (dist <= 0.0) { return BankPos(false, 0.0, 0.0, 0.0); }
        lateral = p.xy / dist;
        verg = 1.0 / dist;
    }
    let tan_t = length(lateral);
    let ft = tan_t / P.bank_tan_max * f32(P.bank_nt - 1u);
    let fv = (verg - P.bank_v0) / P.bank_dv;
    if (ft > f32(P.bank_nt - 1u) || fv < 0.0 || fv > f32(P.bank_nv - 1u)) { return BankPos(false, 0.0, 0.0, 0.0); }
    // The image lies opposite the object.
    return BankPos(true, ft, fv, atan2(-lateral.y, -lateral.x));
}

fn bank_entry(t: u32, v: u32, a: u32, b: u32) -> u32 {
    return ((t * P.bank_nv + v) * P.bank_na + a) * P.bins + b;
}

struct BankPick { entry: u32, rot: f32, reference: vec2<f32> }

// Picks one of the neighbouring bank entries with probability equal to its
// interpolation weight; the reference point is interpolated.
fn bank_pick(bp: BankPos, bin: u32, ut: f32, uv: f32) -> BankPick {
    let t0 = min(u32(bp.ft), P.bank_nt - 1u);
    let t1 = min(t0 + 1u, P.bank_nt - 1u);
    let v0 = min(u32(bp.fv), P.bank_nv - 1u);
    let v1 = min(v0 + 1u, P.bank_nv - 1u);
    let wt = bp.ft - f32(t0);
    let wv = bp.fv - f32(v0);
    // Nearest iris azimuth sample within the symmetry period.
    let step = P.bank_period / f32(P.bank_na);
    let rel = bp.alpha - 0.5 * PI;
    let k = round(rel / step);
    let a = u32(((i32(k) % i32(P.bank_na)) + i32(P.bank_na)) % i32(P.bank_na));
    // Rotating sample a's configuration onto the actual azimuth turns the iris
    // by whole symmetry periods plus at most half a step.
    let rot = rel - f32(a) * step;
    let ref_r = mix(mix(tables[P.bank_refs_off + bank_entry(t0, v0, a, bin)], tables[P.bank_refs_off + bank_entry(t1, v0, a, bin)], wt),
                    mix(tables[P.bank_refs_off + bank_entry(t0, v1, a, bin)], tables[P.bank_refs_off + bank_entry(t1, v1, a, bin)], wt), wv);
    let t = select(t0, t1, ut < wt);
    let v = select(v0, v1, uv < wv);
    let alpha_a = 0.5 * PI + f32(a) * step;
    return BankPick(bank_entry(t, v, a, bin), rot, ref_r * vec2<f32>(cos(alpha_a), sin(alpha_a)));
}

fn rotate2(v: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Entry kernels, centred on the chief-ray point; layout in the lens blob:
// core (sub-pixel cells) and wing (pixel cells, zero inside the core square).
struct Kernel { half: u32, sub: u32, base: u32, size: u32 }
struct Kernels { core: Kernel, wing: Kernel, core_fraction: f32 }

fn bank_kernels(entry: u32) -> Kernels {
    let b = P.bank_layout_off + 6u * entry;
    let hc = lens[b];
    let sub = lens[b + 1u];
    let hw = lens[b + 3u];
    return Kernels(Kernel(hc, sub, P.bank_cdf_off + lens[b + 2u], (2u * hc + 1u) * sub),
                   Kernel(hw, 1u, P.bank_cdf_off + lens[b + 4u], 2u * hw + 1u),
                   bitcast<f32>(lens[b + 5u]));
}

// Offset from the reference of kernel cell `i` at fractional position (jx, jy) in it.
fn bank_offset(k: Kernel, i: u32, jx: f32, jy: f32) -> vec2<f32> {
    let cell = P.pitch / f32(k.sub);
    let start = -(f32(k.half) + 0.5) * P.pitch;
    return vec2<f32>(start + (f32(i % k.size) + jx) * cell, start + (f32(i / k.size) + jy) * cell);
}

fn cdf_find(k: Kernel, u: f32) -> u32 {
    var lo = 0u;
    var hi = k.size * k.size - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (tables[k.base + mid] < u) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

// Sensor point drawn from the PSF of entry `pk`.
fn bank_sample(pk: BankPick, u: f32, jx: f32, jy: f32) -> vec2<f32> {
    let ks = bank_kernels(pk.entry);
    var off: vec2<f32>;
    if (u < ks.core_fraction) {
        off = bank_offset(ks.core, cdf_find(ks.core, u / ks.core_fraction), jx, jy);
    } else {
        off = bank_offset(ks.wing, cdf_find(ks.wing, (u - ks.core_fraction) / (1.0 - ks.core_fraction)), jx, jy);
    }
    return rotate2(pk.reference + off, pk.rot);
}

// Film index (any bin offset added by the caller) of a sensor point, or −1.
fn film_pixel(xs: f32, ys: f32) -> i32 {
    let x = -xs / P.pitch + f32(P.w) * 0.5;
    let y = f32(P.h) * 0.5 + ys / P.pitch;
    if (x < 0.0 || y < 0.0 || x >= f32(P.w) || y >= f32(P.h)) { return -1; }
    return i32(u32(y) * P.w + u32(x));
}

// -------------------------------------------------------------- film

fn film_add(idx: u32, v: f32) {
    FILM_ADD
}

fn job_flux_add(idx: u32, v: f32) {
    JOB_FLUX_ADD
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
    // With the PSF bank, threads extend `psf_margin` pixels beyond the frame
    // so light from just outside can be spread into it.
    let m = P.psf_margin;
    if (gid.x >= P.w + 2u * m || gid.y >= P.h + 2u * m) { return; }
    let px = i32(gid.x) - i32(m);
    let py = i32(gid.y) - i32(m);
    let inside = px >= 0 && py >= 0 && px < i32(P.w) && py < i32(P.h);
    var acc: array<f32, 32>;
    let id = gid.y * (P.w + 2u * m) + gid.x;
    let dz = P.sensor_z - P.rear_z;
    let scale = P.norm * LAMBDA_RANGE / f32(P.bins * P.spp);
    var rng = Rng(hash4(id, P.frame, 0x7a1u, 3u));
    for (var bin = 0u; bin < P.bins; bin++) {
        // Per pixel and bin a fixed random shift of the low-discrepancy sequence.
        var sh: array<u32, 5>;
        for (var k = 0u; k < 5u; k++) { sh[k] = hash4(id, bin, k, 0x9e37u); }
        for (var s = 0u; s < P.spp; s++) {
            let j = P.frame * P.spp + s;
            let u0 = to_unit(sh[0] + r5(j, 0u));
            let u1 = to_unit(sh[1] + r5(j, 1u));
            let u2 = to_unit(sh[2] + r5(j, 2u));
            let u3 = to_unit(sh[3] + r5(j, 3u));
            let u4 = to_unit(sh[4] + r5(j, 4u));
            let xo = (f32(px) + u0 - f32(P.w) * 0.5) * P.pitch;
            let yo = (f32(P.h) * 0.5 - f32(py) - u1) * P.pitch;
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
            let hit = scene_hit(ray.o, ray.d, l);
            let c = hit.radiance * t * (bb.z - bb.x) * (bb.w - bb.y) * c4 / (dz * dz);
            if (c == 0.0) { continue; }
            if (P.bank_on != 0u) {
                let bp = bank_pos(hit.p, hit.at_inf);
                if (bp.ok) {
                    // The light from this scene point goes through its wave-optical PSF.
                    let pk = bank_pick(bp, bin, rnd(&rng), rnd(&rng));
                    let q = bank_sample(pk, rnd(&rng), rnd(&rng), rnd(&rng));
                    let fp = film_pixel(q.x, q.y);
                    if (fp >= 0) { film_add(u32(fp) * P.bins + bin, c * scale); }
                    continue;
                }
            }
            if (inside) { acc[bin] += c; }
        }
    }
    if (!inside) { return; }
    let pix = u32(py) * P.w + u32(px);
    for (var b = 0u; b < P.bins; b++) {
        if (acc[b] != 0.0) { film_add(pix * P.bins + b, acc[b] * scale); }
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
// Task: (light, path, bin, kind), (λ jitter, disk x, disk y, grid jitter x),
// (grid jitter y, weight, tile, grid size), (index of the job's first task, -, -, -).
// Each workgroup traces a 17×17 tile of grid rays and rasterises the 16×16
// quads between them with their flux (after Hullin et al. 2011).

var<workgroup> verts: array<vec4<f32>, 289>;
var<workgroup> region: vec4<f32>;
var<workgroup> region_ok: u32;

// Grid rays of a job: emitter point and normal, grid origin and cell (tile
// vertex units), plane, wavelength, emitted power and event path.
struct Grid { e: vec3<f32>, ne: vec3<f32>, origin: vec2<f32>, cell: vec2<f32>, z: f32, l: f32, power: f32, path: u32 }

// Sensor position (pixels), flux and clip coordinate (w) of the ray through
// tile vertex `v`; w = −1 if it is lost. Unless `soft`, rays blocked by an
// aperture have z = −1 (and w > 1).
fn grid_ray(gr: Grid, v: u32, soft: bool) -> vec4<f32> {
    let lost = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    let q = vec3<f32>(gr.origin + vec2<f32>(f32(v % 17u), f32(v / 17u)) * gr.cell, gr.z);
    let dv = q - gr.e;
    let dist2 = dot(dv, dv);
    let d = dv / sqrt(dist2);
    if (d.z <= 0.0) { return lost; }
    let o = q - d * ((q.z - P.front_z) / d.z);
    if (occluded(o, -d, length(o - gr.e))) { return lost; }
    var ray = Ray(o, d);
    var clip = 0.0;
    let t = trace_clip(&ray, gr.l, gr.path, true, soft, &clip);
    if (t > 0.0 && ray.d.z > 0.0) {
        let p = ray.o + ray.d * ((P.sensor_z - ray.o.z) / ray.d.z);
        let f = gr.power * max(dot(d, gr.ne), 0.0) * d.z / dist2 * t;
        return vec4<f32>(-p.x / P.pitch + f32(P.w) * 0.5, f32(P.h) * 0.5 + p.y / P.pitch, f, clip);
    }
    return select(lost, vec4<f32>(0.0, 0.0, -1.0, clip), clip > 1.0 && !soft);
}

fn passes(v: vec4<f32>) -> bool {
    return v.w >= 0.0 && v.w <= 1.0;
}

// Tile vertex `v`, blocked by an aperture: sensor position, flux and clip
// coordinate linearly extrapolated from the passing vertices beyond it (the
// clip coordinate at least that where it was blocked), else traced through
// the apertures.
fn edge_vertex(gr: Grid, v: u32) -> vec4<f32> {
    let i = i32(v % 17u);
    let j = i32(v / 17u);
    var sum = vec4<f32>(0.0);
    var n = 0.0;
    for (var k = 0; k < 9; k++) {
        let di = k % 3 - 1;
        let dj = k / 3 - 1;
        let i2 = i + 2 * di;
        let j2 = j + 2 * dj;
        if (k == 4 || i2 < 0 || i2 > 16 || j2 < 0 || j2 > 16) { continue; }
        let p = verts[(j + dj) * 17 + i + di];
        let p2 = verts[j2 * 17 + i2];
        if (passes(p) && passes(p2)) {
            sum += vec4<f32>(2.0 * p.xy - p2.xy, p.z, 2.0 * p.w - p2.w);
            n += 1.0;
        }
    }
    if (n == 0.0) { return grid_ray(gr, v, true); }
    return vec4<f32>(sum.xyz / n, max(verts[v].w, sum.w / n));
}

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

// Rasterises a triangle (pixel coordinates, w the clip coordinate) carrying
// `flux` where the interpolated clip coordinate is at most 1. Large
// triangles go to the quarter-resolution film: they only occur in widely
// spread, smooth ghosts and would otherwise dominate the cost.
fn raster_triangle(a0: vec4<f32>, b0: vec4<f32>, c0: vec4<f32>, flux: f32, bin: u32) {
    if (min(a0.w, min(b0.w, c0.w)) > 1.0) { return; }
    let m = (a0 + b0 + c0) / 3.0;
    let area0 = 0.5 * ((b0.x - a0.x) * (c0.y - a0.y) - (c0.x - a0.x) * (b0.y - a0.y));
    if (abs(area0) < 1.0) {
        if (m.w <= 1.0) { splat(m.x, m.y, bin, flux); }
        return;
    }
    let low = abs(area0) > 16.0;
    let k = select(1.0, 0.25, low);
    let a = a0.xy * k;
    let b = b0.xy * k;
    let c = c0.xy * k;
    let area = area0 * k * k;
    let fw = select(P.w, (P.w + 3u) / 4u, low);
    let fh = select(P.h, (P.h + 3u) / 4u, low);
    // Bounds of the passing part: passing corners and edge crossings of clip = 1.
    var lo = vec2<f32>(1e30);
    var hi = vec2<f32>(-1e30);
    for (var e = 0u; e < 3u; e++) {
        var p = a0;
        var q = b0;
        switch e {
            case 1u: { p = b0; q = c0; }
            case 2u: { p = c0; q = a0; }
            default: {}
        }
        if (p.w <= 1.0) {
            lo = min(lo, p.xy);
            hi = max(hi, p.xy);
        }
        if ((p.w <= 1.0) != (q.w <= 1.0)) {
            let x = mix(p.xy, q.xy, (1.0 - p.w) / (q.w - p.w));
            lo = min(lo, x);
            hi = max(hi, x);
        }
    }
    let x0 = max(floor(lo.x * k), 0.0);
    let x1 = min(ceil(hi.x * k), f32(fw));
    let y0 = max(floor(lo.y * k), 0.0);
    let y1 = min(ceil(hi.y * k), f32(fh));
    if (x0 >= x1 || y0 >= y1) { return; }
    if ((x1 - x0) * (y1 - y0) > 65536.0) {
        if (m.w <= 1.0) { splat(m.x, m.y, bin, flux); }
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
            let w2 = 1.0 - w0 - w1;
            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 && w0 * a0.w + w1 * b0.w + w2 * c0.w <= 1.0) {
                let idx = (py * fw + px) * P.bins + bin;
                if (low) { lowres_add(idx, v); } else { film_add(idx, v); }
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn light_grid(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(local_invocation_index) li_idx: u32) {
    // One record per workgroup: a 16×16 tile of one job's grid.
    let task = wg.x + wg.y * 65535u;
    if (task >= P.n_tasks) { return; }
    let job = jobs[task * 4u];
    let jf = bitcast<vec4<f32>>(jobs[task * 4u + 1u]);
    let jf2 = bitcast<vec4<f32>>(jobs[task * 4u + 2u]);
    let tinfo = jobs[task * 4u + 2u];
    let first_task = jobs[task * 4u + 3u].x;
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
    let g = tinfo.w;
    let tiles = g / 16u;
    let tx = tinfo.z % tiles;
    let ty = tinfo.z / tiles;
    let cell = vec2<f32>((reg.z - reg.x) / f32(g), (reg.w - reg.y) / f32(g));
    let bin_width = LAMBDA_RANGE / f32(P.bins);
    let l = LAMBDA_MIN + bin_width * (f32(job.z) + jf.x);
    let e = lg.c + lg.t1 * (jf.y * lg.radius) + lg.t2 * (jf.z * lg.radius);
    let corner = vec2<f32>(f32(tx * 16u) + jf.w - 0.5, f32(ty * 16u) + jf2.x - 0.5);
    let gr = Grid(e, normalize(-lg.c), reg.xy + corner * cell, cell, z_plane, l, light_spectrum(li, l) * PI * lg.radius * lg.radius, job.y);
    for (var v = li_idx; v < 289u; v += 256u) {
        verts[v] = grid_ray(gr, v, false);
    }
    workgroupBarrier();
    // jf2.y weights jobs that sample only some of the wavelength bins.
    let scale = cell.x * cell.y * P.norm * bin_width * jf2.y / (P.pitch * P.pitch);
    let i00 = lid.y * 17u + lid.x;
    var a = verts[i00];
    var b = verts[i00 + 1u];
    var c = verts[i00 + 18u];
    var d = verts[i00 + 17u];
    let cw = vec4<f32>(a.w, b.w, c.w, d.w);
    // Corners whose rays pass the apertures.
    let inside = select(vec4<f32>(0.0), vec4<f32>(1.0), cw >= vec4<f32>(0.0) & cw <= vec4<f32>(1.0));
    // Direct images of lights within the PSF bank: only collect the flux; the
    // psf_splat pass images it through the wave-optical PSF.
    if (P.bank_on != 0u && !ghost && bank_pos(e, false).ok) {
        let flux = 0.25 * dot(vec4<f32>(a.z, b.z, c.z, d.z), inside) * scale;
        if (flux > 0.0) { job_flux_add(first_task, flux); }
        return;
    }
    // Quad cut by an aperture: blocked corners are placed by extrapolation, and
    // the clip coordinate cuts the quad (after Hullin et al. 2011).
    if (any(inside > vec4<f32>(0.0)) && any(cw > vec4<f32>(1.0))) {
        if (a.z < 0.0) { a = edge_vertex(gr, i00); }
        if (b.z < 0.0) { b = edge_vertex(gr, i00 + 1u); }
        if (c.z < 0.0) { c = edge_vertex(gr, i00 + 18u); }
        if (d.z < 0.0) { d = edge_vertex(gr, i00 + 17u); }
    }
    // Cut quads below a pixel keep their unbiased corner samples.
    let ac = c.xy - a.xy;
    let bd = d.xy - b.xy;
    let cut = abs(ac.x * bd.y - ac.y * bd.x) >= 2.0;
    if (all(vec4<f32>(a.w, b.w, c.w, d.w) >= vec4<f32>(0.0)) && (all(cw <= vec4<f32>(1.0)) || cut)) {
        let flux = 0.25 * (a.z + b.z + c.z + d.z) * scale;
        raster_triangle(a, b, c, 0.5 * flux, job.z);
        raster_triangle(a, c, d, 0.5 * flux, job.z);
    } else {
        // Each passing corner carries its quarter.
        for (var k = 0u; k < 4u; k++) {
            var cv = a;
            switch k {
                case 1u: { cv = b; }
                case 2u: { cv = c; }
                case 3u: { cv = d; }
                default: {}
            }
            if (inside[k] > 0.0) { splat(cv.x, cv.y, job.z, 0.25 * cv.z * scale); }
        }
    }
}

// Images the collected flux of in-bank light jobs through the PSF: one
// workgroup per job (its first task).
@compute @workgroup_size(256)
fn psf_splat(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let task = wg.x + wg.y * 65535u;
    if (task >= P.n_tasks || P.bank_on == 0u) { return; }
    let job = jobs[task * 4u];
    let jf = bitcast<vec4<f32>>(jobs[task * 4u + 1u]);
    let tinfo = jobs[task * 4u + 2u];
    if (job.w != 0u || tinfo.z != 0u) { return; }
    let lg = light_geom(job.x);
    let e = lg.c + lg.t1 * (jf.y * lg.radius) + lg.t2 * (jf.z * lg.radius);
    let bp = bank_pos(e, false);
    if (!bp.ok) { return; }
    let jidx = task;
    let flux = JOB_FLUX_LOAD;
    if (flux <= 0.0) { return; }
    var rng = Rng(hash4(task, P.frame, 0x5e1u, 9u));
    let pk = bank_pick(bp, job.z, rnd(&rng), rnd(&rng));
    let ks = bank_kernels(pk.entry);
    for (var part = 0u; part < 2u; part++) {
        var k = ks.core;
        if (part == 1u) { k = ks.wing; }
        let weight = select(ks.core_fraction, 1.0 - ks.core_fraction, part == 1u);
        if (weight <= 0.0) { continue; }
        let n = k.size * k.size;
        for (var i = tid; i < n; i += 256u) {
            var prob = tables[k.base + i];
            if (i > 0u) { prob -= tables[k.base + i - 1u]; }
            if (prob <= 0.0) { continue; }
            // Cells go to the pixel containing their centre.
            let q = rotate2(pk.reference + bank_offset(k, i, 0.5, 0.5), pk.rot);
            let fp = film_pixel(q.x, q.y);
            if (fp >= 0) { film_add(u32(fp) * P.bins + job.z, flux * weight * prob); }
        }
    }
}

// FFT convolution of the spectral film with per-bin diffraction kernels.
// `N` and `LOG2N` are substituted per row length. Complex values are vec2.

struct FftParams {
    w: u32, h: u32, pw: u32, ph: u32,
    bins: u32, b0: u32, b1: u32, inverse: u32,
    rows: u32, cols: u32, kbin: u32, koff: u32,
    khalf: u32, src_rows: u32, batch_rows: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> F: FftParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> twiddle: array<vec2<f32>>;

const N: u32 = FFT_N;
const LOG2N: u32 = FFT_LOG2N;

var<workgroup> sh_re: array<f32, FFT_N>;
var<workgroup> sh_im: array<f32, FFT_N>;

// In-place radix-2 FFT of row `wg.x` (length N) in shared memory.
@compute @workgroup_size(256)
fn fft_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    var base = wg.x * N;
    if (F.batch_rows != 0u) {
        base = (wg.x / F.batch_rows) * N * N + (wg.x % F.batch_rows) * N;
    }
    for (var i = tid; i < N; i += 256u) {
        let v = data[base + i];
        let r = reverseBits(i) >> (32u - LOG2N);
        sh_re[r] = v.x;
        sh_im[r] = v.y;
    }
    workgroupBarrier();
    for (var s = 0u; s < LOG2N; s++) {
        let half = 1u << s;
        for (var k = tid; k < N / 2u; k += 256u) {
            let j = k & (half - 1u);
            let i0 = ((k >> s) << (s + 1u)) + j;
            let i1 = i0 + half;
            var tw = twiddle[j << (LOG2N - 1u - s)];
            if (F.inverse != 0u) { tw.y = -tw.y; }
            let a = vec2<f32>(sh_re[i0], sh_im[i0]);
            let b = vec2<f32>(sh_re[i1], sh_im[i1]);
            let t = vec2<f32>(b.x * tw.x - b.y * tw.y, b.x * tw.y + b.y * tw.x);
            sh_re[i0] = a.x + t.x;
            sh_im[i0] = a.y + t.y;
            sh_re[i1] = a.x - t.x;
            sh_im[i1] = a.y - t.y;
        }
        workgroupBarrier();
    }
    for (var i = tid; i < N; i += 256u) {
        data[base + i] = vec2<f32>(sh_re[i], sh_im[i]);
    }
}

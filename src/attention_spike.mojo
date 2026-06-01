"""Phase-1 go/no-go: hand-written Mojo Metal RoPE + causal GQA attention.

This is the kernel MAX got wrong on its Metal backend (max-backend §8 #2 rung 6)
and the load-bearing risk for the GPU-only thesis (ARCHITECTURE.md §6, §9 #3).
It is built and verified *first, alone*: given Q/K/V fixtures, it runs RoPE +
causal grouped-query attention on the M4 GPU and compares the output to a NumPy
reference (which `attn-capture` has already cross-checked against HF to ~1e-6).

Fixtures live in tests/fixtures/attention/<name>/ as raw little-endian float32:
  q.bin [T,HQ,D]  k.bin/v.bin [T,HKV,D]  expected.bin [T,HQ,D]  (+ meta.txt)
T is derived from q.bin's size. Any fixture exceeding tolerance → non-zero exit,
so `pixi run attn-kernel` is a real gate.

Math (must match reference.py / HF exactly):
  - RoPE split-half, full head_dim rotated, theta=1e6, position = token index.
  - GQA: query head h uses kv head h // (HQ // HKV).
  - causal scaled-dot softmax, scale = 1/sqrt(head_dim); online (flash) softmax.
"""

from std.math import cos, sin, sqrt, exp, log
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceContext
from std.collections import InlineArray
from layout import TileTensor, TensorLayout, row_major

comptime HQ = 14          # query heads
comptime HKV = 2          # key/value heads (GQA)
comptime HEAD_DIM = 64
comptime HALF = HEAD_DIM // 2
comptime GROUP = HQ // HKV  # 7 query heads per kv head
comptime THETA = Float32(1000000.0)

comptime TOL = Float32(2.0e-3)


def attn_kernel[
    LT: TensorLayout
](
    Q: TileTensor[DType.float32, LT, MutAnyOrigin],
    K: TileTensor[DType.float32, LT, MutAnyOrigin],
    V: TileTensor[DType.float32, LT, MutAnyOrigin],
    O: TileTensor[DType.float32, LT, MutAnyOrigin],
    T: Int,
):
    comptime assert Q.flat_rank == 1, "flat buffers"
    var h = thread_idx.x   # query head, 0..HQ-1
    var t = block_idx.x    # query position, 0..T-1
    if h >= HQ or t >= T:
        return
    var kvh = h // GROUP

    # --- load + RoPE the query vector for (t, h) ---
    var qbase = (t * HQ + h) * HEAD_DIM
    var qr = InlineArray[Float32, HEAD_DIM](fill=0.0)
    for d in range(HALF):
        var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
        var ang = Float32(t) * freq
        var c = cos(ang)
        var s = sin(ang)
        var x0 = rebind[Scalar[DType.float32]](Q[qbase + d])
        var x1 = rebind[Scalar[DType.float32]](Q[qbase + d + HALF])
        qr[d] = x0 * c - x1 * s
        qr[d + HALF] = x1 * c + x0 * s

    var scale = 1.0 / sqrt(Float32(HEAD_DIM))

    # --- online-softmax attention over keys j = 0..t (causal) ---
    var m = Float32(-1.0e30)
    var l = Float32(0.0)
    var acc = InlineArray[Float32, HEAD_DIM](fill=0.0)

    for j in range(t + 1):
        var kbase = (j * HKV + kvh) * HEAD_DIM
        # load + RoPE key j, accumulate dot(q_rot, k_rot)
        var score = Float32(0.0)
        for d in range(HALF):
            var freq = exp(-(2.0 * Float32(d) / Float32(HEAD_DIM)) * log(THETA))
            var ang = Float32(j) * freq
            var c = cos(ang)
            var s = sin(ang)
            var x0 = rebind[Scalar[DType.float32]](K[kbase + d])
            var x1 = rebind[Scalar[DType.float32]](K[kbase + d + HALF])
            var kr0 = x0 * c - x1 * s
            var kr1 = x1 * c + x0 * s
            score += qr[d] * kr0 + qr[d + HALF] * kr1
        score *= scale

        var m_new = max(m, score)
        var corr = exp(m - m_new)
        var p = exp(score - m_new)
        l = l * corr + p
        var vbase = (j * HKV + kvh) * HEAD_DIM
        for d in range(HEAD_DIM):
            var vd = rebind[Scalar[DType.float32]](V[vbase + d])
            acc[d] = acc[d] * corr + p * vd
        m = m_new

    var obase = (t * HQ + h) * HEAD_DIM
    for d in range(HEAD_DIM):
        O[obase + d] = rebind[O.ElementType](acc[d] / l)


def read_f32_file(path: String) raises -> List[Float32]:
    var out = List[Float32]()
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var n = len(raw) // 4
        var p = raw.unsafe_ptr().bitcast[Float32]()
        for i in range(n):
            out.append(p[i])
    return out^


def run_fixture(ctx: DeviceContext, dir: String) raises -> Bool:
    var q = read_f32_file(dir + "/q.bin")
    var k = read_f32_file(dir + "/k.bin")
    var v = read_f32_file(dir + "/v.bin")
    var expected = read_f32_file(dir + "/expected.bin")

    var nq = len(q)
    var nk = len(k)
    var nv = len(v)
    var T = nq // (HQ * HEAD_DIM)
    if nq != T * HQ * HEAD_DIM or len(expected) != nq:
        raise Error("fixture shape mismatch in " + dir)

    var q_dev = ctx.enqueue_create_buffer[DType.float32](nq)
    var k_dev = ctx.enqueue_create_buffer[DType.float32](nk)
    var v_dev = ctx.enqueue_create_buffer[DType.float32](nv)
    var o_dev = ctx.enqueue_create_buffer[DType.float32](nq)
    o_dev.enqueue_fill(0.0)

    with q_dev.map_to_host() as mm:
        var mt = TileTensor(mm, row_major(nq))
        comptime assert mt.flat_rank == 1
        for i in range(nq):
            mt[i] = rebind[mt.ElementType](q[i])
    with k_dev.map_to_host() as mm:
        var mt = TileTensor(mm, row_major(nk))
        comptime assert mt.flat_rank == 1
        for i in range(nk):
            mt[i] = rebind[mt.ElementType](k[i])
    with v_dev.map_to_host() as mm:
        var mt = TileTensor(mm, row_major(nv))
        comptime assert mt.flat_rank == 1
        for i in range(nv):
            mt[i] = rebind[mt.ElementType](v[i])

    var q_layout = row_major(nq)
    var qt = TileTensor(q_dev, q_layout)
    var kt = TileTensor(k_dev, row_major(nk))
    var vt = TileTensor(v_dev, row_major(nv))
    var ot = TileTensor(o_dev, row_major(nq))

    comptime kernel = attn_kernel[type_of(q_layout)]
    ctx.enqueue_function[kernel](
        qt, kt, vt, ot, T,
        grid_dim=T,
        block_dim=HQ,
    )
    ctx.synchronize()

    var max_abs = Float32(0.0)
    with o_dev.map_to_host() as m:
        var mt = TileTensor(m, row_major(nq))
        comptime assert mt.flat_rank == 1
        for i in range(nq):
            var got = rebind[Scalar[DType.float32]](mt[i])
            var diff = abs(got - expected[i])
            if diff > max_abs:
                max_abs = diff

    var ok = max_abs < TOL
    var tag = "OK" if ok else "FAIL"
    print(
        "  ", dir, " T=", T, " max_abs=", max_abs, " [", tag, "]", sep=""
    )
    return ok


def main() raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator detected — this is a GPU-only build")

    var ctx = DeviceContext()
    var root = "tests/fixtures/attention/"
    var names = [String("synthetic"), String("real_L0"), String("real_L23")]

    print("attention+RoPE spike — GPU vs NumPy reference (tol", TOL, "):")
    var all_ok = True
    for name in names:
        var ok = run_fixture(ctx, root + name)
        all_ok = all_ok and ok

    if not all_ok:
        raise Error("GPU attention output does NOT match reference — spike FAILED")
    print("OK — Mojo Metal RoPE+GQA attention matches the reference on all fixtures")

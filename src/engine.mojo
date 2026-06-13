"""Model-agnostic decode runtime, generic over the `ModelWeights` trait: the KV-
cache `Session`, prefill (`sess_prefill`, `sess_prefill_suffix`), per-token decode
(`sess_step`), and the greedy / sampled `generate` loops. The model family supplies
the three weight-touching steps (embed a prompt, run one decoder layer, produce
last-position logits) via the trait; everything here — caching, the loop, sampling
— is reused unchanged across families. Adding a family needs no engine change."""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from tensor_ops import DevBuf
from sampling import process_logits, sample, argmax_f
from model_iface import ModelWeights


def upload_ids(ctx: DeviceContext, vals: List[Int]) raises -> DeviceBuffer[DType.int32]:
    var n = len(vals)
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    with d.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](Int32(vals[i]))
    return d^


def argmax_last[W: ModelWeights](ctx: DeviceContext, mut w: W, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> Int:
    """Greedy: last-position logits (via the family's LM head) → argmax."""
    return argmax_f(w.lm_logits(ctx, h, T, dummy))


def logits_last[W: ModelWeights](ctx: DeviceContext, mut w: W, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> List[Float32]:
    """Last-position logits on the host (the family's tied LM head + any softcap)."""
    return w.lm_logits(ctx, h, T, dummy)


# ── decode session: KV caches + position, the per-step primitive ──────────────

@fieldwise_init
struct Session(Movable):
    """Holds the per-layer KV caches and the current position. `prefill` runs the
    prompt and returns the last-position logits; `step` advances one token. Shared
    by greedy/sampled generate and the server's streaming loop."""
    var kcs: List[DevBuf]
    var vcs: List[DevBuf]
    var dummy: DevBuf
    var cache_len: Int
    var pos: Int


def new_session(ctx: DeviceContext, max_seq: Int, nlayers: Int, nkv: Int) raises -> Session:
    var cache_len = max_seq * nkv
    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    for _ in range(nlayers):
        kcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
        vcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
    return Session(kcs^, vcs^, ctx.enqueue_create_buffer[DType.float32](1), cache_len, 0)


def sess_prefill[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session, prompt: List[Int]) raises -> List[Float32]:
    var P = len(prompt)
    var ids_dev = upload_ids(ctx, prompt)
    var h = w.embed_prompt(ctx, ids_dev, P)
    for l in range(w.config().nlayers):
        h = w.run_layer(ctx, l, h, s.kcs[l], s.vcs[l], P, 0, s.cache_len, s.dummy)
    s.pos = P
    return w.lm_logits(ctx, h, P, s.dummy)


# Below this suffix length, prefill is sub-second and frequent (the prefix-cache
# common case), so progress reporting — and its per-layer synchronize — is skipped
# entirely, leaving that hot path byte-for-byte as before.
comptime PROGRESS_MIN_TOK = 2048
comptime PROGRESS_EVERY_NS = 5_000_000_000   # ~5 s between progress lines


def _ktok(n: Int) -> String:
    """Compact token count: 9562 -> '9.5k', 800 -> '800'."""
    if n < 1000:
        return String(n)
    return String(n // 1000) + "." + String((n % 1000) // 100) + "k"

def _dur(secs: Float64) -> String:
    var s = Int(secs + 0.5)
    if s < 90:
        return String(s) + "s"
    return String(s // 60) + "m" + String(s % 60) + "s"


def sess_prefill_suffix[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session,
                        suffix: List[Int], offset: Int, progress: Bool = False) raises -> List[Float32]:
    """Prefill `suffix` tokens at cache position `offset`, reusing the K/V already
    stored in rows [0, offset). Returns the last-row logits. This is the engine
    behind the server's cross-request prefix cache; `sess_prefill` is just the
    offset==0 / whole-prompt special case. RoPE positions come from `offset`, so
    the rotated K and the attention mask stay correct for the reused prefix.

    With `progress` (and a large enough suffix), prints a throttled stdout line
    with percent done + ETA; gated off below PROGRESS_MIN_TOK so frequent tiny
    prefills are untouched."""
    var Q = len(suffix)
    var nlayers = w.config().nlayers
    var ids_dev = upload_ids(ctx, suffix)
    var h = w.embed_prompt(ctx, ids_dev, Q)
    var report = progress and Q >= PROGRESS_MIN_TOK
    var t0 = perf_counter_ns()
    var last = t0
    for l in range(nlayers):
        h = w.run_layer(ctx, l, h, s.kcs[l], s.vcs[l], Q, offset, s.cache_len, s.dummy)
        if report:
            ctx.synchronize()
            var now = perf_counter_ns()
            if Float64(now - last) >= Float64(PROGRESS_EVERY_NS):
                var done = l + 1
                var elapsed = Float64(now - t0) / 1.0e9
                var eta = elapsed * Float64(nlayers - done) / Float64(done)
                print("  prefill ", _ktok(Q), "tok: ", (done * 100) // nlayers,
                      "% (layer ", done, "/", nlayers, "), ~", _dur(eta), " left", sep="")
                last = now
    s.pos = offset + Q
    return w.lm_logits(ctx, h, Q, s.dummy)


def sess_step[W: ModelWeights](ctx: DeviceContext, mut w: W, mut s: Session, token: Int) raises -> List[Float32]:
    var one = upload_ids(ctx, [token])
    var h = w.embed_prompt(ctx, one, 1)
    for l in range(w.config().nlayers):
        h = w.run_layer(ctx, l, h, s.kcs[l], s.vcs[l], 1, s.pos, s.cache_len, s.dummy)
    s.pos += 1
    return w.lm_logits(ctx, h, 1, s.dummy)


def generate[W: ModelWeights](ctx: DeviceContext, mut w: W, prompt: List[Int], max_new: Int) raises -> List[Int]:
    """Greedy decode: prefill the prompt then emit tokens until EOS or max_new."""
    var cfg = w.config()
    var s = new_session(ctx, len(prompt) + max_new + 2, cfg.nlayers, cfg.nkv)
    var nxt = argmax_f(sess_prefill(ctx, w, s, prompt))
    var gen = List[Int]()
    gen.append(nxt)
    while len(gen) < max_new and nxt != cfg.eos1 and nxt != cfg.eos2:
        nxt = argmax_f(sess_step(ctx, w, s, nxt))
        gen.append(nxt)
    return gen^


def generate_sample[W: ModelWeights](ctx: DeviceContext, mut w: W, prompt: List[Int], max_new: Int,
                    temp: Float32, top_k: Int, top_p: Float32, rep_pen: Float32,
                    seed: UInt64) raises -> List[Int]:
    """Greedy-structure decode but draw each token from the processed distribution."""
    var cfg = w.config()
    var s = new_session(ctx, len(prompt) + max_new + 2, cfg.nlayers, cfg.nkv)
    var rng = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)
    var context = prompt.copy()
    var nxt = sample(process_logits(sess_prefill(ctx, w, s, prompt), context, temp, top_k, top_p, rep_pen), rng)
    var gen = List[Int]()
    gen.append(nxt)
    context.append(nxt)
    while len(gen) < max_new and nxt != cfg.eos1 and nxt != cfg.eos2:
        var dist = process_logits(sess_step(ctx, w, s, nxt), context, temp, top_k, top_p, rep_pen)
        nxt = sample(dist, rng)
        context.append(nxt)
        gen.append(nxt)
    return gen^

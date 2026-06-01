"""Generate attention-spike fixtures and validate the NumPy reference vs HF.

Writes tests/fixtures/attention/<name>/{q,k,v,expected}.bin (raw little-endian
float32, C-contiguous [T, H, D]) + meta.txt ("T HQ HKV D theta") for:

  - synthetic         random Q/K/V (broad correctness sweep)
  - real_L0, real_L23 Q/K/V captured from the *actual* Qwen2.5-0.5B-Instruct run
                      on a real prompt (realistic magnitudes — guards against
                      "correct on random, wrong on real")

For the real fixtures it also asserts the NumPy reference reproduces HF's own
attention output (the o_proj input) — i.e. our notion of "correct" matches a
trusted real implementation, not just itself. Run via `pixi run attn-capture`.
"""

import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "attention")
)


def save_fixture(name, q, k, v, theta):
    d = os.path.join(FIX_ROOT, name)
    os.makedirs(d, exist_ok=True)
    expected = ref.attention(q, k, v, theta)
    for fn, arr in [("q", q), ("k", k), ("v", v), ("expected", expected)]:
        np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(d, fn + ".bin"))
    T, HQ, D = q.shape
    HKV = k.shape[1]
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(f"{T} {HQ} {HKV} {D} {theta}\n")
    return expected


def synthetic():
    HQ, HKV, D, theta, T = 14, 2, 64, 1e6, 24
    rng = np.random.RandomState(0)
    q = rng.randn(T, HQ, D).astype(np.float32)
    k = rng.randn(T, HKV, D).astype(np.float32)
    v = rng.randn(T, HKV, D).astype(np.float32)
    save_fixture("synthetic", q, k, v, theta)
    print(f"synthetic: saved T={T} HQ={HQ} HKV={HKV} D={D}")


def real():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    model = model.float().eval()
    cfg = model.config
    HQ = cfg.num_attention_heads
    HKV = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // HQ
    theta = float(getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"])
    print(f"config: HQ={HQ} HKV={HKV} D={D} theta={theta} layers={cfg.num_hidden_layers}")

    msgs = [{"role": "user", "content": "In one sentence, what is the capital of France?"}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    ids = enc["input_ids"]
    T = ids.shape[1]
    print(f"prompt tokens T={T}")

    layers = [0, cfg.num_hidden_layers - 1]
    cap = {}
    hooks = []

    def mk_out(L, which):
        def hook(mod, inp, out):
            cap[(L, which)] = out.detach().float().cpu().numpy()[0]
        return hook

    def mk_ctx(L):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            cap[(L, "ctx")] = x.detach().float().cpu().numpy()[0]
        return hook

    for L in layers:
        attn = model.model.layers[L].self_attn
        hooks.append(attn.q_proj.register_forward_hook(mk_out(L, "q")))
        hooks.append(attn.k_proj.register_forward_hook(mk_out(L, "k")))
        hooks.append(attn.v_proj.register_forward_hook(mk_out(L, "v")))
        hooks.append(attn.o_proj.register_forward_pre_hook(mk_ctx(L), with_kwargs=True))

    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    ok = True
    for L in layers:
        q = cap[(L, "q")].reshape(T, HQ, D)
        k = cap[(L, "k")].reshape(T, HKV, D)
        v = cap[(L, "v")].reshape(T, HKV, D)
        hf_ctx = cap[(L, "ctx")].reshape(T, HQ, D)
        expected = save_fixture(f"real_L{L}", q, k, v, theta)
        max_abs = float(np.abs(expected - hf_ctx).max())
        status = "OK" if max_abs < 1e-3 else "FAIL"
        print(f"real_L{L}: saved T={T}; NumPy-ref vs HF-context max_abs={max_abs:.3e} [{status}]")
        ok = ok and max_abs < 1e-3

    if not ok:
        raise SystemExit("NumPy reference does NOT match HF attention — fix the reference")
    print("OK: NumPy reference reproduces HF attention on real activations")


def main():
    os.makedirs(FIX_ROOT, exist_ok=True)
    synthetic()
    real()


if __name__ == "__main__":
    main()

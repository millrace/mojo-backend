"""From-scratch NumPy reference for Qwen2 attention + RoPE.

This is the *definition of correct* for the Phase-1 spike (ARCHITECTURE.md §6,
§7). It mirrors HF transformers' Qwen2 attention exactly:

- RoPE: split-half ("rotate_half"), full head_dim rotated, theta=1e6.
- GQA: each query head h uses key/value head h // (HQ // HKV).
- causal scaled-dot-product softmax, scale = 1/sqrt(head_dim), softmax in f32.

Everything is float32 to match a CPU/f32 model run, so the same values can be
diffed against both the HF capture (Python) and the Mojo GPU kernel.
"""

import numpy as np

DTYPE = np.float32


def rope_cos_sin(T: int, head_dim: int, theta: float):
    """cos/sin tables of shape [T, head_dim] (second half duplicates the first)."""
    inv_freq = 1.0 / (
        np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )  # [head_dim/2]
    pos = np.arange(T, dtype=np.float32)
    freqs = np.outer(pos, inv_freq)  # [T, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [T, head_dim]
    return np.cos(emb).astype(DTYPE), np.sin(emb).astype(DTYPE)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    """x: [T, H, D]; cos/sin: [T, D] -> [T, H, D]."""
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return (x * cos) + (rotate_half(x) * sin)


def attention(q, k, v, theta: float):
    """q: [T, HQ, D]; k, v: [T, HKV, D]  ->  context [T, HQ, D] (pre-o_proj)."""
    q = q.astype(DTYPE)
    k = k.astype(DTYPE)
    v = v.astype(DTYPE)
    T, HQ, D = q.shape
    HKV = k.shape[1]
    group = HQ // HKV

    cos, sin = rope_cos_sin(T, D, theta)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scale = np.float32(1.0 / np.sqrt(D))
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # True above diagonal = masked
    out = np.zeros((T, HQ, D), dtype=DTYPE)
    for h in range(HQ):
        kvh = h // group
        scores = (q[:, h, :] @ k[:, kvh, :].T) * scale  # [T, T]
        scores = np.where(mask, np.float32(-np.inf), scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        out[:, h, :] = (w.astype(DTYPE) @ v[:, kvh, :]).astype(DTYPE)
    return out

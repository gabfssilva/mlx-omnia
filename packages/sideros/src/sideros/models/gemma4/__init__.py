"""Gemma 4 text (`gemma4` / `gemma4_unified` / `gemma4_assistant`): PLE, KV sharing,
proportional partial-rotary RoPE, dual head_dim, standard (non-zero-centered) norms,
logit softcap, double-wide MLP on KV-shared layers.

Authoritative semantics: transformers' `modeling_gemma4.py`. Four structures have no
Sideros precedent:

- **Per-Layer Embeddings (PLE)** — a second `[vocab, layers*ple_dim]` embedding table
  scaled by `sqrt(ple_dim)`, a context projection scaled by `1/sqrt(hidden)`, combined
  at `1/sqrt(2)` per layer, and a **third residual arm** per block
  (gate → act → mul-by-ple → project → norm → add → `*= layer_scalar`).
- **KV sharing** — `num_kv_shared_layers` trailing layers own no `k_proj`/`v_proj`/
  `k_norm`/`v_norm`; they read full-length K,V published by the last non-sharing layer
  of the same type. `make_cache` returns a `KVCache` per **non-sharing** layer only;
  the sharing layers get a `SharedKVReader` that references the storing layer's cache.
- **Proportional partial-rotary RoPE** on full-attention layers —
  `partial_rotary_factor=0.25` over `global_head_dim=512`: 64 real frequency pairs
  (128 rotated dims) and 192 zero-frequency pairs (identity). The exponent denominator
  is `global_head_dim`, not the rotated count, so `mx.fast.rope` cannot reproduce it;
  manual RoPE with precomputed fp32 cos/sin is required.
- **Dual head_dim** — sliding layers use `head_dim` (256), full layers use
  `global_head_dim` (512). `scale=1.0` (both Q and K are RMSNormed).

Norms are **standard** RMSNorm (`weight`, no `1+w`) — *not* the Gemma 3 zero-centered
fold. `v_norm` and the MoE router norm are scale-less (`RMSNormNoScale`, no `weight`).
No dict-side qkv fusion (shared layers lack k/v; widths vary by layer type). No
`_fold_norm_scales`. The MLP gate/up stay separate (no dict-side concat) to keep
shared layers clean; the tree declares separate `gate_proj`/`up_proj` as the checkpoint
stores them. Logit softcap `tanh(logits/cap)*cap` runs in fp32. `attention_k_eq_v`
(12B path) drops `v_proj` on full layers (`v = k`).
"""

from sideros.models.gemma4.checkpoint import CHECKPOINT
from sideros.models.gemma4.config import (
    Gemma4Config,
    Gemma4RoPE,
    Gemma4RoPEParameters,
    Gemma4TextConfig,
)
from sideros.models.gemma4.model import Gemma4, Gemma4Activations

__all__ = [
    "CHECKPOINT",
    "Gemma4",
    "Gemma4Activations",
    "Gemma4Config",
    "Gemma4RoPE",
    "Gemma4RoPEParameters",
    "Gemma4TextConfig",
]

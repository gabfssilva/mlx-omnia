"""Laguna S 2.1 (`laguna`): interleaved SWA/full attention with per-layer head counts,
softplus attention gating, sigmoid-routed MoE with a per-expert selection bias and a
shared expert.

Authoritative semantics: transformers' `modeling_laguna.py`. What the trunk does that
no previous ported model has all at once:

- **Per-layer head count** (`num_attention_heads_per_layer`): 48 query heads on full
  layers, 72 on sliding. KV heads are constant at 8, so the GQA ratio varies by layer.
- **Per-layer-type RoPE**: full layers use YaRN (factor 128, original 8192) with
  `partial_rotary_factor` 0.5 — the first 64 of 128 dims rotate; the rest pass through.
  Sliding layers use default RoPE (theta 10k) with full rotary. The YaRN `mscale`
  (attention_factor 1.4852) pre-scales only the rotary dims, not the pass-through.
- **Softplus attention gating** (per-head): `softplus(g_proj(x))` in float32 scales each
  head's output before `o_proj`.
- **Sigmoid routing** (not softmax): scores are independent; selection adds a
  `e_score_correction_bias`; weights come from the unbiased sigmoid scores and
  renormalize. A `routed_scaling_factor` (2.5) scales the routed output before the
  shared expert is added. `softmax_topk` does NOT apply — routing is plain ops.

qkv is fused on the output axis, gate‖up row-interleaved for the decode kernel, and
experts stacked `[E, out, in]` via `SwitchLinear` — all at load, dict-side.
"""

from mlx_omnia.engine.models.laguna.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.laguna.config import (
    LagunaConfig,
    LagunaRoPEConfigs,
    LagunaRoPEParameters,
    LagunaYaRNScaling,
)
from mlx_omnia.engine.models.laguna.model import Laguna, LagunaActivations

__all__ = [
    "CHECKPOINT",
    "Laguna",
    "LagunaActivations",
    "LagunaConfig",
    "LagunaRoPEConfigs",
    "LagunaRoPEParameters",
    "LagunaYaRNScaling",
]

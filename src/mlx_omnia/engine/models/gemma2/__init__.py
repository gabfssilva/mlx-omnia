"""Gemma 2: sandwich norms, alternating sliding/full attention, and tanh softcapping on
both the attention scores and the logits.

Authoritative semantics: transformers' modeling_gemma2.py; the numerical reference
implementation does *not* implement the sliding window and is therefore only a
reference inside the first 4096 positions.

Gemma 1's three house rules carry over (zero-centred RMSNorm folded at load, embeddings
scaled by `sqrt(hidden_size)`, `head_dim` decoupled from `hidden_size`). What Gemma 2
adds:

- **Sandwich norms**, the layout `gemma3` already holds.
- **`layer_types` alternating `sliding_attention` / `full_attention`**, starting on
  sliding: transformers derives it as `(i + 1) % 2` when the config omits the list. The
  window lives in the mask, never in the cache — the same rule `gemma3` follows.
- **`attn_logit_softcapping`**: `tanh(scores / c) · c` between the matmul and the
  softmax. That is a boundary `mx.fast.scaled_dot_product_attention` has no seam for, so
  attention is written out here — the one ported block that does not go through the fused
  kernel. The score matmul is the *only* op that has to be manual; the softcap, the mask
  and the softmax then run on the materialized `[…, L, S]` scores.
- **`final_logit_softcapping`**: the same tanh over the head's output.
- **`query_pre_attn_scalar`** (256) sets the scale, and on the 27B it is *not*
  `head_dim` (128) — the scale is `query_pre_attn_scalar ** -0.5` and nothing else.

GQA is expanded by reshaping the queries to `[B, kv_heads, repeats, L, head_dim]` and
broadcasting the keys, as the reference implementation does: a real `repeat` would materialize the
whole cache once per group.
"""

from mlx_omnia.engine.models.gemma2.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.gemma2.config import Gemma2Config
from mlx_omnia.engine.models.gemma2.model import Gemma2, Gemma2Activations

__all__ = [
    "CHECKPOINT",
    "Gemma2",
    "Gemma2Activations",
    "Gemma2Config",
]

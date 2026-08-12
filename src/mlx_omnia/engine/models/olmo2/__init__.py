"""OLMo 2: Llama's leaves with the norms moved after each sublayer, and q/k normed
across the whole projection.

Authoritative semantics: transformers' modeling_olmo2.py.

Two deltas against llama, both about where a norm sits:

- **Post-norm, not pre-norm.** The block is `x + post_attention_layernorm(attn(x))`
  then `h + post_feedforward_layernorm(mlp(h))` — attention reads the *raw* residual
  and the norm is applied to its output. There is no `input_layernorm` leaf, so a
  loader that assumed llama's tree would fail the totality contract on both ends.
- **q/k norm over `heads · head_dim`, not per head.** Qwen3 normalizes each head
  separately (`RMSNorm(head_dim)`); OLMo 2 normalizes the flat projection before it is
  split into heads, which is a different reduction over different elements. It also
  runs before the reshape, so `rope_epilogue` — written for the per-head shape — does
  not apply.
"""

from mlx_omnia.engine.models.olmo2.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.olmo2.config import Olmo2Config
from mlx_omnia.engine.models.olmo2.model import Olmo2

__all__ = [
    "CHECKPOINT",
    "Olmo2",
    "Olmo2Config",
]

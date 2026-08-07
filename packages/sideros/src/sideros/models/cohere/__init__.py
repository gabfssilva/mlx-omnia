"""Cohere Command-R: parallel block, LayerNorm instead of RMSNorm, tied head with a
logit scale.

Authoritative semantics: transformers' modeling_cohere.py.

Four deltas, and the first is structural:

- **One norm per block, two sublayers.** Attention and MLP both read the *same*
  normalized input and their outputs join the residual together:
  `x + attn(norm(x)) + mlp(norm(x))`. There is no `post_attention_layernorm` leaf and
  the MLP does not see the attention's output — the sequential block would give
  different numbers with the same weights.
- **LayerNorm, not RMSNorm**, and with `bias=False` in every published checkpoint:
  the mean is subtracted (RMSNorm does not), so the two are not interchangeable.
- **`traditional=True` RoPE**, pairing `(x[2i], x[2i+1])`.
- **The head is always tied** and its output is multiplied by `logit_scale`
  (0.0625 = 1/16 on Command-R).

`use_qk_norm` (Command-R+ 08-2024) normalizes per head with a `[heads, head_dim]`
weight — a LayerNorm over the head dimension, scaled by a 2-D weight, applied before
the transpose into attention layout.
"""

from sideros.models.cohere.checkpoint import CHECKPOINT
from sideros.models.cohere.config import CohereConfig
from sideros.models.cohere.model import Cohere

__all__ = [
    "CHECKPOINT",
    "Cohere",
    "CohereConfig",
]

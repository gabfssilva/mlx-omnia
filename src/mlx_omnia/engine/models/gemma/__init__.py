"""Gemma 1: the llama block with Gemma's three house rules and nothing else.

Authoritative semantics: transformers' modeling_gemma.py.

The block is llama's — pre-norm, two joins, no sandwich. What is Gemma's:

- **Zero-centred RMSNorm** (`scale = 1 + w`), folded on the dict side at load exactly
  as `gemma3` does, so the tree holds plain `nn.RMSNorm`.
- **Embeddings scaled by `sqrt(hidden_size)`**, with transformers' float32 constant cast
  to the weight dtype (bf16 and fp32 see different numbers; the cast is reproduced, not
  the rounded constant).
- **`head_dim` is decoupled from `hidden_size`** — 256 against 2048 on the 2B, so
  `heads · head_dim` is not the hidden width and the o_proj input is the former.

The head is always tied. `hidden_act` is read rather than assumed: the published
checkpoints ship `"gelu"` (the exact erf form) while the transformers default is the tanh
approximation, and the two differ by ~1e-3 — picking one silently is how a port ends up
off by a rounding no fixture explains.
"""

from mlx_omnia.engine.models.gemma.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.gemma.config import GemmaConfig
from mlx_omnia.engine.models.gemma.model import Gemma

__all__ = ["CHECKPOINT", "Gemma", "GemmaConfig"]

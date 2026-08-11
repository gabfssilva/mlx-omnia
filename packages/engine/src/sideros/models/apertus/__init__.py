"""Apertus: a gate-free MLP with a learned activation, and norms named after what they
precede.

Authoritative semantics: transformers' modeling_apertus.py.

Three deltas against llama:

- **No gate.** The MLP is `down(xielu(up(x)))` — one projection in, not two — so there
  is no gate‖up fusion at load and `SwiGLU` does not apply.
- **xIELU**, a learned activation with two per-layer parameters in the checkpoint
  (`alpha_p`, `alpha_n`, stored with singleton axes and squeezed on the dict side):
  `softplus(alpha_p)·x² + beta·x` above zero, `(expm1(min(x, eps)) - x)·(beta +
  softplus(alpha_n)) + beta·x` below. They are weights, so they live in the tree under
  the checkpoint's own path (`mlp.act_fn.alpha_p`).
- **`attention_layernorm` / `feedforward_layernorm`**, not `input_layernorm` /
  `post_attention_layernorm`. Same two pre-norms, different names.

q/k norm is per head (`RMSNorm(head_dim)`), which is Qwen 3's shape, and `post_norm` is
false on the published checkpoint — a true one would add a layout this tree does not
declare, so it raises.
"""

from sideros.models.apertus.checkpoint import CHECKPOINT
from sideros.models.apertus.config import ApertusConfig
from sideros.models.apertus.model import Apertus

__all__ = [
    "CHECKPOINT",
    "Apertus",
    "ApertusConfig",
]

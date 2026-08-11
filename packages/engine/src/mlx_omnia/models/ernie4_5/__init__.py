"""ERNIE 4.5 dense: llama's block with `traditional=True` RoPE and an optional
`head_dim`.

Authoritative semantics: transformers' modeling_ernie4_5.py.

The only delta that changes numbers is the RoPE pairing: ERNIE rotates `(x[2i],
x[2i+1])` where llama rotates `(x[i], x[i+d/2])`. Same rotation, different pairing, and
silent when wrong.

`use_bias` covers q/k/v/o *and* the MLP in one flag. The published checkpoints set it
false; a true one would need a biased `SwiGLU`, which the shared layer does not have, so
the config raises instead of loading into a tree that would drop the bias.
"""

from mlx_omnia.models.ernie4_5.checkpoint import CHECKPOINT
from mlx_omnia.models.ernie4_5.config import Ernie45Config
from mlx_omnia.models.ernie4_5.model import Ernie45

__all__ = [
    "CHECKPOINT",
    "Ernie45",
    "Ernie45Config",
]

"""Llama 4 Scout (`llama4`): iRoPE, block-chunked local attention, NoPE + temperature
tuning, weightless qk_norm, sigmoid top-1 MoE with an ungated shared expert.

Authoritative semantics: transformers' `modeling_llama4.py`. Five arithmetic
pieces no other ported model has:

- **Interleaved RoPE** (`traditional=True`): every other port uses the half-split
  rotation (`traditional=False`). The two styles produce different rotations on every
  dim pair — `traditional` is load-bearing for parity.
- **Llama3 RoPE scaling** (NTK-by-parts `inv_freq`, factor 16, original 8192): high
  frequencies extrapolate, low frequencies divide by `factor`; the smooth band is
  degenerate for Scout (`low_freq_factor == high_freq_factor == 1.0`), leaving a
  binary split.
- **Block-chunked local attention** (chunk 8192): the window starts at the chunk
  boundary `floor(p/chunk)*chunk`, not at `p-chunk` (that is a sliding window). At a
  boundary a query sees ~1 token; at the end it sees the whole chunk.
- **NoPE + temperature tuning** on full layers (every 4th): `q *=
  log1p(floor(pos/floor_scale)) * attn_scale + 1.0` per position, applied before
  attention, only on NoPE layers.
- **Weightless L2Norm qk_norm**: `mx.fast.rms_norm(weight=None, eps=1e-6)`, no
  checkpoint weight, applied after RoPE, only on RoPE (chunked) layers.

MoE is the simplest in the house: top-1 (argmax + sigmoid, no renorm/bias/scale), a
shared expert that is an ungated dense SwiGLU folded into the residual. Routing
pre-multiplies the sigmoid score into the expert input (transformers
authoritative): `expert(x * sigmoid(logit))`, not `sigmoid(logit) * expert(x)` —
silu is not linear, so the two differ. qkv is fused on the output axis, gate‖up
row-interleaved for the decode kernel, experts stacked `[E, out, in]` — all at
load, dict-side.
"""

from mlx_omnia.models.llama4.checkpoint import CHECKPOINT
from mlx_omnia.models.llama4.config import Llama4Config, Llama4RoPEParameters, Llama4TextConfig
from mlx_omnia.models.llama4.model import Llama4, Llama4Activations

__all__ = [
    "CHECKPOINT",
    "Llama4",
    "Llama4Activations",
    "Llama4Config",
    "Llama4RoPEParameters",
    "Llama4TextConfig",
]

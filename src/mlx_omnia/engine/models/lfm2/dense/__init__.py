"""LFM2 dense: the same hybrid trunk as `lfm2.moe` with a plain SwiGLU in every block.

Authoritative semantics: transformers' modeling_lfm2.py.

The mixers — the gated short conv and the GQA with per-head q/k norm — are the family's
shared `layers/` modules: the checkpoint's leaf names (`conv`, `in_proj`, `out_proj`,
`q_layernorm`) are identical to the MoE variant's, and so is the fused one-token conv
path. What this variant owns is the trunk and the MLP width.
"""

from mlx_omnia.engine.models.lfm2.dense.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.lfm2.dense.model import LFM2

__all__ = ["CHECKPOINT", "LFM2"]

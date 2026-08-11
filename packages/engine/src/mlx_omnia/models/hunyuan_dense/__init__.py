"""Hunyuan dense (`hunyuan_dense` / `hunyuan_v1_dense`): llama with per-head q/k norm
under Tencent's names and an NTK-alpha RoPE base.

Authoritative semantics: transformers' modular_hunyuan_v1_dense.py; the reference is
transformers alone.

Two deltas against llama:

- **`query_layernorm` / `key_layernorm`**, per head, applied to q and k after the
  reshape and before the rotation — Qwen 3's placement, Tencent's naming.
- **NTK-alpha RoPE.** With `rope_parameters.rope_type == "dynamic"` and an `alpha`, the
  base is not `rope_theta`: it is `rope_theta · alpha^(d/(d-2))`, computed once. That is
  a *static* rescaling in transformers too — the "dynamic" name refers to how it was
  derived, not to anything that changes per step — so it folds into `mx.fast.rope`'s
  `base` with no frequency table.

`rope_parameters` is the newer spelling of `rope_scaling` plus `rope_theta` in one block;
both are read. Any other `rope_type` raises.
"""

from mlx_omnia.models.hunyuan_dense.checkpoint import CHECKPOINT
from mlx_omnia.models.hunyuan_dense.config import HunyuanDenseConfig
from mlx_omnia.models.hunyuan_dense.model import HunyuanDense

__all__ = [
    "CHECKPOINT",
    "HunyuanDense",
    "HunyuanDenseConfig",
]

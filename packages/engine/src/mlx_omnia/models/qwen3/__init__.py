"""Qwen3: Qwen2 with per-head q/k RMSNorm, no bias, explicit head_dim.

Authoritative semantics: transformers' modeling_qwen3.py. The norms sit between the
projections and the rotation, and `num_attention_heads * head_dim` is decoupled from
`hidden_size` (0.6B: 16x128 = 2048 against a 1024 trunk).

Three variants share the block and the router and differ only in what the checkpoint
declares: `dense` (`qwen3`), `moe` (`qwen3_moe`) and `vl` (`qwen3_vl`, `qwen3_vl_moe`,
text trunk only).

Two load-time fusions, both on the output axis (row-aligned in every representation,
so dense and packed u32 fuse the same way): q‖k‖v into `qkv_proj` and gate‖up into
`mlp.gate_up_proj` — concatenated on the dense trunk, interleaved row by row on the
sparse one for the decode kernel. The tree declares only the fused names.
"""

from mlx_omnia.models.qwen3 import dense, moe, vl
from mlx_omnia.models.qwen3.config import Qwen3Config, Qwen3MoEConfig
from mlx_omnia.models.qwen3.model import Qwen3, Qwen3MoE

__all__ = [
    "Qwen3",
    "Qwen3Config",
    "Qwen3MoE",
    "Qwen3MoEConfig",
    "dense",
    "moe",
    "vl",
]

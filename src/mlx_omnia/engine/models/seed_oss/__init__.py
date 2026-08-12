"""Seed-OSS: llama's block with an explicit `head_dim` and bias on the q/k/v projections
only.

Authoritative semantics: transformers' modeling_seed_oss.py.

`attention_bias` and `attention_out_bias` are two separate flags, and the published
36B sets the first true and the second false — the load-time qkv fusion carries the bias
vector on the output axis and `o_proj` stays bias-free. `head_dim` (128) is not
`hidden_size // heads` (5120 / 80 = 64), so it is read, never derived.
"""

from mlx_omnia.engine.models.seed_oss.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.seed_oss.config import SeedOssConfig
from mlx_omnia.engine.models.seed_oss.model import SeedOss, SeedOssActivations

__all__ = [
    "CHECKPOINT",
    "SeedOss",
    "SeedOssActivations",
    "SeedOssConfig",
]

"""Qwen2 dense: Qwen3 minus the per-head q/k RMSNorm and minus explicit head_dim,
plus a bias on q/k/v.

Authoritative semantics: transformers' modeling_qwen2.py. `head_dim` is not in the
config — it is `hidden_size // num_attention_heads` (0.5B: 896 // 14 = 64) — and the
three attention projections carry a bias (`o_proj` does not; the MLP does not). Qwen2
is the only architecture in the house with it.

The qkv load-time fusion is a concatenation on the output axis, and rows are the
fusion axis in every representation the checkpoint uses: dense weight, packed u32
plus scales/biases, and the projection bias vector (whose single axis *is* the output
axis). One fusion rule covers all four.
"""

from sideros.models.qwen2.checkpoint import CHECKPOINT
from sideros.models.qwen2.config import Qwen2Config
from sideros.models.qwen2.model import Qwen2, Qwen2Activations

__all__ = ["CHECKPOINT", "Qwen2", "Qwen2Activations", "Qwen2Config"]

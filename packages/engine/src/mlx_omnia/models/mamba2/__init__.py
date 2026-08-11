"""Mamba2 (SSD): a pure-SSM trunk with no attention.

Semantics follow transformers' `modeling_mamba2.py` (the naive `torch_forward` is
the reference), with the reference port's `ssm_kernel`/`ssm_attn` as the MLX mapping. Every
layer is the mixer alone — no MLP, no MoE — so the block is pre-norm + mixer +
residual. The SSD recurrence (`state = dA·state + dt·B·x; y = state·C + D·x`) is
the decode kernel at T=1 and the chunked surrogate-attention scan at prefill.

Property names are the checkpoint's after the loader normalizes the conv1d layout
(`[conv_dim, 1, kernel]` → `[conv_dim, kernel]`). `A_log`, `dt_bias`, `D` stay
float32 across a dtype cast (the decay saturates if they round-trip through bf16).
The gated RMSNorm folds the gate into the norm *input* (`rms_norm(silu(gate)·x)·w`),
not after — the opposite order from qwen3_5.
"""

from mlx_omnia.models.mamba2.checkpoint import CHECKPOINT
from mlx_omnia.models.mamba2.config import Mamba2Config
from mlx_omnia.models.mamba2.model import Mamba2, Mamba2Activations

__all__ = [
    "CHECKPOINT",
    "Mamba2",
    "Mamba2Activations",
    "Mamba2Config",
]

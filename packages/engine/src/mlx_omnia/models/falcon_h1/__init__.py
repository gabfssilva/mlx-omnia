"""Falcon-H1: a parallel within-layer hybrid of Mamba2/SSD and full causal GQA,
with μP multipliers folded at load and a per-layer composite cache (recurrent
state + KV).

Semantics follow transformers' ``modeling_falcon_h1.py``: the gated
RMSNorm groups the variance over ``dim // n_groups`` (the reference implementation
calls bare ``mx.fast.rms_norm`` with no grouping, diverging for ``n_groups > 1`` —
so its 34B output is not a valid reference; only the 7B with ``n_groups=1``
matches). The SSM recurrence (``A_log``, ``d_state=256``, selective scan) runs
in float32; ``A_log`` never leaves it at any weight precision.

Property names are the checkpoint's after the loader folds the nine μP
multipliers and the per-row ``ssm_multipliers`` vector into the weights and
transposes the conv1d from torch's ``[conv_dim, 1, kernel]`` to ``[conv_dim, 1,
kernel] → [conv_dim, kernel, 1] → squeeze → [conv_dim, kernel]`` (the house
layout, matching ``nn.Conv1d``). q/k/v are fused on the output axis, gate‖up
concatenated, both dict-side at load.
"""

from mlx_omnia.models.falcon_h1.checkpoint import CHECKPOINT
from mlx_omnia.models.falcon_h1.config import FalconH1Config
from mlx_omnia.models.falcon_h1.model import FalconH1, FalconH1Activations

__all__ = [
    "CHECKPOINT",
    "FalconH1",
    "FalconH1Activations",
    "FalconH1Config",
]

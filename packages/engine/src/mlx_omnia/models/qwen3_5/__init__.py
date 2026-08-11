"""Qwen3.5/3.6 text: a hybrid decoder where three of every four layers are a gated
DeltaNet and the fourth is GQA with an output gate.

Semantics follow transformers' `modeling_qwen3_5.py`: the DeltaNet's
l2norm carries its eps *inside the sum* (rms_norm x scale would give an effective
eps 128x larger). The decay runs in float32 (`A_log` never leaves it) and so does the
recurrent state.

Property names are the checkpoint's after the loader normalizes the two dialects
(raw HF `model.language_model.*` vs mlx `language_model.model.*`); the projections of
each mixer are concatenated on the output axis into `fused_proj`, dict-side.
"""

from mlx_omnia.models.qwen3_5.checkpoint import CHECKPOINT
from mlx_omnia.models.qwen3_5.config import Qwen35Config, Qwen35RoPEParameters, Qwen35TextConfig
from mlx_omnia.models.qwen3_5.layers.moe import Qwen35MoE
from mlx_omnia.models.qwen3_5.model import (
    MultimodalPrompt,
    Qwen35,
    Qwen35Activations,
    Qwen35Input,
    Qwen35LanguageModel,
    decode_clock,
    multimodal_prompt,
    stream_multimodal_ids,
)

__all__ = [
    "CHECKPOINT",
    "MultimodalPrompt",
    "Qwen35",
    "Qwen35Activations",
    "Qwen35Config",
    "Qwen35Input",
    "Qwen35LanguageModel",
    "Qwen35MoE",
    "Qwen35RoPEParameters",
    "Qwen35TextConfig",
    "decode_clock",
    "multimodal_prompt",
    "stream_multimodal_ids",
]

"""SmolLM3: Llama with the rotation removed from every fourth layer (NoPE).

Authoritative semantics: transformers' modeling_smollm3.py.

The tree is the house's dense decoder (`core/attention.py`), leaf for leaf — the delta is
one bit per layer. `no_rope_layers` ships in the checkpoint as ints (1 = rotate,
0 = NoPE); when absent it is derived from `no_rope_layer_interval`, and the two agree
on SmolLM3-3B (interval 4 over 36 layers).
A NoPE layer attends on content alone, so nothing about the cache or the mask changes.
"""

from mlx_omnia.models.smollm3.checkpoint import CHECKPOINT
from mlx_omnia.models.smollm3.config import SmolLM3Config, SmolLM3RoPEScalingConfig
from mlx_omnia.models.smollm3.model import SmolLM3

__all__ = [
    "CHECKPOINT",
    "SmolLM3",
    "SmolLM3Config",
    "SmolLM3RoPEScalingConfig",
]

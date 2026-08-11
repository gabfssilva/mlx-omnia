"""LFM2: a hybrid trunk of gated short convs and GQA, in a dense and a sparse variant.

Both variants share the mixers in `layers/` — the checkpoint's leaf names are identical —
and differ only in the feed-forward: `dense` is a plain SwiGLU in every block, `moe`
routes 4 of 32 experts past the first two dense layers.
"""

from sideros.models.lfm2 import dense, moe
from sideros.models.lfm2.config import LFM2Config, LFM2MoEConfig, LFM2RoPEParameters
from sideros.models.lfm2.dense import LFM2
from sideros.models.lfm2.moe import LFM2MoE

__all__ = [
    "LFM2",
    "LFM2Config",
    "LFM2MoE",
    "LFM2MoEConfig",
    "LFM2RoPEParameters",
    "dense",
    "moe",
]

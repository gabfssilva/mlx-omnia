"""EXAONE 4: post-norm blocks, per-head q/k norm, and a sliding/global pattern where the
global layers carry no position at all.

Authoritative semantics: transformers' modeling_exaone4.py.

- **Post-norm on both joins**, and no `input_layernorm`: attention reads the raw
  residual, `post_attention_layernorm` normalizes its output, and the MLP reads the join
  unnormalized. Same shape OLMo 2 uses.
- **`sliding_window_pattern`** is a string like `"LLLG"`: `L` layers slide within
  `sliding_window` *and* rotate; `G` layers see everything and are **NoPE** — no
  rotation at all, which is what carries long-range position once the local layers have
  encoded the short range. With no pattern every layer is global *and* rotates.
- The window lives in the mask, never in the cache, the rule `gemma3` set: a masked key
  contributes nothing, and a trimmable cache stays trimmable.

`rope_scaling` accepts the llama3 table (the 32B ships it) and nothing else.
"""

from sideros.models.exaone4.checkpoint import CHECKPOINT
from sideros.models.exaone4.config import Exaone4Config
from sideros.models.exaone4.model import Exaone4

__all__ = [
    "CHECKPOINT",
    "Exaone4",
    "Exaone4Config",
]

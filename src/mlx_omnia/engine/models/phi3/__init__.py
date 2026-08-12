"""Phi-3: llama's block over a checkpoint that already ships qkv and gate‖up fused, plus
LongRoPE.

Authoritative semantics: transformers' modeling_phi3.py.

- **Nothing fuses at load.** `qkv_proj` and `gate_up_proj` are the checkpoint's own
  names, in the output-axis order the tree already declares — the one ported family
  where the layout arrives correct.
- **`partial_rotary_factor`** (1.0 on the published Phi-3, <1 on Phi-4 derivatives)
  bounds how much of the head rotates.
- **LongRoPE** (`rope_scaling.type` in `longrope`/`su`): the per-dimension `long_factor`
  multiplies the period table, and the queries and keys are pre-scaled by
  `sqrt(1 + log(f) / log(original_max_position_embeddings))` before the rotation, with
  `f = max_position_embeddings / original_max_position_embeddings`. `short_factor` is
  never used: it applies below `original_max_position_embeddings`, and switching tables
  mid-stream would rotate the cache and the new key by different frequencies. The
  reference implementation made the same choice; the long table is the one that
  matches a full-length context.

`linear` scaling is `1 / factor` on `mx.fast.rope`'s `scale`. Any other `rope_scaling`
raises rather than silently degrading to the unscaled table.
"""

from mlx_omnia.engine.models.phi3.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.phi3.config import Phi3Config
from mlx_omnia.engine.models.phi3.model import Phi3

__all__ = ["CHECKPOINT", "Phi3", "Phi3Config"]

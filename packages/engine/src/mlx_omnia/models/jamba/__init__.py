"""Jamba: alternating Mamba-**1** and attention layers, each followed by an MLP that is
dense or routed on its own period.

Authoritative semantics: transformers' modeling_jamba.py.

Two periods, independent of each other:

- `layers_block_type`, or `i % attn_layer_period == attn_layer_offset`, decides mamba
  against attention;
- `(i + expert_layer_offset) % expert_layer_period == 0` decides routed against dense for
  that layer's feed-forward.

So a layer is always `x + mixer(input_layernorm(x))` then `h + feed_forward(pre_ff_layernorm(h))`
— two joins, whatever the mixer and whatever the MLP.

**The mixer is Mamba-1, not Mamba-2**, and nothing in `mamba2.py` applies to it:

- `in_proj` splits into `x` and a gate `z` of `intermediate_size` each — no `dt` column;
- the time step and the B/C selectors come from `x_proj` *after* the conv, each with its
  own RMSNorm (`dt_layernorm`, `b_layernorm`, `c_layernorm`) — a Jamba addition that
  vanilla Mamba does not have — and `dt` then goes through `dt_proj` and softplus;
- `A_log` is `[intermediate_size, state_size]`, one decay per channel *and* state, where
  Mamba-2 has one per head;
- the recurrence is a plain sequential scan (`state = dA·state + dt·B·x`, `y = state·C +
  D·x`), no chunked surrogate attention. The prefill cost is linear in the sequence with
  a Python-level loop, which is the honest shape of Mamba-1 and the reference's own.

Attention carries no RoPE at all: position comes from the mamba layers.
"""

from mlx_omnia.models.jamba.checkpoint import CHECKPOINT
from mlx_omnia.models.jamba.config import JambaConfig
from mlx_omnia.models.jamba.model import Jamba

__all__ = [
    "CHECKPOINT",
    "Jamba",
    "JambaConfig",
]

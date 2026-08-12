"""DeepSeek-V2: multi-head latent attention (MLA) and a group-limited softmax router.

Authoritative semantics: transformers' modeling_deepseek_v2.py.

**MLA is the reason this is not another llama.** Attention does not project q/k/v to
three same-shaped tensors:

- the query goes through a low-rank pair (`q_a_proj` → RMSNorm → `q_b_proj`) when
  `q_lora_rank` is set, and a single `q_proj` otherwise;
- the key/value side projects to *one* compressed latent of `kv_lora_rank + qk_rope_head_dim`
  columns. The first part is normed and expanded by `kv_b_proj` into the per-head
  no-position key and the value; the second is a **single** rotary key shared by every
  head, broadcast across them;
- each head's key is `[k_nope ‖ k_pe]`, `qk_nope_head_dim + qk_rope_head_dim` wide, while
  its value is `v_head_dim` wide — the two sides of the cache have different last
  dimensions, which no other ported model does.

This is the decompressed form: the cache holds the expanded per-head keys and values, so
it is the same `KVCache` every other model uses and costs the same bandwidth. The
absorbed form (caching the latent instead) is a different memory profile and is not what
the reference computes.

**YaRN** supplies the rotary table (NTK-by-parts interpolation) and, through
`mscale_all_dim`, a correction that multiplies the *attention scale* by `mscale²` on top
of pre-scaling q/k by `mscale` — two separate uses of the same quantity.

The router is the pre-V3 shape: **softmax** (not sigmoid), no correction bias, no
renormalization after the cut, group limiting by each group's **max** (V3 uses the sum of
its two best), then `routed_scaling_factor`. Shared experts are added ungated.
"""

from mlx_omnia.engine.models.deepseek_v2.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.deepseek_v2.config import DeepseekV2Config
from mlx_omnia.engine.models.deepseek_v2.model import DeepseekV2

__all__ = [
    "CHECKPOINT",
    "DeepseekV2",
    "DeepseekV2Config",
]

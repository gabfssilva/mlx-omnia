"""Qwen3-Next: three gated-DeltaNet layers per full-attention layer, zero-centred norms,
and a routed MLP with a gated shared expert.

Authoritative semantics: transformers' modeling_qwen3_next.py.

This is the backbone `qwen3_5` descends from, and the delta rule is the same one — so
`delta_rule`, `l2norm` and the `gated_delta` kernel are reused rather than restated.
What Qwen3-Next does differently:

- **Two projections, not one.** `in_proj_qkvz` and `in_proj_ba` are separate leaves, and
  `in_proj_qkvz` is laid out **per key head**: reshaped to `[…, num_k_heads, -1]` it
  splits into q, k, v, z, where each key head carries `Hv/Hk` value heads' worth of v and
  z. Qwen3.5 flattened that into one `fused_proj`; here the reordering happens in the
  forward, because the checkpoint's two matrices are the contract.
- **Gated attention output.** `q_proj` is twice as wide: half is the query, half is a
  gate whose `sigmoid` multiplies the attention output before `o_proj`. That is why the
  qkv fusion does not apply.
- **Zero-centred norms** (`scale = 1 + w`) on `input_layernorm`,
  `post_attention_layernorm`, `model.norm`, `q_norm` and `k_norm` — folded on the dict
  side at load, the way `gemma3` does. The DeltaNet's gated norm is *not* zero-centred.
- **`full_attention_interval`** (4): layer `i` attends when `(i + 1) % interval == 0`,
  and is a DeltaNet otherwise.
- **`decoder_sparse_step` and `mlp_only_layers`** decide which layers route; the shared
  expert is gated by `sigmoid(shared_expert_gate(x))`, as in Qwen2-MoE.

MTP leaves are dropped at load.
"""

from mlx_omnia.engine.models.qwen3_next.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.qwen3_next.config import Qwen3NextConfig
from mlx_omnia.engine.models.qwen3_next.model import Qwen3Next, Qwen3NextActivations

__all__ = [
    "CHECKPOINT",
    "Qwen3Next",
    "Qwen3NextActivations",
    "Qwen3NextConfig",
]

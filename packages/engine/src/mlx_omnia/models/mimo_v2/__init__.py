"""MiMo-V2-Flash: sliding layers with attention sinks and twice the kv heads, a value
head narrower than the query head, and a DeepSeek-V3 router with no shared expert.

Authoritative semantics: transformers' modular_mimo_v2_flash.py; the reference is
transformers alone.

What has no counterpart in the other ports:

- **q/k and v have different widths.** `head_dim` is 192, `v_head_dim` is 128, and
  `o_proj` reads `heads · v_head_dim`. That is why q/k/v stay three leaves here: the
  output-axis fusion assumes one head width for all three.
- **The sliding layers carry twice the kv heads** of the full-attention ones, so the two
  layer kinds do not even share a projection shape.
- **Attention sinks on the sliding layers only** — one learned logit per head inside the
  softmax denominator, which `mx.fast.scaled_dot_product_attention` takes natively
  (`sinks=`), the same path `gpt_oss` uses.
- **`attention_value_scale`** (0.707) multiplies the values after the projection.
- **Per-layer-type RoPE**: `rope_parameters` is a dict keyed by `full_attention` /
  `sliding_attention`, each with its own `rope_theta` (5e6 and 1e4) and
  `partial_rotary_factor` (0.334). Two layer kinds, two rotations.
- **`mlp_layer_types`** decides dense against sparse per layer, defaulting to dense only
  on layer 0.

The router is `glm4_moe`'s `noaux_tc` with `n_shared_experts` removed: sigmoid in
float32, `e_score_correction_bias` on the selector, group limiting, then `norm_topk_prob`
and `routed_scaling_factor`.
"""

from mlx_omnia.models.mimo_v2.checkpoint import CHECKPOINT
from mlx_omnia.models.mimo_v2.config import MimoV2Config, MimoV2RoPE, MimoV2RoPEParameters
from mlx_omnia.models.mimo_v2.model import MimoV2, MimoV2Activations

__all__ = [
    "CHECKPOINT",
    "MimoV2",
    "MimoV2Activations",
    "MimoV2Config",
    "MimoV2RoPE",
    "MimoV2RoPEParameters",
]

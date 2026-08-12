"""Bailing MoE (Ling / Ring): the GLM-4-MoE router under a different set of leaf names.

Authoritative semantics: transformers' modeling_bailing_moe.py.

The arithmetic is close to `glm4_moe` — dense first layers, group-limited selection with
an optional `expert_bias` correction, weights from the uncorrected scores, ungated shared
expert. What differs is almost entirely naming, and naming is the contract here:

| this checkpoint | the usual name |
| --- | --- |
| `model.word_embeddings` | `model.embed_tokens` |
| `attention` | `self_attn` |
| `attention.query_key_value` | `self_attn.qkv_proj` (already fused in the file) |
| `attention.dense` | `self_attn.o_proj` |
| `query_layernorm` / `key_layernorm` | `q_norm` / `k_norm` |
| `mlp.gate.weight` + `mlp.gate.expert_bias` | the router |

Two flags with numerical weight:

- **`score_function`** picks sigmoid or softmax before the correction bias; under
  `norm_topk_prob` the kept weights are divided by their sum (with a `1e-20` floor) and
  then scaled by `routed_scaling_factor`.
- **`norm_head`** L2-normalizes the lm_head columns. It is a property of the weights, not
  of the forward pass, so it is applied once on the dict side at load.
"""

from mlx_omnia.engine.models.bailing_moe.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.bailing_moe.config import BailingMoEConfig
from mlx_omnia.engine.models.bailing_moe.model import BailingMoE

__all__ = ["CHECKPOINT", "BailingMoE", "BailingMoEConfig"]

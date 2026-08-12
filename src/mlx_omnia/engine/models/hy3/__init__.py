"""Hunyuan 3 (Hy3, `hy_v3`): a 80-layer sigmoid-MoE trunk with Qwen3-style attention.

Authoritative semantics: transformers' `modeling_hy_v3.py`. What the trunk does:

- **Dense layer 0, sparse layers 1-79** (`mlp_layer_types = ["dense"] + ["sparse"]*79`).
  Layer 0 is a `SwiGLU(4096, 13312)`; layers 1-79 are sigmoid-routed MoE with 192
  experts, 8 per token, plus 1 shared expert (width 1536 = `moe_intermediate_size`).
- **Sigmoid routing** (DeepSeek-V3 family, same as Laguna): `sigmoid(logits)` →
  `+ e_score_correction_bias` (selection only) → `topk(k=8)` → weights from the
  **unbiased** sigmoid scores → renorm → `x router_scaling_factor (2.826)`. The router
  gemv runs in **fp32** (`F.linear(x.float(), w.float())` in transformers), and
  `e_score_correction_bias` is fp32 — both kept fp32 through the dtype cast.
- **Shared expert added raw** (`routed + shared`), no gating. The checkpoint sets
  `enable_moe_fp32_combine: false`, so the sum runs in the model dtype.
- **Attention**: GQA (64/8 heads, head_dim 128), qk-norm **before** RoPE, default RoPE
  (theta 11_158_840) on full head_dim. No sliding window, no sinks. This is Qwen3
  attention.
- **MTP head** (layer 80) is dropped by transformers (`_keys_to_ignore_on_load_unexpected
  = [r"model\\.layers\\.80.*"]`). Inference is 1-token/step AR.

qkv is fused on the output axis, gate‖up row-interleaved for the decode kernel, experts
stacked `[E, out, in]` via `SwitchLinear` — all at load, dict-side. The router weight
(`mlp.gate.weight`) and `e_score_correction_bias` stay fp32 across the dtype cast.
"""

from mlx_omnia.engine.models.hy3.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.hy3.config import Hy3Config, Hy3RoPEParameters
from mlx_omnia.engine.models.hy3.model import Hy3, Hy3Activations

__all__ = [
    "CHECKPOINT",
    "Hy3",
    "Hy3Activations",
    "Hy3Config",
    "Hy3RoPEParameters",
]

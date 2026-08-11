"""OLMoE: OLMo's flat q/k norm with 64 routed experts, 8 per token, on every layer.

Authoritative semantics: transformers' modeling_olmoe.py.

Two deltas against `qwen3_moe`:

- **q/k norm spans `heads · head_dim`**, not each head — the same flat reduction
  `olmo2` uses, and the reason `rope_epilogue` does not apply here either.
- **The block is pre-norm** (`input_layernorm` / `post_attention_layernorm`), unlike
  OLMo 2's post-norm: the two OLMo generations disagree on where the norm sits, so the
  tree is not shared.

`norm_topk_prob` defaults false: the top-k softmax weights are the raw probabilities.
The routed experts are `qwen3_moe`'s `SwitchGLU`.
"""

from sideros.models.olmoe.checkpoint import CHECKPOINT
from sideros.models.olmoe.config import OlmoEConfig
from sideros.models.olmoe.model import OlmoE, OlmoEActivations

__all__ = ["CHECKPOINT", "OlmoE", "OlmoEActivations", "OlmoEConfig"]

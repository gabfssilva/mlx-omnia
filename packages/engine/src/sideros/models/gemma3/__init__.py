"""Gemma 3 text (`gemma3_text`): interleaved sliding/full attention, sandwich norms.

Authoritative semantics: transformers' modeling_gemma3.py. What the trunk does that the
Qwen family does not:

- `layer_types` picks, per layer, a sliding-window mask + `rope_local_base_freq` or a
  full causal mask + `rope_theta`. The cache is never evicted; the window lives in the
  mask (a masked key contributes nothing).
- The attention scale is `query_pre_attn_scalar ** -0.5`, which only coincides with
  `head_dim ** -0.5` on this checkpoint. `num_attention_heads * head_dim` (1024) is
  decoupled from `hidden_size` (640).
- Sandwich norms: each residual arm is normed on the way in *and* on the way out.
- Zero-centered RMSNorm: the scale is `1 + w`. Folded on the dict side at load
  (`fold_norm_scales`), so the tree holds plain `nn.RMSNorm` and no `1 + w` add runs
  per norm per token — 108 extra kernels per step on an 18-layer trunk.
- Embeddings scaled by `sqrt(hidden_size)`; transformers keeps that scalar in float32
  and casts it to the weight dtype, so bf16 sees 25.25 and fp32 25.298221 — the cast is
  reproduced here, not the rounded constant.
- lm_head tied to the embedding table; gelu (tanh approximation) MLP.
"""

from sideros.models.gemma3.checkpoint import CHECKPOINT
from sideros.models.gemma3.config import Gemma3TextConfig
from sideros.models.gemma3.model import Gemma3, Gemma3Activations
from sideros.models.gemma3.tokenizer import Gemma3Tokenizer

__all__ = [
    "CHECKPOINT",
    "Gemma3",
    "Gemma3Activations",
    "Gemma3TextConfig",
    "Gemma3Tokenizer",
]

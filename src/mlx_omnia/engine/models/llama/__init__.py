"""Llama 2/3.x dense: Qwen3 minus the per-head q/k RMSNorm, plus llama3 rope scaling.

Authoritative semantics: transformers' modeling_llama.py. The block is the house's
standard one — norm → attention → add → norm → SwiGLU → add — with plain `nn.RMSNorm`
(not Gemma's `1 + w`) and no bias anywhere.

Three things the dictionary does not already cover:

- **`head_dim` is optional in the config.** Llama-3.1-8B and Llama-2-7b omit the key;
  transformers falls back to `hidden_size // num_attention_heads`. The mirror keeps the
  key optional and resolves the fallback where the dense tree's config is derived.
- **`rope_scaling` of type `llama3`** — the house's fourth formula, after default,
  mrope and YaRN. A static `[head_dim/2]` fp32 table in the *period* convention
  `mx.fast.rope(freqs=)` consumes, so it multiplies where transformers divides. There
  is no `mscale`: unlike gpt-oss's YaRN, q/k are not pre-scaled.
- **Llama-2 serializes `self_attn.rotary_emb.inv_freq`** — one dead `[head_dim/2]`
  tensor per layer, which transformers discards. It leaves the dict before `update`,
  or the totality contract reports 32 unexpected names.

`model_type: "llama"` is also what `Llama-3.1-Nemotron-Nano`, `DeepSeek-R1-Distill-Llama`
and `SmolLM`/`SmolLM2` declare; `smollm3` and `nemotron` are their own entries.
"""

from mlx_omnia.engine.models.llama.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.llama.config import LlamaConfig
from mlx_omnia.engine.models.llama.model import Llama

__all__ = [
    "CHECKPOINT",
    "Llama",
    "LlamaConfig",
]

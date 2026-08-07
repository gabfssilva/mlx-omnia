"""Longcat Flash Lite (ngram): MLA attention, dual-sublayer + shortcut-MoE trunk,
n-gram-augmented input embeddings.

Property names are the checkpoint's (HF ``meituan-longcat/LongCat-Flash-Lite``).
Seven things no other Sideros model has:

- **MLA** (Multi-head Latent Attention): low-rank q/kv projections, a latent KV
  cache (stores the compressed ``kv_lora_rank`` + decoupled ``k_pe``, not full
  K/V), and a load-time ``kv_b_proj`` split into ``embed_q`` (latent → nope key)
  and ``unembed_out`` (latent → value). Decode runs attention in latent space;
  prefill expands the latent to per-head K/V.
- **Decoupled interleaved RoPE** (``traditional=True``): only the rope head dim
  (64) is rotated; the nope head dim (128) is not. The existing ``rope_epilogue``
  kernel is rotate-half only; here ``mx.fast.rope(traditional=True)`` is used.
- **YaRN** with ``mscale_all_dim=1``: the NTK-by-parts table and the mscale that
  pre-scales the attention ``scale`` (not q/k), same op order as ``gpt_oss``.
- **Dual-sublayer + shortcut-MoE**: each logical layer has 2 attentions, 2 dense
  MLPs, and 1 shared MoE. The MoE output (the "shortcut") is computed at
  sublayer 0 and added at the end of sublayer 1.
- **Identity experts**: the router selects from 384 experts (256 routed + 128
  identity). Identity experts are pass-through: their index is clamped to 0 for
  the matmul, their weight is zeroed, and ``input x identity_weight_sum`` is
  added after the expert sum.
- **softmax_bias_topk router**: softmax, add ``e_score_correction_bias`` to the
  scores (not logits), top-k on biased scores, weights from raw scores x
  ``routed_scaling_factor``, no renorm. A dedicated kernel in
  ``core/kernels/softmax_bias_topk.py``.
- **NgramEmbedding**: 12 rolling-hash n-gram tables + 12 projections that augment
  the word embedding. The rolling hash uses int64 modular exponentiation; the
  shift is EOS-aware (resets at every EOS boundary, matching transformers, not
  a simpler zero-pad). ``NgramCache`` carries the last ``n-1 = 3`` ids.

Load fusions (dict-side, before ``update``): stack per-expert ``gate``/``up``/
``down`` into ``switch_mlp`` (interleaved gate‖up), split each sublayer's
``kv_b_proj`` into ``embed_q`` + ``unembed_out`` (dequant if needed, left dense),
rename ``embed_tokens`` → ``ngram_embeddings.word_embeddings``, drop ``mtp.*``.
"""

from sideros.models.longcat_flash_ngram.checkpoint import CHECKPOINT
from sideros.models.longcat_flash_ngram.config import (
    LongcatFlashNgramConfig,
    LongcatFlashNgramRopeScaling,
)
from sideros.models.longcat_flash_ngram.model import (
    LongcatFlashNgram,
    LongcatFlashNgramActivations,
)

__all__ = [
    "CHECKPOINT",
    "LongcatFlashNgram",
    "LongcatFlashNgramActivations",
    "LongcatFlashNgramConfig",
    "LongcatFlashNgramRopeScaling",
]

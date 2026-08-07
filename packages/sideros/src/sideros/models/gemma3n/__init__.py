"""Gemma 3n text: four parallel residual streams (AltUp), a low-rank residual arm
(LAuReL), per-layer embeddings, KV sharing and top-k sparse activations.

Authoritative semantics: transformers' modeling_gemma3n.py.

**The vision and audio towers are not ported** — `vision_tower`, `audio_tower`,
`embed_vision`, `embed_audio` are dropped at load. This is the text trunk of a
multimodal model; it answers text and nothing else.

Four things that exist nowhere else in this repo:

- **AltUp.** The trunk does not carry one hidden state but `altup_num_inputs` (4) of
  them, stacked on a leading axis. Each block *predicts* all four from the active one
  through a `[4, 4]` coefficient matrix routed by `tanh(modality_router(...))`, runs
  attention and the MLP on the active stream only, then *corrects* all four by the
  innovation the block produced. The projections in and out of the stack renormalize the
  extra streams to the active one's magnitude. The coefficient matrices are clipped to
  `altup_coef_clip` and read in float32 — a clip on the *weights*, applied per call in
  the reference, so it is applied once at load here instead.
- **LAuReL**: `x + post_laurel_norm(linear_right(linear_left(x)))`, a second low-rank arm
  joined into the attention residual and then halved with it (`· 2^-0.5`).
- **Top-k sparse activation.** Where `activation_sparsity_pattern[i] > 0`, the MLP gate
  is thresholded at `mean + std · sqrt(2)·erfinv(2s - 1)` before the gelu — a per-token
  cutoff, not a fixed one.
- **`scale = 1.0`.** Attention is *not* divided by `sqrt(head_dim)`; the q/k norms carry
  that. `v_norm` is an RMSNorm with no learned scale.

PLE and KV sharing are the pieces `gemma4` already holds: the per-layer embedding table
and `SharedKVReader` are reused from there. Which layer a shared layer reads is Gemma
3n's own rule — the **last** concrete layer of the same type before the shared range.
"""

from sideros.models.gemma3n.checkpoint import CHECKPOINT
from sideros.models.gemma3n.config import Gemma3nConfig, Gemma3nTextConfig
from sideros.models.gemma3n.model import Gemma3n, Gemma3nActivations

__all__ = [
    "CHECKPOINT",
    "Gemma3n",
    "Gemma3nActivations",
    "Gemma3nConfig",
    "Gemma3nTextConfig",
]

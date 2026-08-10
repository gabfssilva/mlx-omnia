"""Muse-Glimmer: dense 30B VLM — vision tower included, video frontend not ported.
Gemma2 sandwich norms, AfMoE-style sigmoid-gated attention, sliding layers that rotate
and NoPE full-attention layers; a windowed bidirectional ViT feeding the trunk through
a gelu adapter and a weightless RMS norm.

Authoritative semantics: transformers' modeling_muse_glimmer.py (git main 2026-08-10;
snapshot in `reference/muse_glimmer/`).

- **One scaleless QK norm.** Q and K share a single weightless RMSNorm per head, then
  `q *= qk_scale_factor` — folded here into the SDPA scale (rope commutes with it).
- **The embedding is normed.** A weightless RMSNorm over `embed_tokens`, in place of
  the Gemma `sqrt(hidden)` scaling. The head is untied.
- **Two norm conventions in one trunk.** The four block norms are centred (`1 + w`,
  folded at load); `model.norm` is plain-scale and does not fold.
- **The head is scaled then softcapped**: `20 * tanh(logits * output_multiplier / 20)`.
- The tower reorders patches into 32x32 windows internally and undoes it; the 2x2 merge
  is a channel-major pixel shuffle after `ln_post`. Image features replace `<|patch|>`
  rows after the text embed norm, carrying their own (`perception_emb_norm`).
"""

from sideros.models.muse_glimmer.checkpoint import ASSISTANT, CHECKPOINT, load_assistant
from sideros.models.muse_glimmer.config import (
    MuseGlimmerAssistantConfig,
    MuseGlimmerConfig,
    MuseGlimmerTextConfig,
)
from sideros.models.muse_glimmer.dflash import MuseGlimmerAssistant
from sideros.models.muse_glimmer.model import MuseGlimmer

__all__ = [
    "ASSISTANT",
    "CHECKPOINT",
    "MuseGlimmer",
    "MuseGlimmerAssistant",
    "MuseGlimmerAssistantConfig",
    "MuseGlimmerConfig",
    "MuseGlimmerTextConfig",
    "load_assistant",
]

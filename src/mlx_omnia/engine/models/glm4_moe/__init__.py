"""GLM-4.5/4.6 MoE: dense first layers, then group-limited sigmoid routing with a
bias-corrected selector and an ungated shared expert.

Authoritative semantics: transformers' modeling_glm4_moe.py. The recon that
preceded this port is in `docs/models/glm4_moe.md`'s Recon section.

- **`first_k_dense_replace`** layers keep a plain SwiGLU; every later layer routes.
- **The router is DeepSeek-V3's `noaux_tc`**, and it is the only part with real
  structure: score with `sigmoid` in float32, add `e_score_correction_bias`, fold the
  experts into `n_group` groups, keep the `topk_group` groups whose two best experts sum
  highest (zeroing the rest), then take the global top-k of what survives. The **weights
  come from the uncorrected scores** — the bias moves the selection, never the mixing —
  and are renormalized under `norm_topk_prob` and scaled by `routed_scaling_factor`.
- **`use_qk_norm`** is per head, and only some checkpoints carry it, so the leaves are
  declared only when the config asks; `update(strict=True)` then holds either way.
- **Partial rotary** with `traditional=False` (the dense `glm4` uses `traditional=True`
  — the two GLM-4 backbones disagree on the pairing).
- The shared expert is added ungated, with width `moe_intermediate_size ·
  n_shared_experts`.

MTP leaves (`model.layers.{num_hidden_layers}.*`, the extra block GLM ships for
speculative decoding) are dropped at load: this port decodes one token per step.

`dsa/` is the sibling variant: the same router over DeepSeek-V3.2's sparse attention.
"""

from mlx_omnia.engine.models.glm4_moe.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.glm4_moe.config import Glm4MoEConfig
from mlx_omnia.engine.models.glm4_moe.model import Glm4MoE

__all__ = [
    "CHECKPOINT",
    "Glm4MoE",
    "Glm4MoEConfig",
]

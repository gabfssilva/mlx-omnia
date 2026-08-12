"""Step 3.7 Flash (`step3p7`): a 198B VLM with a step3p5 LM trunk, a
perception_encoder ViT, and a multi-tile sliding-window image processor.

Six per-layer features the trunk has that no previous ported model has all at once:

- **Per-layer alternating ``rope_theta``** (5e6 on the 12 full-attention layers,
  1e4 on the 33 sliding) and **per-layer ``partial_rotary``** (0.5 on full, 1.0 on
  sliding): each layer builds its own freqs table and rotates a different fraction of
  ``head_dim``.
- **MFA per-layer head count**: full-attention layers project q to 64·128; sliding
  layers to 96·128 via ``attention_other_setting``. KV is 8·128 for both.
- **Head-wise sigmoid gate** (``g_proj``): ``sigmoid(g_proj(x))`` per-head, multiplied
  across ``head_dim`` on the attention output before ``o_proj``.
- **Per-layer SwiGLU clamp** (``swiglu_limits``): gate upper-only, up symmetric, before
  ``silu(gate)·up``. Only layers 43-44 clamp (limit 7); the shared expert clamps at
  16 on the same layers.
- **+1 RMSNorm**: ``normed = normed * (weight + 1)``. The loader adds 1 to every trunk
  norm weight (``model.norm``, ``input_layernorm``, ``post_attention_layernorm``,
  ``q_norm``, ``k_norm``).
- **Sigmoid-routed 288-expert MoE with shared expert**: selection by
  ``sigmoid(logits) + router_bias`` in fp32 (``need_fp32_gate``), weights from the
  bias-free sigmoid scores, renormalized (``norm_expert_weight``) and scaled by 3.0.
  The shared expert is a dense MLP added to the routed sum on every MoE layer (3-44).

Layers 0-2 are dense SwiGLU MLP (``intermediate_size`` 11264); 3-44 are MoE
(``moe_intermediate_size`` 1280, ``share_expert_dim`` 1280). ``layer_types`` is a
length-48 array (1 full / 3 sliding repeating); only the first 45 are trunk layers.

The vision tower and processor live in ``step3p7/vision.py`` and
``processors/step3p7.py``. Image features are scattered into ``input_ids ==
image_token_id`` positions (``masked_scatter``); the trunk runs 1-D RoPE with a single
position clock — no MRoPE.

Load-time fusions (5): qkv concat, dense gate‖up concat, MoE expert gate‖up
row-interleave, shared expert gate‖up concat, +1 on every RMSNorm weight. No new Metal
kernel: ``sigmoid_topk`` + the affine gate-up/down-combine kernels cover the decode
path; layers 43-44 fall back
to eager (the clamp is not in the kernel).
"""

from mlx_omnia.engine.models.step3p7.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.step3p7.config import Step3p7Config, Step3p7TextConfig
from mlx_omnia.engine.models.step3p7.layers.moe import Step3p7MoE
from mlx_omnia.engine.models.step3p7.model import Step3p7

__all__ = [
    "CHECKPOINT",
    "Step3p7",
    "Step3p7Config",
    "Step3p7MoE",
    "Step3p7TextConfig",
]

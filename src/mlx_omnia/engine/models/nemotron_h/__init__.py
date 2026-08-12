"""Nemotron-H: a four-kind hybrid trunk — Mamba2, attention, ReLU² MLP, and MoE — laid
out by a per-layer pattern string.

Authoritative semantics: transformers' modeling_nemotron_h.py.

- **`hybrid_override_pattern`** gives one character per layer: `M` mamba, `*` attention,
  `-` MLP, `E` MoE. `layers_block_type` is the same list spelled out; either is read.
  Unlike every other hybrid here, an `-`/`E` layer has *no mixer state at all* — its
  cache slot is a bare `LayerCache` that only counts the offset.
- **One norm per layer, one join.** A layer is `x + mixer(norm(x))`, whatever the mixer
  is; there is no second norm and no MLP after attention. The MLP layers are their own
  layers.
- **ReLU² MLP** (`relu(x)²`), gate-free: `up_proj` then `down_proj`, so no gate‖up
  fusion.
- **Grouped gated RMSNorm** in the mamba mixer: the norm runs per group of
  `intermediate_size / n_groups` channels, not over the whole width — a different
  reduction from `mamba2`'s, which is why that mixer is not reused here.
- **The MoE may project to a latent** first (`moe_latent_size`): `fc1_latent_proj` down,
  experts, `fc2_latent_proj` back up. The shared expert reads the *unprojected* input.
- The trunk is `backbone` with `embeddings` and `norm_f`, not `model`/`embed_tokens`/
  `norm`, and the head is never tied.

MTP leaves (`mtp.*`) are dropped at load.
"""

from mlx_omnia.engine.models.nemotron_h.checkpoint import CHECKPOINT, MTP, load_mtp
from mlx_omnia.engine.models.nemotron_h.config import NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.model import NemotronH
from mlx_omnia.engine.models.nemotron_h.mtp import NemotronHMTP

__all__ = [
    "CHECKPOINT",
    "MTP",
    "NemotronH",
    "NemotronHConfig",
    "NemotronHMTP",
    "load_mtp",
]

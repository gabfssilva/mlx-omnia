"""AFMoE: sandwich norms, sigmoid-gated attention output, sliding layers that rotate and
global layers that do not.

Authoritative semantics: transformers' modeling_afmoe.py.

- **The attention output is gated.** A separate `gate_proj` reads the block input and its
  `sigmoid` multiplies the attention result before `o_proj` — an extra `[hidden, heads ·
  head_dim]` matrix per layer, not a norm.
- **Only `sliding_attention` layers rotate.** The full-attention layers are NoPE, the
  same split EXAONE 4 uses; the window lives in the mask.
- **Four norms per block** (input / post-attention / pre-MLP / post-MLP).
- **`mup_enabled`** scales the embeddings by `sqrt(hidden_size)`.
- The router is `noaux_tc`-shaped — sigmoid or softmax in float32, `expert_bias` on the
  selector only, optional group limiting, `route_norm` then `route_scale` — and lives
  under `mlp.router.gate` with `mlp.expert_bias` beside it.

`rotary_emb.inv_freq` leaves the dict at load: it is a derived table, not a weight.
"""

from sideros.models.afmoe.checkpoint import CHECKPOINT
from sideros.models.afmoe.config import AfmoeConfig
from sideros.models.afmoe.model import Afmoe

__all__ = [
    "CHECKPOINT",
    "Afmoe",
    "AfmoeConfig",
]

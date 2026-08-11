"""GLM-4: sandwich norms around each sublayer, partial rotary, traditional RoPE.

Authoritative semantics: transformers' modeling_glm4.py.

Four deltas against llama:

- **Four norms per block, not two.** `post_self_attn_layernorm` and
  `post_mlp_layernorm` normalize each sublayer's *output* before it joins the residual,
  on top of the usual pre-norms — the sandwich the Swift port already met in Step 3.7.
- **Partial rotary.** Only the first `head_dim · partial_rotary_factor` dimensions
  rotate (0.5 on GLM-4-9B); the rest carry no position. `mx.fast.rope` takes that as
  its `dims` argument, so it costs nothing.
- **`traditional=True`.** GLM pairs `(x[2i], x[2i+1])` where llama pairs
  `(x[i], x[i+d/2])`. Same rotation, different pairing — getting it wrong is silent.
- **Bias on q/k/v only.** `o_proj` and the MLP have none, so the load-time qkv fusion
  carries the bias vector and the output projection stays bias-free.

`gate_up_proj` already ships fused in the checkpoint (one `[2·intermediate, hidden]`
matrix), which is the layout `SwiGLU` declares — no fusion runs at load.
"""

from sideros.models.glm4.checkpoint import CHECKPOINT
from sideros.models.glm4.config import Glm4Config
from sideros.models.glm4.model import Glm4

__all__ = [
    "CHECKPOINT",
    "Glm4",
    "Glm4Config",
]

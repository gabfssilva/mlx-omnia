"""ERNIE 4.5 MoE: the dense ERNIE block on some layers, routed experts plus an ungated
shared expert on the rest.

Authoritative semantics: transformers' modeling_ernie4_5_moe.py.

- **Which layers route** is arithmetic, not a list: `(layer + 1) % moe_layer_interval ==
  0` inside `[moe_layer_start_index, moe_layer_end_index]`. Both indices ship as a scalar
  or a per-modality list; the text trunk takes the min of the starts and the max of the
  ends, which is what transformers does.
- **The router runs in float32** — `softmax` or `sigmoid` per `moe_gate_act` — and the
  top-k weights are divided by their own sum with a `1e-12` floor. Under sigmoid that
  normalization is not a no-op, unlike a softmax over all experts.
- **The shared expert is ungated**: its output is added, with no `sigmoid(gate)` factor
  in front (which is what Qwen2-MoE has). Its width is
  `moe_intermediate_size · moe_num_shared_experts`.
- **MTP leaves are dropped at load** (`mtp_block.*`, `mtp_linear_proj.*`,
  `mtp_hidden_norm.*`, `mtp_emb_norm.*`), together with `e_score_correction_bias`: this
  port decodes one token per step, and `update(strict=True)` rejects a name the tree
  lacks.

RoPE is `traditional=True`, as in the dense `ernie4_5`.
"""

from sideros.models.ernie4_5_moe.checkpoint import CHECKPOINT
from sideros.models.ernie4_5_moe.config import Ernie45MoEConfig
from sideros.models.ernie4_5_moe.model import Ernie45MoE, Ernie45MoEActivations

__all__ = [
    "CHECKPOINT",
    "Ernie45MoE",
    "Ernie45MoEActivations",
    "Ernie45MoEConfig",
]

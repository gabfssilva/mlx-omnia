"""Qwen2-MoE: Qwen2 attention (bias on q/k/v, no q/k norm) with routed experts plus a
gated shared expert on every layer.

Authoritative semantics: transformers' modeling_qwen2_moe.py.

Two deltas against `qwen3_moe`:

- **A shared expert runs on every token**, its output scaled by
  `sigmoid(shared_expert_gate(x))` — one logit, not a distribution — and added to the
  routed sum. Its width (`shared_expert_intermediate_size`) is its own.
- **No `norm_topk_prob`.** The top-k weights are the raw softmax probabilities over all
  experts and are not renormalized, so they do not sum to one; that is the model, not an
  omission.

The routed experts take `qwen3_moe`'s `SwitchGLU` — the same row-interleaved gate‖up
stack and the same sorted prefill gather. The shared expert keeps `gate_proj`/`up_proj`
separate: it is one dense MLP per layer, and fusing it would need a second dict-side
rule for a matmul that is not on the hot path.
"""

from sideros.models.qwen2_moe.checkpoint import CHECKPOINT
from sideros.models.qwen2_moe.config import Qwen2MoEConfig
from sideros.models.qwen2_moe.model import Qwen2MoE

__all__ = [
    "CHECKPOINT",
    "Qwen2MoE",
    "Qwen2MoEConfig",
]

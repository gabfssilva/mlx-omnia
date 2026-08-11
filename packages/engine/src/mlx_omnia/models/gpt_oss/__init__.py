"""GPT-OSS: MXFP4 experts, attention sinks, alternating sliding/full attention, YaRN rope.

Four things no other ported model has:

- the experts ship **already MXFP4-packed** (group 32, e2m1 + e8m0 scale, *no* biases):
  nothing is quantized at load, but the leaves are ordinary `SwitchLinear`s and the
  format comes off the tensors like every other checkpoint's — `nn.quantize` only
  swaps in the module that reads them. The load reinterprets `_blocks` (uint8) as
  uint32 — a pure view — and each expert carries a bias `SwitchLinear` has no slot
  for, added after the gather. gate‖up already comes interleaved row by row, the
  layout the decode kernel wants, so that fusion is free;
- **attention sinks**: one learned logit per head inside the softmax denominator.
  mlx's fast SDPA takes them natively (`sinks=`), so this is the reference kernel;
- **sliding(128)/full alternating**, expressed as a *mask* over a full cache, not
  eviction: a masked key contributes nothing, identical to the reference's rotating
  cache;
- **YaRN rope** (theta 150000, factor 32): the NTK-by-parts table and the 1.34657
  `mscale` that pre-scales q/k, computed with the same ops as the reference implementation.

Routing is top-k over the **raw** router logits (the router has a bias) and then a
softmax over just those k — not a renormalized softmax over all 32, which rounds
differently.
"""

from mlx_omnia.models.gpt_oss.checkpoint import CHECKPOINT
from mlx_omnia.models.gpt_oss.config import GPTOSSConfig, GPTOSSRoPEScaling
from mlx_omnia.models.gpt_oss.layers.attention import GPTOSSAttention
from mlx_omnia.models.gpt_oss.layers.rope import yarn_rope
from mlx_omnia.models.gpt_oss.model import GPTOSS

__all__ = [
    "CHECKPOINT",
    "GPTOSS",
    "GPTOSSAttention",
    "GPTOSSConfig",
    "GPTOSSRoPEScaling",
    "yarn_rope",
]

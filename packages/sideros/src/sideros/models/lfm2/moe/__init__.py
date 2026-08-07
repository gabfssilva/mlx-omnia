"""LFM2.5-MoE: a hybrid trunk of gated short convs and GQA, with sigmoid-routed experts.

Property names are the checkpoint's. 18 of the 24 layers mix with a depthwise causal
conv of kernel 3 (cached as a 2-row window, not keys and values); the other 6 are GQA
with q/k-norm and rope theta 5e6. Past the first two dense layers the MLP is 32 experts,
4 per token, selected by `sigmoid(logits) + expert_bias` in float32 and weighted by the
bias-free score.

The conv is unrolled into `kernel` shifted taps accumulated in float32 — bit-exact with
`conv1d`, and the only form whose one-token step is a handful of elementwise ops. The
conv window cannot be rewound; the attention layers keep their KV history and trim
normally.
"""

from sideros.models.lfm2.config import LFM2MoEConfig
from sideros.models.lfm2.moe.checkpoint import CHECKPOINT
from sideros.models.lfm2.moe.model import LFM2MoE

__all__ = ["CHECKPOINT", "LFM2MoE", "LFM2MoEConfig"]

"""BitNet b1.58: a small Llama trunk whose every projection is a ternary ``BitLinear``.

Authoritative semantics: transformers' ``modeling_bitnet.py`` (the BitNet team's own
port). The trunk is Llama — GQA, RoPE, RMSNorm, pre-norm residuals, tied lm_head — with
two deltas: an ``attn_sub_norm`` RMSNorm on the attention output before ``o_proj`` and
an ``ffn_sub_norm`` RMSNorm on the gated MLP product before ``down_proj``, and a
``relu2`` (``relu(x)**2``) activation in place of silu. The lm_head/embed stay dense.

The projections are born ternary: ``weight`` is uint8 packed 4-per-byte along the
output axis with a single per-tensor scalar ``weight_scale`` (no scales/biases, so the
affine ``nn.quantize`` path is a no-op here — the tree never goes through it). The
ternary matmul runs in the ``bitlinear`` Metal kernel; the per-token int8 activation
fake-quant transformers' ``AutoBitLinear`` applies runs in the leaf, before the
dispatch, in fp32. No qkv or gate‖up fusion: each projection carries its own
``weight_scale``, so fusing would collapse independent scales. RoPE is split-half
(``traditional=False``), matching transformers' ``rotate_half``.
"""

from mlx_omnia.engine.models.bitnet.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.bitnet.config import BitNetConfig
from mlx_omnia.engine.models.bitnet.model import BitNet, BitNetActivations

__all__ = [
    "CHECKPOINT",
    "BitNet",
    "BitNetActivations",
    "BitNetConfig",
]

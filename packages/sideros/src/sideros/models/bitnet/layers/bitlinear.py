import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.bitlinear import bitlinear, bitlinear_applies


class BitLinear(nn.Module):
    """A ternary 1.58-bit linear leaf: packed uint8 ``weight`` ``[out//4, in]`` and a
    scalar ``weight_scale`` the kernel multiplies by (autobitlinear). Not an
    ``nn.Linear`` subclass, so the affine ``nn.quantize`` pass leaves it alone."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mx.zeros((out_features // 4, in_features), dtype=mx.uint8)
        self.weight_scale = mx.ones((1,))
        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        row = x.reshape(-1, self.in_features)
        quantized = _act_quant(row).astype(x.dtype)
        if bitlinear_applies(self.out_features, self.in_features, self.weight):
            out = bitlinear(quantized, self.weight, self.weight_scale)
        else:
            out = _ternary_matmul(quantized, self.weight, self.weight_scale)
        if "bias" in self:
            out = out + self.bias
        return out.reshape(*lead, self.out_features)


def _act_quant(x: mx.array) -> mx.array:
    """transformers' ``AutoBitLinear`` per-token int8 absmax fake-quant, in fp32:
    ``round(x * 127/absmax)`` clamped to ``[-128, 127]``, rescaled by ``absmax/127``.
    The matmul that follows runs in the model dtype, so the rescale lands there too."""
    xf = x.astype(mx.float32)
    absmax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    scale = mx.array(127.0, dtype=mx.float32) / mx.maximum(absmax, mx.array(1e-5, dtype=mx.float32))
    quant = mx.round(xf * scale)
    quant = mx.minimum(
        mx.maximum(quant, mx.array(-128.0, dtype=mx.float32)),
        mx.array(127.0, dtype=mx.float32),
    )
    return quant / scale


def _unpack_ternary(weight: mx.array) -> mx.array:
    """``packed [out//4, in]`` uint8 -> ``[out, in]`` float in the logical output order
    transformers' ``unpack_weights`` produces: field ``j`` of byte ``p`` is row
    ``p + j * (out//4)``."""
    out_packs = weight.shape[0]
    in_features = weight.shape[1]
    three = mx.array(3, dtype=mx.uint8)
    fields = [
        (weight & three),
        ((weight >> mx.array(2, dtype=mx.uint8)) & three),
        ((weight >> mx.array(4, dtype=mx.uint8)) & three),
        ((weight >> mx.array(6, dtype=mx.uint8)) & three),
    ]
    stacked = mx.concatenate(fields, axis=0)
    return stacked.astype(mx.float32).reshape(out_packs * 4, in_features) - mx.array(
        1.0, dtype=mx.float32
    )


def _ternary_matmul(x: mx.array, weight: mx.array, weight_scale: mx.array) -> mx.array:
    """The op-chain fallback and parity reference: unpack to float, matmul, scale."""
    w = _unpack_ternary(weight).astype(x.dtype)
    return (x @ w.T) * weight_scale

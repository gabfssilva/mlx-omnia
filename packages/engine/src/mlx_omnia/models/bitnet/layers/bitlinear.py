import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.qmv import Qmv


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
        self._qmv: Qmv | None = None

    def _projection(self) -> Qmv:
        """Resolved once, at the first call — after load, when the weight is final.
        ``Qmv`` stands for the matmul inside this leaf, not for the leaf: the
        activation fake-quant and the additive bias stay here."""
        qmv = self._qmv
        if qmv is None:
            qmv = Qmv(self, kdim=self.in_features, rows=self.out_features)
            self._qmv = qmv
        return qmv

    def __call__(self, x: mx.array) -> mx.array:
        out = self._projection()(_act_quant(x).astype(x.dtype))
        if "bias" in self:
            out = out + self.bias
        return out


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

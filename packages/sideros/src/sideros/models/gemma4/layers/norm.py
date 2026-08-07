import mlx.core as mx
import mlx.nn as nn


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable weight — for `v_norm` and the MoE router norm.

    Upcast to fp32 for the `_norm`, cast back; same as `nn.RMSNorm` minus the
    multiply-by-weight.
    """

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        ms = mx.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * mx.rsqrt(ms + self.eps)
        return normed.astype(orig_dtype)

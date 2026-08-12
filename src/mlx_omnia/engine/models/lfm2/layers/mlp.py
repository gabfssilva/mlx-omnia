import mlx.core as mx
import mlx.nn as nn


class LFM2DenseMLP(nn.Module):
    """w1‖w3 concatenated on the output axis at load; w2 projects back."""

    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.w13 = nn.Linear(hidden, 2 * inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)
        self.inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.w13(x)
        gated = fused[..., : self.inner]
        return self.w2(gated * mx.sigmoid(gated) * fused[..., self.inner :])

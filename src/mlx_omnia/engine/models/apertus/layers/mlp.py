import mlx.core as mx
import mlx.nn as nn

BETA = 0.5
EPS = -1e-6


class XieLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha_p = mx.zeros(())
        self.alpha_n = mx.zeros(())

    def __call__(self, x: mx.array) -> mx.array:
        positive = nn.softplus(self.alpha_p)
        negative = BETA + nn.softplus(self.alpha_n)
        return mx.where(
            x > 0,
            positive * mx.square(x) + BETA * x,
            (mx.expm1(mx.minimum(x, EPS)) - x) * negative + BETA * x,
        )


class ApertusMLP(nn.Module):
    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self.act_fn = XieLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.up_proj(x)))

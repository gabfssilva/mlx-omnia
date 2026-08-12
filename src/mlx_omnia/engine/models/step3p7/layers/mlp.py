import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import SwitchLinear


class Step3p7MLP(nn.Module):
    def __init__(self, hidden: int, inner: int, gate_limit: float, up_limit: float) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self.inner = inner
        self.gate_limit = gate_limit
        self.up_limit = up_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _clamped_swiglu(self.gate_up_proj(x), self.inner, self.gate_limit, self.up_limit)
        )


class Step3p7SwitchGLU(nn.Module):
    def __init__(
        self,
        experts: int,
        hidden: int,
        inner: int,
        gate_limit: float,
        up_limit: float,
    ) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner
        self.gate_limit = gate_limit
        self.up_limit = up_limit

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gate = pairs[..., 0]
        up = pairs[..., 1]
        if self.gate_limit > 0:
            gate = mx.clip(gate, None, self.gate_limit)
        if self.up_limit > 0:
            up = mx.clip(up, -self.up_limit, self.up_limit)
        return gate * mx.sigmoid(gate) * up

    def __call__(
        self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


def _clamped_swiglu(
    fused: mx.array, inner: int, gate_limit: float, up_limit: float
) -> mx.array:
    gate, up = mx.split(fused, [inner], axis=-1)
    if gate_limit > 0:
        gate = mx.clip(gate, None, gate_limit)
    if up_limit > 0:
        up = mx.clip(up, -up_limit, up_limit)
    return gate * mx.sigmoid(gate) * up

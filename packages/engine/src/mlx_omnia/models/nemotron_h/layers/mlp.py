import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import SwitchLinear


def squared_relu(x: mx.array) -> mx.array:
    activated = mx.maximum(x, 0)
    return activated * activated


class NemotronHMLP(nn.Module):
    def __init__(self, hidden: int, inner: int, bias: bool) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, inner, bias=bias)
        self.down_proj = nn.Linear(inner, hidden, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(squared_relu(self.up_proj(x)))


class SwitchMLP(nn.Module):
    """The experts, gate-free: `fc1` up, squared ReLU, `fc2` down."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.fc1 = SwitchLinear(experts, hidden, inner)
        self.fc2 = SwitchLinear(experts, inner, hidden)

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        up = self.fc1(tokens, indices, sorted_indices=sorted_indices)
        return self.fc2(squared_relu(up), indices, sorted_indices=sorted_indices)

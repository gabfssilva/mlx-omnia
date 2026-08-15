import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.down_combine import DownCombine
from mlx_omnia.engine.core.kernels.gate_up import GateUp
from mlx_omnia.engine.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwitchLinear,
    sorted_gather,
)
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.gpt_oss.config import GPTOSSConfig
from mlx_omnia.engine.models.gpt_oss.layers import flags


class GPTOSSExperts(nn.Module):
    """The 32 experts as two stacked `SwitchLinear` leaves. Nothing is quantized at load:
    the checkpoint's packed tensors are what the leaf receives, and the spine reads
    `mode="mxfp4"` off them — a mode with no quantization biases (the per-expert `*_bias`
    here is the affine bias of the projection, not the quantizer's)."""

    def __init__(self, config: GPTOSSConfig) -> None:
        super().__init__()
        experts = config.num_local_experts
        hidden, inner = config.hidden_size, config.intermediate_size
        self.inner = inner
        self.limit = config.swiglu_limit
        self.gate_up_proj: SwitchLinear | QuantizedSwitchLinear = SwitchLinear(
            experts, hidden, 2 * inner
        )
        self.gate_up_proj_bias = mx.zeros((experts, 2 * inner), dtype=mx.bfloat16)
        self.down_proj: SwitchLinear | QuantizedSwitchLinear = SwitchLinear(
            experts, inner, hidden
        )
        self.down_proj_bias = mx.zeros((experts, hidden), dtype=mx.bfloat16)

    def mxfp4(self) -> tuple[QuantizedSwitchLinear, QuantizedSwitchLinear] | None:
        """The pair the decode kernels read `weight`/`scales` off — concrete fields, never
        an opaque format. Only after the load are the leaves quantized at all."""
        gate_up, down = self.gate_up_proj, self.down_proj
        if not isinstance(gate_up, QuantizedSwitchLinear):
            return None
        if not isinstance(down, QuantizedSwitchLinear):
            return None
        return (gate_up, down) if gate_up.mode == "mxfp4" and down.mode == "mxfp4" else None

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False
    ) -> mx.array:
        """`x` is [1, T, 1, 1, hidden] and `indices` [1, T, k]: one gemv per routed pair.
        Under the prefill reorder the pair is flattened instead — [N, 1, hidden] against
        [N] — and `sorted_indices` tells the gather each expert's rows are contiguous."""
        fused = self._gather(
            x, self.gate_up_proj, self.gate_up_proj_bias, indices,
            sorted_indices=sorted_indices,
        )
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gate = mx.clip(pairs[..., 0], None, self.limit)
        up = mx.clip(pairs[..., 1], -self.limit, self.limit)
        act = gate * mx.sigmoid(1.702 * gate) * (up + 1)
        return self._gather(
            act, self.down_proj, self.down_proj_bias, indices,
            sorted_indices=sorted_indices,
        )

    def _gather(
        self,
        x: mx.array,
        projection: SwitchLinear | QuantizedSwitchLinear,
        bias: mx.array,
        indices: mx.array,
        *,
        sorted_indices: bool,
    ) -> mx.array:
        projected = projection(x, indices, sorted_indices=sorted_indices)
        return projected + mx.expand_dims(bias[indices], axis=-2)


class GPTOSSMLP(nn.Module):
    def __init__(self, config: GPTOSSConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.experts = GPTOSSExperts(config)
        self.k = config.num_experts_per_tok
        self.split = config.num_local_experts - self.k
        self.hidden = config.hidden_size
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Top-k over the raw (biased) logits, then a softmax over only those k."""
        logits = self.router(x)
        chosen = mx.argpartition(logits, kth=self.split, axis=-1)[..., self.split :]
        weights = softmax(mx.take_along_axis(logits, chosen, axis=-1), axis=-1, precise=True)
        return chosen, weights

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.experts(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            routed = self.experts(mx.expand_dims(x, (-2, -3)), chosen).squeeze(-2)
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)

    def _kernels(self) -> tuple[GateUp, DownCombine]:
        """Resolved once, at the first T=1 step — after load, when the leaves' formats
        and the projection biases are final."""
        gate_up, down = self._gate_up, self._down
        if gate_up is None or down is None:
            experts = self.experts
            gate_up = GateUp(
                experts.gate_up_proj,
                hidden=self.hidden,
                inner=experts.inner,
                activation="swiglu_oai",
                limit=experts.limit,
                bias=experts.gate_up_proj_bias,
            )
            down = DownCombine(
                experts.down_proj,
                hidden=self.hidden,
                inner=experts.inner,
                bias=experts.down_proj_bias,
            )
            self._gate_up, self._down = gate_up, down
        return gate_up, down

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array | None:
        """The whole routed MLP plus the residual in two dispatches, or `None` off the
        T=1 step. Routing stays outside — the pick is a top-k over the raw logits and
        moving it in-kernel flips near-ties."""
        if not (flags.USE_MXFP4_MOE_GEMV and x.shape[0] == 1 and x.shape[1] == 1):
            return None
        gate_up, down = self._kernels()
        chosen, weights = self.route(x)
        indices = chosen.reshape(-1).astype(mx.uint32)
        act = gate_up(x.reshape(-1), indices)
        return down(act, indices, weights.reshape(-1), residual.reshape(-1)).reshape(
            1, 1, self.hidden
        )

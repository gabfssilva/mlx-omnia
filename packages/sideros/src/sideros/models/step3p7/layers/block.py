import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.down_combine import DownCombine
from sideros.core.kernels.gate_up import GateUp
from sideros.core.kernels.route import Route
from sideros.models.step3p7.config import Step3p7TextConfig
from sideros.models.step3p7.layers.attention import Step3p7Attention
from sideros.models.step3p7.layers.mlp import Step3p7MLP
from sideros.models.step3p7.layers.moe import Step3p7MoE


class Step3p7Block(nn.Module):
    def __init__(self, config: Step3p7TextConfig, layer: int) -> None:
        super().__init__()
        self.self_attn = Step3p7Attention(config, layer)
        if layer in config.moe_layers:
            self.moe = Step3p7MoE(config, layer)
        else:
            self.mlp = Step3p7MLP(
                config.hidden_size,
                config.intermediate_size,
                config.limits[layer],
                config.limits[layer],
            )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._route: Route | None = None
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache, mask)
        h = self.post_attention_layernorm(attended)
        kernels = self._kernels() if x.shape[1] == 1 else None
        if kernels is not None:
            moe = self.moe
            route, gate_up, down = kernels
            row = h.reshape(-1)
            # The gate's precision is the model's: in fp32 it is this block's dispatch,
            # so the row's logits are handed to the router instead of its gemv.
            logits = moe.gate(row.astype(mx.float32)) if moe.need_fp32_gate else None
            chosen, weights = route(row, logits=logits)
            act = gate_up(row, chosen)
            residual = attended + moe.shared_expert(h)
            return down(act, chosen, weights, residual.reshape(-1)).reshape(1, 1, moe.hidden)
        if hasattr(self, "moe"):
            return attended + self.moe(h)
        return attended + self.mlp(h)

    def _kernels(self) -> tuple[Route, GateUp, DownCombine] | None:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final. `None` on the dense layers."""
        if not hasattr(self, "moe"):
            return None
        moe = self.moe
        route, gate_up, down = self._route, self._gate_up, self._down
        if route is None or gate_up is None or down is None:
            switch = moe.switch_mlp
            bias = (
                moe.router_bias
                if hasattr(moe, "router_bias")
                else mx.zeros((moe.gate.weight.shape[0],), mx.float32)
            )
            assert isinstance(bias, mx.array)
            route = Route(
                moe.gate.weight,
                experts=moe.split + moe.k,
                k=moe.k,
                scoring="sigmoid",
                bias=bias,
                normalize=moe.norm_expert_weight,
                scale=moe.scaling,
            )
            gate_up = GateUp(
                switch.gate_up_proj,
                hidden=moe.hidden,
                inner=switch.inner,
                limit=switch.gate_limit or None,
            )
            down = DownCombine(switch.down_proj, hidden=moe.hidden, inner=switch.inner)
            self._route, self._gate_up, self._down = route, gate_up, down
        return route, gate_up, down


class Step3p7Trunk(nn.Module):
    def __init__(self, config: Step3p7TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Step3p7Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

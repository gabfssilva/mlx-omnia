import math
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import DeltaCache, KVCache
from mlx_omnia.engine.core.kernels.add_norm import AddRmsNorm, AddRmsNormStrategy, DefaultAddRmsNorm
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.qwen3_5.config import Qwen35TextConfig
from mlx_omnia.engine.models.qwen3_5.layers import deltanet, flags, moe
from mlx_omnia.engine.models.qwen3_5.layers.attention import Qwen35Attention
from mlx_omnia.engine.models.qwen3_5.layers.deltanet import Qwen35DeltaNet, l2norm
from mlx_omnia.engine.models.qwen3_5.layers.moe import Qwen35MoE

_DeltaStep = Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]
_TailStep = Callable[[mx.array, mx.array], mx.array]


def _trace_key(block: "Qwen35Block") -> tuple[object, ...]:
    """Everything the compiled steps bake in that tests or benches swap at runtime: the
    delegator bindings and the A/B flags. The applicability predicates moved inside the
    delegators' resolution, which is itself keyed on the same bindings and flags, so
    carrying those carries the predicates too. Each binding is read off its owning
    module, which is also where a patch lands."""
    return (
        deltanet.GatedDelta,
        AddRmsNorm,
        DefaultAddRmsNorm,
        moe.Route,
        moe.GateUp,
        moe.DownCombine,
        flags.ADD_RMS_NORM_KERNEL,
        flags.GATED_DELTA_KERNEL,
        isinstance(block.mlp, Qwen35MoE),
    )


class Qwen35Block(nn.Module):
    """One mixer of the two, then the MLP. The MoE variant (35B-A3B) swaps `mlp` for a
    sparse block chosen off the config — the rest of the block is unchanged."""

    def __init__(self, config: Qwen35TextConfig, layer: int) -> None:
        super().__init__()
        self.attends = config.layer_types[layer] == "full_attention"
        if self.attends:
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35DeltaNet(config)
        self.mlp = (
            SwiGLU(config.hidden_size, config.intermediate_size)
            if config.num_experts == 0
            else Qwen35MoE(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._join_strategy: AddRmsNormStrategy | None = None
        self._join_key: tuple[object, ...] | None = None
        self._tail = self._build_tail()
        self._traced = _trace_key(self)
        # `inputs=self.state` keeps the weights this module reads directly implicit
        # inputs of the trace instead of baked constants: a swapped tensor
        # (quantize-on-load, a mutation test) is read on the next call, no retrace
        # needed. Tensors a delegator captured at resolution are baked, which is why
        # resolution is lazy — it happens at the first step, past the loader.
        if self.attends:
            self._tail_step = mx.compile(self._build_tail(), inputs=self.state)
        else:
            self._step = mx.compile(self._build_step(), inputs=self.state)

    def _join(self) -> AddRmsNormStrategy:
        """Resolved once, at the first T=1 step — after load, when the norm leaf's
        format is final. The A/B switch has no place in the declaration (the delegator
        is total and would always find a kernel), so it picks the default strategy
        directly; the cache key carries it, and the two bindings, for the same reason
        `rule()` does."""
        key = (AddRmsNorm, DefaultAddRmsNorm, flags.ADD_RMS_NORM_KERNEL)
        join = self._join_strategy
        if join is None or self._join_key != key:
            post = self.post_attention_layernorm
            join = (
                AddRmsNorm(post, tokens=1)
                if flags.ADD_RMS_NORM_KERNEL
                else DefaultAddRmsNorm.build(post, tokens=1)
            )
            self._join_strategy, self._join_key = join, key
        return join

    def mix(
        self, x: mx.array, cache: KVCache | DeltaCache, positions: mx.array | None
    ) -> mx.array:
        if self.attends:
            assert isinstance(cache, KVCache)
            return self.self_attn(self.input_layernorm(x), cache, positions)
        assert isinstance(cache, DeltaCache)
        return self.linear_attn(self.input_layernorm(x), cache)

    def _build_tail(self) -> _TailStep:
        """The tail every one-token layer ends in: the residual join and the norm that
        reads it (one dispatch when the kernel fits), then the sparse block's routing and
        two fused gemvs — or the dense MLP. Runs uncompiled on the eager path and traced
        inside the compiled steps; the branches resolve at call/trace time, after the
        loader decided what is quantized."""
        mlp = self.mlp

        def tail(x: mx.array, projected: mx.array) -> mx.array:
            attended, normed = self._join()(x, projected)
            if isinstance(mlp, Qwen35MoE):
                return mlp.fused_step(normed, attended)
            return attended + mlp(normed)

        return tail

    def _build_step(self) -> _DeltaStep:
        """The one-token DeltaNet step as a single trace: norm → fused projection →
        unrolled conv → decay → the gated delta rule → gated norm → out projection →
        the tail. The conv window and the recurrent state ride in and out as arrays; the
        ops and dtypes replicate the eager path exactly, so only compile's elementwise
        fusion changes rounding."""
        delta = self.linear_attn
        config = delta.config
        heads = config.linear_num_value_heads
        head_shape = (1, 1, heads, config.linear_value_head_dim)
        key_shape = (1, 1, config.linear_num_key_heads, config.linear_key_head_dim)
        scale = 1 / math.sqrt(config.linear_key_head_dim)
        splits = (
            config.conv_dim,
            config.conv_dim + config.value_dim,
            config.conv_dim + config.value_dim + heads,
        )
        kernel = config.linear_conv_kernel_dim
        norm = self.input_layernorm
        tail = self._build_tail()

        def step(
            x: mx.array, window: mx.array, state: mx.array
        ) -> tuple[mx.array, mx.array, mx.array]:
            fused = delta.fused_proj(norm(x))
            qkv, z, b, a = mx.split(fused, splits, axis=-1)
            padded = mx.concatenate([window, qkv], axis=1)
            taps = delta.conv1d.weight
            mixed = padded[:, :1] * taps[:, 0]
            for j in range(1, kernel):
                mixed = mixed + padded[:, j : j + 1] * taps[:, j]
            mixed = mixed * mx.sigmoid(mixed)
            q, k, v = mx.split(mixed, (config.key_dim, 2 * config.key_dim), axis=-1)
            dt = a.astype(mx.float32) + delta.dt_bias.astype(mx.float32)
            g = -mx.exp(delta.A_log) * nn.softplus(dt)
            out, state_out = delta.rule()(
                l2norm(q.reshape(key_shape), scale, dtype=mx.float32),
                l2norm(k.reshape(key_shape), dtype=mx.float32),
                v.reshape(head_shape).astype(mx.float32),
                mx.exp(g),
                mx.sigmoid(b).astype(mx.float32),
                state,
            )
            out = out.astype(x.dtype)
            lifted = out.astype(mx.float32)
            normed = lifted * mx.rsqrt(
                (lifted * lifted).mean(axis=-1, keepdims=True) + config.rms_norm_eps
            )
            gate = z.reshape(head_shape).astype(mx.float32)
            gated = delta.norm.weight * normed.astype(out.dtype)
            gated = (gated * gate * mx.sigmoid(gate)).astype(x.dtype)
            projected = delta.out_proj(gated.reshape(1, 1, config.value_dim))
            return tail(x, projected), padded[:, 1:], state_out

        return step

    def _rebuild(self) -> None:
        # The traces bake the resolved delegators and the A/B flags in; rebuild when any
        # changes so the mutation tests (and any monkeypatch) reach inside them.
        key = _trace_key(self)
        if self._traced != key:
            self._traced = key
            # A strategy resolved inside a trace holds that trace's tracers as its
            # weight fields; a new trace (or the eager path) reading it evals a dead
            # placeholder. Drop the resolutions so the fresh trace resolves its own.
            self._join_strategy = None
            self._join_key = None
            mlp = self.mlp
            if isinstance(mlp, Qwen35MoE):
                mlp._route = None
                mlp._gate_up = None
                mlp._down = None
            if self.attends:
                self._tail_step = mx.compile(self._build_tail(), inputs=self.state)
            else:
                self._step = mx.compile(self._build_step(), inputs=self.state)

    def _compiled_delta(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.linear_attn.config
        window = cache.window
        if window is None:
            shape = (1, config.linear_conv_kernel_dim - 1, config.conv_dim)
            window = mx.zeros(shape, dtype=x.dtype)
        state = cache.state
        if state is None:
            heads = config.linear_num_value_heads
            shape = (1, heads, config.linear_value_head_dim, config.linear_key_head_dim)
            state = mx.zeros(shape, dtype=mx.float32)
        self._rebuild()
        out, cache.window, cache.state = self._step(x, window, state)
        cache.offset += 1
        return out

    def __call__(
        self, x: mx.array, cache: KVCache | DeltaCache, positions: mx.array | None = None
    ) -> mx.array:
        if x.shape[1] == 1 and flags.COMPILED_STEP:
            if self.attends:
                # Projections stay eager: mx.fast.rope takes the offset as an op
                # attribute and a trace would freeze it at the first token.
                assert isinstance(cache, KVCache)
                projected = self.self_attn(self.input_layernorm(x), cache, positions)
                self._rebuild()
                return self._tail_step(x, projected)
            assert isinstance(cache, DeltaCache)
            return self._compiled_delta(x, cache)
        projected = self.mix(x, cache, positions)
        if x.shape[1] == 1:
            return self._tail(x, projected)
        attended = x + projected
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen35Trunk(nn.Module):
    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen35Block(config, layer) for layer in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

import math
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import (
    DeltaCache,
    FixedDeltaCache,
    FixedKVCache,
    KVCache,
    LayerCache,
)
from mlx_omnia.engine.core.kernels.add_norm import AddRmsNorm
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.qwen3_5.config import Qwen35TextConfig
from mlx_omnia.engine.models.qwen3_5.layers import deltanet, flags, moe
from mlx_omnia.engine.models.qwen3_5.layers.attention import Qwen35Attention
from mlx_omnia.engine.models.qwen3_5.layers.deltanet import Qwen35DeltaNet, Recurring, l2norm
from mlx_omnia.engine.models.qwen3_5.layers.moe import Qwen35MoE

type Qwen35Layer = LayerCache | KVStore | Recurring
"""A layer's cache, alone or standing for one row each: attention reads it through
`core.attend`, and the DeltaNet mixer through `window`/`state`."""

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
        self._join_strategy: AddRmsNorm | None = None
        self._join_key: tuple[object, ...] | None = None
        self._tail = self._build_tail()
        # The uncompiled DeltaNet body, kept beside its trace: a trunk-level compile calls
        # this one, so the outer graph is the only trace.
        self._step_body: _DeltaStep | None = None
        # Whether the delegators were last resolved for a trunk-level graph rather than
        # inside this block's own traces. `_rebuild` reads it; `prepare_decode` sets it.
        self._resolved_outside = False
        # Bumped every time the delegators are dropped, so a graph built over them can
        # tell that what it baked is gone.
        self.epoch = 0
        self._traced = _trace_key(self)
        # `_traced_state()` keeps the weights this module reads directly implicit inputs
        # of the trace instead of baked constants: a swapped tensor (quantize-on-load, a
        # mutation test) is read on the next call, no retrace needed. What a delegator
        # holds is baked instead, and `Qwen35MoE.kernels` is what notices a swap there.
        if self.attends:
            self._tail_step = mx.compile(self._tail, inputs=self._traced_state())
        else:
            self._step_body = self._build_step()
            self._step = mx.compile(self._step_body, inputs=self._traced_state())

    def _join(self) -> AddRmsNorm:
        """Resolved once, at the first T=1 step — after load, when the norm leaf's
        format is final. The A/B switch requests the delegator's reference path; the
        cache key carries it, and the binding, for the same reason `rule()` does."""
        key = (AddRmsNorm, flags.ADD_RMS_NORM_KERNEL)
        join = self._join_strategy
        if join is None or self._join_key != key:
            post = self.post_attention_layernorm
            join = AddRmsNorm(post, tokens=1, reference=not flags.ADD_RMS_NORM_KERNEL)
            self._join_strategy, self._join_key = join, key
        return join

    def mix(self, x: mx.array, cache: Qwen35Layer, positions: mx.array | None) -> mx.array:
        if self.attends:
            assert isinstance(cache, LayerCache)
            return self.self_attn(self.input_layernorm(x), cache, positions)
        assert isinstance(cache, Recurring)
        return self.linear_attn(self.input_layernorm(x), cache)

    def prepare_decode(self) -> None:
        """Every resolution the trunk's decode graph will need, taken fresh and outside
        any trace, before that graph is built.

        Dropped first and not merely resolved: the per-block steps resolve on their first
        call, which happens *inside* their own trace, so a strategy left over from one
        holds that trace's tracers as its weight fields. The mark is what makes the trade
        reversible — a block that streams again through its own step rebuilds instead of
        reading arrays that trace never captured."""
        self._rebuild()
        self._drop_resolutions()
        self._join()
        if not self.attends:
            self.linear_attn.rule()
        mlp = self.mlp
        if isinstance(mlp, Qwen35MoE):
            mlp.kernels()
        self._resolved_outside = True

    def _graph_step(
        self, x: mx.array, cache: FixedKVCache | FixedDeltaCache, positions: mx.array | None
    ) -> mx.array:
        """One token through this block reading every array off a graph-visible cache —
        the body a trunk-level `mx.compile` traces.

        Deliberately not `self._step`/`self._tail_step`: those are traces of their own,
        and a trace inside a trace is either inlined twice or a barrier the outer graph
        cannot fuse across. The arithmetic is the same closure either way, so the two
        paths round identically. `_rebuild` is deliberately absent for the same reason —
        it is host-side work, and `Qwen35.before_trace` is where it happens instead."""
        if self.attends:
            assert isinstance(cache, FixedKVCache)
            return self._tail(x, self.self_attn(self.input_layernorm(x), cache, positions))
        assert isinstance(cache, FixedDeltaCache)
        window, state = cache.window, cache.state
        step = self._step_body
        assert window is not None and state is not None and step is not None
        out, cache.window, cache.state = step(x, window, state)
        return out

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

    def _traced_state(self) -> list[object]:
        """The containers a trace of this block must re-read on every call.

        A sparse MLP is deliberately absent. Its leaves reach the graph through the three
        resolved kernels, which hold them as their own fields — and an array that is both
        a declared input and a strategy's frozen field is exactly what `mx.compile`
        rejects: it swaps the one inside the container for a tracer and the strategy goes
        on reading the original. `Qwen35MoE.kernels` re-resolves when a leaf is replaced,
        so nothing is lost by leaving them baked. A dense MLP stays: the trace reads it
        directly."""
        mixer = self.self_attn if self.attends else self.linear_attn
        state: list[object] = [
            self.input_layernorm.state,
            self.post_attention_layernorm.state,
            mixer.state,
        ]
        if not isinstance(self.mlp, Qwen35MoE):
            state.append(self.mlp.state)
        return state

    def _resolve(self) -> None:
        """The MLP's kernels, bound outside the trace about to read them. A strategy
        decides applicability by inspecting the checkpoint's tensors — the nvfp4 pair
        reads the scale plane — and an array read inside `mx.compile` raises."""
        mlp = self.mlp
        if isinstance(mlp, Qwen35MoE):
            mlp.kernels()

    def _drop_resolutions(self) -> None:
        """Every lazily-bound delegator back to unresolved.

        A strategy resolved inside a trace holds that trace's tracers as its weight
        fields; one resolved outside every trace holds real arrays — and those arrays are
        the module's own leaves, which the per-block traces already declare through
        `inputs=self._traced_state()`, so a trace reading them a second time as constants is the
        uncaptured-input error. Neither resolution survives being borrowed by the other
        path, so both sides drop first and resolve their own.

        `epoch` is how the trunk's graph finds out: it baked whatever was resolved when it
        was traced, and a drop here makes that graph stale."""
        self.epoch += 1
        self._join_strategy = None
        self._join_key = None
        if not self.attends:
            self.linear_attn.unresolve()
        mlp = self.mlp
        if isinstance(mlp, Qwen35MoE):
            mlp.unresolve()

    def _rebuild(self) -> None:
        # The traces bake the resolved delegators and the A/B flags in; rebuild when any
        # changes so the mutation tests (and any monkeypatch) reach inside them — and
        # whenever a trunk-level compile resolved them eagerly, which leaves this block's
        # own traces reading arrays they never captured.
        key = _trace_key(self)
        if self._traced != key or self._resolved_outside:
            self._traced = key
            self._resolved_outside = False
            self._drop_resolutions()
            self._tail = self._build_tail()
            if self.attends:
                self._tail_step = mx.compile(self._tail, inputs=self._traced_state())
            else:
                self._step_body = self._build_step()
                self._step = mx.compile(self._step_body, inputs=self._traced_state())
        # After the rebuild and outside every trace, which is the whole point of doing it
        # here: the step this returns to is about to run compiled.
        self._resolve()

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
        self, x: mx.array, cache: Qwen35Layer, positions: mx.array | None = None
    ) -> mx.array:
        # Both the fused steps and the sparse tail are one row wide; a ragged batch takes
        # the eager path below instead.
        single = x.shape[0] == 1 and x.shape[1] == 1
        if single and isinstance(cache, FixedKVCache | FixedDeltaCache):
            # Promoted caches, so this call is inside a trunk-level trace whether or not it
            # knows it. First, and before the `COMPILED_STEP` pair: a `FixedDeltaCache` is
            # a `DeltaCache`, and taking that branch would nest a compile inside the outer
            # graph and re-resolve delegators the outer graph had already baked.
            return self._graph_step(x, cache, positions)
        if single and flags.COMPILED_STEP:
            if self.attends and isinstance(cache, KVCache):
                # Projections stay eager: mx.fast.rope takes the offset as an op
                # attribute and a trace would freeze it at the first token.
                projected = self.self_attn(self.input_layernorm(x), cache, positions)
                self._rebuild()
                return self._tail_step(x, projected)
            if not self.attends and isinstance(cache, DeltaCache):
                return self._compiled_delta(x, cache)
        projected = self.mix(x, cache, positions)
        if single:
            return self._tail(x, projected)
        attended = x + projected
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen35Trunk(nn.Module):
    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen35Block(config, layer) for layer in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

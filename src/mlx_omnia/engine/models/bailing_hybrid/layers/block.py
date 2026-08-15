from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.core.layers import SwiGLU, l2norm, swish
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.engine.models.bailing_hybrid.layers import flags, kda
from mlx_omnia.engine.models.bailing_hybrid.layers.attention import (
    BailingHybridLatentAttention,
    LatentStore,
)
from mlx_omnia.engine.models.bailing_hybrid.layers.cache import BatchedLatentKVCache
from mlx_omnia.engine.models.bailing_hybrid.layers.kda import KimiDeltaAttention, Recurring
from mlx_omnia.engine.models.bailing_hybrid.layers.moe import BailingHybridMoE

_DeltaStep = Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]
_TailStep = Callable[[mx.array, mx.array], mx.array]

type BailingHybridLayer = LayerCache | LatentStore | Recurring
"""A layer's cache, alone or standing for one row each: the MLA layers read a latent store,
the KDA ones read `window`/`state`."""


def _trace_key(block: "BailingHybridBlock") -> tuple[object, ...]:
    """What the compiled steps bake in that a test or a bench swaps at runtime: the
    kernel binding and the predicate that selects it are read at trace time, so the key
    carries their value, not the function."""
    return (kda.gated_delta, block.delta_fits())


class BailingHybridBlock(nn.Module):
    def __init__(
        self,
        config: BailingHybridConfig,
        attends: bool,
        routes: bool,
        expert_limit: float,
        shared_limit: float,
    ) -> None:
        super().__init__()
        self.attends = attends
        self.attention: BailingHybridLatentAttention | KimiDeltaAttention = (
            BailingHybridLatentAttention(config) if attends else KimiDeltaAttention(config)
        )
        self.mlp: BailingHybridMoE | SwiGLU = (
            BailingHybridMoE(config, expert_limit, shared_limit)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
        self._tail = self._build_tail()
        self._traced = _trace_key(self)
        self._compile_step()

    def _compile_step(self) -> None:
        # `inputs=self.state` keeps the weights implicit inputs of the trace instead of
        # baked constants: a swapped tensor (quantize-on-load, a mutation test) is read
        # on the next call, no retrace needed.
        if self.attends:
            self._tail_step = mx.compile(self._build_tail(), inputs=self.state)
        else:
            self._step = mx.compile(self._build_step(), inputs=self.state)

    def delta_fits(self) -> bool:
        mixer = self.attention
        return isinstance(mixer, KimiDeltaAttention) and mixer.fits()

    def __call__(self, x: mx.array, cache: BailingHybridLayer) -> mx.array:
        # One mixer or the other; mlx.nn.Module's __getattr__ is untyped.
        mixer = self.attention
        # The compiled steps bake the batch axis into their trace, so a ragged batch of more
        # than one row takes the eager path.
        if x.shape[1] == 1 and x.shape[0] == 1 and flags.COMPILED_STEP:
            if self.attends:
                # The MLA projections stay eager: mx.fast.rope takes the offset as an op
                # attribute and a trace would freeze it at the first token.
                assert isinstance(mixer, BailingHybridLatentAttention)
                assert isinstance(cache, KVCache | BatchedLatentKVCache)
                projected = mixer(self.input_layernorm(x), cache)
                self._rebuild()
                return self._tail_step(x, projected)
            if self.delta_fits():
                assert isinstance(cache, Recurring)
                return self._compiled_delta(x, cache)
        normed = self.input_layernorm(x)
        if self.attends:
            assert isinstance(mixer, BailingHybridLatentAttention)
            assert isinstance(cache, KVCache | BatchedLatentKVCache)
            projected = mixer(normed, cache)
        else:
            assert isinstance(mixer, KimiDeltaAttention) and isinstance(cache, Recurring)
            projected = mixer(normed, cache)
        if x.shape[1] == 1:
            return self._tail(x, projected)
        mixed = x + projected
        return mixed + self.mlp(self.post_attention_layernorm(mixed))

    def _build_tail(self) -> _TailStep:
        """The tail every one-token layer ends in: the residual join and the norm that
        reads it (one dispatch when the kernel fits), then the MLP. Runs uncompiled on
        the eager path and traced inside the compiled steps; the branches resolve at
        call/trace time, after the loader decided what is quantized."""
        mlp = self.mlp
        post = self.post_attention_layernorm

        def tail(x: mx.array, projected: mx.array) -> mx.array:
            mixed = x + projected
            return mixed + mlp(post(mixed))

        return tail

    def _build_step(self) -> _DeltaStep:
        """The one-token KDA step as a single trace: norm → the fused projection → the
        unrolled conv → the decay → the gated_delta kernel → the gated norm → the output
        projection → the tail. The conv window and the recurrent state ride in and out as
        arrays; the ops and dtypes replicate the eager path exactly, so only compile's
        elementwise fusion changes rounding."""
        config = self.config
        mixer = self.attention
        assert isinstance(mixer, KimiDeltaAttention)
        norm = self.input_layernorm
        tail = self._build_tail()
        kernel = config.short_conv_kernel_size
        heads, head_dim = config.num_attention_heads, config.head_dim
        width = config.linear_width
        shape = (1, 1, heads, head_dim)

        def step(
            x: mx.array, window: mx.array, state: mx.array
        ) -> tuple[mx.array, mx.array, mx.array]:
            normed = norm(x)
            qkv, f, g, b = mx.split(mixer.fused_proj(normed), mixer.splits, axis=-1)
            padded = mx.concatenate([window, qkv], axis=1)
            taps = mixer.conv1d.weight
            mixed = padded[:, :1] * taps[:, 0]
            for tap in range(1, kernel):
                mixed = mixed + padded[:, tap : tap + 1] * taps[:, tap]
            mixed = swish(mixed)
            q, k, v = (
                part.reshape(shape) for part in mx.split(mixed, (width, 2 * width), axis=-1)
            )
            out, advanced = kda.gated_delta(
                l2norm(q, head_dim**-0.5, dtype=mx.float32),
                l2norm(k, dtype=mx.float32),
                v.astype(mx.float32),
                mixer.decay(f, 1),
                mx.sigmoid(b.astype(mx.float32)),
                state,
            )
            gated = mixer.o_norm(out.astype(x.dtype), g.reshape(shape))
            return (
                tail(x, mixer.o_proj(gated.reshape(1, 1, width))),
                padded[:, 1:],
                advanced,
            )

        return step

    def _rebuild(self) -> None:
        # The traces bake the kernel bindings and the A/B flags in; rebuild when any
        # changes so the mutation tests (and any monkeypatch) reach inside them.
        key = _trace_key(self)
        if self._traced != key:
            self._traced = key
            self._compile_step()

    def _compiled_delta(self, x: mx.array, cache: Recurring) -> mx.array:
        config = self.config
        window = cache.window
        if window is None:
            shape = (1, config.short_conv_kernel_size - 1, 3 * config.linear_width)
            window = mx.zeros(shape, dtype=x.dtype)
        state = cache.state
        if state is None:
            heads, head_dim = config.num_attention_heads, config.head_dim
            state = mx.zeros((1, heads, head_dim, head_dim), dtype=mx.float32)
        self._rebuild()
        out, cache.window, cache.state = self._step(x, window, state)
        cache.offset += 1
        return out


class BailingHybridTrunk(nn.Module):
    def __init__(self, config: BailingHybridConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            BailingHybridBlock(config, attends, routes, expert_limit, shared_limit)
            for attends, routes, expert_limit, shared_limit in zip(
                config.attends,
                config.routes,
                config.expert_limits,
                config.shared_limits,
                strict=True,
            )
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

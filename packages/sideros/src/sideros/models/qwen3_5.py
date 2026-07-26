"""Qwen3.5/3.6 text: a hybrid decoder where three of every four layers are a gated
DeltaNet and the fourth is GQA with an output gate.

Semantics follow transformers' `modeling_qwen3_5.py`, not mlx-lm: the DeltaNet's
l2norm carries its eps *inside the sum* (mlx-lm uses rms_norm x scale, an effective
eps 128x larger). The decay runs in float32 (`A_log` never leaves it) and so does the
recurrent state.

Property names are the checkpoint's after the loader normalizes the two dialects
(raw HF `model.language_model.*` vs mlx `language_model.model.*`); the projections of
each mixer are concatenated on the output axis into `fused_proj`, dict-side.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sideros.core.cache import DeltaCache, KVCache
from sideros.core.kernels.add_rms_norm import add_rms_norm, add_rms_norm_applies
from sideros.core.kernels.gated_delta import gated_delta, gated_delta_applies
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import softmax_topk, softmax_topk_applies
from sideros.core.layers import SORTED_GATHER_MIN, SwiGLU, sorted_gather
from sideros.core.mxcompat import softmax
from sideros.generate import Sampler, greedy
from sideros.models.qwen3_5_vision import (
    Grid,
    Qwen35Vision,
    Qwen35VisionConfig,
    multimodal_positions,
)
from sideros.models.qwen3_moe import QuantizedSwitchLinear, SwitchLinear

_L2_EPS = 1e-6

_DeltaStep = Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]
_TailStep = Callable[[mx.array, mx.array], mx.array]


def _trace_key(block: "Qwen35Block") -> tuple[object, ...]:
    """Everything the compiled steps bake in that tests or benches swap at runtime: the
    kernel bindings, the A/B flags, and the applicability predicates — the traced branches
    read those at trace time, so the key carries their value, not the function."""
    mlp = block.mlp
    return (
        gated_delta,
        add_rms_norm,
        softmax_topk,
        moe_gate_up_act,
        moe_down_combine,
        ADD_RMS_NORM_KERNEL,
        add_rms_norm_applies(block.hidden),
        block.delta_fits(),
        isinstance(mlp, Qwen35MoE) and mlp.fused_step_applies(),
    )

# A/B switches for the bench: set either to False (module attribute, read on every call)
# and the ops chain below takes over — it is also the parity reference. The DeltaNet one
# changes the recurrent state's layout, so flip it before a cache is created.
GATED_DELTA_KERNEL = True
ADD_RMS_NORM_KERNEL = True

# The compiled one-token DeltaNet step (the Swift port's `linearStep`): the ~20
# elementwise ops around the recurrence collapse into a handful of kernels. Rides on
# the gated_delta kernel, so it only engages when that one does.
COMPILED_STEP = True


@dataclass(frozen=True)
class Qwen35MoEParams:
    """What the 35B-A3B adds to the 27B: the MLP of every block is sparse. Routing is
    always softmax over all experts, top-k, renormalize — there is no flag for it."""

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int


@dataclass(frozen=True)
class Qwen35Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    intermediate_size: int
    layer_types: tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    eos_token_id: tuple[int, ...]
    quantized: bool
    mrope_section: tuple[int, int, int]
    image_token_id: int
    moe: Qwen35MoEParams | None = None
    vision: Qwen35VisionConfig | None = None

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


class Conv1dWeight(nn.Module):
    """The depthwise conv's taps, `[conv_dim, kernel]` — the loader squeezes both
    checkpoint dialects into that shape."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.weight = mx.zeros((config.conv_dim, config.linear_conv_kernel_dim))


def _l2norm(x: mx.array, scale: float = 1.0, *, dtype: mx.Dtype | None = None) -> mx.array:
    """transformers' l2norm: the eps sits inside the sum, not in a mean.

    `scale` folds the rule's `1/sqrt(Dk)` into the query, and `dtype` skips the trip back
    to the model dtype: the kernel takes `q` pre-scaled and in float32, which is where
    the ops path does this same arithmetic anyway — rounding it to bfloat16 first is a
    rounding the ops path never pays, and on the 35B it costs 4.3e-2 against a 3.9e-2
    bound.
    """
    lifted = x.astype(mx.float32)
    inv = mx.rsqrt((lifted * lifted).sum(axis=-1, keepdims=True) + _L2_EPS)
    normed = lifted * inv if scale == 1.0 else lifted * inv * scale
    return normed.astype(x.dtype if dtype is None else dtype)


def _delta_rule(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
) -> tuple[mx.array, mx.array]:
    """The recurrent gated delta rule, token by token, entirely in float32.

    `q`, `k`, `v` are `[1, T, H, D]`, `g` the *log* decay and `beta` the write
    strength, `state` `[1, H, Dk, Dv]`. Same recurrence as transformers'
    `torch_recurrent_gated_delta_rule`, which is also what its chunked prefill path
    computes — the chunking is an associative rewrite, not a different rule.
    """
    scale = 1 / math.sqrt(q.shape[-1])
    q32 = q.astype(mx.float32).transpose(0, 2, 1, 3) * scale
    k32 = k.astype(mx.float32).transpose(0, 2, 1, 3)
    v32 = v.astype(mx.float32).transpose(0, 2, 1, 3)
    decay = mx.exp(g.astype(mx.float32)).transpose(0, 2, 1)
    beta32 = beta.astype(mx.float32).transpose(0, 2, 1)

    outputs: list[mx.array] = []
    for t in range(q.shape[1]):
        k_t = k32[:, :, t, :, None]
        state = state * decay[:, :, t, None, None]
        delta = (v32[:, :, t] - (state * k_t).sum(axis=-2)) * beta32[:, :, t, None]
        state = state + k_t * delta[..., None, :]
        outputs.append((state * q32[:, :, t, :, None]).sum(axis=-2))

    out = mx.stack(outputs, axis=2) if outputs else mx.zeros(v32.shape)
    return out.transpose(0, 2, 1, 3).astype(v.dtype), state


class Qwen35DeltaNet(nn.Module):
    """Linear attention: a short causal conv over q‖k‖v feeding a recurrent delta
    rule at float32, with a gated RMSNorm on the way out."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        heads = config.linear_num_value_heads
        self.fused_proj = nn.Linear(
            config.hidden_size, config.conv_dim + config.value_dim + 2 * heads, bias=False
        )
        self.out_proj = nn.Linear(config.value_dim, config.hidden_size, bias=False)
        self.conv1d = Conv1dWeight(config)
        self.norm = nn.RMSNorm(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.A_log = mx.zeros((heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((heads,))
        self.config = config

    def _conv(self, x: mx.array, cache: DeltaCache) -> mx.array:
        """The depthwise causal conv unrolled into taps, then silu. `[1, T, conv_dim]`
        in and out; the window carries the `kernel - 1` rows of history."""
        config = self.config
        kernel = config.linear_conv_kernel_dim
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, config.conv_dim), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for j in range(1, kernel):
            mixed = mixed + padded[:, j : j + length] * taps[:, j]
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        heads = config.linear_num_value_heads
        key_heads = config.linear_num_key_heads
        fused = self.fused_proj(x)
        splits = (
            config.conv_dim,
            config.conv_dim + config.value_dim,
            config.conv_dim + config.value_dim + heads,
        )
        qkv, z, b, a = mx.split(fused, splits, axis=-1)

        mixed = self._conv(qkv, cache)
        q, k, v = mx.split(mixed, (config.key_dim, 2 * config.key_dim), axis=-1)
        key_shape = (1, length, key_heads, config.linear_key_head_dim)
        q, k = q.reshape(key_shape), k.reshape(key_shape)
        v = v.reshape(1, length, heads, config.linear_value_head_dim)

        # g is the log decay: exp(A_log) never leaves float32, or it saturates.
        dt = a.astype(mx.float32) + self.dt_bias.astype(mx.float32)
        g = -mx.exp(self.A_log) * nn.softplus(dt)
        beta = mx.sigmoid(b)

        state = cache.state
        if GATED_DELTA_KERNEL and gated_delta_applies(
            config.linear_key_head_dim, key_heads, heads, config.linear_value_head_dim
        ):
            # The kernel broadcasts the key heads itself and carries the state
            # transposed ([B, Hv, Dv, Dk]), so neither repeat nor the log decay survive.
            scale = 1 / math.sqrt(config.linear_key_head_dim)
            if state is None:
                shape = (1, heads, config.linear_value_head_dim, config.linear_key_head_dim)
                state = mx.zeros(shape, dtype=mx.float32)
            out, state = gated_delta(
                _l2norm(q, scale, dtype=mx.float32),
                _l2norm(k, dtype=mx.float32),
                v.astype(mx.float32),
                mx.exp(g),
                beta.astype(mx.float32),
                state,
            )
            # transformers rounds here (the gated norm reads the rule's output in the
            # model dtype), so the kernel's float32 y does too.
            out = out.astype(v.dtype)
        else:
            q, k = _l2norm(q), _l2norm(k)
            if heads != key_heads:
                # The key heads are broadcast into the value heads *interleaved*
                # (head hv reads hv // (Hv/Hk)), which is what repeat does.
                q = mx.repeat(q, heads // key_heads, axis=2)
                k = mx.repeat(k, heads // key_heads, axis=2)
            if state is None:
                shape = (1, heads, config.linear_key_head_dim, config.linear_value_head_dim)
                state = mx.zeros(shape, dtype=mx.float32)
            out, state = _delta_rule(q, k, v, g, beta, state)
        cache.state = state
        cache.offset += length

        # The gated RMSNorm rounds to the model dtype between the scale and the gate,
        # exactly as Qwen3_5RMSNormGated does; it is the one norm of the family that is
        # not zero-centered.
        lifted = out.astype(mx.float32)
        normed = lifted * mx.rsqrt(
            (lifted * lifted).mean(axis=-1, keepdims=True) + config.rms_norm_eps
        )
        gate = z.reshape(1, length, heads, config.linear_value_head_dim).astype(mx.float32)
        gated = self.norm.weight * normed.astype(out.dtype)
        gated = (gated * gate * mx.sigmoid(gate)).astype(x.dtype)
        return self.out_proj(gated.reshape(1, length, config.value_dim))


@cache
def _mrope_sources(config: Qwen35Config) -> mx.array:
    """Which of T/H/W each rotated dimension reads, as transformers' interleave lays it
    out: frequency `i` belongs to section `i % 3` until that section's quota runs out.
    Frequency `i` drives dims `i` and `i + rope_dims/2` (the half rotation); dims past
    `rope_dims` are not rotated at all, so their section never matters."""
    half = config.rope_dims // 2
    sections = np.zeros(half, dtype=np.int32)
    for section, count in enumerate(config.mrope_section[1:], start=1):
        sections[section : 3 * count : 3] = section
    tail = np.zeros(config.head_dim - config.rope_dims, dtype=np.int32)
    return mx.array(np.concatenate([sections, sections, tail]))


class Qwen35Attention(nn.Module):
    """GQA whose q_proj emits `[query ‖ gate]` per head, with a 256-wide q/k-norm and
    a partial rope over the first quarter of the head."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        queries = config.num_attention_heads * config.head_dim
        key_values = config.num_key_value_heads * config.head_dim
        self.fused_proj = nn.Linear(config.hidden_size, 2 * queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        """Text-only MRoPE is a plain partial rope: the three sections read the same
        position, so the interleave rewrites each frequency with its own value."""
        return mx.fast.rope(
            x, self.config.rope_dims, traditional=False, base=self.config.rope_theta,
            scale=1.0, offset=offset,
        )

    def mrope(self, x: mx.array, positions: mx.array) -> mx.array:
        """Interleaved MRoPE over `[1, L, H, D]`, out in the same layout.

        Viewing `[1, L, H, D]` as `[L, H, 1, D]` is a free reshape (same element order)
        and turns each token into a batch row, which is what lets the rope kernel take
        one offset per token. The interleave then falls out of three calls to that same
        kernel and one select per dimension, without a single new floating-point
        operation — so the text path is identical to `rope` **by construction**, not by
        inspection. A hand-rolled cos/sin would not be: the kernel uses
        `metal::fast::cos` while mlx's elementwise `cos` uses `metal::precise::cos`.
        """
        _, length, heads, dims = x.shape
        rows = x.reshape(length, heads, 1, dims)
        rotated = [
            mx.fast.rope(
                rows, self.config.rope_dims, traditional=False, base=self.config.rope_theta,
                scale=1.0, offset=positions[section],
            )
            for section in range(3)
        ]
        source = _mrope_sources(self.config)
        picked = mx.where(source == 2, rotated[2], mx.where(source == 1, rotated[1], rotated[0]))
        return picked.reshape(1, length, heads, dims)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        config = self.config
        length = x.shape[1]
        queries = config.num_attention_heads * config.head_dim
        key_values = config.num_key_value_heads * config.head_dim
        fused = self.fused_proj(x)
        q_gate, k, v = mx.split(fused, (2 * queries, 2 * queries + key_values), axis=-1)
        q_gate = q_gate.reshape(1, length, config.num_attention_heads, 2 * config.head_dim)
        q, gate = mx.split(q_gate, (config.head_dim,), axis=-1)
        heads = (config.num_key_value_heads, config.head_dim)
        return (
            self.q_norm(q),
            self.k_norm(k.reshape(1, length, *heads)),
            v.reshape(1, length, *heads),
            gate.reshape(1, length, queries),
        )

    def __call__(
        self, x: mx.array, cache: KVCache, positions: mx.array | None = None
    ) -> mx.array:
        config = self.config
        length = x.shape[1]
        queries = config.num_attention_heads * config.head_dim
        offset = cache.offset
        q, k, v, gate = self.split_heads(x)
        if positions is None:
            q, k = self.rope(q.transpose(0, 2, 1, 3), offset), self.rope(
                k.transpose(0, 2, 1, 3), offset
            )
        else:
            q = self.mrope(q, positions).transpose(0, 2, 1, 3)
            k = self.mrope(k, positions).transpose(0, 2, 1, 3)
        keys, values = cache.update_and_fetch(k, v.transpose(0, 2, 1, 3))
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values,
            scale=1 / math.sqrt(config.head_dim),
            mask=None if length == 1 else "causal",
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(1, length, queries)
        return self.o_proj(attended * mx.sigmoid(gate))


def _packed(linear: nn.QuantizedLinear) -> tuple[mx.array, mx.array, mx.array]:
    """A quantized leaf's three tensors, narrowed: `biases` is optional on the type but
    never absent under affine quantization."""
    assert isinstance(linear.biases, mx.array)
    return linear.weight, linear.scales, linear.biases


class Qwen35SharedExpert(nn.Module):
    """Only the shared expert's `down_proj` lives here: its gate and up are row 256 of
    the routed stack, but appending a 257th row to the *down* stack would materialize
    the one tensor that goes mmap'd from the file into the model (5.4 GB at 6 bits)."""

    def __init__(self, config: Qwen35Config, moe: Qwen35MoEParams) -> None:
        super().__init__()
        self.down_proj = nn.Linear(moe.moe_intermediate_size, config.hidden_size, bias=False)


class Qwen35SwitchGLU(nn.Module):
    """gate‖up row-interleaved ([g0,u0,g1,…]) with the shared expert stacked as the
    last row — one gather reads both projections of any slot, routed or shared."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts + 1, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]


class Qwen35MoEInternals(NamedTuple):
    probs: mx.array
    indices: mx.array
    weights: mx.array
    routed: mx.array
    shared: mx.array
    out: mx.array


class Qwen35MoE(nn.Module):
    """256 experts, 8 per token, plus a shared expert scaled by the sigmoid of its own
    logit. That logit is row 256 of the router's matrix, so one gemv produces both."""

    def __init__(self, config: Qwen35Config, moe: Qwen35MoEParams) -> None:
        super().__init__()
        self.experts = moe.num_experts
        self.k = moe.num_experts_per_tok
        self.hidden = config.hidden_size
        self.gate = nn.Linear(config.hidden_size, moe.num_experts + 1, bias=False)
        self.switch_mlp = Qwen35SwitchGLU(
            moe.num_experts, config.hidden_size, moe.moe_intermediate_size
        )
        self.shared_expert = Qwen35SharedExpert(config, moe)

    def route(self, logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """The softmax spans all 256 experts, so the kept weights depend on the dropped
        ones; renormalizing after the cut matches transformers' sorted top-k."""
        probs = softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.experts - self.k, axis=-1)[..., -self.k :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        return probs, chosen, weights / weights.sum(axis=-1, keepdims=True)

    def _routed(self, x: mx.array, chosen: mx.array) -> mx.array:
        """[1, T, k, hidden] — one row per kept expert, unweighted."""
        switch = self.switch_mlp
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                act = switch.activate(switch.gate_up_proj(tokens, experts, sorted_indices=True))
                return switch.down_proj(act, experts, sorted_indices=True)

            return sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        tokens = mx.expand_dims(x, (-2, -3))
        act = switch.activate(switch.gate_up_proj(tokens, chosen))
        return switch.down_proj(act, chosen).squeeze(-2)

    def _shared(self, x: mx.array) -> mx.array:
        """The shared expert reads slot 256 of the gate‖up stack and its own down."""
        switch = self.switch_mlp
        slot = mx.full((*x.shape[:-1], 1), self.experts, dtype=mx.uint32)
        tokens = mx.expand_dims(x, (-2, -3))
        act = switch.activate(switch.gate_up_proj(tokens, slot, sorted_indices=True))
        return self.shared_expert.down_proj(act.squeeze(-2).squeeze(-2))

    def internals(self, x: mx.array) -> Qwen35MoEInternals:
        logits = self.gate(x)
        probs, chosen, weights = self.route(logits[..., : self.experts])
        routed = (self._routed(x, chosen) * mx.expand_dims(weights, -1)).sum(axis=-2)
        shared = mx.sigmoid(logits[..., self.experts :]) * self._shared(x)
        return Qwen35MoEInternals(probs, chosen, weights, routed, shared, routed + shared)

    def fused_step_applies(self) -> bool:
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        shared = self.shared_expert.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and isinstance(shared, nn.QuantizedLinear)
            and (shared.bits, shared.group_size) == (down.bits, down.group_size)
            and moe_gemv_applies(
                self.hidden, self.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(self.experts, self.k)
        )

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array:
        """Four dispatches for the whole sparse block at T=1: the shared expert rides
        along as the ninth slot, so routing, silu, weighting, the expert sum and the
        residual all stay inside the two gemv kernels."""
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        shared = self.shared_expert.down_proj
        assert isinstance(gate_up, QuantizedSwitchLinear)
        assert isinstance(down, QuantizedSwitchLinear)
        assert isinstance(shared, nn.QuantizedLinear)
        chosen, weights = softmax_topk(self.gate(x).reshape(-1), self.k, shared=True)
        act = moe_gate_up_act(
            x.reshape(-1), gate_up.weight, gate_up.scales, gate_up.biases, chosen,
            group_size=gate_up.group_size, bits=gate_up.bits,
        )
        return moe_down_combine(
            act.reshape(-1), down.weight, down.scales, down.biases, chosen, weights,
            residual.reshape(-1), group_size=down.group_size, bits=down.bits,
            shared=_packed(shared),
        ).reshape(1, 1, self.hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.internals(x).out


class Qwen35Block(nn.Module):
    """One mixer of the two, then the MLP. The MoE variant (35B-A3B) swaps `mlp` for a
    sparse block chosen off the config — the rest of the block is unchanged."""

    def __init__(self, config: Qwen35Config, layer: int) -> None:
        super().__init__()
        self.attends = config.layer_types[layer] == "full_attention"
        if self.attends:
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35DeltaNet(config)
        self.mlp = (
            SwiGLU(config.hidden_size, config.intermediate_size)
            if config.moe is None
            else Qwen35MoE(config, config.moe)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps
        self._delta_dims = (
            config.linear_key_head_dim,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )
        self._tail = self._build_tail()
        self._traced = _trace_key(self)
        # `inputs=self.state` keeps the weights implicit inputs of the trace instead of
        # baked constants: a swapped tensor (quantize-on-load, a mutation test) is read
        # on the next call, no retrace needed.
        if self.attends:
            self._tail_step = mx.compile(self._build_tail(), inputs=self.state)
        else:
            self._step = mx.compile(self._build_step(), inputs=self.state)

    def delta_fits(self) -> bool:
        return gated_delta_applies(*self._delta_dims)

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
        post = self.post_attention_layernorm

        def tail(x: mx.array, projected: mx.array) -> mx.array:
            if ADD_RMS_NORM_KERNEL and add_rms_norm_applies(self.hidden):
                attended, normed = add_rms_norm(x, projected, post.weight, self.eps)
            else:
                attended = x + projected
                normed = post(attended)
            if isinstance(mlp, Qwen35MoE) and mlp.fused_step_applies():
                return mlp.fused_step(normed, attended)
            return attended + mlp(normed)

        return tail

    def _build_step(self) -> _DeltaStep:
        """The one-token DeltaNet step as a single trace: norm → fused projection →
        unrolled conv → decay → the gated_delta kernel → gated norm → out projection →
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
            out, state_out = gated_delta(
                _l2norm(q.reshape(key_shape), scale, dtype=mx.float32),
                _l2norm(k.reshape(key_shape), dtype=mx.float32),
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
        # The traces bake the kernel bindings and the A/B flags in; rebuild when any
        # changes so the mutation tests (and any monkeypatch) reach inside them.
        key = _trace_key(self)
        if self._traced != key:
            self._traced = key
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
        if x.shape[1] == 1 and COMPILED_STEP:
            if self.attends:
                # Projections stay eager: mx.fast.rope takes the offset as an op
                # attribute and a trace would freeze it at the first token.
                assert isinstance(cache, KVCache)
                projected = self.self_attn(self.input_layernorm(x), cache, positions)
                self._rebuild()
                return self._tail_step(x, projected)
            if GATED_DELTA_KERNEL and self.delta_fits():
                assert isinstance(cache, DeltaCache)
                return self._compiled_delta(x, cache)
        projected = self.mix(x, cache, positions)
        if x.shape[1] == 1:
            return self._tail(x, projected)
        attended = x + projected
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen35Trunk(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen35Block(config, layer) for layer in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen35Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Qwen35(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen35Trunk(config)
        if config.vision is not None:
            self.visual = Qwen35Vision(config.vision)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | DeltaCache]:
        return [
            KVCache() if kind == "full_attention" else DeltaCache()
            for kind in self.config.layer_types
        ]

    def activations(
        self,
        ids: mx.array,
        cache: list[KVCache | DeltaCache] | None = None,
        *,
        positions: mx.array | None = None,
        embeddings: mx.array | None = None,
    ) -> Qwen35Activations:
        """`embeddings` replaces the token lookup (an image's rows are already spliced
        in) and `positions` is the `[3, L]` MRoPE clock; both default to the text path."""
        cache = cache if cache is not None else self.make_cache()
        x = embeddings if embeddings is not None else self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache, positions)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Qwen35Activations(embedded, blocks, normed, logits)

    def __call__(
        self,
        ids: mx.array,
        cache: list[KVCache | DeltaCache] | None = None,
        *,
        positions: mx.array | None = None,
        embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.activations(ids, cache, positions=positions, embeddings=embeddings).logits


class MultimodalPrompt(NamedTuple):
    """A prompt whose image rows are already spliced in, with the clock they run on.

    `delta` is the trap. After an image the position resumes at
    `pos + max(grid_h, grid_w)/merge`, not at `pos + image_tokens`: a 22x28 image spends
    154 cache rows and 14 positions, so everything after it runs 140 behind the row count
    forever. Removing it does not break greedy — 18 of the 24 layers are DeltaNet and
    carry position through the recurrence, and the rope rotates 64 of 256 dims — the
    model just writes fluent, on-topic, wrong text. What catches it is teacher forcing.
    """

    ids: list[int]
    embeddings: mx.array
    positions: mx.array
    delta: int


def multimodal_prompt(
    model: Qwen35, ids: Sequence[int], images: Sequence[tuple[mx.array, Grid]]
) -> MultimodalPrompt:
    """Runs the tower over each image, scatters its rows onto the placeholder run, and
    builds the 3-D clock the trunk reads."""
    config = model.config
    vision = config.vision
    if vision is None:
        raise ValueError("this checkpoint carries no vision tower")
    tower = model.visual
    assert isinstance(tower, Qwen35Vision)

    tokens = model.model.embed_tokens(mx.array(list(ids)))[None]
    merge = vision.spatial_merge_size
    rows = mx.concatenate(
        [tower(patches.astype(tokens.dtype), grid) for patches, grid in images], axis=0
    ) if images else None
    if rows is not None:
        placeholder = mx.array(list(ids)) == config.image_token_id
        indices = mx.array(np.flatnonzero(np.array(placeholder)))
        if indices.shape[0] != rows.shape[0]:
            raise ValueError(
                f"{rows.shape[0]} image rows for {indices.shape[0]} placeholder tokens"
            )
        tokens[0, indices] = rows.astype(tokens.dtype)

    positions, delta = multimodal_positions(
        list(ids), [grid for _, grid in images], config.image_token_id, merge
    )
    return MultimodalPrompt(list(ids), tokens, mx.array(positions), delta)


def decode_clock(prompt: MultimodalPrompt, row: int) -> mx.array:
    """The 3-D rotation for the token sitting at cache row `row`: transformers'
    `arange(past) + rope_delta`, one row at a time."""
    return mx.full((3, 1), row + prompt.delta, dtype=mx.int32)


def stream_multimodal_ids(
    model: Qwen35,
    prompt: MultimodalPrompt,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: int | None = None,
) -> Iterator[int]:
    """The chassis' lazy loop with the 3-D clock threaded through: prefill reads the
    prompt's own positions, and every step after it rotates by `row + delta`, which is
    what transformers rebuilds as `arange(past) + rope_delta`."""
    cache = model.make_cache()
    row = len(prompt.ids)

    def step(ids: mx.array, positions: mx.array, embeddings: mx.array | None) -> mx.array:
        logits = model(ids, cache, positions=positions, embeddings=embeddings)
        return sampler(logits[:, -1, :])[0]

    def advance(y: mx.array) -> mx.array:
        nonlocal row
        positions = decode_clock(prompt, row)
        out = step(y[None, None], positions, None)
        row += 1
        return out

    y = step(mx.array(prompt.ids)[None], prompt.positions, prompt.embeddings)
    mx.async_eval(y)
    for _ in range(max_tokens):
        next_y = advance(y)
        mx.async_eval(next_y)
        token = y.item()
        assert isinstance(token, int)
        if token == stop:
            return
        yield token
        y = next_y

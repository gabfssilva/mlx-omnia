"""GPT-OSS: MXFP4 experts, attention sinks, alternating sliding/full attention, YaRN rope.

Four things no other ported model has:

- the experts ship **already MXFP4-packed** (group 32, e2m1 + e8m0 scale, *no* biases),
  so they never go through `nn.quantize`: they are bare leaves read in place by
  `gather_qmm(mode="mxfp4")`. The load only reinterprets `_blocks` (uint8) as uint32 —
  a pure view — and each expert carries a bias `SwitchLinear` has no slot for, added
  after the gather. gate‖up already comes interleaved row by row, the layout the
  decode kernel wants, so that fusion is free;
- **attention sinks**: one learned logit per head inside the softmax denominator.
  mlx's fast SDPA takes them natively (`sinks=`), so this is the reference kernel;
- **sliding(128)/full alternating**, expressed as a *mask* over a full cache, not
  eviction: a masked key contributes nothing, identical to the reference's rotating
  cache;
- **YaRN rope** (theta 150000, factor 32): the NTK-by-parts table and the 1.34657
  `mscale` that pre-scales q/k, computed with the same ops as mlx-lm.

Routing is top-k over the **raw** router logits (the router has a bias) and then a
softmax over just those k — not a renormalized softmax over all 32, which rounds
differently. The config dataclass, the TypedDict and `load_gpt_oss` live here until
the integration stage relocates them into config.py/checkpoint.py.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.mxfp4_moe_gemv import (
    mxfp4_down_combine,
    mxfp4_gate_up_act,
    mxfp4_moe_applies,
)
from sideros.core.kernels.sink_attention import sink_attention, sink_attention_applies
from sideros.core.layers import SORTED_GATHER_MIN, sorted_gather, split_qkv
from sideros.core.mxcompat import softmax

_GROUP_SIZE = 32
_BITS = 4
_MODE = "mxfp4"

# A/B switches for the bench stage: set either to False (module attribute, no code edit)
# and the corresponding op path — which is also the parity reference — runs instead.
USE_MXFP4_MOE_GEMV = True
# Measured on the M5 Max (gpt-oss-20b, 435-token prompt, interleaved A/B, median of 5)
# the sink kernel costs ~1.4% of decode against mx.fast.scaled_dot_product_attention
# (107.7 vs 109.3 and 107.9 vs 109.4 in two batteries) — but it stays ON: with it off,
# `test_greedy_matches_mlxlm` diverges from the fixture at index 209. 1.4% does not buy
# a broken parity gate. Re-measure at long context, where MODELS.md expects it to win.
USE_SINK_ATTENTION = True


@dataclass(frozen=True)
class GPTOSSRoPEScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float


@dataclass(frozen=True)
class GPTOSSConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    sliding_window: int
    swiglu_limit: float
    layer_types: tuple[str, ...]
    rope_scaling: GPTOSSRoPEScaling


def yarn_rope(head_dim: int, base: float, scaling: GPTOSSRoPEScaling) -> tuple[mx.array, float]:
    """The NTK-by-parts frequency table and the length scale applied to q/k before the
    rotation. Same op order as mlx-lm's `YarnRoPE`, so the table matches bit for bit."""
    factor = scaling.factor
    original = scaling.original_max_position_embeddings

    def correction(rotations: float) -> float:
        return (head_dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(scaling.beta_fast)), 0)
    high = min(math.ceil(correction(scaling.beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001

    extra = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    inter = factor * extra
    ramp = mx.clip((mx.arange(head_dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
    # yarn_get_mscale(factor, mscale=1) / yarn_get_mscale(factor, mscale_all_dim=0); the
    # denominator is 1 whenever the checkpoint omits mscale_all_dim, as gpt-oss does.
    mscale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return freqs, mscale


class GPTOSSExperts(nn.Module):
    """The 32 experts as MXFP4 stacks read in place. Nothing is quantized at load: the
    leaves are the checkpoint's packed tensors, and `mode="mxfp4"` has no biases (the
    per-expert `*_bias` here is the affine bias of the projection, not the quantizer's)."""

    def __init__(self, config: GPTOSSConfig) -> None:
        super().__init__()
        experts = config.num_local_experts
        hidden, inner = config.hidden_size, config.intermediate_size
        self.inner = inner
        self.limit = config.swiglu_limit
        self.gate_up_proj_weight = mx.zeros((experts, 2 * inner, hidden // 8), dtype=mx.uint32)
        self.gate_up_proj_scales = mx.zeros(
            (experts, 2 * inner, hidden // _GROUP_SIZE), dtype=mx.uint8
        )
        self.gate_up_proj_bias = mx.zeros((experts, 2 * inner), dtype=mx.bfloat16)
        self.down_proj_weight = mx.zeros((experts, hidden, inner // 8), dtype=mx.uint32)
        self.down_proj_scales = mx.zeros((experts, hidden, inner // _GROUP_SIZE), dtype=mx.uint8)
        self.down_proj_bias = mx.zeros((experts, hidden), dtype=mx.bfloat16)

    def _gather(
        self,
        x: mx.array,
        weight: mx.array,
        scales: mx.array,
        bias: mx.array,
        indices: mx.array,
        *,
        sorted_indices: bool,
    ) -> mx.array:
        projected = mx.gather_qmm(
            x, weight, scales, None, rhs_indices=indices, transpose=True,
            group_size=_GROUP_SIZE, bits=_BITS, mode=_MODE, sorted_indices=sorted_indices,
        )
        return projected + mx.expand_dims(bias[indices], axis=-2)

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False
    ) -> mx.array:
        """`x` is [1, T, 1, 1, hidden] and `indices` [1, T, k]: one gemv per routed pair.
        Under the prefill reorder the pair is flattened instead — [N, 1, hidden] against
        [N] — and `sorted_indices` tells the gather each expert's rows are contiguous."""
        fused = self._gather(
            x, self.gate_up_proj_weight, self.gate_up_proj_scales, self.gate_up_proj_bias,
            indices, sorted_indices=sorted_indices,
        )
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gate = mx.clip(pairs[..., 0], None, self.limit)
        up = mx.clip(pairs[..., 1], -self.limit, self.limit)
        act = gate * mx.sigmoid(1.702 * gate) * (up + 1)
        return self._gather(
            act, self.down_proj_weight, self.down_proj_scales, self.down_proj_bias,
            indices, sorted_indices=sorted_indices,
        )


class GPTOSSMLP(nn.Module):
    def __init__(self, config: GPTOSSConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.experts = GPTOSSExperts(config)
        self.k = config.num_experts_per_tok
        self.split = config.num_local_experts - self.k
        self.hidden = config.hidden_size
        self.fusable = mxfp4_moe_applies(config.hidden_size, config.intermediate_size)

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

    def fused_step_applies(self, x: mx.array) -> bool:
        """T=1 only, and only for the MXFP4 layout the kernel is written for. Routing
        stays outside: the pick is a top-k over the raw logits and moving it in-kernel
        flips near-ties."""
        return USE_MXFP4_MOE_GEMV and self.fusable and x.shape[1] == 1

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array:
        """The whole routed MLP plus the residual in two dispatches."""
        experts = self.experts
        chosen, weights = self.route(x)
        indices = chosen.reshape(-1).astype(mx.uint32)
        act = mxfp4_gate_up_act(
            x.reshape(-1),
            experts.gate_up_proj_weight,
            experts.gate_up_proj_scales,
            experts.gate_up_proj_bias,
            indices,
            limit=experts.limit,
        )
        return mxfp4_down_combine(
            act,
            experts.down_proj_weight,
            experts.down_proj_scales,
            experts.down_proj_bias,
            indices,
            weights.reshape(-1),
            residual.reshape(-1),
        ).reshape(1, 1, self.hidden)


class GPTOSSAttention(nn.Module):
    """GQA with bias on every projection, no q/k norm, and a per-head sink logit that
    sits in the softmax denominator. q‖k‖v is one leaf, concatenated at load."""

    def __init__(self, config: GPTOSSConfig, freqs: mx.array, mscale: float) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, queries + 2 * key_values, bias=True)
        self.o_proj = nn.Linear(queries, config.hidden_size, bias=True)
        self.sinks = mx.zeros((self.heads,))
        # Leading underscore: not a parameter, so the strict load does not demand it.
        self._freqs = freqs
        self._mscale = mscale

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        scaled = x * self._mscale if self._mscale != 1.0 else x
        return mx.fast.rope(
            scaled, self.head_dim, traditional=False, base=None, scale=1.0, offset=offset,
            freqs=self._freqs,
        )

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        queries = self._rope(q, offset)
        keys, values = cache.update_and_fetch(self._rope(k, offset), v)
        sinks = self.sinks
        if (
            USE_SINK_ATTENTION
            and not isinstance(mask, str)
            and sinks.dtype == queries.dtype
            and sink_attention_applies(queries, keys)
        ):
            attended = sink_attention(queries, keys, values, sinks, mask, self.scale)
        else:
            attended = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask, sinks=sinks
            )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))


class GPTOSSBlock(nn.Module):
    def __init__(self, config: GPTOSSConfig, freqs: mx.array, mscale: float) -> None:
        super().__init__()
        self.self_attn = GPTOSSAttention(config, freqs, mscale)
        self.mlp = GPTOSSMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        normed = self.post_attention_layernorm(attended)
        if self.mlp.fused_step_applies(normed):
            return self.mlp.fused_step(normed, attended)
        return attended + self.mlp(normed)


class GPTOSSTrunk(nn.Module):
    def __init__(self, config: GPTOSSConfig, freqs: mx.array, mscale: float) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GPTOSSBlock(config, freqs, mscale) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GPTOSSActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class GPTOSS(nn.Module):
    def __init__(self, config: GPTOSSConfig) -> None:
        super().__init__()
        self.config = config
        freqs, mscale = yarn_rope(config.head_dim, config.rope_theta, config.rope_scaling)
        mx.eval(freqs)
        self.model = GPTOSSTrunk(config, freqs, mscale)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        """One full cache per layer: the sliding layers keep every key and mask it,
        which is what an evicting cache would have dropped."""
        return [KVCache() for _ in self.model.layers]

    def _sliding_mask(self, length: int, offset: int) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + window`, built only where it is
        not already something cheaper. No key is old enough for the window to cut while
        `offset + length <= window`, so the band *is* the causal mask there — and at T=1
        the single row is causal by construction, leaving `columns > offset - window`."""
        window = self.config.sliding_window
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> GPTOSSActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if "sliding_attention" in self.config.layer_types:
            sliding = self._sliding_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            x = block(x, full if kind == "full_attention" else sliding, layer_cache)
            blocks.append(x)
        return GPTOSSActivations(blocks, self.lm_head(self.model.norm(x)))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


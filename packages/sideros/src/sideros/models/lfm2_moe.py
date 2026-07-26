"""LFM2.5-MoE: a hybrid trunk of gated short convs and GQA, with sigmoid-routed experts.

Property names are the checkpoint's. 18 of the 24 layers mix with a depthwise causal
conv of kernel 3 (cached as a 2-row window, not keys and values); the other 6 are GQA
with q/k-norm and rope theta 5e6. Past the first two dense layers the MLP is 32 experts,
4 per token, selected by `sigmoid(logits) + expert_bias` in float32 and weighted by the
bias-free score.

The conv is unrolled into `kernel` shifted taps accumulated in float32 — bit-exact with
`conv1d`, and the only form whose one-token step is a handful of elementwise ops. The
conv window cannot be rewound; the attention layers keep their KV history and trim
normally.

Config dataclass, TypedDict and loader live here for now; the integration stage moves
them into `config.py`/`checkpoint.py`.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import ConvCache, KVCache
from sideros.core.kernels.conv_mix import conv_mix, conv_mix_applies
from sideros.core.kernels.moe_gemv_dense import (
    moe_dense_down,
    moe_dense_gate_up,
    moe_gemv_dense_applies,
    moe_route_sigmoid,
)
from sideros.core.layers import SORTED_GATHER_MIN, sorted_gather, split_qkv
from sideros.models.qwen3_moe import SwitchLinear

# A/B switches for the bench: set either to False to fall back to the op path, which
# stays the parity reference. The predicates read them on every call.
CONV_MIX_FUSED = True
MOE_GEMV_DENSE_FUSED = True
# The routing kernel scores from a float32 dot where the op chain rounds the gemv to T.
# In bfloat16 that flips the selection on genuine near-ties (measured: one layer of 22 on
# a synthetic prompt, the 4th and 5th experts 6e-5 apart in a selector quantized to 3e-4).
# Set False to keep the fused gemvs while the op path decides the experts.
MOE_ROUTE_FUSED = True


@dataclass(frozen=True)
class LFM2MoEConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_dense_layers: int
    norm_topk_prob: bool
    use_expert_bias: bool
    routed_scaling_factor: float
    norm_eps: float
    conv_bias: bool
    conv_l_cache: int
    layer_types: tuple[str, ...]
    tie_word_embeddings: bool
    rope_theta: float
    vocab_size: int

    @property
    def head_dim(self) -> int:
        """Implicit, unlike Qwen3: always hidden / heads."""
        return self.hidden_size // self.num_attention_heads


class ShortConvWeight(nn.Module):
    """The depthwise taps, `[hidden, 1, kernel]`. Never convolved as such — the kernel is
    unrolled into shifted products, so this leaf only carries the checkpoint's weight."""

    def __init__(self, hidden: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.weight = mx.zeros((hidden, 1, kernel))
        if bias:
            self.bias = mx.zeros((hidden,))


class LFM2Conv(nn.Module):
    """in_proj splits into B, C, x; a causal depthwise conv runs over B·x; C gates it."""

    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.hidden = hidden
        self.kernel = config.conv_l_cache
        self.conv = ShortConvWeight(hidden, self.kernel, config.conv_bias)
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.conv_bias)
        self.out_proj = nn.Linear(hidden, hidden, bias=config.conv_bias)

    def fused_step_applies(self) -> bool:
        return (
            CONV_MIX_FUSED
            and type(self.in_proj) is nn.Linear
            and "bias" not in self.in_proj
            and "bias" not in self.conv
            and conv_mix_applies(self.hidden, self.kernel, has_bias=False)
        )

    def fused_step(self, x: mx.array, cache: ConvCache) -> mx.array:
        """in_proj, B·x, the three taps and C's gate in one dispatch; out_proj follows."""
        window = cache.window
        if window is None:
            window = mx.zeros((1, self.kernel - 1, self.hidden), dtype=x.dtype)
        weight = self.in_proj.weight
        assert isinstance(weight, mx.array)
        gated, slid = conv_mix(
            x.reshape(-1), weight, self.conv.weight.reshape(-1), window.reshape(2, self.hidden)
        )
        cache.window = slid[None]
        return self.out_proj(gated.reshape(1, 1, self.hidden))

    def __call__(self, x: mx.array, cache: ConvCache) -> mx.array:
        length = x.shape[1]
        # `offset` counts tokens seen, as in every other layer type: the conv keeps only a
        # window, but a layer that never advances breaks the invariant the trunk shares.
        cache.offset += length
        if length == 1 and self.fused_step_applies():
            return self.fused_step(x, cache)
        b, c, v = mx.split(self.in_proj(x), 3, axis=-1)
        bx = b * v
        window = cache.window
        if window is None:
            window = mx.zeros((1, self.kernel - 1, self.hidden), dtype=bx.dtype)
        padded = mx.concatenate([window, bx], axis=1)

        # Accumulated in float32 and rounded once, like the conv kernel: per-tap bfloat16
        # rounding is what would diverge from the reference.
        lifted = padded.astype(mx.float32)
        taps = self.conv.weight[:, 0, :]
        conv = lifted[:, :length, :] * taps[:, 0]
        for j in range(1, self.kernel):
            conv = conv + lifted[:, j : j + length, :] * taps[:, j]
        if "bias" in self.conv:
            conv = conv + self.conv.bias

        cache.window = padded[:, length:, :]
        return self.out_proj(c * conv.astype(bx.dtype))


class LFM2Attention(nn.Module):
    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, queries + 2 * key_values, bias=False)
        self.out_proj = nn.Linear(queries, config.hidden_size, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """q/k rms-normed per head between the projection and the rotation, as in Qwen3."""
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        return self.q_layernorm(q), self.k_layernorm(k), v

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = self.split_heads(x)
        queries = self.rope(q, offset)
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.out_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))


class LFM2DenseMLP(nn.Module):
    """w1‖w3 concatenated on the output axis at load; w2 projects back."""

    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.w13 = nn.Linear(hidden, 2 * inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)
        self.inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.w13(x)
        gated = fused[..., : self.inner]
        return self.w2(gated * mx.sigmoid(gated) * fused[..., self.inner :])


class LFM2Experts(nn.Module):
    """Gate and up block-concatenated ([w1 ‖ w3] on the output axis): read by slice."""

    def __init__(self, count: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.w13 = SwitchLinear(count, hidden, 2 * inner)
        self.w2 = SwitchLinear(count, inner, hidden)
        self.inner = inner

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        fused = self.w13(tokens, indices, sorted_indices=sorted_indices)
        gated = fused[..., : self.inner]
        activated = gated * mx.sigmoid(gated) * fused[..., self.inner :]
        return self.w2(activated, indices, sorted_indices=sorted_indices)


class LFM2SparseMLP(nn.Module):
    """32 experts, 4 per token, sigmoid-routed: the float32 `expert_bias` shifts which
    experts win but never their weights, which come from the bias-free score."""

    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = LFM2Experts(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        if config.use_expert_bias:
            self.expert_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob
        self.scaling = config.routed_scaling_factor

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.sigmoid(self.gate(x))
        selector = scores.astype(mx.float32) + self.expert_bias if "expert_bias" in self else scores
        chosen = mx.argpartition(selector, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-6)
        return chosen, weights * self.scaling

    def fused_step_applies(self) -> bool:
        return MOE_GEMV_DENSE_FUSED and moe_gemv_dense_applies(
            self.hidden, self.experts.inner, self.experts.w13.weight.shape[0], self.k
        )

    def fused_step(self, x: mx.array) -> mx.array:
        """Routing, both gemvs, silu and the routing weight in three dispatches."""
        row = x.reshape(-1)
        gate = self.gate.weight
        assert isinstance(gate, mx.array)
        if MOE_ROUTE_FUSED:
            bias = (
                self.expert_bias
                if "expert_bias" in self
                else mx.zeros(gate.shape[0], mx.float32)
            )
            assert isinstance(bias, mx.array)
            chosen, weights = moe_route_sigmoid(
                row,
                gate,
                bias,
                mx.array(self.scaling, mx.float32),
                self.k,
                normalized=self.norm_topk,
            )
        else:
            selected, scores = self.route(x)
            chosen, weights = selected.reshape(-1).astype(mx.uint32), scores.reshape(-1)
        gate_up = moe_dense_gate_up(row, self.experts.w13.weight, chosen)
        routed = moe_dense_down(gate_up, self.experts.w2.weight, chosen, weights)
        return routed.sum(axis=0).reshape(x.shape)

    def __call__(self, x: mx.array) -> mx.array:
        if x.size == self.hidden and self.fused_step_applies():
            return self.fused_step(x)
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.experts(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.experts(tokens, chosen, sorted_indices=False).squeeze(-2)
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)


class LFM2Block(nn.Module):
    def __init__(self, config: LFM2MoEConfig, layer: int) -> None:
        super().__init__()
        self.attends = config.layer_types[layer] == "full_attention"
        if self.attends:
            self.self_attn = LFM2Attention(config)
        else:
            self.conv = LFM2Conv(config)
        self.feed_forward: LFM2DenseMLP | LFM2SparseMLP = (
            LFM2DenseMLP(config.hidden_size, config.intermediate_size)
            if layer < config.num_dense_layers
            else LFM2SparseMLP(config)
        )
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x: mx.array, cache: KVCache | ConvCache) -> mx.array:
        normed = self.operator_norm(x)
        # A block has one mixer or the other; mlx.nn.Module's __getattr__ is untyped, so
        # the branch is narrowed here.
        if self.attends:
            mixer = self.self_attn
            assert isinstance(mixer, LFM2Attention) and isinstance(cache, KVCache)
            attended = x + mixer(normed, cache)
        else:
            mixer = self.conv
            assert isinstance(mixer, LFM2Conv) and isinstance(cache, ConvCache)
            attended = x + mixer(normed, cache)
        return attended + self.feed_forward(self.ffn_norm(attended))


class LFM2MoETrunk(nn.Module):
    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LFM2Block(config, layer) for layer in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class LFM2MoEActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class LFM2MoE(nn.Module):
    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LFM2MoETrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | ConvCache]:
        return [
            KVCache() if kind == "full_attention" else ConvCache()
            for kind in self.config.layer_types
        ]

    def activations(
        self, ids: mx.array, cache: list[KVCache | ConvCache] | None = None
    ) -> LFM2MoEActivations:
        cache = cache if cache is not None else self.make_cache()
        embeddings = self.model.embed_tokens(ids)
        x = embeddings
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.embedding_norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LFM2MoEActivations(embeddings, blocks, normed, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache | ConvCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


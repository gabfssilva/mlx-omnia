"""Qwen3-MoE: Qwen3 attention with 128 experts, 8 per token, on every layer.

Property names are the checkpoint's (mlx export layout: experts stacked as
`mlp.switch_mlp.*`). qkv is fused on the output axis and gate‖up row-interleaved
at load, dict-side. The T=1 step routes through the fused kernels when the shapes
tile (`moe_gemv_applies`); prefill gathers sorted by expert — a pure reorder.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.core.cache import KVCache
from sideros.core.kernels.add_rms_norm import add_rms_norm, add_rms_norm_applies
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import softmax_topk, softmax_topk_applies
from sideros.core.kernels.rope_epilogue import rope_epilogue, rope_epilogue_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwitchLinear,
    sorted_gather,
    split_qkv,
)
from sideros.core.mxcompat import softmax
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

# Both default off: measured on the M5 Max (Qwen3-30B-A3B-4bit, 435-token prompt,
# interleaved A/B, median of 5) each kernel LOSES decode here — 150.6 (both on) vs
# 159.9 tok/s (both off), reproduced in a second battery (142.7 vs 152.4). The same
# rope_epilogue wins on the dense 0.6B (310.0 vs 303.3), so the loss is specific to
# this step, not to the kernel. Kept wired for re-measurement.
ROPE_EPILOGUE_KERNEL = False
ADD_RMS_NORM_KERNEL = False


@dataclass(frozen=True)
class Qwen3MoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    eos_token_id: tuple[int, ...]


class SwitchGLU(nn.Module):
    """Gate and up fused row-interleaved ([g0,u0,g1,u1,…]) at load: one gather reads both."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class Qwen3MoEMLP(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The softmax spans all experts, so the kept weights depend on the dropped
        ones; renormalizing after the cut matches transformers' sorted topk."""
        probs = softmax(self.gate(x), axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.eps = config.rms_norm_eps
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def _step_heads(self, x: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        """T=1: q/k norm + rotation in one dispatch, v a free slice of the projection."""
        fused = self.qkv_proj(x)
        queries, keys = rope_epilogue(
            fused,
            query_heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm.weight,
            k_norm=self.k_norm.weight,
            offset=offset,
            base=self.rope_theta,
            eps=self.eps,
        )
        values = fused[..., (self.heads + self.kv_heads) * self.head_dim :]
        return (
            queries.reshape(1, self.heads, 1, self.head_dim),
            keys.reshape(1, self.kv_heads, 1, self.head_dim),
            values.reshape(1, self.kv_heads, 1, self.head_dim),
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        if ROPE_EPILOGUE_KERNEL and length == 1 and rope_epilogue_applies(self.head_dim):
            queries, rotated, v = self._step_heads(x, offset)
        else:
            q, k, v = split_qkv(
                self.qkv_proj(x),
                heads=self.heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
            )
            queries = self._rope(self.q_norm(q), offset)
            rotated = self._rope(self.k_norm(k), offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))


class Qwen3MoEBlock(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MoEMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok

    def _fused_step_applies(self) -> bool:
        mlp = self.mlp
        gate_up = mlp.switch_mlp.gate_up_proj
        down = mlp.switch_mlp.down_proj
        return (
            mlp.norm_topk
            and isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            # The gemv kernels read an affine bias per group; MXFP carries none.
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                self.hidden, mlp.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(mlp.split + self.k, self.k)
        )

    def _join(self, x: mx.array, cache: KVCache) -> tuple[mx.array, mx.array]:
        """(x + attention, its post-norm). At T=1 one kernel does the pair."""
        if ADD_RMS_NORM_KERNEL and x.shape[1] == 1 and add_rms_norm_applies(self.hidden):
            normed = mx.fast.rms_norm(x, self.input_layernorm.weight, self.eps)
            return add_rms_norm(
                x,
                self.self_attn(normed, cache),
                self.post_attention_layernorm.weight,
                self.eps,
            )
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended, self.post_attention_layernorm(attended)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended, h = self._join(x, cache)
        if x.shape[1] == 1 and self._fused_step_applies():
            gate_up = self.mlp.switch_mlp.gate_up_proj
            down = self.mlp.switch_mlp.down_proj
            assert isinstance(gate_up, QuantizedSwitchLinear)
            assert isinstance(down, QuantizedSwitchLinear)
            assert gate_up.biases is not None and down.biases is not None
            chosen, weights = softmax_topk(self.mlp.gate(h).reshape(-1), self.k)
            act = moe_gate_up_act(
                h.reshape(-1),
                gate_up.weight,
                gate_up.scales,
                gate_up.biases,
                chosen,
                group_size=gate_up.group_size,
                bits=gate_up.bits,
            )
            return moe_down_combine(
                act.reshape(-1),
                down.weight,
                down.scales,
                down.biases,
                chosen,
                weights,
                attended.reshape(-1),
                group_size=down.group_size,
                bits=down.bits,
            ).reshape(1, 1, self.hidden)

        return attended + self.mlp(h)


class Qwen3MoETrunk(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3MoEBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3MoEActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Qwen3MoE(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoETrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Qwen3MoEActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Qwen3MoEActivations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    eos_token_id: int | list[int]


def _config(path: Path) -> Qwen3MoEConfig:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "qwen3_moe":
        raise ValueError(f"expected model_type qwen3_moe, got {raw['model_type']!r}")
    eos = raw["eos_token_id"]
    return Qwen3MoEConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        norm_topk_prob=raw["norm_topk_prob"],
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _weights(
    directory: Path, config: Qwen3MoEConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: interleave_gate_up(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: Qwen3MoE) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint((
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ), _config, Qwen3MoE, _weights, _composite)

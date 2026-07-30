"""BitNet b1.58: a small Llama trunk whose every projection is a ternary ``BitLinear``.

Authoritative semantics: transformers' ``modeling_bitnet.py`` (the BitNet team's own
port). The trunk is Llama — GQA, RoPE, RMSNorm, pre-norm residuals, tied lm_head — with
two deltas: an ``attn_sub_norm`` RMSNorm on the attention output before ``o_proj`` and
an ``ffn_sub_norm`` RMSNorm on the gated MLP product before ``down_proj``, and a
``relu2`` (``relu(x)**2``) activation in place of silu. The lm_head/embed stay dense.

The projections are born ternary: ``weight`` is uint8 packed 4-per-byte along the
output axis with a single per-tensor scalar ``weight_scale`` (no scales/biases, so the
affine ``nn.quantize`` path is a no-op here — the tree never goes through it). The
ternary matmul runs in the ``bitlinear`` Metal kernel; the per-token int8 activation
fake-quant transformers' ``AutoBitLinear`` applies runs in the leaf, before the
dispatch, in fp32. No qkv or gate‖up fusion: each projection carries its own
``weight_scale``, so fusing would collapse independent scales. RoPE is split-half
(``traditional=True``), matching transformers' ``rotate_half``.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    drop_tied_head,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.core.cache import KVCache
from sideros.core.kernels.bitlinear import bitlinear, bitlinear_applies
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput


def _relu2(x: mx.array) -> mx.array:
    gated = mx.maximum(x, mx.array(0.0, dtype=x.dtype))
    return gated * gated


def _act_quant(x: mx.array) -> mx.array:
    """transformers' ``AutoBitLinear`` per-token int8 absmax fake-quant, in fp32:
    ``round(x * 127/absmax)`` clamped to ``[-128, 127]``, rescaled by ``absmax/127``.
    The matmul that follows runs in the model dtype, so the rescale lands there too."""
    xf = x.astype(mx.float32)
    absmax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    scale = mx.array(127.0, dtype=mx.float32) / mx.maximum(absmax, mx.array(1e-5, dtype=mx.float32))
    quant = mx.round(xf * scale)
    quant = mx.minimum(
        mx.maximum(quant, mx.array(-128.0, dtype=mx.float32)),
        mx.array(127.0, dtype=mx.float32),
    )
    return quant / scale


def _unpack_ternary(weight: mx.array) -> mx.array:
    """``packed [out//4, in]`` uint8 -> ``[out, in]`` float in the logical output order
    transformers' ``unpack_weights`` produces: field ``j`` of byte ``p`` is row
    ``p + j * (out//4)``."""
    out_packs = weight.shape[0]
    in_features = weight.shape[1]
    three = mx.array(3, dtype=mx.uint8)
    fields = [
        (weight & three),
        ((weight >> mx.array(2, dtype=mx.uint8)) & three),
        ((weight >> mx.array(4, dtype=mx.uint8)) & three),
        ((weight >> mx.array(6, dtype=mx.uint8)) & three),
    ]
    stacked = mx.concatenate(fields, axis=0)
    return stacked.astype(mx.float32).reshape(out_packs * 4, in_features) - mx.array(
        1.0, dtype=mx.float32
    )


def _ternary_matmul(x: mx.array, weight: mx.array, weight_scale: mx.array) -> mx.array:
    """The op-chain fallback and parity reference: unpack to float, matmul, scale."""
    w = _unpack_ternary(weight).astype(x.dtype)
    return (x @ w.T) * weight_scale


class BitLinear(nn.Module):
    """A ternary 1.58-bit linear leaf: packed uint8 ``weight`` ``[out//4, in]`` and a
    scalar ``weight_scale`` the kernel multiplies by (autobitlinear). Not an
    ``nn.Linear`` subclass, so the affine ``nn.quantize`` pass leaves it alone."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mx.zeros((out_features // 4, in_features), dtype=mx.uint8)
        self.weight_scale = mx.ones((1,))
        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        row = x.reshape(-1, self.in_features)
        quantized = _act_quant(row).astype(x.dtype)
        if bitlinear_applies(self.out_features, self.in_features, self.weight):
            out = bitlinear(quantized, self.weight, self.weight_scale)
        else:
            out = _ternary_matmul(quantized, self.weight, self.weight_scale)
        if "bias" in self:
            out = out + self.bias
        return out.reshape(*lead, self.out_features)


@dataclass(frozen=True)
class BitNetConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: tuple[int, ...]


class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.q_proj = BitLinear(hidden, queries)
        self.k_proj = BitLinear(hidden, key_values)
        self.v_proj = BitLinear(hidden, key_values)
        self.o_proj = BitLinear(queries, hidden)
        self.attn_sub_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        # Split-half (transformers rotate_half), not mlx-lm's default interleaved.
        return mx.fast.rope(
            x, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q = self.q_proj(x).reshape(1, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries, rotated = self.rope(q, offset), self.rope(k, offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        out = attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        return self.o_proj(self.attn_sub_norm(out))


class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.intermediate_size
        self.gate_proj = BitLinear(hidden, inner)
        self.up_proj = BitLinear(hidden, inner)
        self.down_proj = BitLinear(inner, hidden)
        self.ffn_sub_norm = nn.RMSNorm(inner, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.ffn_sub_norm(_relu2(self.gate_proj(x)) * self.up_proj(x)))


class BitNetBlock(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class BitNetTrunk(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BitNetBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class BitNetActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class BitNet(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BitNetTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[KVCache] | None = None
    ) -> BitNetActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return BitNetActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: NotRequired[int]
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: int | list[int]


def _config(path: Path) -> BitNetConfig:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "bitnet":
        raise ValueError(f"expected model_type bitnet, got {raw['model_type']!r}")
    eos = raw["eos_token_id"]
    heads = raw["num_attention_heads"]
    head_dim = raw.get("head_dim", raw["hidden_size"] // heads)
    return BitNetConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=head_dim,
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _weights(
    directory: Path, config: BitNetConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """The packed ternary ``.weight`` is uint8 and must stay uint8: a ``dtype=`` cast
    keeps the shape (so ``load_weights`` accepts it) and destroys the codes. Cast
    everything else — ``weight_scale``, the embed table, the norms — and leave the
    packed weights alone. No fusions: each projection carries its own per-tensor
    ``weight_scale``, so qkv or gate‖up fusion would collapse independent scales."""
    weights = load_shards(directory)
    reject_dtype_cast(dtype, weights)
    if dtype is not None:
        weights = {
            key: value if value.dtype == mx.uint8 else value.astype(dtype)
            for key, value in weights.items()
        }
    if config.tie_word_embeddings:
        drop_tied_head(weights)
    return weights


def _composite(directory: Path, model: BitNet) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    _config,
    BitNet,
    _weights,
    _composite,
)

"""Gemma 3 text (`gemma3_text`): interleaved sliding/full attention, sandwich norms.

Authoritative semantics: transformers' modeling_gemma3.py. What the trunk does that the
Qwen family does not:

- `layer_types` picks, per layer, a sliding-window mask + `rope_local_base_freq` or a
  full causal mask + `rope_theta`. The cache is never evicted; the window lives in the
  mask (a masked key contributes nothing).
- The attention scale is `query_pre_attn_scalar ** -0.5`, which only coincides with
  `head_dim ** -0.5` on this checkpoint. `num_attention_heads * head_dim` (1024) is
  decoupled from `hidden_size` (640).
- Sandwich norms: each residual arm is normed on the way in *and* on the way out.
- Zero-centered RMSNorm: the scale is `1 + w`. Folded on the dict side at load
  (`_fold_norm_scales`), so the tree holds plain `nn.RMSNorm` and no `1 + w` add runs
  per norm per token — 108 extra kernels per step on an 18-layer trunk.
- Embeddings scaled by `sqrt(hidden_size)`; transformers keeps that scalar in float32
  and casts it to the weight dtype, so bf16 sees 25.25 and fp32 25.298221 — the cast is
  reproduced here, not the rounded constant.
- lm_head tied to the embedding table; gelu (tanh approximation) MLP.
"""

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU, split_qkv
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

_SPACE = "▁"
_DEAD = -1

SLIDING = "sliding_attention"
FULL = "full_attention"


def _byte_fallback(token: str) -> int | None:
    if len(token) != 6 or not token.startswith("<0x") or not token.endswith(">"):
        return None
    try:
        return int(token[3:5], 16)
    except ValueError:
        return None


@dataclass(frozen=True)
class _AddedTokens:
    """Match added tokens against raw text before normalization, longest first."""

    ids: dict[str, int]
    lengths: tuple[int, ...]
    starts: frozenset[str]

    @classmethod
    def build(cls, tokens: dict[str, int]) -> "_AddedTokens":
        return cls(
            ids=tokens,
            lengths=tuple(sorted({len(content) for content in tokens}, reverse=True)),
            starts=frozenset(content[0] for content in tokens if content),
        )

    def split(self, text: str) -> list[str | int]:
        pieces: list[str | int] = []
        pending: list[str] = []
        index = 0
        while index < len(text):
            matched: tuple[int, int] | None = None
            if text[index] in self.starts:
                for length in self.lengths:
                    identifier = self.ids.get(text[index : index + length])
                    if identifier is not None:
                        matched = (length, identifier)
                        break
            if matched is None:
                pending.append(text[index])
                index += 1
                continue
            if pending:
                pieces.append("".join(pending))
                pending = []
            pieces.append(matched[1])
            index += matched[0]
        if pending:
            pieces.append("".join(pending))
        return pieces


@dataclass(frozen=True)
class Gemma3Tokenizer:
    """Gemma 3 SentencePiece-style BPE read from ``tokenizer.json``.

    Spaces normalize to ``▁``. Characters outside the vocabulary fall back to UTF-8
    byte tokens, and the template processor prepends ``<bos>``.
    """

    encoder: dict[str, int]
    decoder: dict[int, str]
    ranks: dict[tuple[int, int], int]
    joined: dict[tuple[int, int], int]
    added: _AddedTokens
    byte_fallbacks: tuple[int, ...]
    bos: int

    @classmethod
    def from_file(cls, path: Path) -> "Gemma3Tokenizer":
        raw = json.loads(path.read_text(encoding="utf-8"))
        vocab: dict[str, int] = raw["model"]["vocab"]
        ranks: dict[tuple[int, int], int] = {}
        joined: dict[tuple[int, int], int] = {}
        for rank, merge in enumerate(raw["model"]["merges"]):
            left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
            whole = vocab.get(left + right)
            first, second = vocab.get(left), vocab.get(right)
            if whole is None or first is None or second is None or (first, second) in ranks:
                continue
            ranks[(first, second)] = rank
            joined[(first, second)] = whole
        return cls(
            encoder=vocab,
            decoder={identifier: token for token, identifier in vocab.items()},
            ranks=ranks,
            joined=joined,
            added=_AddedTokens.build(
                {token["content"]: token["id"] for token in raw["added_tokens"]}
            ),
            byte_fallbacks=tuple(vocab[f"<0x{byte:02X}>"] for byte in range(256)),
            bos=vocab["<bos>"],
        )

    def _merge(self, symbols: list[int]) -> list[int]:
        """Merge with a priority queue over a linked list to avoid quadratic scans."""

        count = len(symbols)
        if count < 2:
            return symbols
        following = list(range(1, count + 1))
        preceding = list(range(-1, count - 1))
        queue = [
            (rank, left, symbols[left], symbols[left + 1])
            for left in range(count - 1)
            if (rank := self.ranks.get((symbols[left], symbols[left + 1]))) is not None
        ]
        heapq.heapify(queue)
        while queue:
            _, left, first, second = heapq.heappop(queue)
            right = following[left]
            if right == count or symbols[left] != first or symbols[right] != second:
                continue
            symbols[left] = self.joined[(first, second)]
            symbols[right] = _DEAD
            after = following[right]
            following[left] = after
            if after != count:
                preceding[after] = left
            before = preceding[left]
            for start, end in ((before, left), (left, after)):
                if start < 0 or end == count:
                    continue
                rank = self.ranks.get((symbols[start], symbols[end]))
                if rank is not None:
                    heapq.heappush(queue, (rank, start, symbols[start], symbols[end]))
        word: list[int] = []
        index = 0
        while index != count:
            word.append(symbols[index])
            index = following[index]
        return word

    def _encode_text(self, text: str) -> list[int]:
        symbols: list[int] = []
        for character in text.replace(" ", _SPACE):
            identifier = self.encoder.get(character)
            if identifier is None:
                symbols.extend(self.byte_fallbacks[byte] for byte in character.encode("utf-8"))
            else:
                symbols.append(identifier)
        return self._merge(symbols)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos]
        for piece in self.added.split(text):
            if isinstance(piece, int):
                ids.append(piece)
            else:
                ids.extend(self._encode_text(piece))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        output = bytearray()
        for identifier in ids:
            token = self.decoder[identifier]
            byte = _byte_fallback(token)
            if byte is None:
                output.extend(token.replace(_SPACE, " ").encode("utf-8"))
            else:
                output.append(byte)
        return bytes(output)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


# `hidden_activation` is gelu_pytorch_tanh: the tanh approximation, which is what
# mlx.nn.gelu_approx computes — as a single compiled kernel, hence the reuse instead of
# the expression inline. `mx.compile` erases the signature from the stubs, so it is
# restated here the way `core/mxcompat.py` restates mlx's own stale ones.
if TYPE_CHECKING:

    def _gelu(x: mx.array) -> mx.array: ...

else:
    _gelu = nn.gelu_approx


@dataclass(frozen=True)
class Gemma3TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_local_base_freq: float
    query_pre_attn_scalar: float
    sliding_window: int
    layer_types: tuple[str, ...]
    tie_word_embeddings: bool
    intermediate_size: int
    eos_token_id: tuple[int, ...]


def causal_mask(queries: int, keys: int, window: int | None) -> mx.array:
    """Which keys each query may attend to: everything up to itself, and within `window`
    positions of it when the layer slides. The queries are the *last* `queries` of the
    `keys` positions, which is what lets a single query attend to a whole cache."""
    rows = mx.arange(keys - queries, keys).reshape(queries, 1)
    columns = mx.arange(keys).reshape(1, keys)
    within = rows >= columns
    if window is None:
        return within
    return within & (rows < columns + window)


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding = layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self.rope_base = config.rope_local_base_freq if self.sliding else config.rope_theta
        self.scale = config.query_pre_attn_scalar**-0.5
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """qkv split back into per-head [1, heads, length, head_dim], normed but
        unrotated — the boundary transformers' q_norm/k_norm hooks expose."""
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        return self.q_norm(q), self.k_norm(k), v

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | None:
        """A lone query attends to everything it can reach; only a sliding layer whose
        cache already exceeds the window still needs a mask at T=1."""
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = self.split_heads(x)
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset),
            keys,
            values,
            scale=self.scale,
            mask=self.mask(length, keys.shape[2]),
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


class Gemma3Block(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_type)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, _gelu)
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cache))
        return attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )


class Gemma3Trunk(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3Block(config, layer_type) for layer_type in config.layer_types]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Gemma3Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Gemma3(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma3Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def embed(self, ids: mx.array) -> mx.array:
        embedded = self.model.embed_tokens(ids)
        scale = mx.array(math.sqrt(self.config.hidden_size), mx.float32).astype(embedded.dtype)
        return embedded * scale

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Gemma3Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.embed(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return Gemma3Activations(embeddings, blocks, normed, self.head(normed))

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
    rope_local_base_freq: float
    query_pre_attn_scalar: float
    sliding_window: int
    layer_types: list[str]
    intermediate_size: int
    tie_word_embeddings: NotRequired[bool]
    eos_token_id: int | list[int]


def _config(path: Path) -> Gemma3TextConfig:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "gemma3_text":
        raise ValueError(f"expected model_type gemma3_text, got {raw['model_type']!r}")
    unknown = set(raw["layer_types"]) - {SLIDING, FULL}
    if unknown:
        raise ValueError(f"unknown layer types {sorted(unknown)}")
    eos = raw["eos_token_id"]
    return Gemma3TextConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        rope_local_base_freq=raw["rope_local_base_freq"],
        query_pre_attn_scalar=raw["query_pre_attn_scalar"],
        sliding_window=raw["sliding_window"],
        layer_types=tuple(raw["layer_types"]),
        # Gemma 3 ships no `tie_word_embeddings`; the transformers config defaults it True
        # and the checkpoint has no lm_head.
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
        intermediate_size=raw["intermediate_size"],
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _fold_norm_scales(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Gemma stores the norm weight centred on zero (`scale = 1 + w`). Folding the sum
    on the dict side keeps the tree on plain `nn.RMSNorm`; left per call it is one extra
    kernel per norm per token (6 per block plus the final one)."""
    for key, value in weights.items():
        if key.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or (
            key == "model.norm.weight"
        ):
            weights[key] = value + 1
    mx.eval(list(weights.values()))
    return weights


def _weights(
    directory: Path, config: Gemma3TextConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
            _fold_norm_scales,
        ],
        dtype,
    )


def _composite(directory: Path, model: Gemma3) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            Gemma3Tokenizer.from_file(directory / "tokenizer.json"),
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
    ), _config, Gemma3, _weights, _composite)

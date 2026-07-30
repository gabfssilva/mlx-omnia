"""GPT-2 as a tree of nn.Module with property names = checkpoint names.

Authoritative semantics: transformers' modeling_gpt2.py. lm_head is tied to wte;
wpe is a learned position table (dense even inside a quantized model).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import mlx.core as mx
import mlx.nn as nn
import regex

from sideros.bpe import BYTE_CHAR, CHAR_BYTE, bpe
from sideros.chat import chat_capabilities
from sideros.checkpoint import checkpoint, reject_dtype_cast
from sideros.core.cache import KVCache
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

_PRETOKEN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float


@dataclass(frozen=True)
class GPT2Tokenizer:
    """GPT-2 byte-level BPE over ``vocab.json`` and ``merges.txt``.

    The tokenizer intentionally has no merge cache: id interning resolved the measured
    bottleneck in the Swift port.
    """

    encoder: dict[str, int]
    decoder: dict[int, str]
    ranks: dict[tuple[str, str], int]

    @classmethod
    def from_files(cls, vocab: Path, merges: Path) -> "GPT2Tokenizer":
        encoder: dict[str, int] = json.loads(vocab.read_text(encoding="utf-8"))
        ranks: dict[tuple[str, str], int] = {}
        for line in merges.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#version"):
                continue
            left, right = line.split()
            ranks[(left, right)] = len(ranks)
        return cls(encoder, {identifier: token for token, identifier in encoder.items()}, ranks)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _PRETOKEN.finditer(text):
            mapped = "".join(BYTE_CHAR[byte] for byte in match.group().encode("utf-8"))
            ids.extend(self.encoder[part] for part in bpe(mapped, self.ranks))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(
            CHAR_BYTE[character] for identifier in ids for character in self.decoder[identifier]
        )

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


def _gelu_new(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = (
            part.reshape(batch, length, self.n_head, -1).transpose(0, 2, 1, 3)
            for part in mx.split(qkv, 3, axis=-1)
        )
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        # A lone query attends to everything: no mask on the T=1 step.
        mask = "causal" if length > 1 else None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask=mask
        )
        return self.c_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(_gelu_new(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))


class GPT2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    ln_f: mx.array
    logits: mx.array


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.h]

    def activations(self, ids: mx.array) -> GPT2Activations:
        x = self.wte(ids) + self.wpe(mx.arange(ids.shape[-1]))
        embeddings = x
        blocks: list[mx.array] = []
        for block in self.h:
            x = block(x)
            blocks.append(x)
        normed = self.ln_f(x)
        return GPT2Activations(embeddings, blocks, normed, self.wte.as_linear(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        if cache is None:
            return self.activations(ids).logits
        offset = cache[0].offset
        x = self.wte(ids) + self.wpe(mx.arange(offset, offset + ids.shape[-1]))
        for block, layer_cache in zip(self.h, cache, strict=True):
            x = block(x, layer_cache)
        return self.wte.as_linear(self.ln_f(x))


class _Json(TypedDict):
    model_type: str
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float


def _config(path: Path) -> GPT2Config:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "gpt2":
        raise ValueError(f"expected model_type gpt2, got {raw['model_type']!r}")
    return GPT2Config(
        vocab_size=raw["vocab_size"],
        n_positions=raw["n_positions"],
        n_embd=raw["n_embd"],
        n_layer=raw["n_layer"],
        n_head=raw["n_head"],
        layer_norm_epsilon=raw["layer_norm_epsilon"],
    )


_CONV1D = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")


def _transpose_conv1d(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """GPT-2's Conv1D ships `[in, out]`. This is the one load-time layout change that
    moves the last axis, so it has to land before any grouping: quantizing the raw
    matrix groups along the output columns."""
    return {
        key: value.T if key.endswith(_CONV1D) else value for key, value in weights.items()
    }


def _weights(directory: Path, config: GPT2Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    raw = mx.load(str(directory / "model.safetensors"))
    assert isinstance(raw, dict)
    reject_dtype_cast(dtype, raw)
    # h.{i}.attn.bias is the checkpoint's causal-mask buffer, not a weight.
    return _transpose_conv1d(
        {
            key: value.astype(dtype if dtype is not None else mx.float32)
            for key, value in raw.items()
            if not key.endswith(".attn.bias")
        }
    )


def _end_of_text(tokenizer: GPT2Tokenizer) -> tuple[int, ...]:
    end = tokenizer.encoder.get("<|endoftext|>")
    return () if end is None else (end,)


def _composite(directory: Path, model: GPT2) -> LanguageModel[ModelInput]:
    tokenizer = GPT2Tokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
    )
    return CompositeModel(
        TextLanguageModel(model, tokenizer, stop=_end_of_text(tokenizer)),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    _config,
    GPT2,
    _weights,
    _composite,
)

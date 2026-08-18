"""The two architectures the quantize suite quantizes, and the drafter beside them.

The model is built here — four leaves and a config — because what the suite covers is the
job around the quantization and not the arithmetic inside it: the engine's own suite holds
`quantize_weights` against the formats. The other half is exactly what no double could
show, so it is real: the entry is written by `mlx_omnia.engine.task`, listed by
`catalog.scan` and opened by `mlx_omnia.load` under the repo id that was asked for.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import TypeIs

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.checkpoint import Checkpoint, Drafter, Pending, attach_weights
from mlx_omnia.engine.parsers import Segment

IDS = mx.array([[3, 1, 4, 1, 5]])
LEAVES = ("attn", "embed", "head", "mlp")
"""Every quantizable leaf of the model below, in the order `inventory` sorts them."""

CALIBRATED = ("awq", "gptq", "oq", "oqe")
"""The four that read a calibration pass. They run here — over the blocked model below,
which is the one with a trunk to intercept."""

HIDDEN = 64
INNER = 128
VOCAB = 64
BLOCKS = 2
CALIBRATION: dict[str, object] = {"sequences": 2, "sequence_length": 64}
"""Enough of the corpus for every statistic to exist and be positive, and small enough that
the pass is part of a unit suite: what is covered here is the job around the pass."""

DRAFT_LEAVES = ("fc", "out")
OUTSIDE = ("embed_tokens", "lm_head")

BLOCK_LEAVES = tuple(
    sorted(
        f"layers.{index}.{leaf}"
        for index in range(BLOCKS)
        for leaf in (
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        )
    )
)
"""The leaves a pass over the trunk observes — and, for GPTQ, exactly the ones that do not
fall back to RTN."""


class Encoder:
    """The corpus is sampled with the model's own tokenizer, so the test model needs one.
    Bytes modulo the vocabulary: every id the pass draws indexes a row that exists."""

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([byte % VOCAB for byte in whole.encode()])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(id % 256 for id in ids)


class Tiny(nn.Module):
    """`norm` is the one tensor no plan touches: `inventory` does not list it, the entry
    carries it unchanged, and it is the difference between what a plan costs and what the
    file it produces weighs."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 64)
        self.attn = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Linear(64, 64, bias=False)
        self.norm = nn.RMSNorm(64)
        self.head = nn.Linear(64, 32, bias=False)

    def __call__(self, ids: mx.array) -> mx.array:
        return self.head(self.norm(nn.silu(self.mlp(self.attn(self.embed(ids))))))


class Backend:
    def __init__(self, model: Tiny) -> None:
        self.model = model
        self.tokenizer = Encoder()
        """A checkpoint has one, and a calibrated method over this model has to fail on what
        it really lacks — a trunk — and not on a tokenizer no double happened to carry."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def tensors(directory: Path, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    loaded = mx.load(str(directory / "model.safetensors"))
    assert isinstance(loaded, dict)
    if dtype is None:
        return loaded
    return {name: value.astype(dtype) for name, value in loaded.items()}


def _tree(directory: Path, dtype: mx.Dtype | None) -> Tiny:
    return attach_weights(Tiny(), tensors(directory, dtype))


def _model(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
    return CompositeModel(Backend(_tree(directory, dtype)), [])


def _pending(directory: Path, dtype: mx.Dtype | None) -> Pending[LanguageModel[ModelInput]]:
    tree = Tiny()
    return Pending(
        tree,
        lambda: tensors(directory, dtype),
        lambda packed: CompositeModel(Backend(attach_weights(tree, packed)), []),
    )


CHECKPOINT = Checkpoint(
    ("config.json", "model.safetensors", "tokenizer.json"),
    _tree,
    _model,
    _pending,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        length = x.shape[-2]
        scores = (self.q_proj(x) @ self.k_proj(x).swapaxes(-1, -2)) / math.sqrt(HIDDEN)
        causal = mx.triu(mx.full((length, length), float("-inf")), k=1)
        return self.o_proj(mx.softmax(scores + causal, axis=-1) @ self.v_proj(x))


class _Gated(nn.Module):
    """The three names AWQ's derivation looks for: the input of `down_proj` is
    `silu(gate) ⊙ up`, so a per-channel scale on `up_proj` passes through it exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INNER, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INNER, bias=False)
        self.down_proj = nn.Linear(INNER, HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(HIDDEN)
        self.self_attn = _Attention()
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN)
        self.mlp = _Gated()

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Blocked(nn.Module):
    """A trunk, which is the whole difference from `Tiny`: `discover_blocks` finds
    `layers`, the pass intercepts each element of it, and `embed_tokens` and `lm_head` sit
    outside — which is exactly the leaf GPTQ has no statistic for and oQ protects."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = [_Layer() for _ in range(BLOCKS)]
        self.norm = nn.RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def __call__(self, ids: mx.array) -> mx.array:
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


class TrunkBackend:
    def __init__(self, model: Blocked) -> None:
        self.model = model
        self.tokenizer = Encoder()

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def _blocked_tree(directory: Path, dtype: mx.Dtype | None) -> Blocked:
    return attach_weights(Blocked(), tensors(directory, dtype))


def _blocked_model(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
    return CompositeModel(TrunkBackend(_blocked_tree(directory, dtype)), [])


def _blocked_pending(directory: Path, dtype: mx.Dtype | None) -> Pending[LanguageModel[ModelInput]]:
    tree = Blocked()
    return Pending(
        tree,
        lambda: tensors(directory, dtype),
        lambda packed: CompositeModel(TrunkBackend(attach_weights(tree, packed)), []),
    )


BLOCKED = Checkpoint(
    ("config.json", "model.safetensors", "tokenizer.json"),
    _blocked_tree,
    _blocked_model,
    _blocked_pending,
)


class Draft(nn.Module):
    """A drafter's shape and the whole of what makes it one: weights, and no way in from a
    token. No embedding, no head, no tokenizer beside it on disk."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.out = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.out(nn.silu(self.fc(hidden)))


def _draft_tree(directory: Path, dtype: mx.Dtype | None) -> Draft:
    return attach_weights(Draft(), tensors(directory, dtype))


def _draft_pending(directory: Path, dtype: mx.Dtype | None) -> Pending[Draft]:
    tree = Draft()
    return Pending(
        tree, lambda: tensors(directory, dtype), lambda packed: attach_weights(tree, packed)
    )


DRAFTER = Drafter(("config.json", "model.safetensors"), _draft_tree, _draft_pending)

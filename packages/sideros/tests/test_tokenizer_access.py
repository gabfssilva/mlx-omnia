"""Reaching the tokenizer of a model that is already loaded.

What `load` hands back is a `CompositeModel` — a protocol over `stream` with the task-level
model behind it — and the tokenizer belongs to that inner model. Counting a prompt's tokens
from outside the generation is what needs the walk down to it, and the alternative is a
second dispatch by architecture over `tokenizer.json`: Gemma 3 reads that file with a
tokenizer of its own, so the file alone does not say which one.
"""

from collections.abc import Iterator
from typing import TypeIs

import mlx.core as mx
import mlx.nn as nn
from conftest import checkpoint_dir, requires_checkpoint

from sideros import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    Gemma3Tokenizer,
    GenerationOptions,
    KVCache,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
    load,
)
from sideros.language import Prefill, tokenizer_of, trunk_of
from sideros.suppress import Segment

GEMMA = "google/gemma-3-270m"

TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


class ConstantLM:
    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return mx.zeros((1, ids.shape[1], 8), dtype=mx.float32)


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b" ".join(b"x" * count for count in ids)


class Tokenizerless:
    """A `LanguageModel` holding no tokenizer at all, which is what a test double is: the
    reason the walk answers `None` rather than asserting its way down."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def test_the_tokenizer_is_reached_through_the_composite_a_template_wraps_the_model_in() -> None:
    """A checkpoint with a chat template loads as a `CompositeModel`, which declares no
    tokenizer: stop the walk at the outermost model and every model that can chat — that is,
    every model the chat window ever counts tokens for — has no tokenizer to count with."""
    tokenizer = WordTokenizer()
    capability = ChatCapability(ChatTemplate.from_source(TEMPLATE))
    model = CompositeModel(TextLanguageModel(ConstantLM(), tokenizer), [capability])

    assert tokenizer_of(model) is tokenizer
    assert tokenizer_of(TextLanguageModel(ConstantLM(), tokenizer)) is tokenizer


def test_a_model_that_holds_no_tokenizer_is_not_given_an_invented_one() -> None:
    assert tokenizer_of(Tokenizerless()) is None
    assert tokenizer_of(CompositeModel(Tokenizerless(), [])) is None


class TreeLM(nn.Module):
    """What a real checkpoint's backend is and `ConstantLM` is not: the `nn.Module` the
    blocks live in."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.embed(ids)


def test_the_tree_a_calibration_pass_runs_is_reached_through_the_same_facades() -> None:
    """One level below the tokenizer: what a data-dependent quantization needs is the
    `nn.Module` itself — to find the blocks in it — and the prefill that supplies each
    block its arguments, which is that same module called for logits."""
    tree = TreeLM()
    model = CompositeModel(TextLanguageModel(tree, WordTokenizer()), [])

    found = trunk_of(model)

    assert found is tree
    forward: Prefill = tree
    assert forward(mx.array([[1, 2, 3]])).shape == (1, 3, 8)


def test_a_backend_that_is_no_tree_answers_nothing_instead_of_the_facade() -> None:
    """`ConstantLM` is a `CausalLM` and no `nn.Module`, which is exactly the double the
    server suites load: the walk ends on it and says so."""
    assert trunk_of(CompositeModel(TextLanguageModel(ConstantLM(), WordTokenizer()), [])) is None
    assert trunk_of(Tokenizerless()) is None


@requires_checkpoint(GEMMA)
def test_gemma3_answers_with_the_ids_of_its_own_tokenizer() -> None:
    """The architecture with a tokenizer that is not the byte-level BPE. Reading
    `tokenizer.json` from outside the model would hand the generic one back for this
    checkpoint: it is the loader that knows the file is SentencePiece-shaped, prepends
    `<bos>` and falls back to byte tokens.
    """
    directory = checkpoint_dir(GEMMA)
    text = "The café owner, naïve about tokenizers"

    found = tokenizer_of(load(directory))
    assert isinstance(found, Gemma3Tokenizer)
    reference = Gemma3Tokenizer.from_file(directory / "tokenizer.json")
    assert found.encode(text) == reference.encode(text)
    # The signature no byte-level BPE over the same file produces: Gemma's own template
    # processor is what puts `<bos>` in front.
    assert found.encode(text)[0] == reference.bos == 2

from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.generate import Meter
from mlx_omnia.engine.language import GenerationOptions, Text, TextLanguageModel
from mlx_omnia.engine.parsers import Segment


class CountingModel:
    continuous_batching = True

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[KVStore]) -> mx.array:
        targets = (ids + 1) % self.vocab
        vocabulary = mx.arange(self.vocab)
        return -mx.abs(vocabulary - targets[..., None]).astype(mx.float32)


class AsciiTokenizer:
    def encode(self, text: str) -> Sequence[int]:
        return tuple(text.encode())

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)


class Unbatched(TextLanguageModel[KVCache]):
    """The same trunk with the batched path refused, which is the stream_ids body."""

    def can_batch(self, options: GenerationOptions) -> bool:
        return False


def test_stream_matches_the_unbatched_path_segment_for_segment() -> None:
    options = GenerationOptions(max_tokens=4)
    batched = TextLanguageModel(CountingModel(128), AsciiTokenizer())
    plain = Unbatched(CountingModel(128), AsciiTokenizer())

    assert batched.can_batch(options)
    assert list(batched.stream(Text("AB"), options)) == list(plain.stream(Text("AB"), options))


def test_stream_yields_the_generated_text_as_content() -> None:
    model = TextLanguageModel(CountingModel(128), AsciiTokenizer())

    segments = list(model.stream(Text("A"), GenerationOptions(max_tokens=3)))

    assert segments == [
        Segment("content", "B"),
        Segment("content", "C"),
        Segment("content", "D"),
    ]


def test_stream_leaves_its_prefix_for_the_next_request() -> None:
    model = TextLanguageModel(CountingModel(128), AsciiTokenizer())
    options = GenerationOptions(max_tokens=2, prefix_budget=1024**2)
    list(model.stream(Text("A"), options))
    meter = Meter()

    list(
        model.stream(
            Text("ABC"),
            GenerationOptions(max_tokens=2, prefix_budget=1024**2, meter=meter),
        )
    )

    assert meter.reused_tokens > 0

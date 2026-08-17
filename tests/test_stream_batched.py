from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.language import GenerationOptions, Text, TextLanguageModel


class CountingModel:

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

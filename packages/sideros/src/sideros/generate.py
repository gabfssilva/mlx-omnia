"""Lazy generation pipeline.

The sampler returns the token as an mx.array (no .item() before the next step is
queued); step n+1 is async-evaluated before step n's sync, so the GPU never idles
between steps.
"""

import codecs
from collections.abc import Callable, Iterator
from typing import Protocol

import mlx.core as mx

from sideros.core.cache import LayerCache
from sideros.tokenizer import GPT2Tokenizer

type Sampler = Callable[[mx.array], mx.array]


class CausalLM[C: LayerCache](Protocol):
    def make_cache(self) -> list[C]: ...

    def __call__(self, ids: mx.array, cache: list[C] | None = None) -> mx.array: ...


def greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def stream_ids[C: LayerCache](
    model: CausalLM[C],
    prompt: list[int],
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: int | None = None,
) -> Iterator[int]:
    cache = model.make_cache()

    def step(ids: mx.array) -> mx.array:
        logits = model(ids[None], cache)
        return sampler(logits[:, -1, :])[0]

    y = step(mx.array(prompt))
    mx.async_eval(y)
    for _ in range(max_tokens):
        next_y = step(y[None])
        mx.async_eval(next_y)
        token = y.item()
        assert isinstance(token, int)
        if token == stop:
            return
        yield token
        y = next_y


def stream_generate[C: LayerCache](
    model: CausalLM[C],
    tokenizer: GPT2Tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: int | None = None,
) -> Iterator[str]:
    """Streams decoded text, holding partial UTF-8 until it completes."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    ids = stream_ids(
        model,
        tokenizer.encode(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        stop=tokenizer.encoder.get("<|endoftext|>") if stop is None else stop,
    )
    for token in ids:
        piece = decoder.decode(tokenizer.decode_bytes([token]))
        if piece:
            yield piece
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail

"""What both speed suites are written against: the house's own checkpoint numbers, and a
generation whose numbers are chosen instead of clocked."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TypeIs

from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    Model,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.services.speed import ModelFacts, Progress

GIGABYTE = 1024**3

DENSE = ModelFacts(
    weight_bytes=17 * GIGABYTE,
    kv_bytes_per_token=98_304,
    attention_window=None,
    checkpoint_bytes=17 * GIGABYTE,
)
"""A 30B mixture at 4 bits, in the house's own numbers: 1.7 GB read per step is the
mixture's slice, 96 KB per token is its cache."""

WINDOWED = replace(DENSE, attention_window=8192)

BF16 = replace(DENSE, weight_bytes=60 * GIGABYTE, checkpoint_bytes=60 * GIGABYTE)
"""The same architecture unquantized: the weights that have to be resident before the cache
gets any room are what turns a shape from tight into impossible."""

BF16_WINDOWED = replace(BF16, attention_window=8192)

MOE = replace(DENSE, weight_bytes=2 * GIGABYTE, checkpoint_bytes=17 * GIGABYTE)


@dataclass
class Scripted:
    """A generation whose numbers are chosen instead of clocked. The script walking off its
    end is how a round nobody asked for fails the test it is in."""

    script: list[tuple[float, float]]
    """(decode tok/s, ttft in seconds) per generation, the load probe first."""

    runs: int = 0

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rate, ttft = self.script[self.runs]
        self.runs += 1
        prompt = input.value
        assert isinstance(prompt, str), "the speed runner hands the model a whole prompt"
        meter.prompt_tokens = len(prompt) // 4
        meter.completion_tokens = options.max_tokens
        meter.prefill_started = 0.0
        meter.first_token = ttft
        meter.last_token = ttft + (options.max_tokens - 1) / rate
        for index in range(options.max_tokens):
            yield Segment("content", str(index))


def composite(
    model: Model[Text, Segment, GenerationOptions],
) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(model, [])


@dataclass
class Handle:
    """The `speed.Task` a job would hand its blocking work, with nothing behind it: the
    measurement writes no row, so what it needs is the loop, the flag and a door."""

    loop: asyncio.AbstractEventLoop
    id: str = "bench"
    cancelled: threading.Event = field(default_factory=threading.Event)
    reported: list[Progress] = field(default_factory=list)

    def report(self, progress: Progress) -> None:
        self.reported.append(progress)

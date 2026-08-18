"""The models this dialect's suites are run over, and the engine that records what it was
asked for. A double that answered with something of its own would leave the translation
untested — the answer here *is* the rendered conversation."""

import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

import mlx.core as mx

from mlx_omnia import (
    TEXT,
    ChatCapability,
    CompositeModel,
    GenerationOptions,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.engine.parsers import FALLBACK, Segment, Segmenter
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Job, Loader
from tests.server.anthropic_script import (
    ANSWERED,
    BIG_PIECES,
    CALL_PIECES,
    CALLING_TEMPLATE,
    CATALOGUED,
    CHUNK,
    GUIDED,
    RESULT,
    REUSED,
    STALL,
    TEMPLATE,
)


@dataclass(frozen=True)
class Echo:
    """Answers with the prompt it was given, cut into pieces of `CHUNK` characters.

    It fills the meter the way `stream_ids` does — the prompt before the first piece, one
    mark per piece — because `usage` and `stop_reason` are read off it: a double that left
    the meter at zero would let both assertions pass on any number at all.
    """

    stall: float = 0.0
    fails: bool = False
    reused: int = 0
    """Rows a prefix cache would have covered. A number rather than a real trie: what this
    suite is about is what the dialect writes down, and a reuse that depended on two turns
    rendering identically would make the assertion about the template."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        if self.stall:
            time.sleep(self.stall)
        meter.prefill(len(input.value), self.reused)
        pieces = [input.value[at : at + CHUNK] for at in range(0, len(input.value), CHUNK)]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=input.value
        )
        for piece in pieces[: options.max_tokens]:
            meter.token()
            yield from segmenter.push(piece)
        yield from segmenter.flush()
        if self.fails:
            raise RuntimeError("the decode thread gave out")


@dataclass(frozen=True)
class Caller:
    """Answers with a call, in the pieces it was given — and with the result of that call once
    the conversation carries one, which is what tells the second turn of a round trip from the
    first: the result only reaches here if the `tool_result` block became a turn the template
    rendered."""

    pieces: tuple[str, ...]

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value))
        result = input.value.partition("<tool>")[2].partition("</tool>")[0]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=input.value
        )
        for piece in (ANSWERED,) if result == RESULT else self.pieces:
            meter.token()
            yield from segmenter.push(piece)

        yield from segmenter.flush()


@dataclass(frozen=True)
class Letters:
    """One id per character. `count_tokens` is about which text was rendered and encoded, not
    about a vocabulary: with this, the count a test asserts is a length it can write down."""

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([ord(character) for character in whole])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)


@dataclass(frozen=True)
class Counted(Echo):
    """The echo, with a tokenizer under it. It is a model of its own rather than a field on
    `Echo` because `tokenizer_of` walks down to whatever holds one: giving every double a
    tokenizer would change what the grammar refusal below is about."""

    tokenizer: Letters = Letters()


def load(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
    match model_id:
        case "echo":
            return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])
        case "counted":
            return CompositeModel(Counted(), [ChatCapability(TEMPLATE)])
        case "cached":
            return CompositeModel(Echo(reused=REUSED), [ChatCapability(TEMPLATE)])
        case "flaky":
            return CompositeModel(Echo(fails=True), [ChatCapability(TEMPLATE)])
        case "slow":
            return CompositeModel(Echo(stall=STALL), [ChatCapability(TEMPLATE)])
        case "base":
            return CompositeModel(Echo(), [])
        case "caller":
            return CompositeModel(Caller(CALL_PIECES), [ChatCapability(CALLING_TEMPLATE)])
        case "mute":
            return CompositeModel(Caller(CALL_PIECES[1:]), [ChatCapability(CALLING_TEMPLATE)])
        case "big":
            return CompositeModel(Caller(BIG_PIECES), [ChatCapability(CALLING_TEMPLATE)])
        case "stranger":
            return CompositeModel(Caller(CALL_PIECES), [ChatCapability(TEMPLATE)])
        case "guided":
            return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])
        case other:
            raise ValueError(f"no model {other!r} in this stand")


class Free:
    """A walk that forbids nothing.

    No model on this stand holds a token table, so a real `Vocabulary` cannot be built over
    one and every schema would end in the same refusal. What a request carrying a schema has
    to prove here is the wiring — that the route compiles it and hands the walk to the
    generation — and that is what this stands in for.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return logits

    def accept(self, token: int) -> bool:
        return True


class Recording(Engine):
    """Keeps the jobs it hands out and the schemas it was asked to compile.

    Only `GUIDED` gets the double above: every other id falls through to the engine's own
    `constrain`, which is what makes the refusal below a real one — nothing under a double has
    a tokenizer, a head width or a stop id to compile against.
    """

    def __init__(self, loader: Loader, daemon: Daemon, metrics: Metrics) -> None:
        super().__init__(loader, daemon, metrics)
        self.jobs: list[Job] = []
        self.compiled: list[Mapping[str, object]] = []

    async def submit(
        self,
        model_id: str,
        input: ModelInput,
        options: GenerationOptions,
        reservation: object | None = None,
        batch_limit: int | None = None,
    ) -> Job:
        job = await super().submit(model_id, input, options, reservation, batch_limit)
        self.jobs.append(job)
        return job

    async def constrain(self, model_id: str, schema: Mapping[str, object]) -> Constraint:
        if model_id != GUIDED:
            return await super().constrain(model_id, schema)
        self.compiled.append(schema)
        return Free()


def fake_hub(root: Path) -> None:
    """One checkpoint in the shape `catalog.scan` recognizes: a snapshot with a config and
    the weights it promises. `refs/main` is left out — the scan falls back to the newest
    revision, and there is only one."""
    snapshot = root / f"models--{CATALOGUED.replace('/', '--')}" / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "tiny"}), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"")

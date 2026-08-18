"""The models under the Gemini stand's engine, and the engine that records what it was asked.

Each one answers with something a checkpoint would decide for itself, so that everything
around it — the template, the segmentation, the frames — stays real.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeIs

import mlx.core as mx

from mlx_omnia import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.engine.parsers import FALLBACK, Segment, Segmenter
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Job, Loader
from tests.server.gemini_names import (
    _TEMPLATE,
    ANSWERED,
    BASE,
    CACHED,
    CALL_PIECES,
    CALLER,
    DOCUMENT,
    FLAKY,
    GUIDED,
    MODEL,
    MUTE,
    PROSE,
    RESULT,
    REUSED,
    STRANGER,
    WRITER,
    WRITTEN,
)


@dataclass(frozen=True)
class Echo:
    """Answers with the prompt it was handed, one line per piece.

    The counts are its own, the way `stream_ids` writes them: one id per character of the
    render, so `promptTokenCount` is a fact about the text that reached the model rather than
    about the text the client sent.
    """

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    fails: bool = False
    """Raises after handing out what it had, which is the one way the worker meets an
    exception: an input the model refuses never becomes a job."""

    reused: int = 0
    """Rows a prefix cache would have covered. A number rather than a real trie: what this
    suite is about is what the dialect writes down, and a reuse that depended on two turns
    rendering identically would make the assertion about the template."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rendered = input.read()
        meter.prefill(len(rendered), self.reused)
        for piece in rendered.splitlines(keepends=True)[: options.max_tokens]:
            meter.token()
            yield Segment("content", piece)
        if self.fails:
            raise RuntimeError("the decode thread gave out")


@dataclass(frozen=True)
class Caller:
    """Answers with a call, in the pieces it was given — and with the result of that call once
    the conversation carries one, which is what tells the second turn of a round trip from the
    first: the result only reaches here if the `functionResponse` part became a turn the
    template rendered."""

    pieces: tuple[str, ...]

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rendered = input.read()
        meter.prefill(len(rendered))
        result = rendered.partition("<tool>")[2].partition("\n")[0]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=rendered
        )
        for piece in (ANSWERED,) if result == RESULT else self.pieces:
            meter.token()
            yield from segmenter.push(piece)

        yield from segmenter.flush()


@dataclass(frozen=True)
class Script:
    """A model whose generation is fixed text, handed out in the pieces it was given. What a
    checkpoint writes when it is asked for a JSON value is the checkpoint's own decision, so
    the text is pinned here and everything around it stays real."""

    pieces: tuple[str, ...]

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rendered = input.read()
        meter.prefill(len(rendered))
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=rendered
        )
        for piece in self.pieces:
            meter.token()
            yield from segmenter.push(piece)

        yield from segmenter.flush()


TEMPLATE = ChatTemplate.from_source(_TEMPLATE)

FOREIGN = ChatTemplate.from_source(_TEMPLATE.replace("tool_call", "call"))
"""The same template with a call spelled in no family's marker, which is what leaves
`parser_of` with nothing to say and the tool channel shut."""


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id == MODEL:
        return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])
    if model_id == BASE:
        return CompositeModel(Echo(), [])
    if model_id == CACHED:
        return CompositeModel(Echo(reused=REUSED), [ChatCapability(TEMPLATE)])
    if model_id == CALLER:
        return CompositeModel(Caller(CALL_PIECES), [ChatCapability(TEMPLATE)])
    if model_id == MUTE:
        return CompositeModel(Caller(CALL_PIECES[1:]), [ChatCapability(TEMPLATE)])
    if model_id == FLAKY:
        return CompositeModel(Echo(fails=True), [ChatCapability(TEMPLATE)])
    if model_id == STRANGER:
        return CompositeModel(Caller(CALL_PIECES), [ChatCapability(FOREIGN)])
    if model_id == WRITER:
        return CompositeModel(Script(WRITTEN), [ChatCapability(TEMPLATE)])
    if model_id == PROSE:
        return CompositeModel(Script(("I would rather not.",)), [ChatCapability(TEMPLATE)])
    if model_id == GUIDED:
        return CompositeModel(Script((json.dumps(DOCUMENT),)), [ChatCapability(TEMPLATE)])
    raise ValueError(f"no model {model_id!r} in this stand")


class Free:
    """A walk that forbids nothing.

    No model on this stand holds a token table, so a real `Vocabulary` cannot be built over
    one and every schema would end in the same refusal. What a constrained request has to
    prove here is the wiring — that the route compiles the schema and hands the walk to the
    generation — and that is what this stands in for.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return logits

    def accept(self, token: int) -> bool:
        return True


class Recording(Engine):
    """Keeps the jobs it hands out, and the schemas it was asked to compile. What the dialect
    made of the sampling knobs is the engine's input and reaches no response body.

    Only `GUIDED` gets the double above: every other id falls through to the engine's own
    `constrain`, which is what makes the refusal below a real one — nothing under a scripted
    model has a tokenizer, a head width or a stop id to compile against.
    """

    def __init__(self, loader: Loader, daemon: Daemon) -> None:
        super().__init__(loader, daemon, Metrics())
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


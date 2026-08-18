import codecs
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Protocol, TypeIs, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.batching import (
    BatchPrefill,
    BatchSequence,
    SequenceCache,
)
from mlx_omnia.engine.batching import (
    step as batch_step,
)
from mlx_omnia.engine.core import api
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.prefix import Prefixes
from mlx_omnia.engine.generate import (
    Constraint,
    Meter,
    Penalty,
    ReasoningBlock,
    ReasoningBudget,
    Sampler,
    greedy,
    stream_ids,
    stream_text,
)
from mlx_omnia.engine.model import (
    AtomicInput,
    ContentType,
    Modality,
    Model,
    ModelInput,
    ModelSignature,
    Wrapping,
)
from mlx_omnia.engine.parsers import FALLBACK, REASONING, Parser, Segment, Segmenter, opened
from mlx_omnia.engine.speculative import Chained

TEXT = ContentType(Modality.TEXT, "text/plain")

type TextContent = str | Iterator[str]
"""A prompt, whole or arriving in pieces.

An `Iterator` and not an `Iterable`, because `str` *is* an `Iterable[str]`: the union would
be equal to its second arm, and code that iterated the field without asking first would take
a whole prompt apart one character at a time — no error, just a different prompt. Iterating
is the distinction, so the type is what carries it.

One `ContentType` for both. What arrives in pieces is the same text as what arrives whole,
and a model that takes one takes the other; making the streamed form its own content type
would put a question in every `native_signature` that no checkpoint has a different answer
to."""


@dataclass(frozen=True)
class Text:
    value: TextContent
    parser: Parser | None = None
    """Which dialect the checkpoint speaks, as the source of the chat template that rendered
    this prompt says. It travels with the prompt because the prompt is all the streamer is
    handed — the template stays on the capability's side of `prepare` — and a `Text` no
    template rendered continues none and declares nothing."""

    @property
    def content_type(self) -> ContentType:
        return TEXT

    def read(self) -> str:
        """The whole prompt, which for one that arrives in pieces means waiting for the last.

        A method and not a property because it spends something: a `Text` built over a
        generator has one read in it, and the second comes back empty. The eager form is a
        `str` and re-reads freely, which is what every caller that came before this had."""
        return self.value if isinstance(self.value, str) else "".join(self.value)


@dataclass(frozen=True)
class LanguagePrompt:
    parts: tuple[AtomicInput, ...]

    @property
    def content_types(self) -> frozenset[ContentType]:
        return frozenset(part.content_type for part in self.parts)


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int
    sampler: Sampler = greedy
    stop: Collection[int] | None = None
    penalty: Penalty | None = None
    constraint: Constraint | None = None
    """The grammar this run decodes under, when the request asked for a strict schema. In the
    comparison, unlike the meter: a run under a grammar is not the generation a free run
    asks for. One walk belongs to one request — what two requests over the same schema share
    is the `Grammar` that opens them, not this."""
    meter: Meter | None = field(default=None, compare=False)
    """Where this run writes its numbers, when someone is counting. It travels with the
    options because it is the only way in: a caller holds the input and the options, and
    everything the count is made of — the rendered prompt, the ids — exists on the other
    side of `stream`. Out of the comparison: two requests asking for the same generation
    are the same options, whoever is measuring them."""
    prefix: Prefixes | None = field(default=None, compare=False)
    """Where this run's spans come from and go, `None` for a caller that keeps none.

    One handle and not a ceiling plus a spill: the store is the daemon's, one for every
    resident model, and what travels per model is which checkpoint wrote the bytes. It is
    here for the reason the meter is — out where a caller stands there is a conversation and
    not yet a prompt — and out of the comparison for the same reason: it says what the run
    may keep, not what it generates."""
    reasoning_budget: int | None = None
    """How many ids the reasoning block of this generation may spend, `None` for no cap.

    A number and not a `ReasoningBudget`, for the reason `prefix` is a handle: out
    where a caller stands there is a conversation and not yet a prompt, and what the closer
    is in ids only exists past the tokenizer. `TextLanguageModel.stream` is where both are,
    so that is where it is turned into the block the loop watches for.

    Part of the comparison: two requests differing only in how long the model may think are
    not the same generation."""
    speculate: bool = field(default=True, compare=False)
    """Whether this run may use the drafter the model was paired with, when it was paired
    with one. It is a permission and not a request: a model with no drafter, and a request
    that cannot be verified greedily, decode the same way either way.

    It travels here because the two levels that decide it — the model's own setting and the
    profile over it — are the caller's, and what holds the drafter is the facade. Out of
    the comparison for the same reason the meter is: two requests asking for the same
    generation are the same options, however each one was proposed."""
    context_limit: int | None = None
    """The checkpoint's own ceiling on prompt and generation together — the config's
    `max_position_embeddings`, when the caller read it. `max_tokens` alone cannot honour
    it: out where a caller stands the prompt is text, and how much of the window it takes
    only exists past `encode`. `None` asks for no cap, which is every caller that came
    before the field."""


class Tokenizer(Protocol):
    """Where a prompt may be cut is a fact of the tokenizer — its pre-tokenizer, its added
    tokens, whether it writes a BOS — so a prompt still arriving is handed over whole and each
    implementation answers for itself. One that can name no safe offset gathers the pieces and
    encodes once, which costs the overlap and never a different prompt."""

    def encode(self, text: TextContent) -> Iterator[int]: ...

    def decode_bytes(self, ids: list[int]) -> bytes: ...


def kept(content: TextContent, seen: list[str]) -> TextContent:
    """The same text with a copy taken, because the tokenizer spends it on the way to ids and
    the segmenter downstream still needs the prompt's tail to know which channel the
    generation starts in."""
    if isinstance(content, str):
        seen.append(content)
        return content

    def watched() -> Iterator[str]:
        for chunk in content:
            seen.append(chunk)
            yield chunk

    return watched()


def reasoning_budget(
    tokens: int | None, prompt: str, tokenizer: Tokenizer
) -> ReasoningBudget | None:
    """The budget the loop watches for: how many ids the block may spend, and that block in ids.

    Which block depends on the prompt. When the template left one open — Qwen3.6 writes
    `<think>` into the generation prompt, harmony opens the analysis channel — the budget is
    already inside it and there is exactly one spelling to close. When it did not, the model
    is the one that will open a block, and every spelling is watched: which of them this
    checkpoint writes is a fact of its vocabulary, and one whose tokenizer has no id for
    `<think>` encodes the marker as pieces the model never emits in that order, so the
    budget over it simply never arms.
    """
    if tokens is None:
        return None
    here = opened(prompt)
    if here is not None:
        opener, closer = here
        return ReasoningBudget(
            tokens,
            (ReasoningBlock(tuple(tokenizer.encode(opener)), tuple(tokenizer.encode(closer))),),
            inside=True,
        )
    blocks = tuple(
        ReasoningBlock(tuple(tokenizer.encode(opener)), tuple(tokenizer.encode(closer)))
        for opener, closer in REASONING
    )
    return ReasoningBudget(tokens, blocks)


@runtime_checkable
class Tokenizing(Protocol):
    tokenizer: Tokenizer


def tokenizer_of(model: object) -> Tokenizer | None:
    """The tokenizer of a loaded model, or `None` when nothing under the facades holds one —
    a test double is a `LanguageModel` and tokenizes nothing.

    What owns the tokenizer is the task-level model, and what `load` hands back is a
    `CompositeModel` over it: counting a prompt's tokens from outside means walking down
    `model` until the checkpoint's own tokenizer appears. Opening `tokenizer.json` instead
    would be a second dispatch by architecture to keep in sync with the loaders' — Gemma 3
    reads that file with a tokenizer of its own, not with the byte-level BPE.
    """
    while not isinstance(model, Tokenizing):
        if not isinstance(model, Wrapping):
            return None
        model = model.model
    return model.tokenizer


class Prefill(Protocol):
    """A trunk asked for its logits alone. The cache the loop threads through is not in the
    signature because nothing holding a `LanguageModel[ModelInput]` can name the element
    type of the list a given trunk builds — and every trunk defaults it away."""

    def __call__(self, ids: mx.array) -> mx.array: ...


def trunk_of(model: object) -> nn.Module | None:
    """The checkpoint's own tree under the facades, or `None` when the walk ends on
    something that is no `nn.Module` — a test double whose backend holds no tree.

    The same descent as `tokenizer_of`, one level lower: what a calibration pass needs is
    the tree the blocks live in, and the facades above it are `stream` and nothing else.
    Callers that also have to *run* it hold the result as a `Prefill`.
    """
    while isinstance(model, Wrapping):
        model = model.model
        if isinstance(model, nn.Module):
            return model
    return None


class LanguageModel[I: ModelInput](Model[I, Segment, GenerationOptions], Protocol):
    """A generation comes out as segments and not as text: what the model wrote, cut at the
    boundaries of the three channels a dialect has to tell apart — content, the reasoning
    block, a tool envelope.

    The channel is decided where the markers are known, which is inside the streamer: it reads
    the family off the prompt the checkpoint's own template rendered, and the channel that
    prompt left open. A consumer handed text and left to find the envelopes again is a second
    state machine over one generation, built out of what the first one already knew and threw
    away."""


def _batch_cache(model: api.LanguageModel[LayerCache]) -> SequenceCache:
    """The batch's cache and not the trunk's: a trunk's `make_cache` answers with its own
    element type, and what the batched path threads is one list of plain layers."""
    return list(model.make_cache())


@dataclass
class TextBatch:
    """Incremental text and parser state for one batched generation."""

    state: BatchSequence
    tokenizer: Tokenizer
    decoder: codecs.IncrementalDecoder
    segmenter: Segmenter
    closed: bool = False

    def push(self, token: int) -> tuple[Segment, ...]:
        piece = self.decoder.decode(self.tokenizer.decode_bytes([token]))
        return () if not piece else self.segmenter.push(piece)

    def finish(self) -> tuple[Segment, ...]:
        """The text side alone. The spans are the decode's own: `batching` closes the last of
        them on the step that retires the sequence, which is where the trunk stands still and
        where the promoted layers still exist."""
        if self.closed:
            return ()
        self.closed = True
        tail = self.decoder.decode(b"", final=True)
        return (*(() if not tail else self.segmenter.push(tail)), *self.segmenter.flush())


@dataclass
class TextPrefill:
    """A batched request whose prompt is still being fed, one block per `advance`.

    Everything the finished `TextBatch` needs is decided when the request arrives; what is
    left to do is the prefill, and it is what the scheduler spends a tick on.
    """

    prefill: BatchPrefill
    tokenizer: Tokenizer
    decoder: codecs.IncrementalDecoder
    segmenter: Segmenter

    def advance(self) -> TextBatch | None:
        state = self.prefill.advance()
        if state is None:
            return None
        return TextBatch(state, self.tokenizer, self.decoder, self.segmenter)


@runtime_checkable
class ContinuousLanguageModel(Protocol):
    def can_batch(self, options: GenerationOptions) -> bool: ...

    def begin_batch(self, input: ModelInput, options: GenerationOptions) -> TextPrefill | None: ...

    def prepare_batch(self, input: ModelInput, options: GenerationOptions) -> TextBatch | None: ...

    def step_batch(self, sequences: Sequence[TextBatch]) -> list[tuple[Segment, ...]]: ...


class TextLanguageModel[I: ModelInput = Text]:
    def __init__(
        self,
        model: api.LanguageModel[LayerCache],
        tokenizer: Tokenizer,
        *,
        stop: Collection[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.stop = stop
        self.drafter: nn.Module | None = None
        """`speculative.Drafting`. Out of the trunk's tree so whoever accounts for memory can
        weigh it: with an MTP head the two are one checkpoint on disk and still two trees
        here, and only one of them is under `self.model`."""
        self._block: int | None = None

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[I]:
        return isinstance(input, Text)

    def speculate_with(self, drafter: nn.Module, *, block_size: int | None = None) -> None:
        """Take an MTP step for this trunk — `core.api.Drafting`.

        Here and not on a family's facade because nothing in it is a family's: the step is a
        `core.api.Step`, and the trunk lends it the two ends of its vocabulary and the rows it
        conditions on through `core.api.Draftable`. A family that implements both speculates
        through this; one that does not is refused by the name of what it is missing, which is
        the only way a checkpoint id can be wrong here.

        A DFlash drafter does not come through this door. Its proposal is one forward over a
        block of mask tokens against layers the *drafter's* own config names, so the pair
        needs that config to build — `models/muse_glimmer` owns it and its own facade.
        """
        if not isinstance(drafter, api.Step):
            raise TypeError(f"{type(drafter).__name__} is not an MTP step for this engine")
        if not isinstance(self.model, api.Draftable):
            raise TypeError(
                f"{type(self.model).__name__} does not lend its embedding table, its head and "
                "the blocks the step conditions on (`core.api.Draftable`), which is the whole "
                "of what an MTP step borrows"
            )
        block = drafter.block if block_size is None else block_size
        if block < 2:
            raise ValueError(f"a block of {block} proposes nothing")
        self.drafter = drafter
        self._block = block

    def _proposer(self, options: GenerationOptions) -> api.Proposer | None:
        """The proposer for this request, or `None` for every request that cannot be verified
        against the target's own argmax.

        Not a refusal: a request that samples, penalizes or decodes under a grammar is
        answered the way it would have been with no head loaded at all. What the switch buys
        is speed, so the case where it buys nothing is a case where it does nothing.

        One per generation — the chain's state is this request's accepted prefix.
        """
        drafter = self.drafter
        if drafter is None or self._block is None or not options.speculate:
            return None
        if options.sampler is not greedy or options.penalty is not None:
            return None
        if options.constraint is not None or options.reasoning_budget is not None:
            return None
        assert isinstance(drafter, api.Step)
        assert isinstance(self.model, api.Draftable)
        # `-1`: the step conditions on the trunk's last block, and `block_outputs` indexes the
        # list Python's way. A trunk that grows a layer still taps the right one.
        return Chained(self.model, drafter, block=self._block, tap=-1)

    def can_batch(self, options: GenerationOptions) -> bool:
        """Always. Continuous batching used to be an attestation every family made and none
        declined; what it needs of a trunk is `make_cache` and the forward, which is the
        whole of what a trunk is."""
        return True

    def begin_batch(self, input: ModelInput, options: GenerationOptions) -> TextPrefill | None:
        """Everything `prepare_batch` decides before the first forward, with the prefill left
        for the caller to spend a block at a time."""
        model = self.model
        if not isinstance(input, Text):
            return None
        prompt = input.read()
        encoded = list(self.tokenizer.encode(prompt))
        budget = options.max_tokens
        if options.context_limit is not None:
            budget = min(budget, max(0, options.context_limit - len(encoded)))
        proposer = self._proposer(options)
        cache = _batch_cache(model)
        # A proposer that keeps a row per position contributes those layers to the walk
        # (`Proposer.caches`), so the spans carry the drafter's rows beside the target's
        # and a resumed prompt resumes both — the round and the prefix compose instead of
        # one dropping the other.
        walk = (
            options.prefix.begin(cache, () if proposer is None else proposer.caches)
            if options.prefix is not None
            else None
        )
        reused = 0 if walk is None else walk.resume(encoded, cache)
        return TextPrefill(
            BatchPrefill(
                model,
                encoded,
                cache,
                0,
                reused,
                budget,
                options.sampler,
                self.stop if options.stop is None else options.stop,
                options.penalty,
                options.constraint,
                options.meter,
                reasoning_budget(options.reasoning_budget, prompt, self.tokenizer),
                proposer,
                walk,
            ),
            self.tokenizer,
            codecs.getincrementaldecoder("utf-8")("replace"),
            Segmenter(FALLBACK if input.parser is None else input.parser, prompt=prompt),
        )

    def prepare_batch(self, input: ModelInput, options: GenerationOptions) -> TextBatch | None:
        """Prepare a text request for continuous batching when its trunk supports it."""
        begun = self.begin_batch(input, options)
        if begun is None:
            return None
        while (batch := begun.advance()) is None:
            pass
        return batch

    def step_batch(self, sequences: Sequence[TextBatch]) -> list[tuple[Segment, ...]]:
        """Advance text requests through one shared decode forward."""
        model = self.model
        active = [index for index, sequence in enumerate(sequences) if not sequence.state.finished]
        emitted: list[list[int]] = [[] for _ in sequences]
        advanced = batch_step(model, [sequences[index].state for index in active]) if active else []
        for index, ids in zip(active, advanced, strict=True):
            emitted[index] = ids
        return [
            (
                *(segment for token in ids for segment in sequence.push(token)),
                *(() if not sequence.state.finished else sequence.finish()),
            )
            for sequence, ids in zip(sequences, emitted, strict=True)
        ]

    def _stream_batched(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        """One batched generation driven synchronously, a tick at a time.

        The same `prepare_batch`/`step_batch` pair the server's scheduler runs, over a batch
        of one: the prefix trie, the clamp, the budget and the meter are all decided inside
        `prepare_batch`, and the final flush is what `step_batch` does when the sequence
        retires.
        """
        state = self.prepare_batch(input, options)
        if state is None:
            return
        while not state.state.finished:
            yield from self.step_batch([state])[0]

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        """The generation, as segments.

        A whole-text prompt goes through `prepare_batch`/`step_batch`; what still decodes
        here is what that path does not answer for — a prompt still arriving as an iterator.
        """
        if isinstance(input.value, str) and self.can_batch(options):
            yield from self._stream_batched(input, options)
            return
        draft = self._proposer(options)
        # Three options are answers about the whole prompt and are read before its first
        # forward: how much window is left, which reasoning block the template left open, and
        # the trie's match. A prompt still being written is waited for when one of them is
        # asked for — it costs the overlap and never a row — and pulled when none is.
        whole = (
            input.read()
            if options.context_limit is not None
            or options.reasoning_budget is not None
            or options.prefix is not None
            or draft is not None
            else None
        )
        seen: list[str] = []
        prompt = kept(whole if whole is not None else input.value, seen)
        # A prompt that is all here has its ids read out now. The tokenizer hands them over
        # lazily whatever it was given, and a lazy prompt is what makes `stream_ids` prefill
        # in the small blocks that overlap a producer — spending those on a prompt with no
        # producer behind it would be paying the latency knob for nothing.
        encoded: Iterable[int] = (
            list(self.tokenizer.encode(prompt))
            if isinstance(prompt, str)
            else self.tokenizer.encode(prompt)
        )
        # The clamp lives here because this is where the prompt's cost in ids first
        # exists: the budget is what the window has left, and a prompt that already
        # spent it generates nothing rather than pushing positions past the config's.
        budget = options.max_tokens
        if options.context_limit is not None and isinstance(encoded, list):
            budget = min(budget, max(0, options.context_limit - len(encoded)))
        ids = stream_ids(
            self.model,
            encoded,
            max_tokens=budget,
            sampler=options.sampler,
            stop=self.stop if options.stop is None else options.stop,
            penalty=options.penalty,
            meter=options.meter,
            prefix=options.prefix,
            constraint=options.constraint,
            reasoning_budget=reasoning_budget(
                options.reasoning_budget, whole if whole is not None else "", self.tokenizer
            ),
            draft=draft,
        )
        # Drawn here rather than inside `stream_text`, and that is what makes the prompt's
        # own text available to it: pulling the first id runs the prefill, which is what
        # consumes the prompt, which is what fills `seen`. The segmenter reads the prompt's
        # tail to know which channel the generation starts in, and a tail read before this
        # would be the tail of a prompt that had not finished arriving.
        first = next(ids, None)
        if first is None:
            return
        # No dialect — unknown, or a prompt no template rendered — still holds the reasoning
        # block back: that one is the model's own spelling and not the dialect's.
        yield from stream_text(
            chain((first,), ids),
            self.tokenizer,
            parser=input.parser,
            prompt="".join(seen),
        )

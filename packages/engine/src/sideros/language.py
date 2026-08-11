from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeIs, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import LayerCache
from sideros.core.prompt_cache import Budget, PromptCache, Spill
from sideros.generate import (
    CausalLM,
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
from sideros.model import (
    AtomicInput,
    ContentType,
    Modality,
    Model,
    ModelInput,
    ModelSignature,
    Wrapping,
)
from sideros.parsers import REASONING, Parser, Segment, opened

TEXT = ContentType(Modality.TEXT, "text/plain")


@dataclass(frozen=True)
class Text:
    value: str
    parser: Parser | None = None
    """Which dialect the checkpoint speaks, as the source of the chat template that rendered
    this prompt says. It travels with the prompt because the prompt is all the streamer is
    handed — the template stays on the capability's side of `prepare` — and a `Text` no
    template rendered continues none and declares nothing."""

    @property
    def content_type(self) -> ContentType:
        return TEXT


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
    prefix_budget: Budget | int = field(default=0, compare=False)
    """How many bytes of prefix cache this run may keep between requests, 0 for none — a
    `Budget` when that ceiling is shared with the other resident models, a bare number when
    this trie is the only one under it.

    It travels here for the reason the meter does — out where a caller stands there is a
    conversation and not yet a prompt — and it is a ceiling rather than the cache itself
    because `PromptCache[KVCache]` and `PromptCache[DeltaCache]` are two types, and nothing
    above `stream` can name which one the model builds. `Budget` is what survives that: it
    holds no cache, only the arithmetic over tries that do. Out of the comparison for the same
    reason as the meter: it says what the run may keep, not what it generates."""
    prefix_spill: Spill[Any] | None = field(default=None, compare=False)
    """Where an evicted prefix goes and where a miss looks, `None` for a trie that only
    forgets. Beside `prefix_budget` and for the same reasons — it is about what the run may
    keep and not about what it generates — with one of its own: what implements it is the
    caller's, because a directory, a key and a ceiling are a daemon's business and a trie of
    caches has none.

    `Any` in the parameter is the same wall `prefix_budget` meets: the element type is the
    trunk's own cache class, which nothing holding a `LanguageModel[ModelInput]` can name.
    What narrows it is `TextLanguageModel`, on the other side of `stream`."""
    reasoning_budget: int | None = None
    """How many ids the reasoning block of this generation may spend, `None` for no cap.

    A number and not a `ReasoningBudget`, for the reason `prefix_budget` is a number: out
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
    def encode(self, text: str) -> list[int]: ...

    def decode_bytes(self, ids: list[int]) -> bytes: ...


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


def prefix_cache[C: LayerCache](
    prefix: PromptCache[C] | None, budget: Budget | int, spill: Spill[C] | None = None
) -> PromptCache[C] | None:
    """The trie this request generates against: none while the budget is 0, the one already
    there when it has not moved, a new one when it has.

    A changed budget drops what was cached instead of trimming to fit, and that is what makes
    the setting `applied` rather than `restart`: the trie's ceiling is fixed at construction,
    so honouring a PATCH means building again. A cache lost costs one prefill; a ceiling that
    only counted for models loaded afterwards would be a number the screen shows and the
    daemon ignores.

    A shared `Budget` is compared by identity and a bare number by value, and both say the
    same thing: the ceiling moved. The daemon hands the same object to every resident model
    for as long as the config holds and builds another when it is PATCHed, so identity is what
    "moved" means there; a caller with a plain int has no object to have kept.

    The spill is not part of that comparison. It is the same object for the life of a
    residency — one model, one directory, one key — so rebuilding on it would be rebuilding
    on identity, and a caller that constructs it per request would drop the trie every time.
    """
    total = budget.total if isinstance(budget, Budget) else budget
    if total <= 0:
        return None
    if prefix is not None:
        kept = (
            prefix.budget is budget
            if isinstance(budget, Budget)
            else prefix.budget.total == budget
        )
        if kept:
            return prefix
    return PromptCache(budget, spill)


class TextLanguageModel[C: LayerCache]:
    def __init__(
        self,
        model: CausalLM[C],
        tokenizer: Tokenizer,
        *,
        stop: Collection[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.stop = stop
        self.prefix: PromptCache[C] | None = None
        """What this model kept from the requests before. It lives on the model and not on
        the engine because its element type is this trunk's own cache, which nothing holding
        a `LanguageModel[ModelInput]` can name — and because dying with the model is exactly
        the invariant "one prompt in two resident models does not cross caches"."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        self.prefix = prefix_cache(self.prefix, options.prefix_budget, options.prefix_spill)
        prompt = self.tokenizer.encode(input.value)
        # The clamp lives here because this is where the prompt's cost in ids first
        # exists: the budget is what the window has left, and a prompt that already
        # spent it generates nothing rather than pushing positions past the config's.
        budget = options.max_tokens
        if options.context_limit is not None:
            budget = min(budget, max(0, options.context_limit - len(prompt)))
        ids = stream_ids(
            self.model,
            prompt,
            max_tokens=budget,
            sampler=options.sampler,
            stop=self.stop if options.stop is None else options.stop,
            penalty=options.penalty,
            meter=options.meter,
            prefix=self.prefix,
            constraint=options.constraint,
            reasoning_budget=reasoning_budget(
                options.reasoning_budget, input.value, self.tokenizer
            ),
        )
        # No dialect — unknown, or a prompt no template rendered — still holds the reasoning
        # block back: that one is the model's own spelling and not the dialect's.
        yield from stream_text(ids, self.tokenizer, parser=input.parser, prompt=input.value)

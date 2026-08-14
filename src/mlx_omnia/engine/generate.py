"""Lazy generation pipeline.

The sampler returns the token as an mx.array (no .item() before the next step is
queued); step n+1 is async-evaluated before step n's sync, so the GPU never idles
between steps.
"""

import codecs
import time
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Protocol, runtime_checkable

import mlx.core as mx

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.cache_file import policy
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.core.prefill import ARRIVING, BLOCK, pulled
from mlx_omnia.engine.core.prompt_cache import PromptCache
from mlx_omnia.engine.parsers import FALLBACK, Parser, Segment, Segmenter
from mlx_omnia.engine.speculative import Acceptance, SpeculationRefused, stream_speculative_ids

type Sampler = Callable[[mx.array], mx.array]
type LogitFilter = Callable[[mx.array], mx.array]
type Penalty = Callable[[mx.array, mx.array], mx.array]


class CausalLM[C: LayerCache](Protocol):
    def make_cache(self) -> list[C]: ...

    def __call__(self, ids: mx.array, cache: list[C] | None = None) -> mx.array: ...


@runtime_checkable
class BlockOutputs[C: LayerCache](Protocol):
    """A forward that also hands back what the trunk held at depths the caller names.

    The same computation as `__call__` — the ids go in once — with a second thing coming
    out of it: `at` selects blocks by index, and the features are their outputs
    concatenated on the last dim, `[batch, length, len(at) * hidden]`. Selection is the
    caller's because the whole trunk is 52 tensors where the reader wants five: returning
    all of them costs 1.4 GB per prefill block against 136 MB for the ones asked for.

    What reads it is a proposer that conditions on the target's own reading of the context
    instead of on its ids (`speculative.Proposer`). Nothing else in the package calls it,
    and a model that does not implement it can still be spoken to and still be drafted for
    — by a proposer that reads no blocks.
    """

    def block_outputs(
        self, ids: mx.array, cache: list[C], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]: ...


@runtime_checkable
class CompiledDecode[C: LayerCache](Protocol):
    def compile_decode(self, cache: list[C]) -> Callable[[mx.array], mx.array]: ...


@runtime_checkable
class CompiledGreedyDecode[C: LayerCache](Protocol):
    def compile_greedy_decode(self, cache: list[C]) -> Callable[[mx.array], mx.array]: ...


class Constraint(Protocol):
    """A grammar's side of the loop: which ids the sampler may see, and the id it drew.

    It is not a `LogitFilter` because a filter has no memory of the run — the mask of step
    n+1 is a function of the id step n drew, and advancing over that id is a side effect per
    token that `LogitFilter` has nowhere to put. `mlx_omnia.engine.grammar` is the implementation.

    `mask` takes the logits and returns them with every forbidden id at -inf; `remaining`
    counts the steps the run still has, this one included, so a constraint can force what
    closes the output before the budget runs out instead of letting it truncate. `accept`
    advances over the id just drawn and answers `False` once the grammar has ended.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array: ...

    def accept(self, token: int) -> bool: ...


class ConstraintConflict(ValueError):
    """Two things asking for the same step. The one pair that exists is a grammar and a
    reasoning budget, and it is named where it is raised."""


@dataclass(frozen=True)
class ReasoningBlock:
    """One spelling of the reasoning block, in ids: what opens it and what closes it.

    Tokenized rather than matched over text because the budget has to act on the step after
    the one it read, and the decoded text of a step exists only past the incremental UTF-8
    decoder in `stream_text`, which is downstream of the loop. A checkpoint whose tokenizer
    has no single id for `<think>` encodes the marker as ordinary pieces the model never
    emits as that sequence, and the budget over it never arms — which is the same thing as
    a checkpoint that does not reason.
    """

    open: tuple[int, ...]
    close: tuple[int, ...]


@dataclass(frozen=True)
class ReasoningBudget:
    """How many ids a generation may spend inside the reasoning block, and how to end it.

    The budget is enforced by *feeding* the closing ids, not by masking the logits into
    them: the closer is a fixed sequence, so once the budget is spent there is nothing left
    to draw and the loop can hand the ids over itself. That is why a budget costs no sync —
    unlike a `Constraint`, whose mask for step n+1 is a function of the id step n drew — and
    all it spends is the one queued step discarded on the turn it arms.

    `inside` says the prompt already left a block open, which is what Qwen3.6's template
    does when it writes `<think>` into the generation prompt. Then `blocks` holds the one
    block it opened; otherwise it holds every spelling the model might open by itself.

    One block per generation: once the reasoning has ended — because the model closed it or
    because this forced it — the budget is spent, and a model that opens a second block
    writes it out in full. Re-entry is rare enough that counting it would be state kept for
    a case nobody has measured.
    """

    tokens: int
    blocks: tuple[ReasoningBlock, ...]
    inside: bool = False


class _Budget:
    """One generation's walk through `ReasoningBudget`: the ids seen, and what is owed."""

    def __init__(self, spec: ReasoningBudget) -> None:
        self._blocks = spec.blocks
        self._left = spec.tokens
        self._block: ReasoningBlock | None = spec.blocks[0] if spec.inside else None
        self._window = max(
            (len(marker) for block in spec.blocks for marker in (block.open, block.close)),
            default=1,
        )
        self._recent: list[int] = []
        self._spent = False

    def emitted(self, token: int) -> tuple[int, ...]:
        """The ids the loop owes after this one: the closer, on the single step the budget
        runs out with the block still open, and nothing on every other step."""
        if self._spent:
            return ()
        self._recent.append(token)
        del self._recent[: -self._window]
        if self._block is None:
            for block in self._blocks:
                if self._tail(block.open):
                    self._block = block
                    break
            return ()
        if self._tail(self._block.close):
            self._spent = True
            return ()
        self._left -= 1
        if self._left > 0:
            return ()
        self._spent = True
        return self._block.close

    def _tail(self, marker: tuple[int, ...]) -> bool:
        return len(self._recent) >= len(marker) and tuple(self._recent[-len(marker) :]) == marker


class Detokenizer(Protocol):
    """What `stream_text` needs, and all the two model facades have in common."""

    def decode_bytes(self, ids: list[int]) -> bytes: ...


class TextTokenizer(Detokenizer, Protocol):
    """What `stream_generate` needs: the round trip, plus the vocabulary it reads the
    default stop out of."""

    @property
    def encoder(self) -> dict[str, int]: ...

    def encode(self, text: str) -> list[int]: ...


def greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def temperature(value: float) -> LogitFilter:
    """Divides the logits: below 1 sharpens the distribution, above 1 flattens it."""
    if value <= 0:
        raise ValueError(f"temperature must be positive: {value}")
    return lambda logits: logits / value


def top_k(k: int) -> LogitFilter:
    """Keeps the k largest logits. A tie at the k-th value keeps the whole tied block:
    breaking it would make the choice depend on the vocabulary's order."""
    if k < 1:
        raise ValueError(f"top_k must be at least 1: {k}")

    def apply(logits: mx.array) -> mx.array:
        kth = mx.min(mx.topk(logits, min(k, logits.shape[-1]), axis=-1), axis=-1, keepdims=True)
        return mx.where(logits < kth, -float("inf"), logits)

    return apply


def top_p(p: float) -> LogitFilter:
    """Keeps the smallest set of tokens whose probabilities add up to `p`."""
    if not 0 < p <= 1:
        raise ValueError(f"top_p must be in (0, 1]: {p}")

    def apply(logits: mx.array) -> mx.array:
        probs = softmax(logits, axis=-1, precise=True)
        ordered = mx.sort(probs, axis=-1)[..., ::-1]
        # Mass covered by the strictly more probable tokens: the first token below `p`
        # is still kept, which is what makes the kept set reach `p` instead of stopping
        # short of it.
        covered = mx.cumsum(ordered, axis=-1) - ordered
        cutoff = mx.min(mx.where(covered < p, ordered, float("inf")), axis=-1, keepdims=True)
        return mx.where(probs < cutoff, -float("inf"), logits)

    return apply


def min_p(p: float) -> LogitFilter:
    """Keeps the tokens at least `p` times as probable as the most probable one."""
    if not 0 <= p < 1:
        raise ValueError(f"min_p must be in [0, 1): {p}")

    def apply(logits: mx.array) -> mx.array:
        probs = softmax(logits, axis=-1, precise=True)
        floor = p * mx.max(probs, axis=-1, keepdims=True)
        return mx.where(probs < floor, -float("inf"), logits)

    return apply


def repetition_penalty(value: float, *, context: int = 20) -> Penalty:
    """Pushes down the ids seen in the last `context` positions, the prompt included.
    A positive logit is divided and a negative one multiplied, so the correction keeps
    its direction on both sides of zero."""
    if value <= 0:
        raise ValueError(f"repetition_penalty must be positive: {value}")

    def penalize(logits: mx.array, history: mx.array) -> mx.array:
        if history.size == 0:
            return logits
        recent = history[-context:][None]
        seen = mx.take_along_axis(logits, recent, axis=-1)
        return mx.put_along_axis(
            logits, recent, mx.where(seen > 0, seen / value, seen * value), axis=-1
        )

    return penalize


def sampler(*filters: LogitFilter, seed: int | None = None) -> Sampler:
    """Draws from the distribution the filters shape, in the order given. The draw runs
    in fp32: in bf16 the tail the filters just carved rounds back into the pile."""
    key = None if seed is None else mx.random.key(seed)

    def sample(logits: mx.array) -> mx.array:
        nonlocal key
        shaped = logits.astype(mx.float32)
        for stage in filters:
            shaped = stage(shaped)
        if key is None:
            return mx.random.categorical(shaped, axis=-1)
        keys = mx.random.split(key)
        key = keys[0]
        return mx.random.categorical(shaped, axis=-1, key=keys[1])

    return sample


def _default_stop(tokenizer: TextTokenizer) -> tuple[int, ...]:
    end = tokenizer.encoder.get("<|endoftext|>")
    return () if end is None else (end,)


@dataclass
class Meter:
    """Where a generation writes its numbers: the ids counted, and the clock marks that
    separate prefill from decode.

    It is filled inside the loop because that is where the numbers exist: above it the
    prompt is still a conversation the chat template has to render, and the output is text
    a detokenizer has already joined. Reading the clock costs no synchronization — every
    mark is taken around the `.item()` the loop already pays for, so the lazy step keeps
    its one sync per token.
    """

    prompt_tokens: int = 0
    reused_tokens: int = 0
    """How much of `prompt_tokens` came out of a prefix cache instead of a forward. It is
    beside the prompt and not subtracted from it: the prompt is what the request sent, and
    that is what a usage count means. What it changes is every rate whose denominator is the
    prefill — `ttft` covers the tail alone, so dividing the whole prompt by it measures
    nothing — and it is the only number that says whether the reuse hit at all: a near miss
    and a cold conversation are the same request from the outside."""
    kept_prefix: bool = False
    """Whether this run left anything the next turn can start from. Beside `reused_tokens`
    because zero reuse has three causes that look identical from outside — the budget is off,
    the conversation diverged, or this trunk cannot keep a prefix at all — and only the last
    one is permanent. A trunk whose layers neither rewind nor serialize keeps nothing, and
    nothing anywhere said so: the request answered normally and the next one prefilled whole,
    for ever."""
    completion_tokens: int = 0
    prefill_started: float | None = None
    first_token: float | None = None
    last_token: float | None = None
    speculation: Acceptance = field(default_factory=Acceptance)
    """What a drafter proposed and how much of it survived, all zeroes for a run that had
    none. Here and not beside it because a caller measuring a generation is measuring this
    too: whether the run speculated at all is not visible in the rate — a model paired with
    a drafter still decodes plainly under a sampler, and the difference between the two is
    a number and not a setting."""

    def prefill(self, prompt_tokens: int, reused: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.reused_tokens = reused
        self.prefill_started = time.perf_counter()

    def token(self) -> None:
        """One id emitted. Counted before it is yielded: a consumer that walks away has
        still received it, and the count is what its usage is made of."""
        now = time.perf_counter()
        self.completion_tokens += 1
        if self.first_token is None:
            self.first_token = now
        self.last_token = now

    @property
    def ttft(self) -> float | None:
        """Prompt in, first token out: the prefill, plus the step that draws from it."""
        if self.prefill_started is None or self.first_token is None:
            return None
        return self.first_token - self.prefill_started

    @property
    def decode_seconds(self) -> float | None:
        if self.first_token is None or self.last_token is None:
            return None
        return self.last_token - self.first_token

    @property
    def tokens_per_second(self) -> float | None:
        """Decode only. The first token belongs to `ttft`, so the rate covers the ones
        after it — the same split `omnia-bench interleaved` reports."""
        elapsed = self.decode_seconds
        if elapsed is None or elapsed <= 0 or self.completion_tokens < 2:
            return None
        return (self.completion_tokens - 1) / elapsed


def _boundary[C: LayerCache](model: CausalLM[C], cache: list[C]) -> list[C] | None:
    """A second cache standing where this one stands now, or `None` when the trunk has no
    use for one — every layer rewinds, so the trie can cut a longer entry back to this point
    whenever it needs to.

    Nothing is copied. `stored()` hands over the rows in use, and they are rows nobody writes
    again: `KVCache` only ever writes at `offset` and past it, and the recurrent layers
    reassign their state rather than filling it in place — which is what `LayerCache
    .checkpoint` already relies on. What this costs is holding those buffers alive, which is
    exactly what storing a prefix means.

    The `mx.eval` is the house rule about lazy arrays and not a formality: unevaluated, the
    entry would keep the whole prefill graph alive until some later request read it.
    """
    if all(layer.is_trimmable for layer in cache):
        return None
    if not all(layer.is_storable for layer in cache):
        return None
    standing = model.make_cache()
    if policy(standing) != policy(cache):
        # The same check the disk key makes, made here because in memory there is no key to
        # make it: a cache the live trunk is not the shape of any more — a ring promoted for
        # a compiled decode is the case — hands over bytes a fresh layer would read under its
        # own rules. Same names, same shapes, a rotation taken for a history.
        return None
    for target, source in zip(standing, cache, strict=True):
        target.restore(source.rows, source.stored())
    mx.eval([tensor for layer in standing for tensor in layer.tensors])
    return standing


def stream_ids[C: LayerCache, D: LayerCache](
    model: CausalLM[C],
    prompt: Iterable[int],
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    meter: Meter | None = None,
    draft: CausalLM[D] | None = None,
    lookahead: int = 4,
    acceptance: Acceptance | None = None,
    prefix: PromptCache[C] | None = None,
    constraint: Constraint | None = None,
    reasoning_budget: ReasoningBudget | None = None,
) -> Generator[int]:
    """`penalty` reads the ids as an mx.array — prompt first, newest last — so the token
    just drawn joins the history without a round trip to the host: reading it back here
    would cost the step of lookahead the loop is built around.

    `meter`, when someone is counting, is filled as the loop runs: the prompt before the
    first step, then one mark per id emitted.

    `prefix` is where the cache comes from and where it goes back: the longest stored prefix
    of `prompt` is taken over and only the rest is prefilled, and at the end — the end a
    consumer that walks away imposes included — the cache is inserted back under the ids
    that in fact entered it. The match is here because here is where the ids are: above this
    the prompt is still a conversation the chat template has to render, which is the same
    wall the `meter` met. Nothing is taken until the first `next`: the generator's body runs
    then, so building the iterator and dropping it leaves the trie untouched.

    Keying on the ids is what makes the template, the effort asked for and the declared tools
    part of the key without this ever naming them: they change the render, the render changes
    the ids, and the match ends where the ids do. One trie belongs to one model — the ids of
    two checkpoints are two alphabets, and nothing in a cache says which one wrote it.

    A `reasoning_budget` takes the ids over instead of shaping them: when the block runs out
    of budget the closer is fed the way a drawn id is, so the cache gets its rows and the
    consumer reads the marker the segmenter is waiting for. It costs the step queued behind
    the id it arms on, once, and no sync — the value of every id was already read back.

    A `constraint` is the one option that changes the shape of the loop rather than the
    numbers inside it: its mask for step n+1 is a function of the id step n drew, so the
    `.item()` that used to happen behind the queue of n+1 now happens in front of it. The
    GPU idles for the length of that sync once per token, and that is what a constrained
    request costs — it is the contract, not an implementation detail.

    A `draft` moves the whole run onto the speculative loop, which emits this same greedy
    stream token for token — `lookahead` and `acceptance` are read only there. Without one
    nothing below changes.

    `prompt` may still be arriving. A sequence is prefilled the way it always was; anything
    else is pulled block by block, so a caller whose prompt is being written by something
    slower — a caption coming out of a second model — has this one's forwards running against
    that production instead of after it. Two things want the whole list and therefore wait for
    it: `prefix`, whose trie matches ids it has to have, and `draft`. What a lazy prompt pays
    them is the overlap, never a row."""
    if draft is not None:
        if sampler is not greedy:
            raise SpeculationRefused(
                "speculation is greedy-only: the acceptance rule that keeps a sampled "
                "distribution needs the draft's and the target's probabilities, and a "
                "`Sampler` hands back an id and no distribution at all"
            )
        if penalty is not None:
            raise SpeculationRefused(
                "speculation and `penalty` do not compose: a verification row would have to "
                "be penalized against a history the round has not committed yet"
            )
        if prefix is not None:
            raise SpeculationRefused(
                "speculation and prefix reuse are not wired together: a round rewinds the "
                "target's cache and the draft's in step, and the trie holds one of the two"
            )
        if constraint is not None:
            raise SpeculationRefused(
                "speculation and a grammar are not wired together: every draft id would "
                "need its own mask, and a rejected round would have to rewind the matcher "
                "by exactly what it rewinds the caches"
            )
        if reasoning_budget is not None:
            raise SpeculationRefused(
                "speculation and a reasoning budget are not wired together: the closer would "
                "have to be fed into the target and the draft in step, and a rejected round "
                "would have to give back ids the budget has already counted"
            )
        yield from stream_speculative_ids(
            model,
            draft,
            prompt,
            max_tokens=max_tokens,
            lookahead=lookahead,
            stop=stop,
            meter=meter,
            # A caller counting the run gets the counts without asking: whether a request
            # speculated at all is not in its rate, and `acceptance` is the only place the
            # answer exists. An explicit one still wins — it is a caller with somewhere
            # else to put them.
            acceptance=acceptance if acceptance is not None or meter is None else meter.speculation,
        )
        return

    cache = model.make_cache()
    if prefix is not None and not isinstance(prompt, Sequence):
        # The trie descends the ids one by one and `take` sizes its match against the whole
        # list, so a prompt still being produced is waited for here rather than matched
        # against a piece of itself — a match cut short at a block boundary would be reuse
        # silently lost on exactly the long conversations the trie exists for.
        prompt = list(prompt)
    # The fresh cache goes in as well as out: a trie backed by disk fills *this* list rather
    # than handing one back, because a file has no cache in it — only the rows one would
    # hold, and which class holds them is the trunk's answer and nobody else's.
    reuse = None if prefix is None else prefix.take(prompt, into=cache)
    if reuse is not None:
        cache = reuse.caches
    # What the cache holds, row for row, for the insert at the end: the prompt, whether its
    # head was reused or prefilled, plus every id the loop feeds back in. It is filled as the
    # prompt arrives, which for a sequence is all at once.
    covered: list[int] = []
    prompt_length = 0
    history = mx.array([], dtype=mx.int32)
    if meter is not None:
        # The clock starts here whatever shape the prompt is in: for one still arriving, the
        # wait for its producer is part of the TTFT this measures. The count is set below,
        # where it exists.
        known = len(prompt) if isinstance(prompt, Sequence) else 0
        meter.prefill(known, 0 if reuse is None else reuse.length)

    decode: Callable[[mx.array], mx.array] | None = None

    def step(ids: mx.array, seen: mx.array) -> mx.array:
        logits = model(ids[None], cache)[:, -1, :] if decode is None else decode(ids)
        if penalty is not None:
            logits = penalty(logits, seen)
        if constraint is not None:
            # Last, so nothing downstream can put a forbidden id back: a filter divides and
            # compares, and -inf survives both.
            logits = constraint.mask(logits, max_tokens - (len(covered) - prompt_length))
        return sampler(logits)[0]

    def advance(drawn: mx.array) -> mx.array:
        nonlocal history
        if penalty is not None:
            history = mx.concatenate([history, drawn[None]])
        queued = step(drawn[None], history)
        mx.async_eval(queued)
        return queued

    if reasoning_budget is not None and constraint is not None:
        # Refused rather than made inert: the closer is fed past the mask, and the matcher —
        # which is advanced over every id this yields — would then be asked to accept an id
        # it had forbidden. A grammar leaves no reasoning block to budget anyway: it forces
        # the document from the first step, so the block a template opened never closes.
        raise ConstraintConflict(
            "a reasoning budget and a grammar do not compose: the forced closer bypasses the "
            "mask the matcher is advanced under"
        )
    budget = None if reasoning_budget is None else _Budget(reasoning_budget)
    boundary: list[C] | None = None
    """Bound before the `try`, because the `finally` reads it and a forward that raises on
    the first block never reaches the line that fills it."""
    standing: list[C] | None = None
    """The growing layers as they stood at the end of the prompt, kept aside when a compile
    below promotes `cache` in place. Promotion copies the rows into fixed buffers and never
    writes these again, so holding them is the prompt's entry for free — the trie stores
    this, rewindable, and never the promoted shape it cannot rewind."""
    promoted = False
    owed: list[int] = []
    """The closer, between the step the budget armed on and the steps that feed it out."""

    try:
        # What the reuse did not already cover, fed in blocks: only the last block's logits
        # are read, and the blocks before it never compute the head at all.
        arriving = iter(prompt)
        if reuse is not None:
            covered.extend(islice(arriving, reuse.length))

        def feed(ids: list[int]) -> object:
            covered.extend(ids)
            return model(mx.array(ids)[None], cache)

        head = pulled(
            feed, arriving, cache, block=BLOCK if isinstance(prompt, Sequence) else ARRIVING
        )
        covered.extend(head)
        prompt_length = len(covered)
        history = mx.array(covered)
        if meter is not None:
            # The count only exists now for a prompt that was arriving; `prefill` above
            # started the clock, and the wait for the producer belongs inside the TTFT it
            # measures.
            meter.prompt_tokens = prompt_length
        y = step(mx.array(head), history)
        mx.async_eval(y)
        # The cut the next turn will actually match, on a trunk that cannot be rewound to it
        # later. The entry this loop leaves behind ends in the ids the model wrote, and a
        # client re-rendering the conversation reproduces them only when it echoes every one
        # back — a reasoning block that the template reopens empty diverges at the first id
        # past the prompt. `KVCache` recovers by rewinding; recurrent state has nothing to
        # rewind to, so the boundary is taken while the cache is standing on it.
        boundary = None if prefix is None else _boundary(model, cache)
        compile_step: Callable[[list[C]], Callable[[mx.array], mx.array]] | None = None
        if (
            sampler is greedy
            and penalty is None
            and constraint is None
            and isinstance(model, CompiledGreedyDecode)
        ):
            compile_step = model.compile_greedy_decode
        elif isinstance(model, CompiledDecode):
            compile_step = model.compile_decode
        # With a prefix the compile is conditional on the trie having something to keep: a
        # promotion hands the decode a shape the trie cannot rewind, so the prompt's state
        # must stand somewhere else first — `boundary` for a trunk that cannot rewind at
        # all, the pre-promotion layers for one that can. A trunk with neither stays on the
        # eager step, which is the one whose cache the inserts below still describe.
        if compile_step is not None and (
            prefix is None
            or boundary is not None
            or all(layer.is_trimmable for layer in cache)
        ):
            before = list(cache)
            decode = compile_step(cache)
            promoted = prefix is not None and any(
                source is not target for source, target in zip(before, cache, strict=True)
            )
            if promoted and boundary is None:
                standing = before
        for emitted in range(max_tokens):
            # Free, step n+1 is queued before n is read back and the GPU never idles;
            # constrained, n+1's mask needs the value of n, so the queue waits behind the
            # sync it was there to hide.
            following = advance(y) if constraint is None else None
            token = y.item()
            assert isinstance(token, int)
            # The step above fed it, so the cache has its row whether or not it is emitted —
            # the stop token below included. A claim longer than the rows is the bug that
            # reads as a wrong answer later: the trie would rewind nothing and the next run
            # would skip prefilling a position no layer ever saw.
            covered.append(token)
            if token in stop:
                return
            if meter is not None:
                meter.token()
            yield token
            if constraint is not None and not constraint.accept(token):
                # The id that completed the document is output — it is the closing brace —
                # and what is left after it is the grammar's stop. Drawing that would be a
                # step spent on an id the loop does not emit.
                return
            if budget is not None:
                owed.extend(budget.emitted(token))
            if emitted % 256 == 0:
                # The first pass hands back the last prefill block's transients, which no
                # decode step will need again; the ones after it keep the pool from drifting
                # up over a long generation.
                mx.clear_cache()
            if owed:
                # The next id is not drawn but owed. `token` still has to be fed — the cache
                # owes it a row either way — so the free path's queued step is spent and its
                # result dropped, which is the whole cost of arming.
                if following is None:
                    advance(y)
                y = mx.array(owed.pop(0))
                continue
            y = advance(y) if following is None else following
    except Exception:
        # A forward that raised leaves the trunk at disagreeing offsets — some layers wrote
        # the row, some did not — and no length describes that cache. It is dropped instead
        # of stored.
        prefix = None
        raise
    finally:
        if prefix is not None:
            # An entry outlives the thread that generated it, and a stream belongs to the
            # thread that made it: an array still carrying queued work can only be evaluated
            # here. Whoever writes it out later — an eviction on the request thread, a drain
            # on the way down — would otherwise meet `no Stream(gpu, 1) in current thread`.
            # The last steps are what this is really for: `advance` queues one the loop never
            # reads back, and its rows are in the cache all the same.
            mx.eval(
                [
                    tensor
                    for layers in (cache, standing if standing is not None else [])
                    for layer in layers
                    for tensor in layer.tensors
                ]
            )
        if meter is not None:
            # A trunk that neither rewinds nor stands still is one the entry below can never
            # be matched against: the next prompt would have to rewind into it, and it has no
            # answer for that. It is stored anyway — the byte count is honest either way —
            # but a caller reading zero reuse deserves to know which zero it is.
            meter.kept_prefix = prefix is not None and (
                boundary is not None
                or standing is not None
                or all(layer.is_trimmable for layer in cache)
            )
        if prefix is not None and boundary is not None:
            # The prompt, as a `user` entry: it is where the conversation stopped being the
            # model's own writing, and it outranks an assistant turn under eviction for the
            # same reason. Beside the longer one below and not instead of it — which of the
            # two the next turn matches is the client's business, not this loop's — and the
            # two share their buffers, so what the second one really costs is the recurrent
            # state. The byte count double-counts what they share, which evicts early and
            # never over-promises.
            prefix.insert(
                prompt,
                boundary,
                role="user",
                nbytes=sum(layer.nbytes for layer in boundary),
            )
        if prefix is not None and standing is not None:
            # The prompt's entry, out of the layers the promotion abandoned. The trim is for
            # a promotion that replaced only some of them: a layer still live in `cache` had
            # the decode's rows written into it, past the prompt this entry claims.
            for layer in standing:
                layer.trim(prompt_length)
            prefix.insert(
                prompt,
                standing,
                role="user",
                nbytes=sum(layer.nbytes for layer in standing),
            )
        if prefix is not None and not promoted:
            # One entry, and an `assistant` one: what a generation leaves behind ends in ids
            # it wrote itself, which is the role the trie drains first. Cutting the prompt at
            # its role boundaries would take boundaries the render does not hand out here.
            prefix.insert(
                covered,
                cache,
                role="assistant",
                nbytes=sum(layer.nbytes for layer in cache),
            )


def stream_text(
    ids: Iterable[int],
    tokenizer: Detokenizer,
    *,
    parser: Parser | None = None,
    prompt: str = "",
) -> Iterator[Segment]:
    """Ids in, segments out: the incremental decoder holds a partial UTF-8 sequence and the
    segmenter holds what could still become a marker, until neither can.

    What leaves carries the channel it came out on, because this is the only place that can
    say it. A consumer that reads text and re-segments it runs a second machine over the same
    generation, and the two do not agree: this one knows the channel the prompt left open and
    the family the checkpoint's own template spells, and one built downstream of it knows
    neither — so an envelope written inside a reasoning block is prose here and a call there.

    Nothing is filtered: the segments concatenate back into what the model wrote, markers
    included, so joining their text gives the string this used to return.

    Every streamer the server reaches goes through this — a facade that grows its own
    detokenization loop is a client reading half a marker again.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    segmenter = Segmenter(FALLBACK if parser is None else parser, prompt=prompt)
    for token in ids:
        piece = decoder.decode(tokenizer.decode_bytes([token]))
        if piece:
            yield from segmenter.push(piece)
    tail = decoder.decode(b"", final=True)
    if tail:
        yield from segmenter.push(tail)
    yield from segmenter.flush()


def stream_generate[C: LayerCache](
    model: CausalLM[C],
    tokenizer: TextTokenizer,
    prompt: str,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] | None = None,
    parser: Parser | None = None,
) -> Iterator[Segment]:
    """Streams decoded text on the channel it came out on, holding partial UTF-8 until it
    completes and a marker's prefix until it stops being one."""
    ids = stream_ids(
        model,
        tokenizer.encode(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        stop=_default_stop(tokenizer) if stop is None else stop,
    )
    yield from stream_text(ids, tokenizer, parser=parser, prompt=prompt)

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
from typing import Protocol

import mlx.core as mx

from mlx_omnia.engine.core.api import LanguageModel, Tracing
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.decode import compiled_decode, plan_of
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.core.prefill import ARRIVING, BLOCK, landing, pulled
from mlx_omnia.engine.core.prefix import Prefixes, Reuse
from mlx_omnia.engine.parsers import FALLBACK, Parser, Segment, Segmenter
from mlx_omnia.engine.speculative import (
    Acceptance,
    Proposer,
    SpeculationRefused,
    stream_speculative_ids,
)

type Sampler = Callable[[mx.array], mx.array]
type LogitFilter = Callable[[mx.array], mx.array]
type Penalty = Callable[[mx.array, mx.array], mx.array]


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


BUDGET_WITH_GRAMMAR = (
    "a reasoning budget and a grammar do not compose: the forced closer bypasses the "
    "mask the matcher is advanced under"
)


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


class ReasoningWalk:
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
    prefix: Reuse = field(default_factory=Reuse)
    """What the store answered with, for whoever is measuring it. Beside `reused_tokens` and
    not instead of it: the count is what a usage report carries, and this is which tier
    answered, how deep the anchor was and what the run wrote back — the difference between a
    30 ms read off an SSD and a 0 ms hit, which no rate distinguishes."""
    kept_prefix: bool = False
    """Whether this run left anything the next turn can start from. Beside `reused_tokens`
    because zero reuse has three causes that look identical from outside — the budget is off,
    the conversation diverged, or this trunk cannot keep a prefix at all — and only the last
    one is permanent. A trunk whose layers neither rewind nor serialize keeps nothing, and
    nothing anywhere said so: the request answered normally and the next one prefilled whole,
    for ever."""
    completion_tokens: int = 0
    prefilled_tokens: int = 0
    """Fresh prompt rows the trunk has taken, block by block, while it is still taking them.
    Written on the boundaries the prefill already stops at, so it costs no synchronization
    and no extra eval. It is what makes a long prompt legible while it is being read: until
    the first token exists there is no `ttft` to divide by, and a 40k prefill is a minute in
    which every rate below is `None`."""
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

    def fed(self, rows: int) -> None:
        """`rows` fresh prompt rows are now in the cache. Absolute and not a delta: every
        prefill in this engine already carries the position it stands at."""
        self.prefilled_tokens = rows

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
    def prefill_seconds(self) -> float | None:
        """The span `ttft` measures while it is still being measured: the clock since the
        prefill started, ending at the first token once there is one."""
        if self.prefill_started is None:
            return None
        end = time.perf_counter() if self.first_token is None else self.first_token
        return end - self.prefill_started

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


def stream_ids[D: LayerCache](
    model: LanguageModel[LayerCache],
    prompt: Iterable[int],
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    meter: Meter | None = None,
    draft: LanguageModel[D] | Proposer | None = None,
    lookahead: int = 4,
    acceptance: Acceptance | None = None,
    prefix: Prefixes | None = None,
    constraint: Constraint | None = None,
    reasoning_budget: ReasoningBudget | None = None,
) -> Generator[int]:
    """`penalty` reads the ids as an mx.array — prompt first, newest last — so the token
    just drawn joins the history without a round trip to the host: reading it back here
    would cost the step of lookahead the loop is built around.

    `meter`, when someone is counting, is filled as the loop runs: the prompt before the
    first step, then one mark per id emitted.

    `prefix` is where the spans come from and where they go: the longest run of stored spans
    of `prompt` is copied in and only the rest is prefilled, and every span this run closes —
    in the prefill, on each boundary the decode crosses, and at the end a consumer that walks
    away imposes — is stored. Nothing is copied out of the store into anybody's ownership: a
    span is immutable, and two requests over the same system prompt read the same one. The
    match is here because here is where the ids are: above this the prompt is still a
    conversation the chat template has to render, which is the same wall the `meter` met.
    Nothing is read until the first `next`: the generator's body runs then, so building the
    iterator and dropping it touches nothing.

    Keying on the ids is what makes the template, the effort asked for and the declared tools
    part of the key without this ever naming them: they change the render, the render changes
    the ids, and the match ends where the ids do. The checkpoint is in the key as well, so
    two resident models never read each other's spans.

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
                "speculation is greedy-only: no sampled acceptance rule is wired here yet. "
                "The one that accepts most needs both distributions and a `Sampler` hands "
                "back an id and neither; the exact-but-weaker one — draw the target's own "
                "token per verification row, accept on equality — needs nothing new and is "
                "unwritten. `high-temp-draft.md`"
            )
        if penalty is not None:
            raise SpeculationRefused(
                "speculation and `penalty` do not compose: a verification row would have to "
                "be penalized against a history the round has not committed yet"
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
            prefix=prefix,
        )
        return

    # `list[LayerCache]` and not `list[C]`: the compiled decode promotes in place, and what
    # it writes back is a fixed buffer or a ring — never the element type the trunk built.
    cache: list[LayerCache] = list(model.make_cache())
    walk = None
    if prefix is not None:
        # The chain is over ids it has to have, so a prompt still being produced is waited
        # for here rather than matched against a piece of itself — a match cut short at a
        # block boundary would be reuse silently lost on exactly the long conversations this
        # exists for.
        prompt = prompt if isinstance(prompt, Sequence) else list(prompt)
        walk = prefix.begin(cache)
    # What the cache holds, row for row: the prompt, whether its head was resumed or
    # prefilled, plus every id the loop feeds back in. It is filled as the prompt arrives,
    # which for a sequence is all at once.
    covered: list[int] = []
    prompt_length = 0
    reused = 0
    fed = 0
    """Rows the trunk has actually taken. The spans are cut against this and never against
    `len(covered)`: the two differ by the one id a constrained step has drawn and not yet
    fed, and a span claiming a row no layer wrote is the bug that reads as a wrong answer a
    turn later."""
    history = mx.array([], dtype=mx.int32)
    if walk is not None and isinstance(prompt, Sequence):
        reused = walk.resume(prompt, cache)
        fed = reused
    if meter is not None:
        # The clock starts here whatever shape the prompt is in: for one still arriving, the
        # wait for its producer is part of the TTFT this measures. The count is set below,
        # where it exists.
        known = len(prompt) if isinstance(prompt, Sequence) else 0
        meter.prefill(known, reused)

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
        nonlocal history, fed
        if penalty is not None:
            history = mx.concatenate([history, drawn[None]])
        queued = step(drawn[None], history)
        fed += 1
        mx.async_eval(queued)
        return queued

    if reasoning_budget is not None and constraint is not None:
        # Refused rather than made inert: the closer is fed past the mask, and the matcher —
        # which is advanced over every id this yields — would then be asked to accept an id
        # it had forbidden. A grammar leaves no reasoning block to budget anyway: it forces
        # the document from the first step, so the block a template opened never closes.
        raise ConstraintConflict(BUDGET_WITH_GRAMMAR)
    budget = None if reasoning_budget is None else ReasoningWalk(reasoning_budget)
    owed: list[int] = []
    """The closer, between the step the budget armed on and the steps that feed it out."""

    def close(rows: int) -> None:
        """Store every span the trunk has finished, standing where it stands.

        Called only where the trunk *is* standing — between prefill blocks, on the cut
        before the last one, between decode steps — because a recurrent layer's state has no
        expression in the middle of a forward.
        """
        if walk is not None:
            walk.commit(covered, cache, rows)

    try:
        # What the resume did not already cover, fed in blocks: only the last block's logits
        # are read, and the blocks before it never compute the head at all.
        arriving = iter(prompt)
        covered.extend(islice(arriving, reused))

        def feed(ids: list[int]) -> object:
            nonlocal fed
            covered.extend(ids)
            fed += len(ids)
            return model(mx.array(ids)[None], cache)

        def stopped(_: int) -> None:
            close(fed)
            if meter is not None:
                meter.fed(fed - reused)

        head = pulled(
            feed,
            arriving,
            cache,
            block=BLOCK if isinstance(prompt, Sequence) else ARRIVING,
            after=stopped,
        )
        if walk is not None and walk.chain.anchored:
            # The last block is cut on the last span boundary so the trunk stops on one. A
            # recurrent state exists only while the layer is stopped, and without this cut a
            # turn whose prefill ends mid-span — which is nearly every turn, and every turn
            # of a replay asking for one token — would leave no resume point at all.
            cut = landing(fed, fed + len(head), walk.chain.span) - fed
            if cut > 0:
                feed(head[:cut])
                mx.eval([tensor for layer in cache for tensor in layer.tensors])
                stopped(fed)
                head = head[cut:]
        covered.extend(head)
        fed += len(head)
        prompt_length = len(covered)
        history = mx.array(covered)
        if meter is not None:
            # The count only exists now for a prompt that was arriving; `prefill` above
            # started the clock, and the wait for the producer belongs inside the TTFT it
            # measures.
            meter.prompt_tokens = prompt_length
        y = step(mx.array(head), history)
        mx.async_eval(y)
        # After the last block's forward — `pulled` hands that one back rather than feeding
        # it — and before the compile below. A promotion turns a sliding layer into a ring,
        # and a ring has already dropped every row outside its window: a span left
        # uncommitted here is a span the next boundary the decode crosses cannot be cut from.
        stopped(fed)
        # The lease: the trunk is handed the promoted caches themselves, which is what keeps
        # a fused single-row kernel on. `Tracing` is the claim that its forward survives
        # that — graph-visible positions, and the unwritten columns of a fixed buffer masked
        # by the trunk itself. `is_fixable` is the layers' half of the same question.
        #
        # Screened only when nothing else will read the row: outside its argmax it is not
        # logits, so a sampler, a penalty or a grammar over it would read invented numbers.
        if isinstance(model, Tracing) and cache and all(layer.is_fixable for layer in cache):
            decode = compiled_decode(
                plan_of(
                    model,
                    screened=sampler is greedy and penalty is None and constraint is None,
                ),
                cache,
            )
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
            if walk is not None and fed // walk.chain.span > walk.covered // walk.chain.span:
                # A boundary crossed between steps, which is the only place the decode ever
                # stands still. What it buys is the client that echoes the ids it was sent:
                # for one that re-renders them, the span this closes is on a branch the next
                # turn never walks, and the prompt's own anchor is untouched by it.
                close(fed)
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
        # the row, some did not — and no length describes that cache. Nothing more is stored.
        walk = None
        raise
    finally:
        if meter is not None:
            meter.reused_tokens = reused
            meter.kept_prefix = walk is not None
            if walk is not None:
                meter.prefix = walk.reuse
        if walk is not None:
            # The last spans, and the `mx.eval` inside them is what makes them portable: a
            # span outlives the thread that generated it, and a stream belongs to the thread
            # that made it. Whoever writes it out later — an eviction on the request thread,
            # a drain on the way down — would otherwise meet `no Stream(gpu, 1) in current
            # thread`. The last steps are what this is really for: `advance` queues one the
            # loop never reads back, and its rows are in the cache all the same.
            close(fed)


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


def stream_generate(
    model: LanguageModel[LayerCache],
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

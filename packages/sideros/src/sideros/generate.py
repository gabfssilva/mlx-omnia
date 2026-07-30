"""Lazy generation pipeline.

The sampler returns the token as an mx.array (no .item() before the next step is
queued); step n+1 is async-evaluated before step n's sync, so the GPU never idles
between steps.
"""

import codecs
import time
from collections.abc import Callable, Collection, Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from sideros.core.cache import LayerCache
from sideros.core.mxcompat import softmax
from sideros.core.prompt_cache import PromptCache
from sideros.speculative import Acceptance, SpeculationRefused, stream_speculative_ids
from sideros.suppress import Segment, Segmenter
from sideros.tools import ToolFamily

type Sampler = Callable[[mx.array], mx.array]
type LogitFilter = Callable[[mx.array], mx.array]
type Penalty = Callable[[mx.array, mx.array], mx.array]


class CausalLM[C: LayerCache](Protocol):
    def make_cache(self) -> list[C]: ...

    def __call__(self, ids: mx.array, cache: list[C] | None = None) -> mx.array: ...


class Constraint(Protocol):
    """A grammar's side of the loop: which ids the sampler may see, and the id it drew.

    It is not a `LogitFilter` because a filter has no memory of the run — the mask of step
    n+1 is a function of the id step n drew, and advancing over that id is a side effect per
    token that `LogitFilter` has nowhere to put. `sideros.grammar` is the implementation.

    `mask` takes the logits and returns them with every forbidden id at -inf; `remaining`
    counts the steps the run still has, this one included, so a constraint can force what
    closes the output before the budget runs out instead of letting it truncate. `accept`
    advances over the id just drawn and answers `False` once the grammar has ended.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array: ...

    def accept(self, token: int) -> bool: ...


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
    completion_tokens: int = 0
    prefill_started: float | None = None
    first_token: float | None = None
    last_token: float | None = None

    def prefill(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
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
        after it — the same split `bench/interleaved.py` reports."""
        elapsed = self.decode_seconds
        if elapsed is None or elapsed <= 0 or self.completion_tokens < 2:
            return None
        return (self.completion_tokens - 1) / elapsed


def stream_ids[C: LayerCache, D: LayerCache](
    model: CausalLM[C],
    prompt: list[int],
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

    Keying on the ids is what makes the template, `enable_thinking` and the declared tools
    part of the key without this ever naming them: they change the render, the render changes
    the ids, and the match ends where the ids do. One trie belongs to one model — the ids of
    two checkpoints are two alphabets, and nothing in a cache says which one wrote it.

    A `constraint` is the one option that changes the shape of the loop rather than the
    numbers inside it: its mask for step n+1 is a function of the id step n drew, so the
    `.item()` that used to happen behind the queue of n+1 now happens in front of it. The
    GPU idles for the length of that sync once per token, and that is what a constrained
    request costs — it is the contract, not an implementation detail.

    A `draft` moves the whole run onto the speculative loop, which emits this same greedy
    stream token for token — `lookahead` and `acceptance` are read only there. Without one
    nothing below changes."""
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
        yield from stream_speculative_ids(
            model,
            draft,
            prompt,
            max_tokens=max_tokens,
            lookahead=lookahead,
            stop=stop,
            meter=meter,
            acceptance=acceptance,
        )
        return

    cache = model.make_cache()
    if prefix is not None and not all(layer.is_trimmable for layer in cache):
        # Declared refusal, not one optimization fewer: a recurrent state and a conv window
        # are not a history to cut at a common prefix, and the wrong state is exactly what
        # survives a decode and answers fluently.
        prefix = None
    reuse = None if prefix is None else prefix.take(prompt)
    if reuse is not None:
        cache = reuse.caches
    # What the cache holds, row for row, for the insert at the end: the prompt, whether its
    # head was reused or prefilled, plus every id the loop feeds back in.
    covered = list(prompt)
    history = mx.array(prompt)
    if meter is not None:
        meter.prefill(len(prompt))

    def step(ids: mx.array, seen: mx.array) -> mx.array:
        logits = model(ids[None], cache)[:, -1, :]
        if penalty is not None:
            logits = penalty(logits, seen)
        if constraint is not None:
            # Last, so nothing downstream can put a forbidden id back: a filter divides and
            # compares, and -inf survives both.
            logits = constraint.mask(logits, max_tokens - (len(covered) - len(prompt)))
        return sampler(logits)[0]

    def advance(drawn: mx.array) -> mx.array:
        nonlocal history
        if penalty is not None:
            history = mx.concatenate([history, drawn[None]])
        queued = step(drawn[None], history)
        mx.async_eval(queued)
        return queued

    try:
        y = step(history if reuse is None else mx.array(prompt[reuse.length :]), history)
        mx.async_eval(y)
        for _ in range(max_tokens):
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
            y = advance(y) if following is None else following
    except Exception:
        # A forward that raised leaves the trunk at disagreeing offsets — some layers wrote
        # the row, some did not — and no length describes that cache. It is dropped instead
        # of stored.
        prefix = None
        raise
    finally:
        if prefix is not None:
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
    tools: ToolFamily | None = None,
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
    segmenter = Segmenter(tools, prompt=prompt)
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
    tools: ToolFamily | None = None,
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
    yield from stream_text(ids, tokenizer, tools=tools, prompt=prompt)

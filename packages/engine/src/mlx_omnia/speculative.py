"""Speculative decoding: a small draft proposes, the target verifies in one forward.

A round costs one read of the big weights however many proposals survive it, and what comes
out is the target's own script: acceptance is equality against the target's argmax, so a
draft can only change the speed. Anything else is a bug, not a tradeoff.

Greedy only, and not as a first step towards the rest: the rule that leaves a *sampled*
distribution intact is a ratio between the draft's and the target's probabilities for the
drawn token, with a rejection redrawing from the residual `max(0, p - q)`. `Sampler` is an
opaque `logits -> id` callable — neither distribution is reachable through it, and accepting
on token equality under a temperature would bias the output without saying so. The
non-greedy path is refused by name (in `generate.stream_ids`) rather than approximated.
"""

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import LayerCache
from mlx_omnia.core.prefill import prefill

if TYPE_CHECKING:
    # `generate` imports this module to reach the speculative path; the names below are
    # annotations only, so they stay out of the runtime cycle.
    from mlx_omnia.generate import CausalLM, Meter


class SpeculationRefused(Exception):
    """A draft that cannot be verified is refused before the first token. There is no
    degraded mode: a wrong number costs more than the speculation was worth."""


@runtime_checkable
class Drafting(Protocol):
    """A model facade that can be handed a second checkpoint to speculate with.

    The pairing is not the loader's: what drafts for a checkpoint is a decision made
    outside the engine — a setting, a measurement, a checkpoint quantized this morning —
    and `mlx_omnia.load` takes an id and no opinion about it. So the facade is loaded first
    and given its drafter after, by whoever knows.

    `drafter` is out so that whoever accounts for the memory can weigh it: two checkpoints
    are resident and only one of them is under the model's own tree.
    """

    drafter: nn.Module | None

    def speculate_with(self, drafter: nn.Module, *, block_size: int | None = None) -> None:
        """Take this checkpoint as the drafter, or refuse it by name. A tree of the wrong
        architecture, or one whose shapes do not meet the target's, is a `TypeError` or a
        `ValueError` here rather than a wrong number in the middle of a generation."""
        ...


@runtime_checkable
class Speculable(Protocol):
    """A target that lends the two ends of its vocabulary, raw.

    A proposer whose input and output are hidden rows — a DFlash block, an MTP step — has no
    embedding table and no head of its own: the ids it is handed become rows through the
    target's lookup, and the rows it writes become ids through the target's. That borrowing
    is the only reason such a pair is typed to a concrete target, and this is what it needs
    in place of one.

    **Raw, and the name is the point.** A facade's `embed` and `head` are usually not the
    bare tensors — Muse-Glimmer's embed applies a scaleless RMS and its head applies
    `output_multiplier` and a softcap. A drafter is trained against the bare ones, and under
    greedy both missing transforms are monotone: substituting the cooked pair changes no
    token any test would catch, and changes what the checkpoint is. Where the two coincide a
    family answers with the same tensor twice; where they do not, the difference has a name.

    `block_outputs` is deliberately not here. It is `generate.BlockOutputs`, and the round
    itself checks for it — the consumers are different, the round reads blocks while the
    pair borrows the vocabulary, and a proposer can want either without the other.
    """

    def raw_embed(self, ids: mx.array) -> mx.array:
        """The embedding lookup alone: `ids` of any shape, plus the hidden dim."""
        ...

    def raw_logits(self, hidden: mx.array) -> mx.array:
        """Hidden rows through the output head, unscaled and uncapped. A method and not the
        head itself, because a trunk with `tie_word_embeddings` has no `lm_head` to hand
        over — there the logits are `embed_tokens.as_linear(hidden)`."""
        ...


@runtime_checkable
class Proposer(Protocol):
    """What writes the ids a round verifies. One round asks for one proposal, whatever the
    thing behind it costs to produce: a small language model run `width` times, or a
    block-diffusion drafter run once.

    The round owns the target and nothing else; a proposer owns whatever state it needs
    between rounds — its own cache, and the target's reading of the context when it reads
    one — and rewinds it in `settle`.

    `taps` is the whole of what a proposer asks the target for beyond its ids: the blocks
    whose output it conditions on, empty for one that conditions on ids alone. A non-empty
    `taps` is what makes the round take `BlockOutputs` instead of a plain forward, and a
    target that does not implement it refuses the draft by name.
    """

    @property
    def taps(self) -> Sequence[int]:
        """Which of the target's blocks this reads, by index. Empty asks for none."""
        ...

    @property
    def width(self) -> int:
        """Ids proposed per round."""
        ...

    def absorb(self, features: mx.array) -> None:
        """The target's reading of the positions its cache has just taken and kept, in
        order, `[1, new, len(taps) * hidden]`. Exactly the positions this has not been
        given before, and `committed[-1]` in the next `propose` is the one token past
        them. Never called when `taps` is empty."""
        ...

    def propose(self, committed: Sequence[int]) -> mx.array:
        """`width` ids continuing `committed`, whose last entry the target's cache has not
        seen yet."""
        ...

    def settle(self, length: int) -> None:
        """The round kept `length` ids. Whatever this wrote past them describes a sequence
        that never happened and goes now."""
        ...


@dataclass
class Acceptance:
    """Read out of the loop, never estimated: `proposed` is what the draft wrote, `accepted`
    is how much of it the target's own argmax confirmed."""

    rounds: int = 0
    proposed: int = 0
    accepted: int = 0

    @property
    def rate(self) -> float | None:
        return None if self.proposed == 0 else self.accepted / self.proposed


class Autoregressive[D: LayerCache]:
    """A small language model as a proposer: `width` steps, each one conditioned on the id
    the last one drew. It reads no blocks of the target — its whole input is the ids — so
    the round never asks the target for features.

    It is the original speculative draft, and it is a `Proposer` and not the special case
    the loop is written around: the round is single, and what a checkpoint costs to propose
    with is the checkpoint's business.
    """

    def __init__(self, model: "CausalLM[D]", width: int) -> None:
        if width < 1:
            raise ValueError(f"lookahead must be at least 1: {width}")
        self._model = model
        self._cache = model.make_cache()
        self._width = width
        _must_rewind(self._cache, "draft")

    @property
    def taps(self) -> Sequence[int]:
        return ()

    @property
    def width(self) -> int:
        return self._width

    def absorb(self, features: mx.array) -> None:
        raise AssertionError("an autoregressive draft reads no blocks of the target")

    def propose(self, committed: Sequence[int]) -> mx.array:
        """Fed whatever committed ids its own cache has not seen — after a full acceptance
        that is one token, and round one is where it meets the whole prompt."""
        catchup = mx.array(committed[self._cache[0].offset :])
        window = prefill(
            lambda block: self._model(catchup[block][None], self._cache),
            catchup.size,
            self._cache,
        )
        ids = catchup[window][None]
        proposals: list[mx.array] = []
        for _ in range(self._width):
            token = mx.argmax(self._model(ids, self._cache)[:, -1, :], axis=-1)
            # Queued as it is built: the GPU runs draft step n while the CPU builds step
            # n + 1, instead of idling until the round's single sync.
            mx.async_eval(token)
            proposals.append(token)
            ids = token[None]
        return mx.concatenate(proposals)

    def settle(self, length: int) -> None:
        for layer in self._cache:
            layer.trim(length)


@runtime_checkable
class Verifiable[C: LayerCache](Protocol):
    """A target that can undo a verification forward for less than it costs to run it again.

    The generic rewind for a trunk that cannot trim is restore-and-replay, and the replay is
    another forward of the whole model. On a sparse trunk that is the expensive half twice:
    every row routes to its own experts, so a four-row forward reads about four times the MoE
    bytes of one, and the replay pays it again.

    A trunk that knows which of its layers actually kept state can do better, because the
    forward it just ran already produced every layer's input. `verify` returns what
    `BlockOutputs.block_outputs` returns plus a third thing — a callable that takes how many
    of the rows survived and puts the cache back to exactly that, however this trunk can:
    trimming the keys, counting the offsets, replaying only the recurrent mixers.

    Optional. A target without it is rewound by the round's own replay, which is correct and
    slower; a target with it is asserting that the two land on the same state, and the
    equality test on a hybrid trunk is what holds it to that.
    """

    def verify(
        self, ids: mx.array, cache: list[C], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array, Callable[[int], None]]: ...


@runtime_checkable
class CompiledVerify[C: LayerCache](Protocol):
    """A target whose verification forward compiles: every round feeds exactly
    `width + 1` rows, so the shape is fixed and the graph is built once. The rewind
    that comes with it costs no forward — per-row state checkpoints inside the
    recurrent layers' own kernel, a moved position for the attention buffers.

    `compile_verify` promotes the cache the way a compiled decode does, so it is taken
    once, after the prefill, and the returned pair replaces both `Verifiable.verify`
    and the round's replay for as long as the generation runs."""

    def compile_verify(
        self, cache: list[C], *, width: int, at: Sequence[int]
    ) -> (
        tuple[
            Callable[[mx.array], tuple[mx.array, mx.array]],
            Callable[[int], None],
        ]
        | None
    ): ...


@runtime_checkable
class ChainCompilable(Protocol):
    """A step that can compile its own chained drafting: one graph reused for every
    link, a fixed KV buffer of `width` rows, and a reset that rewinds it between
    rounds. `embed`/`head` are the target's raw vocabulary ends, baked in as
    constants."""

    def compile_chain(
        self,
        embed: Callable[[mx.array], mx.array],
        head: Callable[[mx.array], mx.array],
        width: int,
    ) -> (
        tuple[
            Callable[[mx.array, mx.array], tuple[mx.array, mx.array]],
            Callable[[], None],
        ]
        | None
    ): ...


@runtime_checkable
class Step[S: LayerCache](Protocol):
    """One multi-token-prediction step: the pair *(what the trunk held at a position, the
    embedding of the token after it)* to a hidden row for the position after **that**.

    A tree and not a model: no embedding table, no head, no tokenizer — the two ends of the
    vocabulary are the target's, which is what `Speculable` lends. `models/nemotron_h/mtp.py`
    is the implementation, and the shape is the family's own blocks with a fusion in front.
    """

    @property
    def block(self) -> int:
        """How many ids a round is worth proposing with this step when nobody says.

        The step's own, because it is a property of the *pair* and not of the round: what a
        wider block costs is one more chained run and one more row in the target's forward,
        and what it buys is the acceptance the head still has at that depth. Both are the
        checkpoint's, and neither is in its config.
        """
        ...

    def make_cache(self) -> list[S]: ...

    def __call__(
        self, embeddings: mx.array, hidden: mx.array, cache: list[S] | None = None
    ) -> mx.array: ...


class Chained[S: LayerCache]:
    """An MTP step as a proposer, run `width` times over its own output.

    The step predicts one token. Past the first, the only hidden there is to condition on is
    the one the step itself just produced — the target has not run — so a wider block is the
    same weights chained, and acceptance decays faster than the first step's rate to the
    power of the depth. On Nemotron-3.5 that is measured: 0.830 for one, then 0.621 and
    0.390, against 0.689 and 0.572 for the naive power.

    **The cache is per round and the head keeps nothing between them.** A round builds one,
    the chain's steps share it so step two attends to step one, and it goes when the round
    does — which is why `settle` has nothing to do. That it is *allowed* to go is not
    obvious and is not deduced: it is the shape the acceptance above was measured in, an
    empty cache at every round's first step. Carrying the head's keys across rounds the way
    vllm does is the untested variant, and it can only move speed.

    What crosses a round is one row: the target's own reading of the last position it kept,
    handed over by `absorb`. `taps` is the block that row comes from.
    """

    def __init__(
        self,
        target: Speculable,
        step: Step[S],
        *,
        block: int,
        tap: int,
    ) -> None:
        if block < 2:
            raise ValueError(f"a block of {block} proposes nothing")
        if not isinstance(target, Speculable):
            raise SpeculationRefused(
                f"{type(target).__name__} does not lend its embedding table and its head "
                "(`speculative.Speculable`), which is the whole of an MTP step's vocabulary"
            )
        self._target = target
        self._step = step
        self._width = block - 1
        self._tap = tap
        self._hidden: mx.array | None = None
        self._chain = (
            step.compile_chain(target.raw_embed, target.raw_logits, self._width)
            if isinstance(step, ChainCompilable)
            else None
        )

    @property
    def taps(self) -> Sequence[int]:
        return (self._tap,)

    @property
    def width(self) -> int:
        return self._width

    def absorb(self, features: mx.array) -> None:
        """Only the last row is kept. The step conditions on the position right before the
        token it is given, and the round hands over exactly the positions the target's cache
        took — so the last of them is that position, every time."""
        self._hidden = features[:, -1:, :]

    def propose(self, committed: Sequence[int]) -> mx.array:
        """`committed[-1]` is the token the target's cache has not seen, and `self._hidden`
        is the target's reading of the one before it: the pair the step is defined on.

        Both borrowed pieces are the raw ones (`Speculable`), and the head is read once per
        drafted token — which on a 131k vocabulary is most of what a round's draft costs.
        """
        assert self._hidden is not None, "the round feeds the proposer before it asks"
        hidden = self._hidden
        if self._chain is not None:
            chain, reset = self._chain
            reset()
            token = mx.array([committed[-1]])
            drafted: list[mx.array] = []
            for _ in range(self._width):
                token, hidden = chain(token, hidden)
                mx.async_eval(token)
                drafted.append(token)
            return mx.concatenate(drafted)
        token = mx.array([committed[-1]])[None]
        cache = self._step.make_cache()
        proposals: list[mx.array] = []
        for _ in range(self._width):
            hidden = self._step(self._target.raw_embed(token), hidden, cache)
            token = mx.argmax(self._target.raw_logits(hidden)[:, -1, :], axis=-1)[None]
            mx.async_eval(token)
            proposals.append(token[0])
        return mx.concatenate(proposals)

    def settle(self, length: int) -> None:
        """Nothing: the chain's cache belonged to the round that just ended, and the row that
        crosses to the next one arrives through `absorb` already cut to what was kept."""


def stream_speculative_ids[C: LayerCache, D: LayerCache](
    target: "CausalLM[C]",
    draft: "CausalLM[D] | Proposer",
    prompt: Iterable[int],
    *,
    max_tokens: int,
    lookahead: int = 4,
    stop: Collection[int] = (),
    meter: "Meter | None" = None,
    acceptance: Acceptance | None = None,
) -> Iterator[int]:
    """The target's greedy stream, token for token, with the draft only paying for speed.

    A `Proposer` says for itself how many ids a round proposes; a plain language model is
    wrapped in `Autoregressive` and `lookahead` is what it proposes. `acceptance` collects
    what the rounds actually accepted, for whoever is measuring.
    """
    from mlx_omnia.generate import BlockOutputs

    proposer: Proposer = draft if isinstance(draft, Proposer) else Autoregressive(draft, lookahead)
    taps = proposer.taps
    if taps and not isinstance(target, BlockOutputs):
        raise SpeculationRefused(
            f"this draft conditions on the target's blocks {list(taps)} and the target has "
            "no `block_outputs`: what it would read instead is nothing at all"
        )
    target_cache = target.make_cache()
    replays = _rewinds(target_cache, "target")

    # Read here, ahead of the meter that wants its length: a round rewinds both caches in
    # step, so this loop needs the whole prompt before its first forward whatever shape it
    # arrived in — there is no block for a lazy one to overlap against.
    committed = list(prompt)
    if meter is not None:
        meter.prefill(len(committed))

    def forward(ids: mx.array) -> tuple[mx.array, mx.array | None]:
        """One forward of the target: its logits, and what the proposer reads of it when it
        reads anything. The verification forward needs both out of the same pass, which is
        also why it can never take the compiled decode path."""
        if not taps:
            return target(ids, target_cache), None
        assert isinstance(target, BlockOutputs)
        return target.block_outputs(ids, target_cache, at=taps)

    def verify(ids: mx.array) -> tuple[mx.array, mx.array | None, Callable[[int], None] | None]:
        """The same forward, plus the target's own way back when it has one."""
        if taps and isinstance(target, Verifiable):
            return target.verify(ids, target_cache, at=taps)
        logits, features = forward(ids)
        return logits, features, None

    def feed(ids: mx.array) -> mx.array:
        """Prefill: every row a block writes is kept, so the proposer gets all of them."""
        logits, features = forward(ids)
        if features is not None:
            proposer.absorb(features)
        return logits

    # The first token is the target's alone: the prompt's own forward already yields one
    # (that is ttft), and the draft has nothing to propose before it.
    ids = mx.array(committed)
    window = prefill(lambda block: feed(ids[block][None]), ids.size, target_cache)
    pending = _ints(mx.argmax(feed(ids[window][None])[:, -1, :], axis=-1))
    committed += pending

    compiled: tuple[Callable[[mx.array], tuple[mx.array, mx.array]], Callable[[int], None]] | None
    compiled = (
        target.compile_verify(target_cache, width=proposer.width, at=taps)
        if taps and isinstance(target, CompiledVerify)
        else None
    )

    emitted = 0
    while emitted < max_tokens:
        if not pending:
            if compiled is not None and target_cache[0].offset == len(committed) - 1:
                pending, accepted = _compiled_round(compiled, proposer, committed)
            else:
                pending, accepted = _round(
                    verify, forward, proposer, target_cache, committed, replays
                )
            committed += pending
            if acceptance is not None:
                acceptance.rounds += 1
                acceptance.proposed += proposer.width
                acceptance.accepted += accepted
        token = pending.pop(0)
        if token in stop:
            return
        if meter is not None:
            meter.token()
        yield token
        emitted += 1


def _compiled_round(
    compiled: tuple[
        Callable[[mx.array], tuple[mx.array, mx.array]], Callable[[int], None]
    ],
    proposer: Proposer,
    committed: list[int],
) -> tuple[list[int], int]:
    """One round through the compiled fixed-shape verify: the gap is always the one
    committed token the cache has not seen, so every round feeds `width + 1` rows, and
    a rejection is a slot pick instead of a replay."""
    verify_fn, rewind_fn = compiled
    drafted = proposer.propose(committed)
    width = drafted.size
    verified = mx.concatenate([mx.array([committed[-1]], dtype=drafted.dtype), drafted])
    logits, features = verify_fn(verified)
    judged = mx.argmax(logits, axis=-1)
    mx.eval(judged, drafted)

    proposed, predicted = _ints(drafted), _ints(judged)
    accepted = 0
    while accepted < width and proposed[accepted] == predicted[accepted]:
        accepted += 1
    if accepted < width:
        rewind_fn(1 + accepted)
    proposer.absorb(features[:, : 1 + accepted])
    proposer.settle(len(committed) + accepted)
    return [*proposed[:accepted], predicted[accepted]], accepted


def _round[C: LayerCache](
    verify: Callable[[mx.array], tuple[mx.array, mx.array | None, Callable[[int], None] | None]],
    forward: Callable[[mx.array], tuple[mx.array, mx.array | None]],
    proposer: Proposer,
    target_cache: list[C],
    committed: list[int],
    replays: bool,
) -> tuple[list[int], int]:
    """One round, returning the tokens it settled and how many of them were the draft's.

    The target is fed whatever committed ids its cache has not seen, then the proposals.
    Row j of its logits predicts the token after `verified[j]`, so the last `width + 1`
    rows judge every proposal and hand back the target's own token past the accepted run.

    A rejected proposal leaves a cache describing a sequence that never happened, and there
    are two ways back. A cache of keys drops the rows: `trim`. A cache holding recurrent
    state cannot, because the recurrence kept no history to subtract — so the round takes a
    `checkpoint()` before the forward and, when anything was rejected, restores it and runs
    the kept rows again. Both land on the same offset; only the second costs a forward, and
    only when the draft was wrong.

    What the target *kept* is what the proposer is given to read — the gap plus the accepted
    proposals, never the rejected tail. Causality is why the first forward's features serve:
    row j's output cannot depend on rows past it, so the rows a replay would recompute are
    the rows already in hand.
    """
    drafted = proposer.propose(committed)
    width = drafted.size
    gap = mx.array(committed[target_cache[0].offset :], dtype=drafted.dtype)
    verified = mx.concatenate([gap, drafted])
    # Before the forward, which is the only moment a restore point means anything — and only
    # when the target has no way of its own, because a `Verifiable` takes its own inside
    # `verify` for the layers it decides need them.
    restores = [layer.checkpoint() for layer in target_cache] if replays else ()
    logits, features, rewind = verify(verified[None])
    judged = mx.argmax(logits[0, -(width + 1) :], axis=-1)
    mx.eval(judged, drafted)

    proposed, predicted = _ints(drafted), _ints(judged)
    accepted = 0
    while accepted < width and proposed[accepted] == predicted[accepted]:
        accepted += 1

    if features is not None:
        proposer.absorb(features[:, : gap.size + accepted])
    length = len(committed) + accepted
    if accepted == width:
        # Nothing to undo when every proposal survived: the cache is already the sequence.
        pass
    elif rewind is not None:
        rewind(gap.size + accepted)
        mx.eval([tensor for layer in target_cache for tensor in layer.tensors])
    elif not replays:
        for layer in target_cache:
            layer.trim(length)
    else:
        for restore in restores:
            restore()
        kept = gap.size + accepted
        if kept:
            forward(verified[None, :kept])
            mx.eval([tensor for layer in target_cache for tensor in layer.tensors])
    proposer.settle(length)
    return [*proposed[:accepted], predicted[accepted]], accepted


def _rewinds(cache: Sequence[LayerCache], role: str) -> bool:
    """How this cache goes back, or a refusal. `False` is `trim`, `True` is replay.

    Trim first, because it is free where it applies. A trunk with one recurrent layer in it
    takes the second road for *all* of its layers — the restore is per layer but the replay
    is one forward, so a mixed list cannot be half-rewound.
    """
    if all(layer.is_trimmable for layer in cache):
        return False
    if all(layer.is_replayable for layer in cache):
        return True
    return _refuse(role)


def _must_rewind(cache: Sequence[LayerCache], role: str) -> None:
    """Trim or nothing, for a caller with no forward to replay through — which is the
    autoregressive draft, whose own model the round never holds."""
    if not all(layer.is_trimmable for layer in cache):
        _refuse(role)


def _refuse(role: str) -> Never:
    raise SpeculationRefused(
        f"the {role}'s cache keeps recurrent state and cannot rewind: a rejected proposal "
        "would leave it describing a sequence that never happened"
    )


def _ints(values: mx.array) -> list[int]:
    listed = values.tolist()
    assert isinstance(listed, list)
    ids: list[int] = []
    for value in listed:
        assert isinstance(value, int)
        ids.append(value)
    return ids

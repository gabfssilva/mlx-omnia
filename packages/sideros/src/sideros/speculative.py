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

from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import LayerCache
from sideros.core.prefill import prefill

if TYPE_CHECKING:
    # `generate` imports this module to reach the speculative path; the names below are
    # annotations only, so they stay out of the runtime cycle.
    from sideros.generate import CausalLM, Meter


class SpeculationRefused(Exception):
    """A draft that cannot be verified is refused before the first token. There is no
    degraded mode: a wrong number costs more than the speculation was worth."""


@runtime_checkable
class Drafting(Protocol):
    """A model facade that can be handed a second checkpoint to speculate with.

    The pairing is not the loader's: what drafts for a checkpoint is a decision made
    outside the engine — a setting, a measurement, a checkpoint quantized this morning —
    and `sideros.load` takes an id and no opinion about it. So the facade is loaded first
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


def stream_speculative_ids[C: LayerCache, D: LayerCache](
    target: "CausalLM[C]",
    draft: "CausalLM[D] | Proposer",
    prompt: list[int],
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
    from sideros.generate import BlockOutputs

    proposer: Proposer = draft if isinstance(draft, Proposer) else Autoregressive(draft, lookahead)
    taps = proposer.taps
    if taps and not isinstance(target, BlockOutputs):
        raise SpeculationRefused(
            f"this draft conditions on the target's blocks {list(taps)} and the target has "
            "no `block_outputs`: what it would read instead is nothing at all"
        )
    target_cache = target.make_cache()
    _must_rewind(target_cache, "target")

    if meter is not None:
        meter.prefill(len(prompt))

    def forward(ids: mx.array) -> tuple[mx.array, mx.array | None]:
        """One forward of the target: its logits, and what the proposer reads of it when it
        reads anything. The verification forward needs both out of the same pass, which is
        also why it can never take the compiled decode path."""
        if not taps:
            return target(ids, target_cache), None
        assert isinstance(target, BlockOutputs)
        return target.block_outputs(ids, target_cache, at=taps)

    def feed(ids: mx.array) -> mx.array:
        """Prefill: every row a block writes is kept, so the proposer gets all of them."""
        logits, features = forward(ids)
        if features is not None:
            proposer.absorb(features)
        return logits

    committed = list(prompt)
    # The first token is the target's alone: the prompt's own forward already yields one
    # (that is ttft), and the draft has nothing to propose before it.
    ids = mx.array(committed)
    window = prefill(lambda block: feed(ids[block][None]), ids.size, target_cache)
    pending = _ints(mx.argmax(feed(ids[window][None])[:, -1, :], axis=-1))
    committed += pending

    emitted = 0
    while emitted < max_tokens:
        if not pending:
            pending, accepted = _round(forward, proposer, target_cache, committed)
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


def _round[C: LayerCache](
    forward: Callable[[mx.array], tuple[mx.array, mx.array | None]],
    proposer: Proposer,
    target_cache: list[C],
    committed: list[int],
) -> tuple[list[int], int]:
    """One round, returning the tokens it settled and how many of them were the draft's.

    The target is fed whatever committed ids its cache has not seen, then the proposals.
    Row j of its logits predicts the token after `verified[j]`, so the last `width + 1`
    rows judge every proposal and hand back the target's own token past the accepted run.

    A rejected proposal leaves keys describing a sequence that never happened: the target's
    cache is rewound to the accepted prefix, and the proposer is told the same length. What
    the target *kept* is what the proposer is given to read — the gap plus the accepted
    proposals, never the rejected tail.
    """
    drafted = proposer.propose(committed)
    width = drafted.size
    gap = mx.array(committed[target_cache[0].offset :], dtype=drafted.dtype)
    verified = mx.concatenate([gap, drafted])
    logits, features = forward(verified[None])
    judged = mx.argmax(logits[0, -(width + 1) :], axis=-1)
    mx.eval(judged, drafted)

    proposed, predicted = _ints(drafted), _ints(judged)
    accepted = 0
    while accepted < width and proposed[accepted] == predicted[accepted]:
        accepted += 1

    if features is not None:
        proposer.absorb(features[:, : gap.size + accepted])
    length = len(committed) + accepted
    for layer in target_cache:
        layer.trim(length)
    proposer.settle(length)
    return [*proposed[:accepted], predicted[accepted]], accepted


def _must_rewind(cache: Sequence[LayerCache], role: str) -> None:
    if all(layer.is_trimmable for layer in cache):
        return
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

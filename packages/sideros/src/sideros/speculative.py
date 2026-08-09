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

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

from sideros.core.cache import LayerCache
from sideros.core.prefill import prefill

if TYPE_CHECKING:
    # `generate` imports this module to reach the speculative path; the names below are
    # annotations only, so they stay out of the runtime cycle.
    from sideros.generate import CausalLM, Meter


class SpeculationRefused(Exception):
    """A draft that cannot be verified is refused before the first token. There is no
    degraded mode: a wrong number costs more than the speculation was worth."""


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


def stream_speculative_ids[C: LayerCache, D: LayerCache](
    target: "CausalLM[C]",
    draft: "CausalLM[D]",
    prompt: list[int],
    *,
    max_tokens: int,
    lookahead: int = 4,
    stop: Collection[int] = (),
    meter: "Meter | None" = None,
    acceptance: Acceptance | None = None,
) -> Iterator[int]:
    """The target's greedy stream, token for token, with the draft only paying for speed.

    `lookahead` is how many tokens a round proposes; `acceptance` collects what the rounds
    actually accepted, for whoever is measuring.
    """
    if lookahead < 1:
        raise ValueError(f"lookahead must be at least 1: {lookahead}")
    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    _must_rewind(target_cache, "target")
    _must_rewind(draft_cache, "draft")

    if meter is not None:
        meter.prefill(len(prompt))

    committed = list(prompt)
    # The first token is the target's alone: the prompt's own forward already yields one
    # (that is ttft), and the draft has nothing to propose before it.
    ids = mx.array(committed)
    window = prefill(lambda block: target(ids[block][None], target_cache), ids.size, target_cache)
    pending = _ints(mx.argmax(target(ids[window][None], target_cache)[:, -1, :], axis=-1))
    committed += pending

    emitted = 0
    while emitted < max_tokens:
        if not pending:
            pending, accepted = _round(
                target, draft, target_cache, draft_cache, committed, lookahead
            )
            committed += pending
            if acceptance is not None:
                acceptance.rounds += 1
                acceptance.proposed += lookahead
                acceptance.accepted += accepted
        token = pending.pop(0)
        if token in stop:
            return
        if meter is not None:
            meter.token()
        yield token
        emitted += 1


def _round[C: LayerCache, D: LayerCache](
    target: "CausalLM[C]",
    draft: "CausalLM[D]",
    target_cache: list[C],
    draft_cache: list[D],
    committed: list[int],
    lookahead: int,
) -> tuple[list[int], int]:
    """One round, returning the tokens it settled and how many of them were the draft's.

    Each model is fed whatever committed ids its own cache has not seen — after a full
    acceptance the draft is exactly one token behind the target. Row j of the target's
    logits predicts the token after `verified[j]`, so the last `lookahead + 1` rows judge
    every proposal and hand back the target's own token past the accepted run.

    A rejected proposal leaves keys in both caches describing a sequence that never
    happened: both are rewound to the accepted prefix before the round returns.
    """
    catchup = mx.array(committed[draft_cache[0].offset :])
    # Round one is where the draft meets the whole prompt; every round after it is a token
    # or two behind, and the split collapses to the single block it already was.
    window = prefill(
        lambda block: draft(catchup[block][None], draft_cache), catchup.size, draft_cache
    )
    ids = catchup[window][None]
    proposals: list[mx.array] = []
    for _ in range(lookahead):
        token = mx.argmax(draft(ids, draft_cache)[:, -1, :], axis=-1)
        # Queued as it is built: the GPU runs draft step n while the CPU builds step n + 1,
        # instead of idling until the round's single sync below.
        mx.async_eval(token)
        proposals.append(token)
        ids = token[None]

    drafted = mx.concatenate(proposals)
    gap = mx.array(committed[target_cache[0].offset :], dtype=drafted.dtype)
    verified = mx.concatenate([gap, drafted])
    judged = mx.argmax(target(verified[None], target_cache)[0, -(lookahead + 1) :], axis=-1)
    mx.eval(judged, drafted)

    proposed, predicted = _ints(drafted), _ints(judged)
    accepted = 0
    while accepted < lookahead and proposed[accepted] == predicted[accepted]:
        accepted += 1

    length = len(committed) + accepted
    for layer in target_cache:
        layer.trim(length)
    for layer in draft_cache:
        layer.trim(length)
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

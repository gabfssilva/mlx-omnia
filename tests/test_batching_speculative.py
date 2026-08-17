"""A drafting slot inside the tick emits what the same slot emits without a draft.

The doubles are the ones `test_speculative` is written against, with the cache the batched
path needs: logits are a function of the row's absolute position, so a rewind that kept a
rejected row spells a different id here rather than a different speed.
"""

from collections.abc import Collection, Sequence

import mlx.core as mx
import pytest

from mlx_omnia.engine.batching import (
    BatchedKVCache,
    BatchSequence,
    prepare_batch_sequence,
    step,
)
from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.generate import Meter, greedy
from mlx_omnia.engine.speculative import Autoregressive, Proposer

VOCAB = 64
PROMPT = [0]
SCRIPT = [10 + index for index in range(16)]
HOSTILE = [token + 1 for token in SCRIPT]
ALTERNATING = [token if index % 2 == 0 else token + 1 for index, token in enumerate(SCRIPT)]
LOOKAHEAD = 2


class ScriptCache(KVCache):
    """A growing KV buffer that declines a fixed shape, so these doubles stay on the eager
    path they are written against.

    Truthful and not a trick: the double picks its logits from the host-side offset of the
    row it is answering for, which is a read a traced graph cannot make. A real trunk reads
    its cache through the graph and promotes."""

    @property
    def is_fixable(self) -> bool:
        return False


class DraftLM:
    """The draft's own trunk: logits by absolute position, read from its cache's offset."""

    def __init__(self, script: list[int], vocab: int) -> None:
        self.script = script
        self.vocab = vocab

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        assert cache is not None, "the round always carries a cache"
        offset = cache[0].offset
        length = ids.shape[-1]
        last = len(self.script) - 1
        rows = mx.array([self.script[min(offset + index, last)] for index in range(length)])
        consumed = mx.zeros((1, 1, length, 1))
        cache[0].update_and_fetch(consumed, consumed)
        distance = mx.abs(mx.arange(self.vocab)[None] - rows[:, None]).astype(mx.float32)
        return (-distance)[None]


class ScriptedBatchLM:
    """The target, readable through both shapes the batched path hands it: a slot's own cache
    list on the drafting round, and the ragged adapter on the shared forward."""


    def __init__(self, script: list[int], vocab: int) -> None:
        self.script = script
        self.vocab = vocab

    def make_cache(self) -> list[ScriptCache]:
        return [ScriptCache()]

    def __call__(self, ids: mx.array, cache: Sequence[KVStore]) -> mx.array:
        layer = cache[0]
        rows, length = ids.shape[0], ids.shape[1]
        offsets = _offsets(layer, rows)
        consumed = mx.zeros((rows, 1, length, 1))
        if isinstance(layer, BatchedKVCache):
            layer.attend(consumed, keys=consumed, values=consumed, scale=1.0, mask=None)
        else:
            assert isinstance(layer, KVCache)
            layer.update_and_fetch(consumed, consumed)
        last = len(self.script) - 1
        targets = mx.array(
            [[self.script[min(offset + i, last)] for i in range(length)] for offset in offsets]
        )
        distance = mx.abs(mx.arange(self.vocab)[None, None] - targets[..., None])
        return -distance.astype(mx.float32)


def _offsets(layer: KVStore, rows: int) -> list[int]:
    offset = layer.offset
    if not isinstance(offset, mx.array):
        return [offset] * rows
    listed = offset.tolist()
    assert isinstance(listed, list)
    positions: list[int] = []
    for value in listed:
        assert isinstance(value, int)
        positions.append(value)
    return positions


def _proposer(script: list[int]) -> Proposer:
    return Autoregressive(DraftLM(script, VOCAB), LOOKAHEAD)


def _sequence(
    model: ScriptedBatchLM,
    *,
    prompt: list[int] = PROMPT,
    max_tokens: int,
    proposer: Proposer | None = None,
    stop: Collection[int] = (),
    meter: Meter | None = None,
) -> BatchSequence:
    return prepare_batch_sequence(
        model,
        prompt,
        max_tokens=max_tokens,
        sampler=greedy,
        stop=stop,
        meter=meter,
        proposer=proposer,
    )


def _drain(model: ScriptedBatchLM, sequence: BatchSequence) -> tuple[list[int], list[int]]:
    """Every id the slot emitted, and how many each tick was worth."""
    emitted: list[int] = []
    ticks: list[int] = []
    while not sequence.finished:
        ids = step(model, [sequence])[0]
        ticks.append(len(ids))
        emitted.extend(ids)
    return emitted, ticks


@pytest.mark.parametrize("script", [SCRIPT, HOSTILE, ALTERNATING])
def test_a_drafting_slot_emits_the_undrafted_stream(script: list[int]) -> None:
    """The hostile draft is the discriminating one: every proposal differs from the target's
    script, so a slot that accepted one would spell a different id here."""
    model = ScriptedBatchLM(SCRIPT, VOCAB)
    drafted, _ = _drain(model, _sequence(model, max_tokens=8, proposer=_proposer(script)))
    plain, _ = _drain(model, _sequence(model, max_tokens=8))

    assert drafted == plain
    assert drafted == SCRIPT[:8]


def test_a_stop_token_ends_the_round_it_lands_in() -> None:
    """The round settles several ids at once and the stop token is one of them: what the slot
    owes is the ids before it, never the stop token itself."""
    model = ScriptedBatchLM(SCRIPT, VOCAB)
    stop = {SCRIPT[3]}
    drafted, _ = _drain(
        model, _sequence(model, max_tokens=8, stop=stop, proposer=_proposer(SCRIPT))
    )
    plain, _ = _drain(model, _sequence(model, max_tokens=8, stop=stop))

    assert drafted == plain
    assert drafted == SCRIPT[:3]


def test_one_tick_settles_more_than_one_id() -> None:
    """What the batched round buys: with a draft the target agrees with, a single `step` is
    worth the accepted proposals plus the target's own token."""
    model = ScriptedBatchLM(SCRIPT, VOCAB)
    emitted, ticks = _drain(model, _sequence(model, max_tokens=8, proposer=_proposer(SCRIPT)))

    assert emitted == SCRIPT[:8]
    assert max(ticks) > 1
    assert len(ticks) < len(emitted)


def test_a_drafting_slot_and_a_plain_one_share_a_tick() -> None:
    """Mixed batch: the drafting slot runs its own round, the other joins the shared forward,
    and neither reads the other's positions — the two prompts have different lengths, so a
    slot that borrowed the wrong offset spells a different script."""
    model = ScriptedBatchLM(SCRIPT, VOCAB)
    drafting = _sequence(model, max_tokens=6, proposer=_proposer(SCRIPT))
    plain = _sequence(model, prompt=[0, 1, 2], max_tokens=6)
    drafted_ids: list[int] = []
    plain_ids: list[int] = []
    while not (drafting.finished and plain.finished):
        active = [sequence for sequence in (drafting, plain) if not sequence.finished]
        advanced = step(model, active)
        for sequence, ids in zip(active, advanced, strict=True):
            (drafted_ids if sequence is drafting else plain_ids).extend(ids)

    solo_drafted, _ = _drain(model, _sequence(model, max_tokens=6))
    solo_plain, _ = _drain(model, _sequence(model, prompt=[0, 1, 2], max_tokens=6))

    assert drafted_ids == solo_drafted == SCRIPT[:6]
    assert plain_ids == solo_plain == SCRIPT[2:8]
    assert drafted_ids != plain_ids


def test_the_meter_counts_the_rounds_and_the_ids() -> None:
    """The acceptance the bench reads is the round's own tally, and the completion count is
    still one mark per id handed out — not per id a round settled."""
    model = ScriptedBatchLM(SCRIPT, VOCAB)
    meter = Meter()
    emitted, _ = _drain(
        model, _sequence(model, max_tokens=6, proposer=_proposer(SCRIPT), meter=meter)
    )

    assert emitted == SCRIPT[:6]
    assert meter.speculation.rounds > 0
    assert meter.speculation.accepted == meter.speculation.proposed
    assert meter.completion_tokens == 6
    assert meter.prompt_tokens == len(PROMPT)

    plain_meter = Meter()
    _drain(model, _sequence(model, max_tokens=6, meter=plain_meter))

    assert plain_meter.speculation.rounds == 0


def test_a_rejecting_draft_still_settles_the_targets_own_token() -> None:
    """Nothing accepted is still a token: the round hands back the target's argmax past the
    gap, which is exactly what an undrafted tick would have drawn."""
    model = ScriptedBatchLM(SCRIPT, VOCAB)
    meter = Meter()
    emitted, ticks = _drain(
        model, _sequence(model, max_tokens=5, proposer=_proposer(HOSTILE), meter=meter)
    )

    assert emitted == SCRIPT[:5]
    assert ticks == [1] * len(ticks)
    assert meter.speculation.accepted == 0
    assert meter.speculation.proposed == meter.speculation.rounds * LOOKAHEAD

"""Speculative decoding: a small draft proposes, the target verifies in one forward.

A round costs one read of the big weights however many proposals survive it, and what comes
out is the target's own script: acceptance is equality against the target's argmax, so a
draft can only change the speed. Anything else is a bug, not a tradeoff.

Greedy only for now, and the reason is narrower than this docstring used to claim. Two
rules leave a *sampled* distribution intact, and they differ in acceptance, not in law:
draw the target's own token on each verification row and accept a proposal on equality —
the emitted token is that draw on both branches, so any `q` is exact — or accept with
probability `min(1, p/q)` and redraw a rejection from the residual `max(0, p - q)`. The
second is the maximal coupling of `p` and `q` and accepts `sum(min(p, q)) = 1 - TV(p, q)`,
which no exact rule beats; the first couples them independently and accepts `sum(p * q)`,
always less. Greedy is where the two coincide, which is why the equality below is both.

Neither is wired. The ratio needs both distributions and `Sampler` is an opaque
`logits -> id` callable that reaches neither; the equality rule needs nothing new and is
simply not written. So `generate.stream_ids` refuses the non-greedy path by name because
nothing samples here yet — not because token equality would bias the output, which is what
this said and is false. `high-temp-draft.md` carries the argument and the path.
"""

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never

import mlx.core as mx

from mlx_omnia.engine.core.api import Draftable, LanguageModel, Proposer, Step
from mlx_omnia.engine.core.cache import LayerCache, Layout, Rows, Snapshot
from mlx_omnia.engine.core.decode import compiled_verify, state_of
from mlx_omnia.engine.core.prefill import landing, prefill
from mlx_omnia.engine.core.prefix import Prefix, Prefixes

if TYPE_CHECKING:
    # `generate` imports this module to reach the speculative path; the names below are
    # annotations only, so they stay out of the runtime cycle.
    from mlx_omnia.engine.generate import Meter


class SpeculationRefused(Exception):
    """A draft that cannot be verified is refused before the first token. There is no
    degraded mode: a wrong number costs more than the speculation was worth."""


def _lends_vocabulary(target: object, block: int) -> Draftable[LayerCache]:
    """The two refusals `Chained` and `Persistent` share, before either holds state."""
    if block < 2:
        raise ValueError(f"a block of {block} proposes nothing")
    if not isinstance(target, Draftable):
        raise SpeculationRefused(
            f"{type(target).__name__} does not lend its embedding table and its head "
            "(`core.api.Draftable`), which is the whole of an MTP step's vocabulary"
        )
    return target


def taps_refused(taps: Collection[int]) -> SpeculationRefused:
    return SpeculationRefused(
        f"this draft conditions on the target's blocks {list(taps)} and the target has "
        "no `block_outputs`: what it would read instead is nothing at all"
    )


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


class Contributed(LayerCache):
    """One drafter layer as the prefix walk sees it: the inner rows, branded.

    The wrapper exists for the seed. A span written with a drafter's rows must never match
    a chain built without them — or built over another drafter — and `signature` is what
    `core.cache_file.policy` folds into the chain's seed; a plain KV buffer's own is empty.
    The brand is the step's class name, which moves when the head's shape does; which
    *checkpoint* trained it is already in the seed as the model and its stamp.

    `stored` refuses a range past what the inner layer holds, where a plain buffer would
    hand back stale rows: the drafter's rows lag the trunk's, and a short answer here is
    the walk's own "not written yet" — the span waits for the next commit.
    """

    def __init__(self, inner: LayerCache, brand: str) -> None:
        super().__init__()
        self._inner = inner
        self._brand = brand

    @property
    def layout(self) -> Mapping[str, Layout]:
        return self._inner.layout

    @property
    def signature(self) -> dict[str, object]:
        return {"drafter": self._brand, **self._inner.signature}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        if self._inner.rows < stop:
            return {}
        return self._inner.stored(start, stop)

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self._inner.restore(offset, tensors)


class _Paired[S: LayerCache](Contributed):
    """`Persistent`'s layer on the walk, shifted one row so every span is a function of its
    own ids.

    The inner cache holds pair *i* — the target's hidden at *i* with the id at *i + 1* — at
    row *i*, so an unshifted span's last pair would depend on the first id past the span,
    which is one id more than the span's key is taken over. Walk row *j* is therefore pair
    *j - 1*: a function of ids `..j`, all inside the key. The pair that closes on a boundary
    is not stored as rows at all — the boundary's `seed` snapshot is its hidden half, and
    the resumed run builds the pair in its first catch-up from the tail's first id. Row 0
    holds nothing, which is right: no pair ends at the first token.
    """

    def __init__(
        self, proposer: "Persistent[S]", inner: LayerCache, brand: str, *, seeded: bool
    ) -> None:
        super().__init__(inner, brand)
        self._proposer = proposer
        self._seeded = seeded

    @property
    def layout(self) -> Mapping[str, Layout]:
        inner = self._inner.layout
        return {**inner, "seed": Snapshot()} if self._seeded else inner

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        held: dict[str, mx.array] = {}
        if start < stop:
            if self._inner.rows < stop - 1:
                return {}
            held.update(self._inner.stored(max(start - 1, 0), stop - 1))
        if self._seeded:
            hidden = self._proposer.hidden_at(stop)
            if hidden is not None:
                held["seed"] = hidden
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        rows = {name: held for name, held in tensors.items() if name != "seed"}
        self._inner.restore(offset - 1, rows)
        if self._seeded:
            self._proposer.seed(tensors.get("seed"))


class Autoregressive[D: LayerCache]:
    """A small language model as a proposer: `width` steps, each one conditioned on the id
    the last one drew. It reads no blocks of the target — its whole input is the ids — so
    the round never asks the target for features.

    It is the original speculative draft, and it is a `Proposer` and not the special case
    the loop is written around: the round is single, and what a checkpoint costs to propose
    with is the checkpoint's business.
    """

    def __init__(self, model: LanguageModel[D], width: int) -> None:
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

    @property
    def caches(self) -> Sequence[LayerCache]:
        """Nothing: its whole input is ids, and round one feeds it whatever the target's
        cache already covers. That catch-up is a prefill of the draft's own small
        checkpoint, and it happens after the first token — outside the TTFT the prefix
        exists to cut."""
        return ()

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


class Chained:
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
    vllm does is `Persistent`, and on Qwen3.8's head it measured worth it — this stays as
    the shape for a head whose acceptance was measured empty, and as the compiled-chain
    fast path a fresh fixed buffer makes possible.

    What crosses a round is one row: the target's own reading of the last position it kept,
    handed over by `absorb`. `taps` is the block that row comes from.
    """

    def __init__(
        self,
        target: Draftable[LayerCache],
        step: Step[LayerCache],
        *,
        block: int,
        tap: int,
    ) -> None:
        self._target = _lends_vocabulary(target, block)
        self._step = step
        self._width = block - 1
        self._tap = tap
        self._hidden: mx.array | None = None
        self._cache: list[LayerCache] = list(step.make_cache())
        self._chain: Callable[[mx.array, mx.array], tuple[mx.array, mx.array]] | None = None

    @property
    def taps(self) -> Sequence[int]:
        return (self._tap,)

    @property
    def width(self) -> int:
        return self._width

    @property
    def caches(self) -> Sequence[LayerCache]:
        """Nothing: the one row it carries is the target's reading of the last position,
        and a resumed prompt still runs a forward over its tail — one row is the least
        that forward can produce."""
        return ()

    def absorb(self, features: mx.array) -> None:
        """Only the last row is kept. The step conditions on the position right before the
        token it is given, and the round hands over exactly the positions the target's cache
        took — so the last of them is that position, every time."""
        self._hidden = features[:, -1:, :]

    def propose(self, committed: Sequence[int]) -> mx.array:
        """`committed[-1]` is the token the target's cache has not seen, and `self._hidden`
        is the target's reading of the one before it: the pair the step is defined on.

        Both borrowed pieces are the raw ones (`core.api.Draftable`), and the head is read
        once per drafted token — which on a 131k vocabulary is most of what a round's draft
        costs.

        Every round starts the step's cache empty, which is the shape the acceptance above
        was measured in, and `LayerCache.restore(0, {})` is that: no tensors and position
        zero. The first link of the first round runs eagerly, and not as a warm-up that
        costs a forward — its output is the round's first proposal. It exists because a
        fixed buffer is promoted from rows that exist and a cache built a moment ago has
        none, so the shapes the graph needs arrive with the link that produces them.
        """
        assert self._hidden is not None, "the round feeds the proposer before it asks"
        hidden = self._hidden
        token = mx.array([committed[-1]])
        for layer in self._cache:
            layer.restore(0, {})
        drafted: list[mx.array] = []
        for _ in range(self._width):
            if self._chain is not None:
                token, hidden = self._chain(token, hidden)
            else:
                hidden = self._step(self._target.raw_embed(token[None]), hidden, self._cache)
                token = mx.argmax(self._target.raw_logits(hidden)[:, -1, :], axis=-1)
                self._compile()
            mx.async_eval(token)
            drafted.append(token)
        return mx.concatenate(drafted)

    def settle(self, length: int) -> None:
        """Nothing: the chain's cache belonged to the round that just ended, and the row that
        crosses to the next one arrives through `absorb` already cut to what was kept."""

    def _compile(self) -> None:
        """One graph for every link after the first, over a fixed buffer of `width` rows.

        The only thing that moves between links is the position, which lives in the buffer's
        own container — so the same trace serves the whole chain, and `restore(0, {})` is
        what puts it back between rounds. A step whose cache cannot present a fixed shape
        keeps the eager links and this stays `None`.
        """
        if not all(layer.is_fixable for layer in self._cache):
            return
        slots = [layer.fixed(self._width) for layer in self._cache]
        mx.eval(*(tensor for layer in slots for tensor in layer.tensors))
        self._cache = slots
        embed, head, step = self._target.raw_embed, self._target.raw_logits, self._step

        def forward(token: mx.array, hidden: mx.array) -> tuple[mx.array, mx.array]:
            out = step(embed(token[None]), hidden, slots)
            return mx.argmax(head(out)[:, -1, :], axis=-1), out

        state = state_of(slots)
        self._chain = mx.compile(forward, inputs=state, outputs=state)


class Persistent[S: LayerCache]:
    """An MTP step as a proposer whose keys survive the round — vllm's own shape: the head
    attends to the whole history, rotated at absolute positions, instead of to its round's
    few rows alone. On Qwen3.8's head the history is worth real acceptance — 0.73 to 0.85
    at depth one on the nvfp4 27B entry — which buys more than `Chained`'s compiled chain
    saves in dispatch.

    The drafter's cache holds one row per committed position, and every surviving row was
    written from the *target's* hidden: a round's chained links append rows built from the
    step's own guesses, and `settle` trims straight back past them. The accepted positions
    re-enter on the next round's catch-up — the rows the round handed `absorb` — so a guess
    never conditions anything beyond the round that made it.

    Catch-up is one forward over every pair the cache is missing: pair *i* is *(the
    target's hidden at i, the embedding of the committed token at i+1)*, it occupies cache
    row *i*, and its last output row is the first proposal. That works only for a cache
    that can trim, which an MTP step's KV can and a recurrent drafter's cannot —
    `_must_rewind` refuses the rest.
    """

    def __init__(
        self,
        target: Draftable[LayerCache],
        step: Step[S],
        *,
        block: int,
        tap: int,
    ) -> None:
        self._target = _lends_vocabulary(target, block)
        self._step = step
        self._width = block - 1
        self._tap = tap
        self._cache = step.make_cache()
        _must_rewind(self._cache, "draft")
        self._pending: mx.array | None = None
        self._anchor = 0
        """Rows of the cache built from the target's hidden — where `settle` trims to."""
        for layer in self._cache:
            kinds = layer.layout.values()
            if not kinds or any(
                not isinstance(kind, Rows) or kind.stride != 1 or kind.keep is not None
                for kind in kinds
            ):
                raise SpeculationRefused(
                    f"{type(layer).__name__} does not hold one plain row per pair, and "
                    "`_Paired`'s one-row shift is written against exactly that: a step "
                    "whose cache pools, slides or snapshots needs its own walk adapter"
                )
        self._walked = tuple(
            _Paired(self, layer, type(step).__name__, seeded=index == 0)
            for index, layer in enumerate(self._cache)
        )

    @property
    def taps(self) -> Sequence[int]:
        return (self._tap,)

    @property
    def width(self) -> int:
        return self._width

    @property
    def caches(self) -> Sequence[LayerCache]:
        """Its layers, on the walk. Every pair is built from the target's hidden at that
        position — which a resumed prompt never produces — so the pairs ride the
        conversation's spans instead of being rebuilt: one layer against the target's
        dozens, and `absorb` is then handed only the tail."""
        return self._walked

    def seed(self, hidden: mx.array | None) -> None:
        """The boundary pair's hidden half, restored: the target's reading of the last
        resumed position, which the tail's forward will never produce. It lands in
        `_pending` ahead of the tail's own rows, so the first catch-up builds the boundary
        pair from it and the tail's first id — the same arithmetic as any other pair."""
        assert self._pending is None, "seeded after features already arrived"
        assert hidden is not None, "an anchored resume always carries the boundary hidden"
        self._pending = hidden

    def hidden_at(self, boundary: int) -> mx.array | None:
        """The target's reading of `boundary - 1`, when the absorbed rows reach exactly
        there — which is the only moment the walk asks: a snapshot is cut while the trunk
        stands on the boundary, and standing there means everything before it was fed and
        absorbed."""
        held = self._pending
        if held is None:
            return None
        covered = self._cache[0].rows + held.shape[1]
        return held[:, -1:, :] if covered == boundary else None

    def absorb(self, features: mx.array) -> None:
        """Every row is kept, not the last one: each is the hidden half of a pair the
        cache does not have yet, and the next catch-up runs over all of them."""
        self._pending = (
            features
            if self._pending is None
            else mx.concatenate([self._pending, features], axis=1)
        )

    def propose(self, committed: Sequence[int]) -> mx.array:
        assert self._pending is not None, "the round feeds the proposer before it asks"
        offset = self._cache[0].offset
        pairs = len(committed) - 1 - offset
        assert pairs == self._pending.shape[1], (
            f"the cache holds {offset} pairs and {self._pending.shape[1]} rows arrived: "
            f"together they must reach position {len(committed) - 1} exactly"
        )
        # Blocked like any prefill, and for the same reason: after a whole prompt was
        # absorbed, the catch-up is the step's attention over every pair at once, and fed
        # whole that materializes scores over the full square — ~127 GB at 40k on the 27B.
        # `prefill` feeds all but the last block and the last one's output is the proposal's.
        tokens = mx.array(committed[offset + 1 :])[None]
        pending = self._pending
        window = prefill(
            lambda block: self._step(
                self._target.raw_embed(tokens[:, block]), pending[:, block], self._cache
            ),
            pairs,
            self._cache,
        )
        out = self._step(
            self._target.raw_embed(tokens[:, window]), pending[:, window], self._cache
        )
        self._pending = None
        self._anchor = offset + pairs
        hidden = out[:, -1:, :]
        token = mx.argmax(self._target.raw_logits(hidden)[:, -1, :], axis=-1)[None]
        mx.async_eval(token)
        proposals = [token[0]]
        for _ in range(self._width - 1):
            hidden = self._step(self._target.raw_embed(token), hidden, self._cache)
            token = mx.argmax(self._target.raw_logits(hidden)[:, -1, :], axis=-1)[None]
            mx.async_eval(token)
            proposals.append(token[0])
        return mx.concatenate(proposals)

    def settle(self, length: int) -> None:
        """Back past the round's guessed rows: only pairs built from the target's own
        hidden stay, and the accepted positions return through the next `absorb`."""
        for layer in self._cache:
            layer.trim(self._anchor)


def stream_speculative_ids[D: LayerCache](
    target: LanguageModel[LayerCache],
    draft: LanguageModel[D] | Proposer,
    prompt: Iterable[int],
    *,
    max_tokens: int,
    lookahead: int = 4,
    stop: Collection[int] = (),
    meter: "Meter | None" = None,
    acceptance: Acceptance | None = None,
    prefix: "Prefixes | None" = None,
) -> Iterator[int]:
    """The target's greedy stream, token for token, with the draft only paying for speed.

    A `Proposer` says for itself how many ids a round proposes; a plain language model is
    wrapped in `Autoregressive` and `lookahead` is what it proposes. `acceptance` collects
    what the rounds actually accepted, for whoever is measuring.

    `prefix` composes with the rounds, and the invariant that makes it compose is the one the
    rounds already keep: between them the target's rows are the settled ids and never a
    proposal, so the spans a boundary closes are this conversation's. Nothing is ever stored
    inside a round — the verification forward writes `width + 1` rows before it knows how
    many survive — and a round that would step over a boundary the anchor needs is shortened
    to stop on it. A proposer whose state is per position contributes its layers to the walk
    (`Proposer.caches`), so the spans carry the drafter's rows beside the target's and a
    resumed prompt resumes both.
    """
    proposer: Proposer = draft if isinstance(draft, Proposer) else Autoregressive(draft, lookahead)
    taps = proposer.taps
    if taps and not isinstance(target, Draftable):
        raise taps_refused(taps)
    target_cache = list(target.make_cache())
    replays = rewinds(target_cache, "target")

    # Read here, ahead of the meter that wants its length: a round rewinds both caches in
    # step, so this loop needs the whole prompt before its first forward whatever shape it
    # arrived in — there is no block for a lazy one to overlap against.
    committed = list(prompt)
    walk = prefix.begin(target_cache, proposer.caches) if prefix is not None else None
    reused = 0 if walk is None else walk.resume(committed, target_cache)
    if meter is not None:
        meter.prefill(len(committed), reused)
        meter.kept_prefix = walk is not None
        if walk is not None:
            meter.prefix = walk.reuse

    def forward(ids: mx.array) -> tuple[mx.array, mx.array | None]:
        """One forward of the target: its logits, and what the proposer reads of it when it
        reads anything. The verification forward needs both out of the same pass, which is
        also why it can never take the compiled decode path."""
        if not taps:
            return target(ids, target_cache), None
        assert isinstance(target, Draftable)
        return target.block_outputs(ids, target_cache, at=taps)

    def feed(ids: mx.array) -> mx.array:
        """Prefill: every row a block writes is kept, so the proposer gets all of them."""
        logits, features = forward(ids)
        if features is not None:
            proposer.absorb(features)
        return logits

    def close(rows: int) -> None:
        if walk is not None:
            walk.commit(committed, target_cache, rows)

    def stopped(at: int) -> None:
        close(reused + at)
        if meter is not None:
            meter.fed(at)

    # The first token is the target's alone: the prompt's own forward already yields one
    # (that is ttft), and the draft has nothing to propose before it.
    ids = mx.array(committed[reused:])
    window = prefill(
        lambda block: feed(ids[block][None]),
        ids.size,
        target_cache,
        after=stopped,
    )
    if walk is not None and walk.chain.anchored:
        # The last block cut on the last span boundary, so the trunk stops on one — the same
        # cut the plain loop makes, and for the same reason: a recurrent state exists only
        # while the layer is stopped.
        cut = landing(reused + window.start, len(committed), walk.chain.span) - reused
        if cut > window.start:
            feed(ids[window.start : cut][None])
            mx.eval([tensor for layer in target_cache for tensor in layer.tensors])
            stopped(cut)
            window = slice(cut, window.stop)
    pending = _ints(mx.argmax(feed(ids[window][None])[:, -1, :], axis=-1))
    if meter is not None:
        meter.fed(ids.size)
    committed += pending

    # After the prefill, which is where a promotion belongs: the shapes are final and the
    # rows the graph will read are written. A target that declines leaves this `None` and
    # every round below restores and replays.
    compiled = (
        compiled_verify(target, target_cache, rows=proposer.width + 1, taps=taps)
        if taps and isinstance(target, Draftable)
        else None
    )

    emitted = 0
    while emitted < max_tokens:
        if not pending:
            settled = len(committed) - 1
            limit = room(walk, settled, proposer.width)
            if (
                compiled is not None
                and limit is None
                and target_cache[0].offset == settled
            ):
                pending, accepted = _compiled_round(compiled, proposer, committed)
            else:
                pending, accepted = one_round(
                    forward,
                    proposer,
                    target_cache,
                    committed,
                    replays,
                    limit=limit,
                    # The promoted layers carry the round's checkpoints either way, so a
                    # shortened round rewinds by picking one rather than replaying.
                    rewind=None if compiled is None else compiled[1],
                )
            committed += pending
            if acceptance is not None:
                acceptance.rounds += 1
                acceptance.proposed += proposer.width
                acceptance.accepted += accepted
            if walk is not None:
                # Between rounds, and never inside one: the rows are the settled ids exactly
                # here and nowhere else in the loop.
                close(len(committed) - 1)
        token = pending.pop(0)
        if token in stop:
            return
        if meter is not None:
            meter.token()
        yield token
        emitted += 1
    if walk is not None:
        close(len(committed) - 1)


def _compiled_round(
    compiled: tuple[
        Callable[[mx.array], tuple[mx.array, mx.array]], Callable[[int, int], None]
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
        rewind_fn(1 + accepted, len(committed) + accepted)
    proposer.absorb(features[:, : 1 + accepted])
    proposer.settle(len(committed) + accepted)
    return [*proposed[:accepted], predicted[accepted]], accepted


def room(walk: "Prefix | None", settled: int, width: int) -> int | None:
    """How many proposals a round may make without stepping over a boundary the anchor needs,
    or `None` for a round nothing constrains.

    Only a trunk with recurrent layers constrains one. A span of rows can be cut out of the
    buffers at any later moment, so a round that runs past a boundary loses nothing; a
    snapshot cannot, because the state it would capture only exists while the layer is
    stopped, and a round walks `1 + accepted` positions in one forward.

    A shortened round is still a round: at zero it feeds the gap alone and settles one token,
    landing exactly on the boundary. One round per span at four wide and three accepted on
    average is one in eighty-five, which is what the cap costs.
    """
    if walk is None or not walk.chain.anchored:
        return None
    span = walk.chain.span
    boundary = settled // span * span + span
    return None if boundary - settled - 1 >= width else max(0, boundary - settled - 1)


def one_round[C: LayerCache](
    forward: Callable[[mx.array], tuple[mx.array, mx.array | None]],
    proposer: Proposer,
    target_cache: list[C],
    committed: list[int],
    replays: bool,
    *,
    limit: int | None = None,
    rewind: Callable[[int, int], None] | None = None,
) -> tuple[list[int], int]:
    """One round, returning the tokens it settled and how many of them were the draft's.

    `limit` cuts the draft down so the round stops on a boundary a snapshot needs (`_room`).
    It is applied to the proposal and not to the acceptance: the proposer wrote what it wrote,
    and the rows the target never sees are rows it never has to undo.

    The target is fed whatever committed ids its cache has not seen, then the proposals.
    Row j of its logits predicts the token after `verified[j]`, so the last `width + 1`
    rows judge every proposal and hand back the target's own token past the accepted run.

    A rejected proposal leaves a cache describing a sequence that never happened, and there
    are three ways back. `rewind` is the cheapest and is passed in when the layers are the
    promoted ones a compiled round already made room in — a slot pick and a moved position,
    no forward. Without it, a cache of keys drops the rows: `trim`. A cache holding recurrent
    state cannot, because the recurrence kept no history to subtract — so the round takes a
    `checkpoint()` before the forward and, when anything was rejected, restores it and runs
    the kept rows again. All three land on the same offset; only the last costs a forward,
    and only when the draft was wrong.

    What the target *kept* is what the proposer is given to read — the gap plus the accepted
    proposals, never the rejected tail. Causality is why the first forward's features serve:
    row j's output cannot depend on rows past it, so the rows a replay would recompute are
    the rows already in hand.
    """
    drafted = proposer.propose(committed)
    if limit is not None and limit < drafted.size:
        drafted = drafted[:limit]
    width = drafted.size
    gap = mx.array(committed[target_cache[0].offset :], dtype=drafted.dtype)
    verified = mx.concatenate([gap, drafted])
    # Before the forward, which is the only moment a restore point means anything — and only
    # when nothing cheaper is on offer, because a promoted cache records the round's rows as
    # it walks them.
    restores = (
        [layer.checkpoint() for layer in target_cache] if replays and rewind is None else ()
    )
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
    if accepted == width:
        # Nothing to undo when every proposal survived: the cache is already the sequence.
        pass
    elif rewind is not None:
        rewind(gap.size + accepted, length)
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


def rewinds(cache: Sequence[LayerCache], role: str) -> bool:
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

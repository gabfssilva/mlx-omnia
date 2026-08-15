"""What a model family declares. Nothing here knows how the engine executes it.

Three audiences, today spread across five modules, and this file is the first:

- the **trunk** (``models/<family>/model.py``) — here;
- the **cache classes** — next door, in ``cache`` and ``attend``, because that is where
  the code reading them already lives;
- the **facade** (``TextLanguageModel``) — no family writes one; it is ``model.Model``
  and lives in ``engine.language`` with the types of the layer above.

The sieve that decides where something belongs:

- identical across every family -> a core function;
- varies by kind of state       -> a ``LayerCache`` method;
- varies by architecture        -> a method here.

That is why continuous batching, compiled decode and prefix caching do not appear in
this file even though all three are mandatory. None of them is a question about the
model's arithmetic; all three are questions about state — whether it can present itself
as N rows (``LayerCache.batched``), whether it has a fixed shape (``LayerCache.fixed``),
whether it can cut itself into spans (``LayerCache.layout``/``stored``/``restore``).
They are mandatory because they are a base class's contract, not because anyone declares
them here. The fixed-shape step that consumes them — ``Decoding``, ``DecodePlan``, the
lease over compiled graphs — belongs to ``decode``, because it is the same for every
family.

Prefix caching is absent for a second reason as well: it is not a decode mode. The resume
writes into the layers before the first step, and after that a resumed decode and a cold
decode are the same graph. The disk tier (``server.prefixes``) asks the model for nothing
— it knows keys and bytes, and the only constraint on this side is a dtype safetensors
can write.

It lives in ``core`` because nothing in it knows a family, which is the whole of what
``core`` means. It imports ``cache`` and nothing else, and the direction never reverses:
``decode`` reads it from beside, ``engine.generate``, ``engine.batching`` and
``engine.speculative`` from above, and the families implement it from below. Nothing here
may ever import ``engine.models``, and nothing here may reach up.

Two items left open on purpose:

1. **The name.** ``LanguageModel`` already exists in ``engine.language`` for the facade,
   the one thing no family writes. If this name stays, that one has to give it back.
2. **Resident containers.** A family whose graph reads a global buffer needs it to enter
   ``mx.compile`` as an input, and a change to it to force a rebuild
   (``models/laguna/layers/attention.py:232``). By the sieve above that varies by family
   and would push a third mandatory member. The way out is to invert it — whoever owns
   the buffer registers it, and ``decode`` collects from the registry — and it is what
   remains to be decided before this file is wired to anything.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache

__all__ = [
    "Draftable",
    "Drafting",
    "LanguageModel",
    "Proposer",
    "Step",
]


# ── the trunk ────────────────────────────────────────────────────────────


@runtime_checkable
class LanguageModel[C: LayerCache](Protocol):
    """The minimum, and the only mandatory surface.

    Two members, and they are the only two that vary by architecture: which layer carries
    which kind of cache, and what the step computes. Everything else a family used to
    declare is either the ``LayerCache`` contract (batching, fixed shape, spans) or the
    core's own work (promotion, compilation, leasing, padding, buffer growth).
    """

    def make_cache(self) -> list[C]:
        """Build one cache per layer.

        Returns
        -------
        list[C]
            A fresh cache per layer, in the order the step walks them. That order is
            contract: restoring a prefix, promoting to fixed shape and assembling a
            ragged batch all address layers by index.
        """
        ...

    def __call__(self, ids: mx.array, cache: Sequence[C]) -> mx.array:
        """Run the trunk over ``ids``, appending to ``cache``.

        Parameters
        ----------
        ids : mx.array
            Token ids, ``[rows, tokens]``. The length varies, and this same forward
            serves prefill.
        cache : Sequence[C]
            One entry per layer, in ``make_cache`` order. Growing caches, fixed-shape
            caches and a batch's ragged adapters all answer the same read contract
            (``core.attend.attend``), which is why there is a single ``__call__``.

        Returns
        -------
        mx.array
            Logits, ``[rows, tokens, vocab]``.

        Notes
        -----
        The fixed-shape step is not a second method: it is this one, traced by
        ``core.decode`` over already-promoted caches. The split between the two is a
        precondition of ``mx.compile``, not a decision the family makes.
        """
        ...


@runtime_checkable
class Draftable[C: LayerCache](Protocol):
    """Buys being the target of speculative decode. The trunk's only optional surface.

    ``raw_embed`` and ``raw_logits`` are the two raw ends of the vocabulary. A drafter
    that speaks in hidden rows owns neither an embedding table nor a head, and borrows
    the target's. **Raw** is the point: a facade's ``embed`` usually applies an RMS and
    its ``head`` a multiplier and a softcap, and the drafter was trained against the bare
    tensors. Where the two coincide a family answers with the same tensor twice; where
    they do not, the difference has a name.

    Rewind is not here. Undoing a rejected round is a question about state, and the
    answer belongs to the cache: ``trim`` for what keeps history, ``checkpoint`` for what
    keeps recurrent state, and a fixed-shape rewind for what has been promoted.
    """

    def raw_embed(self, ids: mx.array) -> mx.array:
        """Look ``ids`` up in the bare embedding table, before any facade scaling."""
        ...

    def raw_logits(self, hidden: mx.array) -> mx.array:
        """Project ``hidden`` through the bare head, before any multiplier or softcap."""
        ...

    def block_outputs(
        self, ids: mx.array, cache: Sequence[C], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """Run the forward and tap the blocks the caller names.

        Parameters
        ----------
        ids : mx.array
            Token ids, ``[rows, tokens]``, as in ``LanguageModel.__call__``.
        cache : Sequence[C]
            One entry per layer, in ``make_cache`` order.
        at : Sequence[int]
            Block indices to tap. The caller selects because the whole trunk is dozens of
            tensors where the reader wants five.

        Returns
        -------
        tuple[mx.array, mx.array]
            Logits ``[rows, tokens, vocab]``, and the tapped block outputs concatenated
            on the last axis as ``[rows, tokens, len(at) * hidden]``.
        """
        ...


# ── the drafter ──────────────────────────────────────────────────────────
#
# Another object, not another declaration of the same one: what follows is implemented by
# a tree that is not the trunk, and by a facade almost no family writes.


@runtime_checkable
class Step[S: LayerCache](Protocol):
    """One multi-token-prediction step.

    Maps the pair *(what the trunk held at a position, the embedding of the token after
    it)* to the hidden row of the position after **that** one.

    A tree and not a model: no embedding table, no head, no tokenizer — the two ends of
    the vocabulary belong to the target, which lends them through ``Draftable``.
    """

    @property
    def block(self) -> int:
        """How many ids are worth proposing per round when no one says otherwise.

        It belongs to the step because it is a property of the *pair* and not of the
        round: a wider round costs one more chaining and one more row in the target's
        forward, and buys the acceptance the head still has at that depth. Neither number
        is in the checkpoint's config.
        """
        ...

    def make_cache(self) -> list[S]:
        """Build this tree's own caches, one per layer it walks."""
        ...

    def __call__(
        self, embeddings: mx.array, hidden: mx.array, cache: Sequence[S] | None = None
    ) -> mx.array:
        """Advance one position.

        Parameters
        ----------
        embeddings : mx.array
            The next token's embedding, from the target's ``raw_embed``.
        hidden : mx.array
            What the target held at the current position.
        cache : Sequence[S] | None, optional
            This tree's state. ``None`` runs the step stateless.

        Returns
        -------
        mx.array
            The hidden row of the position after the current one.
        """
        ...


@runtime_checkable
class Proposer(Protocol):
    """Whatever writes the ids a round verifies.

    A round asks for a proposal, whatever it costs to produce one: a small model run
    ``width`` times, or a block drafter run once. The round owns the target and nothing
    else; the proposer owns whatever state it needs between rounds, and undoes it in
    ``settle``.
    """

    @property
    def taps(self) -> Sequence[int]:
        """Which of the target's blocks this one reads, by index.

        Empty asks for none, which is what makes the round use the ordinary forward
        instead of ``Draftable.block_outputs``.
        """
        ...

    @property
    def width(self) -> int:
        """Ids proposed per round."""
        ...

    @property
    def resumes(self) -> bool:
        """Whether this can start against a target whose prompt came from the prefix
        store instead of having been prefilled.

        The question is what this one needed from the prefill. Something that reads no
        block needed nothing; something that keeps only the target's last row catches up
        on the tail a resumed prompt still runs. Something that accumulates one row per
        *position* does not: the resumed positions never produced features and will not
        without a forward over them, which is exactly the prefill the resume avoided.
        That one answers ``False``, and the caller drops the prefix rather than the draft
        — a round saves one read of the weights on every token of every turn, and a
        prefix saves one turn's prefill.
        """
        ...

    def absorb(self, features: mx.array) -> None:
        """Take the target's reading of the positions its cache just accepted and stored.

        Parameters
        ----------
        features : mx.array
            ``[1, new, len(taps) * hidden]``, in order — exactly the positions this one
            has not received yet. Never called when ``taps`` is empty.
        """
        ...

    def propose(self, committed: Sequence[int]) -> mx.array:
        """Write the ids the next round verifies.

        Parameters
        ----------
        committed : Sequence[int]
            The sequence so far, whose last entry the target's cache has not seen.

        Returns
        -------
        mx.array
            ``width`` ids continuing ``committed``.
        """
        ...

    def settle(self, length: int) -> None:
        """The round kept ``length`` ids. Whatever this one wrote past that describes a
        sequence that never happened, and goes away now."""
        ...


@runtime_checkable
class Drafting(Protocol):
    """The facade that accepts a second checkpoint.

    Pairing is not the loader's call: what drafts for a checkpoint is decided outside the
    engine — a setting, a measurement, a checkpoint quantized this morning — and
    ``mlx_omnia.load`` takes an id and no opinion about it. So the facade is loaded first
    and receives the drafter afterwards, from whoever knows.

    ``drafter`` stays out of the trunk's tree so that whoever accounts for memory can
    weigh it: there are two resident checkpoints and only one of them sits under the
    model.

    Implemented by ``TextLanguageModel``. A family overrides only to pick a different
    ``Proposer`` — qwen3_5 swaps the chained one for the persistent one, measured.
    """

    drafter: nn.Module | None

    def speculate_with(self, drafter: nn.Module, *, block_size: int | None = None) -> None:
        """Take this checkpoint as the drafter, or refuse it by name.

        Parameters
        ----------
        drafter : nn.Module
            The tree that will propose.
        block_size : int | None, optional
            Ids per round. Defaults to the drafter's own ``Step.block``.

        Raises
        ------
        TypeError
            The drafter is not a ``Step``, or this target does not lend the two ends of
            its vocabulary. Refused here, by the name of what is missing, rather than as
            a wrong number in the middle of a generation.
        """
        ...

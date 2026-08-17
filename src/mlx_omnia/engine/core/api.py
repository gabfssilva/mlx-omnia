"""What a model family declares. Nothing here knows how the engine executes it.

Three audiences, and this file is the first:

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
them here. The fixed-shape step that consumes them — the promotion, the trace, the lease
over compiled graphs — belongs to ``decode`` and ``engine.batching``, because it is the
same for every family.

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

``engine.language`` has a ``LanguageModel`` of its own; the two are told apart by module
path. That one is the *facade* — prompts in, segments out, what ``load`` returns — and no
family writes it. This one is the checkpoint's own tree, which every family writes. Only
the facade is re-exported from ``mlx_omnia``.

Every declaration a trunk makes is in this file. Nothing about undoing a speculative round
or about compiling a step is here: those are the ``LayerCache`` contract
(``readable``/``checkpoints``/``rewind``), except for the one bit no cache can answer,
which is ``Draftable.checkpoints``.
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
    "Screened",
    "Step",
    "Tracing",
]


# ── the trunk ────────────────────────────────────────────────────────────


@runtime_checkable
class LanguageModel[C: LayerCache](Protocol):
    """The minimum, and the only mandatory surface.

    Two members, and they are the only two that vary by architecture: which layer carries
    which kind of cache, and what the step computes. Anything else is either the
    ``LayerCache`` contract (batching, fixed shape, spans) or the core's own work
    (promotion, compilation, leasing, padding, buffer growth).
    """

    def make_cache(self) -> Sequence[C]:
        """Build one cache per layer.

        Returns
        -------
        Sequence[C]
            A fresh cache per layer, in the order the step walks them. That order is
            contract: restoring a prefix, promoting to fixed shape and assembling a
            ragged batch all address layers by index.

            A `Sequence` and not a `list` so that a family may answer with the element
            type it actually builds; whoever needs to replace a layer in place — a
            promotion does — copies it into one of its own first.
        """
        ...

    def __call__(self, ids: mx.array, cache: Sequence[C]) -> mx.array:
        """Run the trunk over ``ids``, appending to ``cache``.

        Serves prefill, decode and a batch alike, over growing caches, fixed-shape caches
        and ragged adapters: all three answer the same read contract (``core.attend``).

        Parameters
        ----------
        ids : mx.array
            Token ids, ``[rows, tokens]``.
        cache : Sequence[C]
            One entry per layer, in ``make_cache`` order.

        Returns
        -------
        mx.array
            Logits, ``[rows, tokens, vocab]``.

        Notes
        -----
        Must be traceable by ``mx.compile``, which runs the function's Python once and its
        graph on every call after. Two consequences bind the implementation:

        - positions are read off the graph, never host-side. ``int(offset.item())`` is
          frozen at the first token; ``LayerCache.span`` is the static column count.
        - the result may not depend on how many times the function has been called.
        """
        ...


@runtime_checkable
class Draftable[C: LayerCache](LanguageModel[C], Protocol):
    """Buys being the target of speculative decode. The trunk's only optional surface.

    It extends ``LanguageModel`` rather than sitting beside it because there is nothing
    else it could be attached to: every member here is about lending a drafter what the
    trunk holds, and a thing with no ``make_cache`` and no forward holds nothing. A family
    declares this one *instead of* ``LanguageModel``, not as well.

    ``raw_embed`` and ``raw_logits`` are the two raw ends of the vocabulary. A drafter
    that speaks in hidden rows owns neither an embedding table nor a head, and borrows
    the target's. **Raw** is the point: a facade's ``embed`` usually applies an RMS and
    its ``head`` a multiplier and a softcap, and the drafter was trained against the bare
    tensors — under greedy both missing transforms are monotone, so substituting the cooked
    pair changes no token a test would catch and changes what the checkpoint is. Where the
    two coincide a family answers with the same tensor twice; where they do not, the
    difference has a name.

    ``block_outputs`` is here rather than in a protocol of its own because every target
    that lends its vocabulary also taps (muse_glimmer, nemotron_h, qwen3_5), and a proposer
    that taps nothing — a whole small model, ``speculative.Autoregressive`` — borrows
    neither end and never reaches this protocol at all.

    ``checkpoints`` is the whole of what a rewind asks a family for. The rest of undoing a
    rejected round belongs to the cache: ``LayerCache.checkpoints`` makes the room and
    ``LayerCache.rewind`` picks it back up.
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
            tensors where the reader wants five: returning all of them costs about 1.4 GB
            per prefill block against 136 MB for the ones asked for.

        Returns
        -------
        tuple[mx.array, mx.array]
            Logits ``[rows, tokens, vocab]``, and the tapped block outputs concatenated
            on the last axis as ``[rows, tokens, len(at) * hidden]``.
        """
        ...

    def checkpoints(self, cache: Sequence[C], rows: int) -> bool:
        """Whether a ``rows``-row forward over ``cache`` records what a rejected round is
        undone from.

        A round writes ``width + 1`` rows before it knows how many survive. Dropping the
        rest is free for a layer that keeps keys and impossible for one that keeps recurrent
        state, so the only way back to row *k* is to have kept row *k*.
        ``LayerCache.checkpoints`` makes room for them; whether *this* trunk's mixers write
        into that room is a property of the kernels they resolved, which no cache can see.

        Parameters
        ----------
        cache : Sequence[C]
            The layers the round will run over, in ``make_cache`` order.
        rows : int
            Rows that round will feed, ``width + 1``.

        Returns
        -------
        bool
            ``True`` asserts that a rewind by ``LayerCache.rewind`` lands on the state a
            forward over the kept rows alone would have produced. ``False`` — the default —
            sends the round down restore-and-replay, correct everywhere and one forward more
            expensive when the draft was wrong.
        """
        del cache, rows
        return False


@runtime_checkable
class Tracing[C: LayerCache](LanguageModel[C], Protocol):
    """A trunk whose ordinary forward survives being traced over promoted caches.

    Declaring it asserts two things about ``__call__``, neither of which the engine can
    check:

    - every position it reads comes off the graph. ``FixedKVCache.position`` is a tensor
      and ``LayerCache.offset`` is the host's stale view of it.
    - the columns a promoted cache has not written are cut before they are attended.
      ``update_and_fetch`` there returns the whole capacity; ``LayerCache.readable`` is the
      cut, and ``core.attend`` applies it for any layer that reads through it.

    What it buys is the one-row lease: the trunk is handed the promoted caches directly and
    its fused single-row kernels stay resolved. The alternative — a ragged adapter over one
    row — masks correctly and turns them off, measured at 0.872x on Laguna-XS.

    Optional. Without it a trunk still gets a compiled decode under the batched path, where
    the adapter does the masking; what it does not get is the lease.
    """

    def before_trace(self, cache: Sequence[C]) -> Sequence[object]:
        """Settle what the trace cannot, and name the containers it must be given.

        Parameters
        ----------
        cache : Sequence[C]
            The promoted slots this graph will read, in ``make_cache`` order. The shapes
            are final here, which is what lets a kernel be chosen against them.

        Returns
        -------
        Sequence[object]
            Live containers the compile takes as ``inputs`` alongside the caches — a
            resident table the step reads and something outside the graph rewrites.
            Weights do not belong here: they bake into the trace as constants, and a
            captured container whose arrays a strategy also holds by reference trips the
            uncaptured-input check. Empty for a trunk that only had to resolve.
        """
        ...

    @property
    def trace_epoch(self) -> tuple[int, ...]:
        """A token that changes when ``before_trace`` would now settle differently.

        Polled once per step; a change rebuilds the graph in the same room. The default is
        constant, which is right for a trunk whose resolution depends only on shapes — they
        cannot change without a rebuild anyway. A trunk whose kernels are re-resolved by
        something *outside* the decode (a per-block eager step in between) answers with
        what would tell it apart.
        """
        return ()


@runtime_checkable
class Screened[C: LayerCache](LanguageModel[C], Protocol):
    """A trunk that can answer the greedy id without projecting a logits row.

    An output head is often the largest single read of a decode step — laguna's is a bf16
    ``[100352, 2048]``, 411 MB against a checkpoint whose blocks are 4-bit — and a greedy
    step reads exactly one number off it. ``core.kernels.lm_head`` screens instead: a coarse
    int5 pass, a certificate that rules out most of the vocabulary, and the full bf16 rows
    only for what survives. Measured at 1.108x the plain projection on Laguna-XS.

    The engine takes this path only when nothing else will read the row — no sampler but
    greedy, no penalty, no grammar — because outside its argmax the row is not logits.
    """

    def screened(self, ids: mx.array, cache: Sequence[C]) -> mx.array:
        """The same shape and the same argmax as ``__call__``, and not the same numbers.

        Parameters
        ----------
        ids : mx.array
            ``[rows, 1]``. A decode step and only a decode step: the screen is a
            single-token head, so there is no prefill form of this and no block form.

        Returns
        -------
        mx.array
            ``[rows, 1, vocab]``. Its argmax along the last axis is the stock head's; every
            other slot is an approximation of a logit that was never computed. A softmax, a
            temperature, a top-k or a logprob over it returns invented numbers, so the
            caller may read nothing but the argmax. A trunk whose screen serves one row at
            a time answers a batch row by row.
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

    def make_cache(self) -> Sequence[S]:
        """Build this tree's own caches, one per layer it walks.

        A `Sequence` and not a `list`, for the reason `LanguageModel.make_cache` gives: a
        tree answers with the element type it builds, and whoever promotes one into a fixed
        shape copies it into a list of its own first.
        """
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

"""The contract every layer cache answers, and the adapter for one that holds nothing."""

from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.attend import AttentionMask
from mlx_omnia.engine.core.cache.layout import Layout


class LayerCache:
    """Per-layer cache contract: `offset` counts rows written; `trim` rewinds only
    when the layer keeps enough history — check `is_trimmable` first.

    Three things past holding rows, and every feature the engine has past a plain eager
    forward rests on one of them. `fixed` is what a compiled decode needs, `batched` is
    what continuous batching needs, `layout`/`stored`/`restore` is what a prefix cache
    needs. They live here and not on the model because none of them is a question about a
    model's arithmetic: they are questions about state, and the state is what knows.
    """

    offset: int | mx.array
    """Rows written. An `int` for a cache of one sequence; the per-row positions for a
    ragged adapter over several, and a graph tensor for a layer whose position moved inside
    a trace. A reader that needs the number narrows; one that only passes it on does not."""

    def __init__(self) -> None:
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array, /) -> tuple[mx.array, mx.array]:
        """This step's rows in, the dense history out — the read `core.attend` takes for a
        layer that is not `Attending`.

        Refused here rather than absent, so that every cache satisfies one read contract
        and an attention layer can be handed any of them. A layer that keeps no keys and
        values is a real thing (a recurrent mixer's, a stateless block's); handing one to
        an attention is not."""
        raise NotImplementedError(
            f"{type(self).__name__} has no `update_and_fetch`: it keeps no dense keys and "
            "values to hand back"
        )

    def readable(self, mask: AttentionMask, queries: int, /) -> AttentionMask:
        """`mask`, cut to the columns this layer actually wrote.

        Called after `update_and_fetch`, over the `queries` rows it just took. The default
        returns `mask` untouched, which is correct for any cache whose fetch is exactly what
        it wrote. A fixed-shape buffer's is not — it returns its whole capacity, and the
        columns past its position hold whatever was there before — so it answers with the
        intersection.

        Here rather than at the reader so that one forward serves both. A reader that
        skipped the cut attends the buffer's stale rows: wrong logits, no error.
        """
        return mask

    @property
    def is_fixable(self) -> bool:
        """Whether `fixed` can answer. A layer holding no tensors already has a fixed
        shape; one holding tensors has to say how they become a graph's own."""
        return not self.tensors

    def fixed(self, capacity: int) -> "LayerCache":
        """This layer in the shape a traced graph can hold, sized for `capacity` rows.

        Idempotent: a layer already fixed at this capacity returns itself, and one sitting
        below it regrows. What comes back replaces this layer in the model's cache list,
        which is why it is a new object and not a mutation — the growing form is what a
        prefill wants and the fixed form is what a decode wants, and a conversation walks
        between the two.
        """
        if not self.is_fixable:
            raise NotImplementedError(
                f"{type(self).__name__} cannot present itself in a fixed shape"
            )
        return self

    def batched(self, rows: Sequence["LayerCache"]) -> "LayerCache":
        """These rows of this layer as one ragged layer, for a batch stepping together.

        Called on the first row with every row including itself, so a class answers for
        its own kind and refuses another's — nothing generic can know how a family's cache
        stacks, and a dispatch that learned them one import at a time would go stale on
        the next family.
        """
        if self.tensors:
            raise NotImplementedError(f"{type(self).__name__} has no ragged batch adapter")
        return BatchedLayerCache(uniform(rows, LayerCache))

    @property
    def span(self) -> int:
        """Columns a mask over this layer must cover.

        A host-side int even where the positions are graph tensors: buffer shapes do not
        change inside a step, so a family can build a band under `mx.compile` without
        evaluating anything. Before this a ragged mask read `int(mx.max(offset).item())`,
        which is the one thing a trace cannot do.
        """
        offset = self.offset
        if not isinstance(offset, int):
            raise NotImplementedError(
                f"{type(self).__name__} carries per-row positions and does not name one span"
            )
        return offset + 1

    @property
    def containers(self) -> list[mx.array] | None:
        """The list a compiled decode passes as its `inputs` and `outputs`, or `None` for a
        layer holding nothing the graph writes.

        A list and not the tensors: `mx.compile` swaps the arrays inside a captured
        container for tracers in place, so what the graph and the cache share has to be the
        container itself. A layer that rebinds an attribute instead is a layer whose write
        happened once, at trace time.
        """
        return None

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        """What this layer holds, for whoever budgets caches that outlive a request. Every
        subclass with a tensor in it answers for its own; a layer holding only its offset
        weighs nothing."""
        return 0

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        """What this layer holds, for whoever needs the cache evaluated without evaluating
        what the same forward also produced — `core.prefill` is the caller, and the thing it
        is avoiding is the head. Same contract as `nbytes`: every subclass with a tensor of
        its own answers for it, one that only borrows another layer's answers nothing, and a
        layer holding only its offset has nothing to evaluate."""
        return ()

    @property
    def is_replayable(self) -> bool:
        """Whether `checkpoint()` is a *true* restore point — whether running the layer again
        over the same rows lands on the state it would have had.

        The default is yes, because that is what `checkpoint()` below promises and what every
        layer honours whose writes only land past the offset or which reassigns its tensors
        rather than mutating them. A layer that rotates its buffer, or whose position lives
        somewhere the base does not capture, answers `False` — and it is a property and not a
        docstring because a caller that rewinds by replay has to be able to ask.

        Read over the whole list the way `is_trimmable` is: one layer that cannot restore
        makes the trunk's rewind a lie, and a lie here is a decode running off state that
        describes a sequence that never happened.
        """
        return True

    def checkpoint(self) -> Callable[[], None]:
        """A restore point at a call boundary, for a replay that runs the layer again over
        the same input. Unlike `trim`, this needs no kept history: the base restores the
        offset alone, exact for a cache whose writes only land past it (KVCache overwrites
        the stale rows on the next update); a subclass that reassigns tensors restores
        those references too."""
        offset = self.offset

        def restore() -> None:
            self.offset = offset

        return restore

    @property
    def rows(self) -> int:
        """How many rows this layer holds *now*.

        The same number as `offset` for every cache whose counter is a host-side int. For one
        whose position lives in the graph it is not: `offset` there is the value the decode
        was traced with and stays frozen, because a forward that read the live count would
        have to evaluate it and a compiled trace cannot. So the forward asks `offset` and
        whoever writes the cache out asks here, which costs one evaluation and is paid once
        per file rather than once per token.
        """
        offset = self.offset
        if not isinstance(offset, int):
            raise NotImplementedError(f"{type(self).__name__} holds no single row count")
        return offset

    @property
    def signature(self) -> dict[str, object]:
        """What the bytes of `stored()` mean beyond their shapes.

        Empty for a cache holding rows as the model produced them. A cache that encodes them,
        or one that rotates them, answers with what it did — and that answer goes into the
        key of the file, which is what keeps a cache written under one policy from being read
        back by a trunk built under another. Inside the layer and not in a caller's config
        because the layer is the only thing that knows: the trunk decides the policy, and the
        engine that files the bytes never sees it.
        """
        return {}

    @property
    def layout(self) -> Mapping[str, Layout]:
        """How each tensor of `stored()` composes along the spans of one conversation.

        Empty for a layer with no bytes of its own — a shared reader, a stateless block in a
        hybrid's pattern — which is not the same as a layer that cannot be stored: it still
        takes the offset back, and a trunk restored short of one layer's offset is a cache
        that decodes fluently off state that never existed.
        """
        return {}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """This layer's tensors, named, for the absolute token range `[start, stop)`.

        The range is in tokens and never in rows, because the two are the same number only
        for a plain KV buffer: a pooled cache writes one row per `ratio` tokens, a compressed
        one splits the range between a dense head and packed codes, and a ring holds the last
        window of it rotated. Each class does that arithmetic itself — nothing generic can,
        and a caller that guessed would be reading another layer's rows.

        A `Snapshot` tensor ignores the range: it is the whole value, and it is only valid
        while the layer stands on `stop`.

        Named rather than positional because the bytes outlive the process that wrote them,
        and a tuple's order is not a contract anybody can read back.
        """
        return {}

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """The inverse, into a layer a model has just made: the tensors of one conversation's
        spans, already composed, and the position they leave the layer at.

        The base takes the offset and nothing else, which is the whole of what a layer with no
        tensors has to come back to."""
        self.offset = offset

    def trim(self, length: int) -> None:
        raise NotImplementedError("this cache keeps no history to rewind to")

    def checkpoints(self, rows: int) -> bool:
        """Make room to record this layer's state after each of the next `rows` writes, and
        answer whether it can.

        A layer that keeps only keys needs no room — its rejected rows are stale and the
        next write covers them — and answers `True` having done nothing. A layer whose state
        the recurrence consumed can only go back to a row it kept, so it answers `False`
        unless it has somewhere to keep them.

        Read over the whole list, the way `is_trimmable` is: one layer that cannot land the
        round on a kept row makes the trunk's rewind a lie.
        """
        del rows
        return True

    def rewind(self, kept: int, position: int) -> None:
        """Undo all but the first `kept` rows of the round `checkpoints` made room for, and
        stand at absolute `position`.

        The two are different facts: `kept` indexes the round's own rows and picks which
        recorded state survives; `position` is where the layer ends up in the conversation.
        A layer holding nothing but a counter needs only the second.
        """
        del kept
        self.offset = position


class BatchedLayerCache(LayerCache):
    """N stateless layers as one: nothing to stack, only offsets to keep in step.

    A bare `LayerCache` is what a family hands the layers that keep no state — an MLP-only
    block in a hybrid pattern. The rows are ragged, so what a write propagates is the
    delta, the same rule as `BatchedConvCache.offset`.
    """

    def __init__(self, caches: Sequence[LayerCache]) -> None:
        self._caches = tuple(caches)

    @property
    def offset(self) -> int:
        return self._caches[0].offset

    @offset.setter
    def offset(self, value: int) -> None:
        advance = value - self._caches[0].offset
        for cache in self._caches:
            cache.offset += advance


def batch(caches: Sequence[Sequence[LayerCache]]) -> list[LayerCache]:
    """Transpose per-sequence caches into per-layer ragged batch adapters.

    Parameters
    ----------
    caches : Sequence[Sequence[LayerCache]]
        Cache layers grouped by sequence.

    Returns
    -------
    list[LayerCache]
        One adapter per model layer, each built by the layer's own first row.
    """
    return [layers[0].batched(layers) for layers in zip(*caches, strict=True)]


def row_mask(mask: AttentionMask, index: int, count: int, span: int) -> mx.array | None:
    """One row's slice of the batch mask, cut to the row's own history span. The string
    forms are batch-wide claims that do not survive ragged rows — the adapters rebuild
    what they meant per row, so here they narrow to `None`."""
    if mask is None or isinstance(mask, str):
        return None
    sliced = mask[index : index + 1] if mask.shape[0] == count else mask
    return sliced[..., :span]


def uniform[T](layers: Sequence[LayerCache], kind: type[T] | tuple[type[T], ...]) -> list[T]:
    """Every row of one batched layer, required to be the kind the first row is."""
    rows: list[T] = []
    for layer in layers:
        if not isinstance(layer, kind):
            raise mixed(layer)
        rows.append(layer)
    return rows


def mixed(layer: LayerCache) -> TypeError:
    return TypeError(f"a batched layer mixes {type(layer).__name__} with another cache kind")

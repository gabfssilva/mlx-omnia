"""A KV cache that stores its rows in one of the house's quantization formats.

The format is a parameter and not a decision this module makes: `Affine`, `MXFP` and `NVFP`
already exist with a quantizer and a dequantizer in ops, and a cache built around one codec
would have to be rewritten to accept the next. K and V take their own format — it is the
axis a fidelity sweep varies, and unifying them now and separating them later rewrites the
read.

**Quantized along `head_dim`, never along tokens.** Packing tokens into the same `uint32`
would make `trim` impossible anywhere but a multiple of the group, and `trim` is what the
prefix trie rewinds a stored cache with — so the reuse of a conversation would die with it.
Along `head_dim` every row is independent, `is_trimmable` stays `True`, and the trie goes on
working over a compressed cache.

A shape that does not close the format's groups is a **named error and never a rounding**:
`admits` is arithmetic, not availability. The concrete case is not hypothetical — the latent
cache of `bailing_hybrid` is 576 wide (`kv_lora_rank` 512 + `qk_rope_head_dim` 64), which
closes groups of 16, 32 and 64 and does not close 128.

The read never materializes the whole history. Attention runs as a blocked softmax: one
block of keys is dequantized, scored, folded into a running maximum and sum, and dropped.
The transient is one block rather than the context, which is the whole point — a cache that
handed back dense rows would spend exactly the bytes the compression saved. What this is
*not* is fast: it is ops, and one dispatch per block. The kernel is 57.4's, and whether the
feature survives the decode cost at all is 57.3's gate.
"""

from collections.abc import Callable, Mapping

import mlx.core as mx

from mlx_omnia.engine.core.attend import Attending, AttentionMask
from mlx_omnia.engine.core.cache import LayerCache, Layout, Rows, reserve
from mlx_omnia.engine.quant.quantization import MXFP, NVFP, Affine, Quantization, admits

_BLOCK = 256
"""Rows a buffer grows by, and rows a read scores at a time. The same figure `core.cache`
grows by: a block is the unit the padding is bounded at either way."""


class ShapeRefused(Exception):
    """The head width does not close the format's groups. Named because the alternative is a
    silent substitution — a format quietly widened to one the shape admits is a fidelity
    number measured against a policy nobody asked for."""


class FormatRefused(Exception):
    """The format cannot describe a cache at all, whatever the shape.

    One member of the ADT is in here and it is `NVFP`. Its scale has a second level over the
    whole tensor, so a row's codes depend on which other rows were quantized with it: 48 rows
    in one call and 48 calls of one row do not agree. A weight is quantized once and the level
    is free; a cache is written a step at a time, and there it means a prefill and a decode of
    the same sequence hold different bytes for the same token.
    """


class QuantizedKVCache(LayerCache, Attending):
    """`KVCache`, with the rows stored under `k_format` and `v_format`.

    `start_tokens` keeps the head of the context dense. Below it a conversation never pays
    the rounding at all, and above it the dense head is the region a sliding policy would
    keep anyway — which is why the read below combines two regions from the first day, with
    one of them allowed to be empty.
    """

    def __init__(
        self,
        k_format: Quantization,
        v_format: Quantization,
        *,
        start_tokens: int = 0,
    ) -> None:
        super().__init__()
        for side, format in (("k", k_format), ("v", v_format)):
            if isinstance(format, NVFP):
                raise FormatRefused(
                    f"nvfp4 cannot store a {side} cache: its scale has a second, whole-tensor "
                    "level, so the same row quantizes differently depending on how many rows "
                    "arrived with it. A prefill and a stepwise decode of one sequence would "
                    "then disagree permanently, which is the invariant this house does not "
                    "trade. Measured, not assumed — see test_quantized_kv_cache.py."
                )
        self.k_format = k_format
        self.v_format = v_format
        self.start_tokens = start_tokens
        self._dense_keys: mx.array | None = None
        self._dense_values: mx.array | None = None
        self._dense = 0
        """Rows held dense, which is `min(offset, start_tokens)`."""
        self._keys: _Packed | None = None
        self._values: _Packed | None = None

    @property
    def is_trimmable(self) -> bool:
        """True, and that is the point of packing along `head_dim`: every row stands on its
        own, so rewinding is moving the offset exactly as the dense cache does."""
        return True

    @property
    def is_fixable(self) -> bool:
        """Once it holds a row. The fixed form takes its shapes from what is already there —
        the code width of a format is not a thing to guess at, and a prefill that never ran
        has nothing worth compiling over."""
        return self.offset > 0

    def fixed(self, capacity: int) -> "LayerCache":
        """Copy a completed prefill into fixed buffers. No row is quantized again: the codes
        are already the state, and re-rounding them would be a second loss the format never
        asked for."""
        if capacity < self.offset:
            raise ValueError(f"capacity {capacity} is below cache length {self.offset}")
        return _fitted(
            self.k_format,
            self.v_format,
            start_tokens=self.start_tokens,
            capacity=capacity,
            rows=self.offset,
            dense_keys=self._dense_keys,
            dense_values=self._dense_values,
            k_parts=None if self._keys is None else self._keys.tensors,
            v_parts=None if self._values is None else self._values.tensors,
        )

    @property
    def nbytes(self) -> int:
        """Where the saving shows. Counting the dense buffer here would be a cache that
        compresses and reports the figure the trie budgets against unchanged — which moves no
        ceiling at all."""
        dense = sum(
            buffer.nbytes for buffer in (self._dense_keys, self._dense_values) if buffer is not None
        )
        packed = sum(held.nbytes for held in (self._keys, self._values) if held is not None)
        return dense + packed

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        held: list[mx.array] = [
            buffer for buffer in (self._dense_keys, self._dense_values) if buffer is not None
        ]
        for packed in (self._keys, self._values):
            if packed is not None:
                held.extend(packed.tensors)
        return tuple(held)

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)
        self._dense = min(self._dense, self.offset)

    @property
    def layout(self) -> Mapping[str, Layout]:
        """Rows, every one of them, and the two regions declared apart.

        The codes are the state: a span holds the compression itself, which is the whole
        reason it is worth storing — the bytes are already the small ones, and reading them
        back needs no quantizer to run again. The dense head and the packed tail are separate
        tensors because they are separate row spaces, and a span contributes to whichever of
        the two its own tokens fall in — the first span of a conversation to both, every span
        after `start_tokens` to only one, and the concatenation does not care.
        """
        held: dict[str, Layout] = {"dense_keys": Rows(), "dense_values": Rows()}
        for side, format in (("keys", self.k_format), ("values", self.v_format)):
            held[f"{side}.codes"] = Rows()
            held[f"{side}.scales"] = Rows()
            if isinstance(format, Affine):
                held[f"{side}.biases"] = Rows()
        return held

    @property
    def signature(self) -> dict[str, object]:
        """The two formats and where the dense head ends — everything a reader would have to
        assume otherwise. It goes in the key rather than in the file because the failure it
        prevents happens *before* the file is opened: the trunk builds its cache from its own
        policy, and a candidate written under another one must not be a candidate at all. A
        descriptor inside the file would answer the same question one mmap too late."""
        return {
            "k": _spelled(self.k_format),
            "v": _spelled(self.v_format),
            "start_tokens": self.start_tokens,
        }

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """The two regions' share of `[start, stop)`.

        A row's region is decided by its absolute position and never by anything else — the
        same rule `_write` follows — so the split of a range is arithmetic over
        `start_tokens` and needs nothing read back."""
        held: dict[str, mx.array] = {}
        head, tail = min(start, self._dense), min(stop, self._dense)
        if tail > head and self._dense_keys is not None and self._dense_values is not None:
            held["dense_keys"] = self._dense_keys[..., head:tail, :]
            held["dense_values"] = self._dense_values[..., head:tail, :]
        first = max(0, start - self.start_tokens)
        last = max(0, stop - self.start_tokens)
        if last > first:
            for side, packed in (("keys", self._keys), ("values", self._values)):
                if packed is None:
                    continue
                held |= {
                    f"{side}.{name}": tensor
                    for name, tensor in packed.stored(first, last).items()
                }
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """The split is recomputed and never read from the file: `_write` puts a row in the
        dense head exactly when its absolute position is below `start_tokens`, so the two
        counts are a function of the offset and the policy this cache was built with. Reading
        them back instead would be a second copy of the same rule, free to disagree."""
        self.offset = offset
        self._dense = min(offset, self.start_tokens)
        self._dense_keys = tensors.get("dense_keys")
        self._dense_values = tensors.get("dense_values")
        compressed = offset - self._dense
        self._keys = _restored(self.k_format, "keys", tensors, compressed)
        self._values = _restored(self.v_format, "values", tensors, compressed)

    def attend(
        self,
        queries: mx.array,
        *,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: AttentionMask,
        sinks: mx.array | None = None,
        softcap: float | None = None,
    ) -> mx.array:
        if sinks is not None:
            # The blocked read has no sink column, and attending without one is a
            # different model — the compression probe turns this into a policy refusal.
            raise TypeError("a compressed cache cannot attend with sinks")
        if softcap is not None:
            # Same refusal: the blocked read runs the fused kernel, which has no cap.
            raise TypeError("a compressed cache cannot attend with a softcap")
        first = self.offset
        self._write(keys, values)
        rows, length = queries.shape[2], self.offset

        def additive(start: int, size: int) -> mx.array:
            return _mask(mask, start, size, first, rows, length)

        return _blocked(queries, self._regions(), scale=scale, mask_at=additive)

    def _write(self, keys: mx.array, values: mx.array) -> None:
        """The rows this step produced, into whichever region they belong to. The split is by
        absolute position and never by what is convenient: a row that started dense stays
        dense, or a `trim` back into the head would find rounded rows where it left exact
        ones."""
        rows = keys.shape[2]
        head = max(0, min(self.start_tokens - self.offset, rows))
        if head:
            dense_keys = reserve(self._dense_keys, self._dense + head, keys)
            dense_values = reserve(self._dense_values, self._dense + head, values)
            dense_keys[..., self._dense : self._dense + head, :] = keys[..., :head, :]
            dense_values[..., self._dense : self._dense + head, :] = values[..., :head, :]
            self._dense_keys, self._dense_values = dense_keys, dense_values
            self._dense += head
        if head < rows:
            self._keys = _append(self._keys, keys[..., head:, :], self.k_format)
            self._values = _append(self._values, values[..., head:, :], self.v_format)
        self.offset += rows

    def _regions(self) -> list[tuple[int, Callable[[int, int], tuple[mx.array, mx.array]]]]:
        """The cache as a list of `(rows, read)`, oldest first. The dense head is a region
        like any other and is absent when `start_tokens` is 0, which is what keeps the
        combining path exercised instead of dead."""
        regions: list[tuple[int, Callable[[int, int], tuple[mx.array, mx.array]]]] = []
        if self._dense:
            dense_keys, dense_values = self._dense_keys, self._dense_values
            assert dense_keys is not None and dense_values is not None

            def dense(start: int, stop: int) -> tuple[mx.array, mx.array]:
                return dense_keys[..., start:stop, :], dense_values[..., start:stop, :]

            regions.append((self._dense, dense))
        compressed = self.offset - self._dense
        if compressed:
            packed_keys, packed_values = self._keys, self._values
            assert packed_keys is not None and packed_values is not None

            def unpacked(start: int, stop: int) -> tuple[mx.array, mx.array]:
                return packed_keys.rows(start, stop), packed_values.rows(start, stop)

            regions.append((compressed, unpacked))
        return regions


class FixedQuantizedKVCache(LayerCache, Attending):
    """`QuantizedKVCache` in the shape a traced decode can hold.

    Everything that moved per step in the growing form is a tensor here: the position lives
    in `graph` and the step's row is packed by `mx.quantize` and written with
    `mx.slice_update` at that tensor index. The packing is what made this possible at all —
    the formats quantize along `head_dim`, never along tokens, so one row packed inside the
    graph is byte-for-byte the row the growing writer would have appended.

    The two regions are split **statically**, at promotion: a dense pair of
    `min(start_tokens, capacity)` rows and packed triples for the rest, one of which is
    empty whenever the policy says so. Both writes then run every step, unconditionally,
    and the one that does not belong to this row lands where no read covers it — the dense
    buffer's one extra slot, or packed slot 0 while the head is still filling, which the
    step that reaches it overwrites. A branch instead would be a branch on a value the
    graph owns.

    The read is the same blocked softmax over a capacity that is now static, so its loop
    unrolls at trace. Blocks past the position score `-inf` and contribute exactly zero
    through the running maximum; the first block always holds column 0, which is written
    before any read, so the running maximum is never `-inf` against `-inf`.
    """

    offset: int

    def __init__(
        self,
        k_format: Quantization,
        v_format: Quantization,
        *,
        start_tokens: int,
        capacity: int,
        dense: tuple[mx.array, mx.array] | None,
        keys: tuple[mx.array, ...],
        values: tuple[mx.array, ...],
        position: int,
    ) -> None:
        super().__init__()
        self.k_format = k_format
        self.v_format = v_format
        self.start_tokens = start_tokens
        self.capacity = capacity
        self.dense_capacity = min(start_tokens, capacity)
        self.packed_capacity = capacity - self.dense_capacity
        self.offset = position
        head: list[mx.array] = [] if dense is None else [dense[0], dense[1]]
        self._keys_at = len(head)
        self._values_at = self._keys_at + len(keys)
        self.graph = [*head, *keys, *values, mx.array([position], dtype=mx.int32)]
        """What the compiled decode passes as its inputs and outputs: the buffers, and the
        position last. A list, updated functionally — `mx.compile` swaps the arrays inside
        a captured container for tracers in place, so the container is what the graph and
        the cache share."""

    @property
    def position(self) -> mx.array:
        return self.graph[-1]

    @property
    def is_fixable(self) -> bool:
        return True

    def fixed(self, capacity: int) -> "LayerCache":
        """Idempotent at the capacity it stands on, a copy into larger buffers below it —
        what a generation that outgrows its room pays once per doubling."""
        if capacity == self.capacity:
            return self
        dense, packed = self._dense(), self._packed()
        return _fitted(
            self.k_format,
            self.v_format,
            start_tokens=self.start_tokens,
            capacity=capacity,
            rows=self.rows,
            dense_keys=None if dense is None else dense[0],
            dense_values=None if dense is None else dense[1],
            k_parts=None if packed is None else packed[0],
            v_parts=None if packed is None else packed[1],
        )

    @property
    def containers(self) -> list[mx.array] | None:
        return self.graph

    @property
    def span(self) -> int:
        return self.capacity

    @property
    def rows(self) -> int:
        return int(self.position.item())

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def is_replayable(self) -> bool:
        """No, for `FixedKVCache`'s reason: the position that moves is inside the graph, and
        the base's `checkpoint()` captures the host-side `offset` the compiled decode never
        touches."""
        return False

    @property
    def nbytes(self) -> int:
        return sum(held.nbytes for held in self.graph)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(self.graph)

    def readable(self, mask: AttentionMask, queries: int, /) -> AttentionMask:
        """`LayerCache.readable`: the band `column <= the position this row landed on`, over
        the whole capacity. Unused by `attend` below, which masks its own blocks off the
        same expression — it is here because the contract is what a reader asks, and a
        reader that skipped it would attend the columns nothing has written yet."""
        columns = mx.arange(self.capacity)
        band = (columns[None, :] <= self._at(queries)).reshape(1, 1, queries, self.capacity)
        return band if not isinstance(mask, mx.array) else band & mask

    def attend(
        self,
        queries: mx.array,
        *,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: AttentionMask,
        sinks: mx.array | None = None,
        softcap: float | None = None,
    ) -> mx.array:
        if sinks is not None:
            raise TypeError("a compressed cache cannot attend with sinks")
        if softcap is not None:
            raise TypeError("a compressed cache cannot attend with a softcap")
        self._write(keys, values)
        rows = queries.shape[2]

        def additive(start: int, size: int) -> mx.array:
            return self._band(mask, start, size, rows)

        return _blocked(queries, self._regions(), scale=scale, mask_at=additive)

    def _at(self, queries: int) -> mx.array:
        """Where the `queries` rows just written sit, as a column vector: they end at the
        position, so row *j* is at `position - queries + j`."""
        return self.position - queries + mx.arange(queries)[:, None]

    def _band(self, mask: AttentionMask, start: int, size: int, queries: int) -> mx.array:
        """The block's additive term, off the position tensor.

        Fill and causality in one expression, which is why a string mask is replaced rather
        than intersected: `"causal"` aligns the queries to the end of the keys, and the end
        of a fixed buffer is capacity and not position. An array mask is the boolean band a
        fixed-shape decode builds, and it intersects.
        """
        columns = mx.arange(start, start + size)
        visible = columns[None, :] <= self._at(queries)
        term = visible.reshape(1, 1, queries, size)
        if mask is not None and not isinstance(mask, str):
            term = term & mask[..., start : start + size]
        return mx.where(term, 0.0, -mx.inf).astype(mx.float32)

    def _write(self, keys: mx.array, values: mx.array) -> None:
        """This step's row into both regions, at indices clamped so that the one it does not
        belong to writes where no read covers.

        The dense pair carries one slot past its head for exactly that, and packed slot 0
        takes the spurious rows of a still-filling head — every one of them absolute
        position `dense_capacity`, which the mask hides until the step that writes it for
        real arrives.
        """
        assert keys.shape[2] == 1, "the fixed form is the one-row decode step"
        position = self.position
        if self._keys_at:
            at = mx.minimum(position, self.dense_capacity) if self.packed_capacity else position
            self.graph[0] = mx.slice_update(self.graph[0], keys, at, axes=(2,))
            self.graph[1] = mx.slice_update(self.graph[1], values, at, axes=(2,))
        if self.packed_capacity:
            at = mx.maximum(position - self.dense_capacity, 0) if self.dense_capacity else position
            for first, rows, format in (
                (self._keys_at, keys, self.k_format),
                (self._values_at, values, self.v_format),
            ):
                for index, part in enumerate(_quantize(rows, format)):
                    self.graph[first + index] = mx.slice_update(
                        self.graph[first + index], part, at, axes=(2,)
                    )
        self.graph[-1] = position + 1

    def _dense(self) -> tuple[mx.array, mx.array] | None:
        return None if self._keys_at == 0 else (self.graph[0], self.graph[1])

    def _packed(self) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...]] | None:
        if self._values_at == len(self.graph) - 1:
            return None
        return (
            tuple(self.graph[self._keys_at : self._values_at]),
            tuple(self.graph[self._values_at : -1]),
        )

    def _regions(self) -> list[tuple[int, Callable[[int, int], tuple[mx.array, mx.array]]]]:
        """The capacity as `(rows, read)`, oldest first — the same shape the growing form
        hands `_blocked`, with both counts static."""
        regions: list[tuple[int, Callable[[int, int], tuple[mx.array, mx.array]]]] = []
        dense = self._dense()
        if dense is not None and self.dense_capacity:
            dense_keys, dense_values = dense

            def head(start: int, stop: int) -> tuple[mx.array, mx.array]:
                return dense_keys[..., start:stop, :], dense_values[..., start:stop, :]

            regions.append((self.dense_capacity, head))
        packed = self._packed()
        if packed is not None:
            k_parts, v_parts = packed

            def tail(start: int, stop: int) -> tuple[mx.array, mx.array]:
                return (
                    _dequantized(self.k_format, k_parts, start, stop),
                    _dequantized(self.v_format, v_parts, start, stop),
                )

            regions.append((self.packed_capacity, tail))
        return regions

    @property
    def signature(self) -> dict[str, object]:
        return {
            "k": _spelled(self.k_format),
            "v": _spelled(self.v_format),
            "start_tokens": self.start_tokens,
        }

    @property
    def layout(self) -> Mapping[str, Layout]:
        """The growing form's, unchanged: a decode that was compiled is the one conversation
        that would otherwise keep nothing, and what it holds is the same two row spaces."""
        held: dict[str, Layout] = {"dense_keys": Rows(), "dense_values": Rows()}
        for side, format in (("keys", self.k_format), ("values", self.v_format)):
            held[f"{side}.codes"] = Rows()
            held[f"{side}.scales"] = Rows()
            if isinstance(format, Affine):
                held[f"{side}.biases"] = Rows()
        return held

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """The two regions' share of `[start, stop)`. The caller cuts against `rows` and not
        `offset`: the position moved inside the compiled decode and only the graph followed
        it."""
        held: dict[str, mx.array] = {}
        dense = self._dense()
        head, tail = min(start, self.dense_capacity), min(stop, self.dense_capacity)
        if tail > head and dense is not None:
            held["dense_keys"] = dense[0][..., head:tail, :]
            held["dense_values"] = dense[1][..., head:tail, :]
        packed = self._packed()
        first = max(0, start - self.dense_capacity)
        last = max(0, stop - self.dense_capacity)
        if last > first and packed is not None:
            for side, parts in (("keys", packed[0]), ("values", packed[1])):
                for name, part in zip(("codes", "scales", "biases"), parts, strict=False):
                    held[f"{side}.{name}"] = part[..., first:last, :]
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """Into the buffers this cache was built with, which is where it differs from the
        growing one: the capacity is fixed at promotion and the rows land inside it."""
        dense = self._dense()
        dense_keys, dense_values = tensors.get("dense_keys"), tensors.get("dense_values")
        if dense is not None and dense_keys is not None and dense_values is not None:
            dense[0][..., : dense_keys.shape[2], :] = dense_keys
            dense[1][..., : dense_values.shape[2], :] = dense_values
        packed = self._packed()
        if packed is not None:
            for side, parts in (("keys", packed[0]), ("values", packed[1])):
                for name, part in zip(("codes", "scales", "biases"), parts, strict=False):
                    held = tensors.get(f"{side}.{name}")
                    if held is not None:
                        part[..., : held.shape[2], :] = held
        self.graph[-1] = mx.array([offset], dtype=mx.int32)
        self.offset = offset

    def trim(self, length: int) -> None:
        raise NotImplementedError("a compiled cache cannot be rewound")

    def rewind(self, kept: int, position: int) -> None:
        """`LayerCache.rewind`. Both counters move — the graph's and the host's — and the
        rows past the position are stale, which the next write covers."""
        del kept
        self.graph[-1] = mx.array([position], dtype=mx.int32)
        self.offset = position


def _fitted(
    k_format: Quantization,
    v_format: Quantization,
    *,
    start_tokens: int,
    capacity: int,
    rows: int,
    dense_keys: mx.array | None,
    dense_values: mx.array | None,
    k_parts: tuple[mx.array, ...] | None,
    v_parts: tuple[mx.array, ...] | None,
) -> FixedQuantizedKVCache:
    """Buffers of `capacity`, carrying `rows` over — the one builder both a promotion and a
    regrow go through, because the split they compute is the same arithmetic over
    `start_tokens` and neither is free to answer it differently."""
    dense_capacity = min(start_tokens, capacity)
    packed_capacity = capacity - dense_capacity
    held = min(rows, dense_capacity)
    tail = max(0, rows - dense_capacity)
    dense: tuple[mx.array, mx.array] | None = None
    if dense_capacity:
        assert dense_keys is not None and dense_values is not None
        # One slot past the head, and never read: it is where a step past the dense region
        # puts the row it also writes packed.
        slots = dense_capacity + (1 if packed_capacity else 0)
        dense = (_carried(dense_keys, slots, held), _carried(dense_values, slots, held))
    keys: tuple[mx.array, ...] = ()
    values: tuple[mx.array, ...] = ()
    if packed_capacity:
        keys = _slots(k_format, k_parts, dense_keys, packed_capacity, tail)
        values = _slots(v_format, v_parts, dense_values, packed_capacity, tail)
    fixed = FixedQuantizedKVCache(
        k_format,
        v_format,
        start_tokens=start_tokens,
        capacity=capacity,
        dense=dense,
        keys=keys,
        values=values,
        position=rows,
    )
    mx.eval(*fixed.graph)
    return fixed


def _carried(source: mx.array, slots: int, rows: int) -> mx.array:
    """`source`'s first `rows` rows in a buffer of `slots`, shaped and typed by the source —
    each side from its own tensor, because keys and values are not always the same width."""
    shape = list(source.shape)
    shape[2] = slots
    buffer = mx.zeros(shape, dtype=source.dtype)
    if rows:
        buffer[..., :rows, :] = source[..., :rows, :]
    return buffer


def _slots(
    format: Quantization,
    parts: tuple[mx.array, ...] | None,
    like: mx.array | None,
    slots: int,
    rows: int,
) -> tuple[mx.array, ...]:
    """The format's parts as buffers of `slots`, carrying `rows` over.

    A cache promoted before it packed anything has no parts to take the code shapes from,
    so they come from quantizing one dense row — which is also where the shape refusal
    lands, at promotion rather than at the first traced step.
    """
    if parts is None:
        assert like is not None, "a cache with no rows at all has nothing to promote"
        if not admits(like.shape, format):
            raise _refuse(like.shape[-1], format)
        parts = _quantize(like[..., :1, :], format)
        rows = 0
    return tuple(_carried(part, slots, rows) for part in parts)


class _Packed:
    """One side of the cache, quantized along the last axis and grown in blocks.

    The three formats differ only in whether there are biases, so the wrapper carries the
    triple and the dequantizer is `mx.dequantize` under the format's own arguments. A codec
    of its own per format would be three copies of one slice.
    """

    def __init__(self, format: Quantization) -> None:
        self.format = format
        self.codes: mx.array | None = None
        self.scales: mx.array | None = None
        self.biases: mx.array | None = None
        self.rows_written = 0

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(held for held in (self.codes, self.scales, self.biases) if held is not None)

    @property
    def nbytes(self) -> int:
        return sum(held.nbytes for held in self.tensors)

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """The triple, cut to the rows of `[start, stop)`. Named here rather than assembled
        by the cache because which of the three exist is the format's business: `biases` is
        present for `Affine` and absent for the others, and that absence is the format, not
        a gap."""
        if self.codes is None or self.scales is None:
            return {}
        held = {
            "codes": self.codes[..., start:stop, :],
            "scales": self.scales[..., start:stop, :],
        }
        if self.biases is not None:
            held["biases"] = self.biases[..., start:stop, :]
        return held

    def rows(self, start: int, stop: int) -> mx.array:
        assert self.codes is not None and self.scales is not None
        return _dequantized(self.format, self.tensors, start, stop)


def _restored(
    format: Quantization, side: str, tensors: Mapping[str, mx.array], rows: int
) -> "_Packed | None":
    codes, scales = tensors.get(f"{side}.codes"), tensors.get(f"{side}.scales")
    if codes is None or scales is None:
        return None
    packed = _Packed(format)
    packed.codes = codes
    packed.scales = scales
    packed.biases = tensors.get(f"{side}.biases")
    packed.rows_written = rows
    return packed


def _spelled(format: Quantization) -> str:
    """The format as one token for the key. Three fields and not the dataclass, because what
    goes in the digest has to be the same string next release — a repr is not that."""
    return f"{_name(format)}/{format.bits}/{format.group_size}"


def _name(format: Quantization) -> str:
    """What to call the format in an error. `Affine` has no `mode` — it is the one the ADT
    spells by its own class rather than by a string."""
    return "affine" if isinstance(format, Affine) else format.mode


def _quantize(rows: mx.array, format: Quantization) -> tuple[mx.array, ...]:
    """The rows as the format's parts, codes first — two for the exponent formats and three
    for `Affine`, which is the only one with a bias.

    One quantizer for both forms of the cache, which is what makes the fixed form's claim
    checkable rather than argued: the growing writer and the traced step call the same
    call with the same arguments, so a row packed a step at a time is the bytes a prefill
    would have written for it.
    """
    arguments = {"group_size": format.group_size, "bits": format.bits}
    match format:
        case Affine():
            codes, scales, biases = mx.quantize(rows, **arguments)
            return codes, scales, biases
        case MXFP() | NVFP():
            codes, scales = mx.quantize(rows, mode=format.mode, **arguments)
            return codes, scales


def _dequantized(
    format: Quantization, parts: tuple[mx.array, ...], start: int, stop: int
) -> mx.array:
    """The inverse, over the rows of `[start, stop)`."""
    cut = [part[..., start:stop, :] for part in parts]
    arguments = {"group_size": format.group_size, "bits": format.bits}
    if isinstance(format, Affine):
        return mx.dequantize(cut[0], cut[1], cut[2], **arguments)
    return mx.dequantize(cut[0], cut[1], mode=format.mode, **arguments)


def _refuse(width: int, format: Quantization) -> ShapeRefused:
    return ShapeRefused(
        f"a head of {width} does not close {_name(format)} groups of "
        f"{format.group_size} at {format.bits} bits: a cache under this policy would be "
        "quantizing a shape the format cannot describe"
    )


def _append(held: _Packed | None, rows: mx.array, format: Quantization) -> _Packed:
    if not admits(rows.shape, format):
        raise _refuse(rows.shape[-1], format)
    packed = _Packed(format) if held is None else held
    parts = _quantize(rows, format)
    codes, scales = parts[0], parts[1]
    biases = parts[2] if len(parts) == 3 else None
    needed = packed.rows_written + rows.shape[2]
    held_codes = reserve(packed.codes, needed, codes)
    held_scales = reserve(packed.scales, needed, scales)
    held_codes[..., packed.rows_written : needed, :] = codes
    held_scales[..., packed.rows_written : needed, :] = scales
    packed.codes, packed.scales = held_codes, held_scales
    if biases is not None:
        held_biases = reserve(packed.biases, needed, biases)
        held_biases[..., packed.rows_written : needed, :] = biases
        packed.biases = held_biases
    packed.rows_written = needed
    return packed


def _blocked(
    queries: mx.array,
    regions: list[tuple[int, Callable[[int, int], tuple[mx.array, mx.array]]]],
    *,
    scale: float,
    mask_at: Callable[[int, int], mx.array],
) -> mx.array:
    """Attention as a running softmax over blocks of keys.

    `mask_at` is asked for the additive term of the block starting at absolute column
    `start` and `size` wide. A callback and not the mask itself because the two forms of
    this cache answer it differently: the growing one off a host-side query offset, the
    fixed one off the position tensor its graph owns. The loop is the same either way, and
    at a static capacity it unrolls at trace.

    The arithmetic is the standard online one: a running maximum, a running denominator and a
    running numerator, each block correcting what came before by `exp(m_old - m_new)`. What
    it buys here is that no block outlives its own iteration — the whole history is never
    dense at once, which is the only reason a compressed cache is worth having.

    In fp32 throughout. The comparison is not the point here, the accumulation is: a running
    sum over hundreds of blocks in bf16 loses the tail of the distribution to the dtype
    rather than to the quantizer, and the fidelity number would then be measuring the
    accumulator.
    """
    heads = queries.shape[1]
    q = queries.astype(mx.float32)
    running_max = mx.full((*q.shape[:3], 1), -mx.inf, dtype=mx.float32)
    denominator = mx.zeros((*q.shape[:3], 1), dtype=mx.float32)
    numerator = mx.zeros((*q.shape[:3], q.shape[3]), dtype=mx.float32)
    at = 0
    for size, read in regions:
        for start in range(0, size, _BLOCK):
            stop = min(start + _BLOCK, size)
            keys, values = read(start, stop)
            k = _expand(keys.astype(mx.float32), heads)
            v = _expand(values.astype(mx.float32), heads)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            scores = scores + mask_at(at + start, stop - start)
            block_max = mx.maximum(scores.max(axis=-1, keepdims=True), -mx.inf)
            new_max = mx.maximum(running_max, block_max)
            correction = mx.exp(running_max - new_max)
            weights = mx.exp(scores - new_max)
            denominator = denominator * correction + weights.sum(axis=-1, keepdims=True)
            numerator = numerator * correction + weights @ v
            running_max = new_max
        at += size
    return (numerator / denominator).astype(queries.dtype)


def _expand(rows: mx.array, heads: int) -> mx.array:
    """Grouped-query attention repeats each key-value head over the query heads that read it.
    `mx.fast.scaled_dot_product_attention` does this internally; here it is explicit because
    the matmul below is."""
    kv_heads = rows.shape[1]
    if kv_heads == heads:
        return rows
    repeats = heads // kv_heads
    return mx.repeat(rows, repeats, axis=1)


def _mask(
    mask: AttentionMask,
    start: int,
    size: int,
    query_offset: int,
    rows: int,
    length: int,
) -> mx.array:
    """The block's slice of the mask, as an additive term.

    `"causal"` is computed rather than sliced: the queries of this step sit at absolute
    positions `query_offset ..`, the keys of this block at `start ..`, and a key past its
    query is the only thing hidden. A decode step is one query at the end of the context, so
    nothing is — which is the branch `mx.fast.scaled_dot_product_attention` takes by passing
    no mask at all.
    """
    if mask is None:
        return mx.zeros((1, 1, 1, 1), dtype=mx.float32)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unknown mask {mask!r}")
        if rows == 1 and query_offset + 1 == length:
            return mx.zeros((1, 1, 1, 1), dtype=mx.float32)
        keys = mx.arange(start, start + size)
        queries = mx.arange(query_offset, query_offset + rows)[:, None]
        return mx.where(keys[None, :] > queries, -mx.inf, 0.0).astype(mx.float32)[None, None]
    return mask[..., start : start + size].astype(mx.float32)

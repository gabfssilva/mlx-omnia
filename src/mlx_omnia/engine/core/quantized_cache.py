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

from mlx_omnia.engine.core.attend import AttentionMask
from mlx_omnia.engine.core.cache import LayerCache
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


class QuantizedKVCache(LayerCache):
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
    def nbytes(self) -> int:
        """Where the saving shows. Counting the dense buffer here would be a cache that
        compresses and reports the figure the trie budgets against unchanged — which moves no
        ceiling at all."""
        dense = sum(
            buffer.nbytes
            for buffer in (self._dense_keys, self._dense_values)
            if buffer is not None
        )
        packed = sum(held.nbytes for held in (self._keys, self._values) if held is not None)
        return dense + packed

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        held: list[mx.array] = [
            buffer
            for buffer in (self._dense_keys, self._dense_values)
            if buffer is not None
        ]
        for packed in (self._keys, self._values):
            if packed is not None:
                held.extend(packed.tensors)
        return tuple(held)

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)
        self._dense = min(self._dense, self.offset)

    @property
    def is_storable(self) -> bool:
        """The codes are the state. Writing them out is writing the compression itself, which
        is the whole reason it is worth a file: the bytes on disk are already the small ones,
        and reading them back needs no quantizer to run again."""
        return True

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

    def stored(self) -> dict[str, mx.array]:
        """The rows in use of both regions, cut the way `KVCache.stored` cuts: the packed
        buffers grow by the same block and the padding is not state."""
        held: dict[str, mx.array] = {}
        if self._dense and self._dense_keys is not None and self._dense_values is not None:
            held["dense_keys"] = self._dense_keys[..., : self._dense, :]
            held["dense_values"] = self._dense_values[..., : self._dense, :]
        for side, packed in (("keys", self._keys), ("values", self._values)):
            if packed is None:
                continue
            held |= {f"{side}.{name}": tensor for name, tensor in packed.stored().items()}
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
        return _blocked(
            queries,
            self._regions(),
            scale=scale,
            mask=mask,
            query_offset=first,
            length=self.offset,
        )

    def _write(self, keys: mx.array, values: mx.array) -> None:
        """The rows this step produced, into whichever region they belong to. The split is by
        absolute position and never by what is convenient: a row that started dense stays
        dense, or a `trim` back into the head would find rounded rows where it left exact
        ones."""
        rows = keys.shape[2]
        head = max(0, min(self.start_tokens - self.offset, rows))
        if head:
            self._dense_keys = _reserve(self._dense_keys, self._dense + head, keys)
            self._dense_values = _reserve(self._dense_values, self._dense + head, values)
            self._dense_keys[..., self._dense : self._dense + head, :] = keys[..., :head, :]
            self._dense_values[..., self._dense : self._dense + head, :] = values[..., :head, :]
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
        return tuple(
            held for held in (self.codes, self.scales, self.biases) if held is not None
        )

    @property
    def nbytes(self) -> int:
        return sum(held.nbytes for held in self.tensors)

    def stored(self) -> dict[str, mx.array]:
        """The triple cut to the rows written. Named here rather than assembled by the cache
        because which of the three exist is the format's business: `biases` is present for
        `Affine` and absent for the others, and that absence is the format, not a gap."""
        if self.codes is None or self.scales is None:
            return {}
        held = {
            "codes": self.codes[..., : self.rows_written, :],
            "scales": self.scales[..., : self.rows_written, :],
        }
        if self.biases is not None:
            held["biases"] = self.biases[..., : self.rows_written, :]
        return held

    def rows(self, start: int, stop: int) -> mx.array:
        assert self.codes is not None and self.scales is not None
        arguments = {"group_size": self.format.group_size, "bits": self.format.bits}
        if self.biases is None:
            assert not isinstance(self.format, Affine)
            return mx.dequantize(
                self.codes[..., start:stop, :],
                self.scales[..., start:stop, :],
                mode=self.format.mode,
                **arguments,
            )
        return mx.dequantize(
            self.codes[..., start:stop, :],
            self.scales[..., start:stop, :],
            self.biases[..., start:stop, :],
            **arguments,
        )


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


def _append(held: _Packed | None, rows: mx.array, format: Quantization) -> _Packed:
    if not admits(rows.shape, format):
        raise ShapeRefused(
            f"a head of {rows.shape[-1]} does not close {_name(format)} groups of "
            f"{format.group_size} at {format.bits} bits: a cache under this policy would be "
            "quantizing a shape the format cannot describe"
        )
    packed = _Packed(format) if held is None else held
    arguments = {"group_size": format.group_size, "bits": format.bits}
    biases: mx.array | None
    match format:
        case Affine():
            # The only one with a bias: a scale and an offset per group, against one
            # exponent for the other two.
            codes, scales, biases = mx.quantize(rows, **arguments)
        case MXFP() | NVFP():
            codes, scales = mx.quantize(rows, mode=format.mode, **arguments)
            biases = None
    needed = packed.rows_written + rows.shape[2]
    packed.codes = _reserve(packed.codes, needed, codes)
    packed.scales = _reserve(packed.scales, needed, scales)
    packed.codes[..., packed.rows_written : needed, :] = codes
    packed.scales[..., packed.rows_written : needed, :] = scales
    if biases is not None:
        packed.biases = _reserve(packed.biases, needed, biases)
        packed.biases[..., packed.rows_written : needed, :] = biases
    packed.rows_written = needed
    return packed


def _reserve(buffer: mx.array | None, needed: int, like: mx.array) -> mx.array:
    """`core.cache._reserving` over the token axis of a `[batch, heads, tokens, width]`
    buffer. Written again rather than imported because what is grown here is three buffers of
    different widths under one row count, and the public entry takes one."""
    if buffer is not None and buffer.shape[2] >= needed:
        return buffer
    capacity = (needed + _BLOCK - 1) // _BLOCK * _BLOCK
    shape = list(like.shape)
    shape[2] = capacity
    grown = mx.zeros(shape, dtype=like.dtype)
    if buffer is not None:
        grown[..., : buffer.shape[2], :] = buffer
    return grown


def _blocked(
    queries: mx.array,
    regions: list[tuple[int, Callable[[int, int], tuple[mx.array, mx.array]]]],
    *,
    scale: float,
    mask: AttentionMask,
    query_offset: int,
    length: int,
) -> mx.array:
    """Attention as a running softmax over blocks of keys.

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
    rows = q.shape[2]
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
            scores = scores + _mask(mask, at + start, stop - start, query_offset, rows, length)
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

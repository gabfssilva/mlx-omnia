"""How a layer's tensors compose along the spans of one conversation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rows:
    """History: a span holds the rows its own tokens produced, and spans concatenate.

    `stride` is tokens per row — 1 for a KV buffer, the compression ratio for a pooled one —
    and it is what a span length has to be a multiple of, or a span would end in the middle
    of a row nobody can cut.

    `keep` is the suffix that suffices to resume: the sliding window of a layer whose mask
    proves the rows before it are never read again. Spans deeper than that are not fetched at
    all and come back as zeros of the right shape, which keeps every row at its absolute
    position — the only thing the mask and the rotary tables are written against.
    """

    axis: int = 2
    stride: int = 1
    keep: int | None = None


@dataclass(frozen=True)
class Snapshot:
    """State: the whole value as it stood on a span's boundary.

    It does not compose — resuming reads the last one and nothing before it — and it only
    exists while the layer is stopped on the boundary, because a recurrence in the middle of
    a forward has no state outside the kernel.
    """


type Layout = Rows | Snapshot

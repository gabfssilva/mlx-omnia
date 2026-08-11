"""Numbers and names as they are read, for the ones more than one screen reads.

Everything here was written twice or more, or was written in a screen another screen then
had to import. A formatter used by a single view stays in that view.
"""

from __future__ import annotations

import re

GIB = 1024**3

# A quantization suffix in a repository name is not part of the model's name.
PACKED = re.compile(r"-(?:\d+bit|mxfp4|nf4|bf16|fp16|fp32|int[48])$", re.IGNORECASE)


def gb(byte_count: float, digits: int = 1) -> str:
    return f"{byte_count / GIB:.{digits}f}"


def tokens(count: int) -> str:
    return f"{round(count / 1000)}k" if count >= 1000 else str(count)


def display_name(model: str) -> str:
    return PACKED.sub("", [part for part in model.split("/") if part][-1])


def owner(identifier: str) -> str:
    """The hub organisation, or the directory a quantized entry was written into."""
    parts = [p for p in identifier.split("/") if p]
    return parts[-2] if len(parts) > 1 else "local"


def size(amount: float) -> str:
    div, unit = _scale(amount)
    return f"{amount / div:.2f} {unit}"


def size_pair(done: float, total: float) -> str:
    """Both sides in the total's unit: "2.31 / 4.48 GB"."""
    div, unit = _scale(total)
    return f"{done / div:.2f} / {total / div:.2f} {unit}"


def rate(bytes_per_second: float) -> str:
    return f"{round(bytes_per_second / 1e6)} MB/s"


def left(seconds: float) -> str:
    """Rounded to the unit above the one it is about to leave: a countdown that reads
    "119s" is a number nobody subtracts from."""
    if seconds < 90:
        return f"{round(seconds)}s left"
    minutes = round(seconds / 60)
    return f"{minutes}m left" if minutes < 90 else f"{round(minutes / 60)}h left"


def _scale(amount: float) -> tuple[float, str]:
    return (1e9, "GB") if amount >= 1e9 else (1e6, "MB")

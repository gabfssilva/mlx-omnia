"""The prompt a battery runs, tiled to a token count.

Prompt length is an axis of its own — prefill is quadratic in it and the KV cache decode reads
grows with it — so a bench that wants 8192 ids has to get them without changing the
distribution of tokens underneath. Repeating the text until the slice holds keeps that
distribution; padding or truncating a different text does not.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

BENCH_PROMPT = Path(__file__).parent / "bench_prompt.txt"
"""Ships with the package, so a number taken from an installed copy is a number over the same
text as one taken from the repository."""


def tile(
    encode: Callable[[str], Iterable[int]],
    text: str,
    target: int,
    *,
    limit: int | None = None,
) -> list[int]:
    """`limit` is the model's context: a prompt that does not leave room to generate is not a
    shorter measurement, it is a failed one, so the target gives way to the context."""
    if limit is not None and target > limit:
        target = limit
    ids = [int(value) for value in encode(text)]
    if not ids:
        raise ValueError("the prompt text encoded to no ids")
    if len(ids) < target:
        ids = [int(value) for value in encode(text * (-(-target // len(ids)) + 1))]
    return ids[:target]

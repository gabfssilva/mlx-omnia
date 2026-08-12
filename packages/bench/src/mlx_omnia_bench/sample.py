"""What a timed round leaves behind: what was counted, not the rates derived from it.

A rate is a division, and the two axes do not divide the same way. Decode is over the steps
*after* the first, because the first one is inside the ttft. Prefill is the whole prompt over
that ttft, which therefore carries one decode step with it — the same term on both sides of a
ratio, so the comparison is honest while the absolute number is not prefill alone.

Storing `97.4 tok/s` throws away which of those it was and over how many tokens. The second
one is not bookkeeping: two arms that generated different counts are not two measurements of
the same work, and a report that cannot see the counts divides them anyway.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Sample:
    prompt_tokens: int
    ttft: float
    """Prompt in, first id out: the prefill plus the step that draws from it."""
    generated: int
    decode_s: float
    """The steps after the first, which is `generated - 1` of them."""

    @property
    def decode(self) -> float:
        return (self.generated - 1) / self.decode_s

    @property
    def prefill(self) -> float:
        return self.prompt_tokens / self.ttft

    def as_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "decode": self.decode, "prefill": self.prefill}

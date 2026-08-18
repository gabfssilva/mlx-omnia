"""The shapes the register answers with: one request, one model, the daemon, the frame."""

from dataclasses import dataclass
from typing import Literal

RequestState = Literal["queued", "running", "completed", "cancelled", "error"]
"""The engine's states; `engine.Job.state` is this and the record carries whichever one the
request ended in."""


@dataclass(frozen=True)
class Speculation:
    rounds: int
    proposed: int
    accepted: int


@dataclass(frozen=True)
class Sample:
    """One request's numbers, live or finished.

    `bytes_per_token` is absent when the model has no `nn.Module` under it — a test double
    holds no tensors — and `ceiling_fraction` with it: a percentage of a ceiling nobody
    computed is a number nobody can read. `tokens_per_second` is the meter's, decode only,
    and absent before the second token exists to measure a rate between.

    `load_seconds` is `Job.load_seconds`: the seconds this request spent putting its model in
    memory, absent when it found it resident. It is outside `ttft` — the meter starts at the
    prefill, with the weights already there.
    """

    model: str
    state: RequestState
    prompt_tokens: int
    reused_tokens: int
    """How much of `prompt_tokens` a prefix cache covered. Zero is a real answer — the trie
    was off, or the render did not reproduce what the last turn wrote — and telling the two
    apart is the whole reason it is published beside the prompt."""
    kept_prefix: bool
    """Whether this run left the next turn anything to start from. It separates the third
    zero from the other two: a trunk whose layers neither rewind nor serialize keeps nothing
    at all, and every turn on it is a cold prefill no setting will fix."""
    prefilled_tokens: int
    """Fresh prompt rows the trunk has taken so far. It is the whole of `prompt_tokens -
    reused_tokens` once the prefill is over, and before that it is where the prefill is —
    the only thing that moves during a 40k prompt, which is a minute in which the request is
    otherwise indistinguishable from a stalled one."""
    completion_tokens: int
    started_at: float
    load_seconds: float | None
    ttft: float | None
    tokens_per_second: float | None
    prefill_tokens_per_second: float | None
    """The prompt's own rate. Once there is a first token it is the fresh prompt over `ttft`;
    while the trunk is still reading it, the blocks fed so far over the clock they were fed
    in — the same ratio, measured at where it has got to."""
    bytes_per_token: int | None
    ceiling_fraction: float | None
    speculation: Speculation | None
    """What a drafter proposed and how much of it the target confirmed, `None` for a request
    that did not speculate — no drafter loaded, or one this request could not be verified
    against. The rate above never says which: a paired model under a sampler decodes exactly
    as an unpaired one."""


@dataclass(frozen=True)
class Aggregate:
    """Every request on one model since boot, ring or no ring.

    `tokens_per_second` is the ratio of the totals rather than the mean of the rates: a
    four-token request and a four-hundred-token one weigh what they generated. `ttft` is a
    mean, because a prefill is one measurement per request whatever its length.
    """

    model: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    ttft: float | None
    prefill_tokens_per_second: float | None
    """The fresh prompt rows this model read over the time it took to read them — the same
    ratio one request reports, summed. What a prefix cache handed over is out of the
    numerator for the reason `prefill_rate` gives."""
    tokens_per_second: float | None
    bytes_per_token: int | None
    ceiling_fraction: float | None


@dataclass(frozen=True)
class Totals:
    """Every request this daemon has served, whatever model answered it.

    `tokens_per_second` is the same ratio-of-totals an `Aggregate` reports, over every model:
    a per-request decode rate weighted by what each request generated, and not the daemon's
    throughput — two requests batched together each decode at their own rate while the wall
    clock runs once. `ceiling_fraction` is those rates against each model's own ceiling,
    weighted the same way, because bytes per token is a property of the checkpoint and there
    is no single denominator to divide the sum by.
    """

    requests: int
    running: int
    """In flight right now — queued requests are the engine's `state`, not the register's."""
    prompt_tokens: int
    completion_tokens: int
    ttft: float | None
    prefill_tokens_per_second: float | None
    tokens_per_second: float | None
    ceiling_fraction: float | None


@dataclass(frozen=True)
class Snapshot:
    """The one shape both windows answer with: the `GET` and every frame of the stream."""

    live: list[Sample]
    """Every request being served, newest first. A list because the engine batches: two
    conversations on one model, or two models answering at once, are two live measurements
    and a register that published one of them would be describing the other's screen."""
    requests: list[Sample]
    """Newest first."""
    models: list[Aggregate]
    totals: Totals

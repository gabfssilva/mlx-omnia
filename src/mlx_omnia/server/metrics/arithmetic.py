"""The counters the register keeps and the divisions that turn them into the models'
shapes. Rates are kept as their two sums so that adding a request is an addition and never a
re-average."""

from collections.abc import Iterable
from dataclasses import dataclass

from mlx_omnia.engine.footprint import ceiling
from mlx_omnia.engine.generate import Meter
from mlx_omnia.server.metrics.models import (
    Aggregate,
    RequestState,
    Sample,
    Speculation,
    Totals,
)


@dataclass
class ModelTotals:
    """The counters an `Aggregate` is divided out of."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    ttft_seconds: float = 0.0
    ttft_requests: int = 0
    prefill_tokens: int = 0
    """Fresh rows, over the requests that got as far as a first token — the numerator
    `ttft_seconds` is the denominator of. A request that was cancelled mid-prefill is in
    neither: it has rows and no time to divide them by."""
    decode_seconds: float = 0.0
    decode_tokens: int = 0
    bytes_per_token: int | None = None


@dataclass(frozen=True)
class Live:
    """The request being generated. It holds the meter itself rather than a copy of its
    numbers, which is what makes a snapshot taken mid-generation say where it is."""

    model: str
    meter: Meter
    bytes_per_token: int | None
    started_at: float
    load_seconds: float | None


def _fraction(rate: float | None, bytes_per_token: int | None) -> float | None:
    if rate is None or bytes_per_token is None:
        return None
    return rate / ceiling(bytes_per_token)


def prefill_rate(prompt_tokens: int, ttft: float | None, reused: int = 0) -> float | None:
    """The prompt against the time it took to get a token out of it — the same denominator
    mlx-lm's "prompt tok/s" uses, and the only one this process can measure: separating the
    prefill from the step that draws the first token would cost a sync the decode loop does
    not pay. So it is a floor on the prefill rate, by one decode step.

    The rows a prefix cache handed over are taken out of the numerator, because `ttft` did
    not pay for them: 62k tokens over the time it took to read the 300 that were left is a
    rate no hardware here has. At its widest the reuse still leaves one row — `take` keeps
    a position for the sampler to read — so the difference is never zero in this process,
    and `None` covers the empty prompt and whatever else could make it so.
    """
    fresh = prompt_tokens - reused
    if ttft is None or ttft <= 0 or fresh <= 0:
        return None
    return fresh / ttft


def _prefill_rate(meter: Meter) -> float | None:
    """The finished request's rate, or — while the prompt is still being read — the rows the
    trunk has taken over the clock it has been taking them in."""
    settled = prefill_rate(meter.prompt_tokens, meter.ttft, meter.reused_tokens)
    if settled is not None:
        return settled
    elapsed = meter.prefill_seconds
    if elapsed is None or elapsed <= 0 or meter.prefilled_tokens <= 0:
        return None
    return meter.prefilled_tokens / elapsed


def measure(live: Live, state: RequestState) -> Sample:
    meter = live.meter
    rate = meter.tokens_per_second
    return Sample(
        model=live.model,
        state=state,
        prompt_tokens=meter.prompt_tokens,
        reused_tokens=meter.reused_tokens,
        kept_prefix=meter.kept_prefix,
        prefilled_tokens=meter.prefilled_tokens,
        completion_tokens=meter.completion_tokens,
        started_at=live.started_at,
        load_seconds=live.load_seconds,
        ttft=meter.ttft,
        tokens_per_second=rate,
        prefill_tokens_per_second=_prefill_rate(meter),
        bytes_per_token=live.bytes_per_token,
        ceiling_fraction=_fraction(rate, live.bytes_per_token),
        speculation=(
            None
            if meter.speculation.rounds == 0
            else Speculation(
                rounds=meter.speculation.rounds,
                proposed=meter.speculation.proposed,
                accepted=meter.speculation.accepted,
            )
        ),
    )


def _mean_prefill(prefill_tokens: int, ttft_seconds: float) -> float | None:
    if ttft_seconds <= 0 or prefill_tokens <= 0:
        return None
    return prefill_tokens / ttft_seconds


def aggregate(model: str, totals: ModelTotals) -> Aggregate:
    ttft = totals.ttft_seconds / totals.ttft_requests if totals.ttft_requests else None
    rate = totals.decode_tokens / totals.decode_seconds if totals.decode_seconds > 0 else None
    return Aggregate(
        model=model,
        requests=totals.requests,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        ttft=ttft,
        prefill_tokens_per_second=_mean_prefill(totals.prefill_tokens, totals.ttft_seconds),
        tokens_per_second=rate,
        bytes_per_token=totals.bytes_per_token,
        ceiling_fraction=_fraction(rate, totals.bytes_per_token),
    )


def overall(totals: Iterable[ModelTotals], running: int) -> Totals:
    held = list(totals)
    ttft_seconds = sum(one.ttft_seconds for one in held)
    ttft_requests = sum(one.ttft_requests for one in held)
    decode_seconds = sum(one.decode_seconds for one in held)
    decode_tokens = sum(one.decode_tokens for one in held)
    rate = decode_tokens / decode_seconds if decode_seconds > 0 else None
    shares = [
        (one.decode_tokens, one.decode_tokens / one.decode_seconds / ceiling(one.bytes_per_token))
        for one in held
        if one.bytes_per_token is not None and one.decode_seconds > 0
    ]
    weight = sum(tokens for tokens, _ in shares)
    return Totals(
        requests=sum(one.requests for one in held),
        running=running,
        prompt_tokens=sum(one.prompt_tokens for one in held),
        completion_tokens=sum(one.completion_tokens for one in held),
        ttft=ttft_seconds / ttft_requests if ttft_requests else None,
        prefill_tokens_per_second=_mean_prefill(
            sum(one.prefill_tokens for one in held), ttft_seconds
        ),
        tokens_per_second=rate,
        ceiling_fraction=(
            sum(tokens * share for tokens, share in shares) / weight if weight else None
        ),
    )

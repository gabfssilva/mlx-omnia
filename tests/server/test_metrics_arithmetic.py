"""The arithmetic of the register, checked off `Metrics` directly.

The meters here have their marks written by hand: over HTTP the rates would be whatever the
machine did, and the question here is whether a four-token request and a hundred-token one
weigh the same.
"""

import pytest

from mlx_omnia.engine.footprint import SUSTAINED_GBS
from mlx_omnia.engine.generate import Meter
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.metrics import register as register_module
from tests.server.metrics_stand import BIG_ACTIVE_BYTES, TINY_ACTIVE_BYTES, TINY_CEILING


def test_the_aggregate_weighs_each_request_by_what_it_generated() -> None:
    """Two requests on one model: two tokens in a second, and a hundred in four. The rate of
    the pair is 102/5 = 20.4 tok/s — the ratio of the totals — and not 13.5, the mean of the
    two rates, which would let a four-token request weigh as much as a hundred-token one.

    ttft is the other way round: one prefill per request, whatever its length, so it averages.
    """
    register = Metrics()
    short = Meter(
        prompt_tokens=5, completion_tokens=3, prefill_started=0.0, first_token=0.5, last_token=1.5
    )
    long = Meter(
        prompt_tokens=7, completion_tokens=101, prefill_started=0.0, first_token=1.5, last_token=5.5
    )

    for meter in (short, long):
        register.begin("big", meter, BIG_ACTIVE_BYTES)
        register.end("completed")

    aggregate = register.snapshot().models[0]
    assert aggregate.model == "big"
    counts = (aggregate.requests, aggregate.prompt_tokens, aggregate.completion_tokens)
    assert counts == (2, 12, 104)
    assert aggregate.tokens_per_second == pytest.approx(20.4)
    assert aggregate.ttft == pytest.approx(1.0)
    # The prefill is the other ratio of totals: twelve fresh rows over the two seconds of
    # prefill that read them, and not the mean of 10 and 4.67 tok/s.
    assert aggregate.prefill_tokens_per_second == pytest.approx(12 / 2.0)
    assert aggregate.bytes_per_token == BIG_ACTIVE_BYTES
    assert aggregate.ceiling_fraction == pytest.approx(
        20.4 / (SUSTAINED_GBS * 1e9 / BIG_ACTIVE_BYTES), rel=1e-3
    )

    # The daemon-wide totals are the same arithmetic over every model, and with one model
    # they are that model's: what a global row must not do is average the two rates.
    overall = register.snapshot().totals
    assert (overall.requests, overall.running) == (2, 0)
    assert (overall.prompt_tokens, overall.completion_tokens) == (12, 104)
    assert overall.ttft == pytest.approx(1.0)
    assert overall.tokens_per_second == pytest.approx(20.4)
    assert overall.ceiling_fraction == pytest.approx(aggregate.ceiling_fraction)


def test_a_reused_prefix_does_not_inflate_the_prefill_rate() -> None:
    """The bug in the shape of a test. `ttft` covers the rows the prefill actually read, so a
    turn whose head came out of the trie pays for its tail alone — and dividing the whole
    prompt by that time publishes a rate no machine here reached. Same prompt, same model:
    the only difference is how many rows were read, and the warm run must not look faster at
    prefilling than the cold one."""
    register = Metrics()
    cold = Meter(prompt_tokens=600, completion_tokens=1, prefill_started=0.0, first_token=0.15)
    warm = Meter(
        prompt_tokens=600,
        reused_tokens=580,
        completion_tokens=1,
        prefill_started=0.0,
        first_token=0.01,
    )

    for meter in (cold, warm):
        register.begin("m", meter, None)
        register.end("completed")

    hot, fresh = register.snapshot().requests
    assert (hot.prompt_tokens, hot.reused_tokens) == (600, 580)
    assert (fresh.prompt_tokens, fresh.reused_tokens) == (600, 0)
    assert hot.prefill_tokens_per_second == pytest.approx(20 / 0.01)
    assert fresh.prefill_tokens_per_second == pytest.approx(600 / 0.15)
    hot_rate, fresh_rate = hot.prefill_tokens_per_second, fresh.prefill_tokens_per_second
    assert hot_rate is not None and fresh_rate is not None
    assert hot_rate < fresh_rate


def test_the_ring_forgets_old_requests_and_the_totals_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What is bounded is the detail, never the count: the ring is the window a dashboard
    draws and the aggregates are counters since boot. A total kept by summing the ring would
    start going *down* on the request that pushed the first one off the end."""
    monkeypatch.setattr(register_module, "_HISTORY", 2)
    register = Metrics()

    for tokens in (1, 2, 3):
        register.begin("m", Meter(prompt_tokens=1, completion_tokens=tokens), None)
        register.end("completed")

    current = register.snapshot()
    assert [record.completion_tokens for record in current.requests] == [3, 2]
    assert current.models[0].requests == 3
    assert current.models[0].completion_tokens == 6
    assert current.models[0].bytes_per_token is None
    assert current.models[0].ceiling_fraction is None


def test_a_request_that_is_running_is_the_live_entry_and_then_a_record() -> None:
    """`live` is the one field a snapshot cannot answer from history, and it is what the
    stream is watched for. It holds the meter itself — a copy taken at the start would freeze
    the numbers at zero — and it is gone once the request ends, so a finished generation never
    reads as still running."""
    register = Metrics()
    meter = Meter()

    register.begin("m", meter, None)
    meter.prefill(4)
    meter.token()
    live = register.snapshot().live
    assert len(live) == 1
    assert (live[0].state, live[0].prompt_tokens, live[0].completion_tokens) == ("running", 4, 1)
    assert register.snapshot().totals.running == 1

    register.end("cancelled")
    ended = register.snapshot()
    assert ended.live == []
    assert ended.requests[0].state == "cancelled"
    assert ended.requests[0].completion_tokens == 1


def test_a_prefill_still_running_publishes_the_rate_it_has_reached() -> None:
    """A 40k prompt is a minute in which `ttft` does not exist, and every rate defined
    against it is `None`: from outside, a request reading its prompt and a request stuck are
    the same row. The rows the trunk has taken are published instead, and divided by the
    clock they were taken in — the same ratio the settled rate is, measured where it has got
    to.

    Mutation: dropping the live branch of `_prefill_rate` leaves the sample at `None` until
    the first token, and the first assertion below fails.
    """
    register = Metrics()
    meter = Meter()
    register.begin("m", meter, None)
    meter.prefill(4096)
    meter.fed(2048)

    running = register.snapshot().live[0]
    assert running.prefilled_tokens == 2048
    assert running.ttft is None
    assert running.prefill_tokens_per_second is not None
    assert running.prefill_tokens_per_second > 0

    # And once there is a first token the settled definition takes over: the whole fresh
    # prompt over `ttft`, and not the blocks counted on the way.
    meter.prefill_started = 0.0
    meter.first_token = 2.0
    meter.completion_tokens = 1
    settled = register.snapshot().live[0]
    assert settled.prefill_tokens_per_second == pytest.approx(4096 / 2.0)


def test_the_totals_are_every_model_at_once() -> None:
    """The global row. Two models with different ceilings: the requests and the tokens add
    up, the rate is the ratio of the totals rather than the mean of the two, and the share of
    the ceiling is weighted by what each model decoded — bytes per token is the checkpoint's,
    so there is no single denominator to divide one sum by."""
    register = Metrics()
    slow = Meter(
        prompt_tokens=5, completion_tokens=3, prefill_started=0.0, first_token=0.5, last_token=1.5
    )
    fast = Meter(
        prompt_tokens=7, completion_tokens=101, prefill_started=0.0, first_token=1.5, last_token=5.5
    )
    register.begin("slow", slow, BIG_ACTIVE_BYTES)
    register.end("completed")
    register.begin("fast", fast, TINY_ACTIVE_BYTES)
    register.end("completed")

    overall = register.snapshot().totals
    assert (overall.requests, overall.running) == (2, 0)
    assert (overall.prompt_tokens, overall.completion_tokens) == (12, 104)
    assert overall.ttft == pytest.approx(1.0)
    assert overall.prefill_tokens_per_second == pytest.approx(12 / 2.0)
    assert overall.tokens_per_second == pytest.approx(102 / 5)
    shares = [
        (2, 2 / 1.0 / (SUSTAINED_GBS * 1e9 / BIG_ACTIVE_BYTES)),
        (100, 100 / 4.0 / TINY_CEILING),
    ]
    assert overall.ceiling_fraction == pytest.approx(
        sum(tokens * share for tokens, share in shares) / 102
    )


def test_the_totals_count_every_request_in_flight() -> None:
    """Two conversations at once is what the engine batches for, and `live` is a list for the
    same reason: a register that published one of them would be describing the other's
    screen."""
    register = Metrics()
    first = register.begin("m", Meter(prompt_tokens=2), None)
    register.begin("n", Meter(prompt_tokens=5), None)

    current = register.snapshot()
    assert [record.model for record in current.live] == ["n", "m"], "newest first"
    assert current.totals.running == 2

    register.end("completed", first)
    assert [record.model for record in register.snapshot().live] == ["n"]
    assert register.snapshot().totals.running == 1


def test_the_beat_republishes_a_running_request_and_nothing_else() -> None:
    """`begin` and `end` are the register's only edges, and ttft, the decode rate and the
    acceptance all land between them. The beat is what a reader watching a generation sees
    it move by; with nothing running it must not wake anybody."""
    register = Metrics()
    raised: list[int] = []
    register.on_change = lambda: raised.append(1)

    register.beat()
    assert raised == []

    meter = Meter()
    register.begin("m", meter, None)
    edges = len(raised)
    meter.prefill(4)
    meter.token()
    register.beat()
    assert len(raised) == edges + 1
    live = register.snapshot().live
    assert len(live) == 1 and live[0].completion_tokens == 1

    register.end("completed")
    after = len(raised)
    register.beat()
    assert len(raised) == after


def test_concurrent_requests_end_against_their_own_meters() -> None:
    register = Metrics()
    first = Meter(prompt_tokens=2, completion_tokens=3)
    second = Meter(prompt_tokens=5, completion_tokens=7)
    first_key = register.begin("m", first, None)
    second_key = register.begin("m", second, None)

    register.end("completed", first_key)
    register.end("cancelled", second_key)

    requests = register.snapshot().requests
    assert [(request.state, request.completion_tokens) for request in requests] == [
        ("cancelled", 7),
        ("completed", 3),
    ]

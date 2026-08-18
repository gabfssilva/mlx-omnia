"""/admin/metrics: the register of what each request cost.

The arithmetic of the aggregates is checked off the register directly in
`test_metrics_arithmetic.py`; what runs here goes through a real server, because what it
asks is whether the register keeps a record per request and where the load is charged.
"""

import pytest

from tests.server.metrics_stand import (
    COLD_LOAD,
    TINY_ACTIVE_BYTES,
    TINY_CEILING,
    Stand,
    entries,
    number,
    run,
    snapshot,
    stand,
    totals,
)

__all__ = ["stand"]


def test_two_sequential_requests_are_two_records_with_their_own_numbers(stand: Stand) -> None:
    """The register is a register and not a last-value: the second request must not overwrite
    the first, and each record must carry what *that* request did — same model, different
    prompt, different length. Newest first, which is the order a dashboard reads."""
    before = totals(stand, "quick")

    run(stand, "quick", "abc", 4)
    run(stand, "quick", "abcdefg", 9)

    recent = entries(snapshot(stand), "requests")[:2]
    assert [record["model"] for record in recent] == ["quick", "quick"]
    assert [record["prompt_tokens"] for record in recent] == [7, 3]
    assert [record["completion_tokens"] for record in recent] == [9, 4]
    assert [record["state"] for record in recent] == ["completed", "completed"]
    for record in recent:
        assert number(record, "ttft") > 0
        rate = number(record, "tokens_per_second")
        assert rate > 0
        assert record["bytes_per_token"] == TINY_ACTIVE_BYTES
        assert number(record, "ceiling_fraction") == pytest.approx(rate / TINY_CEILING)

    after = totals(stand, "quick")
    assert (after[0] - before[0], after[1] - before[1], after[2] - before[2]) == (2, 10, 13)


def test_the_load_is_the_first_request_s_and_no_other_s(stand: Stand) -> None:
    """A cold checkpoint is seconds and every request after it is none, so the load belongs to
    the request that paid it: reported on the first and absent on the second. It sits outside
    `ttft`, which starts at the prefill with the weights already in memory — a load folded
    into it would read as a prefill twenty times the size of the prompt.

    The prefill rate is asserted as the ratio it is defined as, not as a floor: `prefill_rate`
    reporting the decode's denominator by mistake would pass any `> 0`.
    """
    run(stand, "cold", "abcdefgh", 4)
    first = entries(snapshot(stand), "requests")[0]
    run(stand, "cold", "abcdefgh", 4)
    second = entries(snapshot(stand), "requests")[0]

    assert number(first, "load_seconds") >= COLD_LOAD
    assert number(first, "ttft") < COLD_LOAD, "the load leaked into the meter"
    assert second["load_seconds"] is None
    for record in (first, second):
        assert number(record, "prefill_tokens_per_second") == pytest.approx(
            number(record, "prompt_tokens") / number(record, "ttft")
        )


def test_a_model_with_no_tree_reports_no_bytes_and_no_ceiling(stand: Stand) -> None:
    """`footprint` reads tensors and a model the walk cannot reach has none. The field is
    empty rather than zero: a percentage of a ceiling nobody computed is a number the
    dashboard would print as if it meant something."""
    run(stand, "opaque", "hi", 3)

    record = entries(snapshot(stand), "requests")[0]

    assert record["model"] == "opaque"
    assert record["completion_tokens"] == 3
    assert record["bytes_per_token"] is None
    assert record["ceiling_fraction"] is None
    assert number(record, "tokens_per_second") > 0, "the rate is the meter's, not the tree's"

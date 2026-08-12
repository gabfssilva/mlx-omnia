"""The stopwatch every arm shares, and the cut it stops at."""

import time
from collections.abc import Iterator, Sequence

import pytest

from mlx_omnia_bench.arm import NothingToTime, arm
from mlx_omnia_bench.sample import Sample

STEP = 0.005


def paced(ids: Sequence[int], *, prefill: float = STEP, step: float = STEP):
    def generate(
        prompt: Sequence[int], script: Sequence[int] | None, limit: int
    ) -> Iterator[int]:
        time.sleep(prefill)
        source = ids if script is None else script
        for index in range(min(limit, len(source))):
            if index:
                time.sleep(step)
            yield source[index]

    return generate


def test_sample_derives_the_two_rates_from_what_was_counted() -> None:
    sample = Sample(prompt_tokens=1000, ttft=0.5, generated=128, decode_s=1.27)
    assert sample.prefill == pytest.approx(2000.0)
    assert sample.decode == pytest.approx(127 / 1.27)


def test_the_first_id_is_inside_the_ttft_and_outside_the_decode() -> None:
    measured = arm("one", paced([1, 2, 3, 4]), tokens=4).timed([9], None)
    assert measured.generated == 4
    assert measured.ttft >= STEP
    assert measured.decode_s >= 2 * STEP
    assert measured.prompt_tokens == 1


def test_the_cut_is_the_token_count_even_when_the_engine_would_go_on() -> None:
    def endless(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        index = 0
        while True:
            yield index
            index += 1

    assert arm("endless", endless, tokens=6).timed([1], None).generated == 6
    assert arm("endless", endless, tokens=6).stream([1]) == [0, 1, 2, 3, 4, 5]


def test_the_script_replaces_what_a_free_run_would_have_produced() -> None:
    one = arm("scripted", paced([1, 2, 3, 4]), tokens=4)
    assert one.stream([9]) == [1, 2, 3, 4]
    seen: list[Sequence[int] | None] = []

    def watching(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        seen.append(script)
        yield from (script or [7, 7, 7])

    arm("watching", watching, tokens=3).timed([9], [5, 6, 7])
    assert seen == [[5, 6, 7]]


def test_a_free_arm_is_never_handed_a_script() -> None:
    seen: list[Sequence[int] | None] = []

    def watching(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        seen.append(script)
        yield from [1, 2, 3]

    arm("drafted", watching, tokens=3, free=True).timed([9], [5, 6, 7])
    assert seen == [None]


def test_the_timed_loop_never_reads_an_id() -> None:
    """Converting one is a device synchronization in a lazy engine. `stream` converts because
    it wants the ids; the round under the clock must not add a sync the measured loop does not
    already pay."""

    class Explosive:
        def __int__(self) -> int:
            raise AssertionError("the timed loop read an id")

    def generate(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        yield from (Explosive() for _ in range(limit))

    assert arm("lazy", generate, tokens=4).timed([1], None).generated == 4


def test_an_arm_that_says_almost_nothing_has_no_rate() -> None:
    def one(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        yield 1

    with pytest.raises(NothingToTime, match="emitted 1"):
        arm("mute", one, tokens=8).timed([1], None)


def test_a_token_count_below_two_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        arm("tiny", paced([1]), tokens=1)

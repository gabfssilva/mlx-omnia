"""What makes rounds comparable: the rotation, one script across the arms, and the refusal to
divide two runs that did different amounts of work."""

from collections.abc import Iterator, Sequence

import pytest

from mlx_omnia_bench.arm import arm
from mlx_omnia_bench.battery import Incomparable, interleaved
from mlx_omnia_bench.gate import Cool

UNGATED = Cool(None)


def scripted(name: str, ids: Sequence[int], order: list[str], *, free: bool = False):
    def generate(
        prompt: Sequence[int], script: Sequence[int] | None, limit: int
    ) -> Iterator[int]:
        order.append(name)
        yield from (ids if script is None else script)[:limit]

    return arm(name, generate, tokens=len(ids), free=free)


def test_the_order_rotates_so_no_arm_always_runs_last() -> None:
    order: list[str] = []
    arms = [scripted(name, [1, 2, 3], order) for name in ("a", "b", "c")]
    interleaved(arms, [7], runs=3, gate=UNGATED, log=lambda _: None)
    assert order[:3] == ["a", "b", "c"], "the free streams come first, in arm order"
    assert order[3:] == ["a", "b", "c", "b", "c", "a", "c", "a", "b"]


def test_every_arm_is_gated_before_every_round() -> None:
    waits = []

    class Counting:
        def wait(self) -> float | None:
            waits.append(len(waits))
            return None

    order: list[str] = []
    arms = [scripted(name, [1, 2], order) for name in ("a", "b")]
    interleaved(arms, [7], runs=4, gate=Counting(), log=lambda _: None)
    assert len(waits) == 8


def test_one_script_reaches_every_arm_that_takes_one() -> None:
    seen: list[Sequence[int] | None] = []

    def watching(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        seen.append(None if script is None else list(script))
        yield from (script or [9, 9, 9])

    order: list[str] = []
    result = interleaved(
        [scripted("reference", [1, 2, 3], order), arm("other", watching, tokens=3)],
        [7],
        runs=1,
        gate=UNGATED,
        log=lambda _: None,
    )
    assert result.reference == "reference"
    assert seen == [None, [1, 2, 3]], "its own stream first, then the reference's"


def test_a_divergent_arm_is_reported_at_the_id_it_parts_on() -> None:
    order: list[str] = []
    result = interleaved(
        [scripted("a", [1, 2, 3, 4], order), scripted("b", [1, 2, 9, 4], order)],
        [7],
        runs=1,
        gate=UNGATED,
        log=lambda _: None,
    )
    assert result.divergence("a") is None
    assert result.divergence("b") == 2
    assert "parts from a at id 2" in result.render()


def test_a_free_arm_of_a_different_length_is_refused() -> None:
    """Forced arms are pinned to one script, so their free streams may differ. A free arm's
    round is its own generation, and a shorter one is a rate over different work."""
    order: list[str] = []
    with pytest.raises(Incomparable, match="stopped early"):
        interleaved(
            [scripted("a", [1, 2, 3, 4], order), scripted("draft", [1, 2], order, free=True)],
            [7],
            runs=1,
            gate=UNGATED,
            log=lambda _: None,
        )


def test_arms_that_decoded_different_step_counts_get_no_ratio() -> None:
    """A grammar closes its document when the document is done, script or no script, and two
    runs of different lengths have no ratio between them."""

    def closing(prompt: Sequence[int], script: Sequence[int] | None, limit: int):
        yield from [1, 2]

    order: list[str] = []
    result = interleaved(
        [scripted("free", [1, 2, 3, 4], order), arm("bound", closing, tokens=4)],
        [7],
        runs=1,
        gate=UNGATED,
        log=lambda _: None,
    )
    assert not result.comparable
    assert "no ratio" in result.render()
    assert "ratio: decode" not in result.render()


def test_the_reference_is_the_numerator_of_the_ratios() -> None:
    order: list[str] = []
    result = interleaved(
        [scripted("ours", [1, 2, 3], order), scripted("theirs", [1, 2, 3], order)],
        [7],
        runs=1,
        gate=UNGATED,
        log=lambda _: None,
    )
    assert "(ours/theirs; >1 = ours faster)" in result.render()
    assert result.as_dict()["reference"] == "ours"


def test_two_arms_of_the_same_name_are_refused() -> None:
    order: list[str] = []
    with pytest.raises(ValueError, match="share a name"):
        interleaved(
            [scripted("a", [1, 2], order), scripted("a", [1, 2], order)],
            [7],
            runs=1,
            gate=UNGATED,
        )

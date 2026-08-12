"""The gate decides whether a measurement happens at all, and every branch of it is a policy
about a machine nobody can see from a test — so the sensor is handed over instead."""

from dataclasses import dataclass, field

import pytest

from mlx_omnia_bench.gate import Cool, Hot


@dataclass
class Readings:
    """A scripted sensor. Past the end it keeps answering the last reading, so a test that
    wants "hot for ever" writes one number."""

    series: list[float | None]
    taken: int = 0

    def temperature(self) -> float | None:
        reading = self.series[min(self.taken, len(self.series) - 1)]
        self.taken += 1
        return reading


@dataclass
class Slept:
    seconds: list[float] = field(default_factory=list)

    def __call__(self, duration: float) -> None:
        self.seconds.append(duration)


def cool(series: list[float | None], **policy: float) -> tuple[Cool, Slept, list[str]]:
    slept, logged = Slept(), []
    return Cool(Readings(series), sleep=slept, log=logged.append, **policy), slept, logged


def test_already_cool_starts_at_once() -> None:
    gate, slept, _ = cool([38.0])
    assert gate.wait() == 38.0
    assert slept.seconds == []


def test_waits_until_it_reaches_the_gate() -> None:
    gate, slept, logged = cool([60.0, 50.0, 39.0])
    assert gate.wait() == 39.0
    assert slept.seconds == [10.0, 10.0]
    assert any("passed at 39.0" in line for line in logged)


def test_hot_and_not_cooling_aborts_instead_of_measuring_a_loaded_machine() -> None:
    gate, _, _ = cool([60.0])
    with pytest.raises(Hot, match="not cooling"):
        gate.wait()


def test_jitter_around_a_plateau_is_not_progress() -> None:
    """59.9 is below 60.0 and above the epsilon that separates a real drop from sensor
    noise. Counting it as progress would keep resetting the stall clock and the gate would
    wait out its hard ceiling on a machine that is not cooling at all."""
    gate, _, _ = cool([60.0, 59.9] * 200)
    with pytest.raises(Hot, match="not cooling"):
        gate.wait()


def test_still_cooling_but_never_arriving_hits_the_hard_ceiling() -> None:
    """A drop every poll keeps the stall detector satisfied for ever; something has to end
    the wait anyway."""
    gate, _, _ = cool([100.0 - step * 0.5 for step in range(200)], max_s=100.0)
    with pytest.raises(Hot, match="did not reach"):
        gate.wait()


def test_an_unreadable_sensor_gives_up_rather_than_blocking() -> None:
    gate, _, logged = cool([None])
    assert gate.wait() is None
    assert any("no usable temperature" in line for line in logged)


def test_a_reading_that_recovers_is_not_counted_as_a_failure() -> None:
    gate, _, _ = cool([None, None, 30.0])
    assert gate.wait() == 30.0


def test_an_implausible_reading_says_so_and_does_not_block() -> None:
    gate, _, logged = cool([3.7])
    assert gate.wait() == 3.7
    assert any("implausible" in line for line in logged)


def test_no_sensor_is_an_ungated_run() -> None:
    assert Cool(None).wait() is None

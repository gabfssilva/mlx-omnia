"""The rules that decide whether a change is accepted, which is the whole point of the two
batteries: a wrong direction here rejects a good change or lets a regression through, and
neither shows up as an error anywhere else."""

import json

import pytest

from mlx_omnia.bench.ceiling import ceiling
from mlx_omnia.bench.report import Axis, Calibration, Outcome, axis, verdict


def test_direction_needs_disjoint_ranges() -> None:
    assert axis([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]).direction == 1
    assert axis([13.0, 14.0, 15.0], [10.0, 11.0, 12.0]).direction == -1


def test_touching_ranges_are_not_a_movement() -> None:
    """The candidate is faster at the median and its worst round ties the baseline's best.
    That is the noise the machine has anyway, and reading it as a gain is how a lucky
    measurement becomes a result."""
    moved = axis([10.0, 11.0, 12.0], [12.0, 13.0, 14.0])
    assert moved.speedup > 1
    assert moved.direction == 0


def test_speedup_is_median_over_median() -> None:
    assert axis([10.0, 20.0, 30.0], [40.0, 40.0, 40.0]).speedup == 2.0


def test_below_the_floor_rejects_even_while_rising() -> None:
    """A floor is not a tiebreak: an axis under it rejects whatever the ranges say."""
    assert verdict(Axis(0.90, 1), Axis(1.0, 0)).outcome is Outcome.REJECTED


def test_up_with_nothing_down_accepts() -> None:
    assert verdict(Axis(1.2, 1), Axis(1.0, 0)).outcome is Outcome.ACCEPTED


def test_down_with_nothing_up_rejects() -> None:
    assert verdict(Axis(1.0, 0), Axis(0.99, -1)).outcome is Outcome.REJECTED


def test_one_of_each_is_a_tradeoff_nobody_averages() -> None:
    decided = verdict(Axis(1.10, 1), Axis(0.97, -1))
    assert decided.outcome is Outcome.TRADEOFF
    assert "decode up" in decided.detail and "prefill down" in decided.detail


def test_neither_axis_moved() -> None:
    assert verdict(Axis(1.001, 0), Axis(0.999, 0)).outcome is Outcome.NEUTRAL


def test_calibration_reports_drift_against_what_was_stored(tmp_path) -> None:
    store = Calibration(tmp_path / "nested" / "calibration.json", band=0.05)
    assert store.record("qwen3@abc", 100.0) is None
    drift = store.record("qwen3@abc", 90.0)
    assert drift is not None
    assert drift.fraction == pytest.approx(-0.10)
    assert drift.outside
    assert json.loads((tmp_path / "nested" / "calibration.json").read_text()) == {
        "qwen3@abc": 90.0
    }


def test_calibration_inside_the_band_is_not_flagged(tmp_path) -> None:
    store = Calibration(tmp_path / "calibration.json", band=0.05)
    store.record("k", 100.0)
    drift = store.record("k", 102.0)
    assert drift is not None and not drift.outside


def test_calibration_keys_do_not_see_each_other(tmp_path) -> None:
    store = Calibration(tmp_path / "calibration.json")
    store.record("a", 100.0)
    assert store.record("b", 50.0) is None
    assert store.record("a", 100.0) is not None


def test_ceiling_is_bandwidth_over_bytes() -> None:
    assert ceiling(1_000_000_000, 610.0) == pytest.approx(610.0)
    assert ceiling(2_000_000_000, 610.0) == pytest.approx(305.0)

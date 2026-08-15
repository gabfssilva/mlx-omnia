"""The side that runs in its own interpreter, and the check that it ran the tree it was told
to. No model and no GPU: what is under test is the process boundary, not a decode."""

import json
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from mlx_omnia.bench import arm, worker
from mlx_omnia.bench.gate import Throttled
from mlx_omnia.bench.paired import Side, paired, run_side
from mlx_omnia.bench.report import Outcome

FAKE = '''
from mlx_omnia.bench import arm

MARK = "baseline"


def build(step: float = 0.0, tokens: int = 4):
    import time

    def generate(prompt, script, limit):
        time.sleep(step)
        for index in range(limit):
            time.sleep(step)
            yield script[min(index, len(script) - 1)] if script else 100 + index

    return arm("fake", generate, tokens=tokens)
'''


def tree(root: Path, name: str, *, step: float = 0.0) -> Path:
    where = root / name
    where.mkdir()
    (where / "engine.py").write_text(FAKE.replace('step: float = 0.0', f"step: float = {step}"))
    return where


def side(label: str, where: Path, **config: object) -> Side:
    return Side(
        label,
        build="engine:build",
        config=config,
        tree=where,
        roots=("",),
        verify=("engine",),
    )


def test_both_sides_run_and_the_candidate_is_forced_to_the_baseline_stream(tmp_path) -> None:
    result = paired(
        side("baseline", tree(tmp_path, "base")),
        side("current", tree(tmp_path, "head")),
        [1, 2, 3],
        runs=2,
        gated=False,
        log=lambda _: None,
    )
    assert result.streams["baseline"] == [100, 101, 102, 103]
    assert result.divergence is None
    assert [sample.generated for sample in result.samples["current"]] == [4, 4]
    assert result.prompt_tokens == 3
    assert result.verdict.outcome in set(Outcome)


def test_a_slower_candidate_reads_as_a_regression(tmp_path) -> None:
    result = paired(
        side("baseline", tree(tmp_path, "base", step=0.001)),
        side("current", tree(tmp_path, "head", step=0.02)),
        [1, 2, 3],
        runs=2,
        gated=False,
        log=lambda _: None,
    )
    assert result.decode.speedup < 0.95
    assert result.decode.direction == -1
    assert result.verdict.outcome is Outcome.REJECTED


def test_a_side_that_imported_the_wrong_tree_fails_instead_of_measuring_twice(
    tmp_path,
) -> None:
    """Without this the baseline silently benches the candidate's code and the ratio comes
    out at 1.000 with nothing anywhere to say it was a mistake."""
    elsewhere = tree(tmp_path, "elsewhere")
    lying = Side(
        "baseline",
        build="engine:build",
        config={},
        tree=elsewhere,
        roots=("",),
        verify=("engine", "pytest"),
    )
    with pytest.raises(RuntimeError, match="baseline"):
        paired(lying, side("current", tree(tmp_path, "head")), [1, 2], runs=1, gated=False,
               log=lambda _: None)


def test_two_sides_cannot_share_a_label(tmp_path) -> None:
    where = tree(tmp_path, "base")
    with pytest.raises(ValueError, match="labelled"):
        paired(side("same", where), side("same", where), [1], runs=1, gated=False)


def window(*clocks: int, ratio: float = 1.0) -> list[tuple[float, int, float]]:
    """One sample per 100 ms, all inside the round `settled_mhz` is asked about."""
    return [(1.0 + index * 0.1, mhz, ratio) for index, mhz in enumerate(clocks)]


def test_the_climb_out_of_the_idle_clock_is_not_a_throttled_round() -> None:
    """The trace of a real gated laguna-xs round: idle at 338, ~250 ms to reach 1620, and
    every sample after that pinned. Reading it at its minimum calls 338 the round's clock."""
    assert worker.settled_mhz(window(338, 711, 1620, 1620, 1610), 0.0, 9.0) == 1610


def test_a_clock_that_drops_after_settling_is_the_throttle_the_floor_is_for() -> None:
    assert worker.settled_mhz(window(1620, 1620, 900, 950), 0.0, 9.0) == 900


def test_a_window_that_only_ever_rises_is_read_at_its_last_sample() -> None:
    """Nothing here settled, so there is no ramp to drop — and dropping all of it would hand
    a round that never left 1100 MHz back as no evidence of anything."""
    assert worker.settled_mhz(window(338, 711, 900, 1100), 0.0, 9.0) == 1100


def test_a_plateau_under_the_floor_is_not_a_climb() -> None:
    """A clock that sat still and then rose was *held* there — the climb out of idle passes
    through rising samples, never through a plateau. Counting one as part of it accepts a
    round that spent its whole prefill at 900 MHz."""
    assert worker.settled_mhz(window(900, 900, 1620, 1620), 0.0, 9.0) == 900


def test_a_round_already_at_speed_keeps_its_first_sample() -> None:
    assert worker.settled_mhz(window(1620, 1610, 1620), 0.0, 9.0) == 1610


def test_samples_outside_the_round_are_not_the_round() -> None:
    samples = [*window(338, 1620, 1620), (50.0, 400, 1.0)]
    assert worker.settled_mhz(samples, 1.0, 1.3) == 1620


def test_an_idle_sample_inside_the_window_is_not_evidence() -> None:
    """`gpu_active_ratio` below the bar is the GPU between rounds, not a clock the round
    ran at."""
    assert worker.settled_mhz(window(1620, 1620) + window(400, ratio=0.1), 0.0, 9.0) == 1620


def test_a_round_with_no_loaded_sample_is_not_evidence_of_throttling() -> None:
    assert worker.settled_mhz(window(400, 400, ratio=0.1), 0.0, 9.0) is None


def test_a_throttled_round_keeps_retrying_until_its_clock_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gate:
        waits = 0

        def wait(self) -> float:
            self.waits += 1
            return 30.0

    class Clocks:
        samples: ClassVar[list[tuple[float, int, float]]] = []

        def stop(self) -> None:
            pass

    readings = iter((1000, 1100, 1400))
    gate = Gate()
    monkeypatch.setattr(worker, "find_macmon", lambda: "/fake/macmon")
    monkeypatch.setattr(worker, "Macmon", lambda path: path)
    monkeypatch.setattr(worker, "Cool", lambda sensor: gate)
    monkeypatch.setattr(worker, "Clocks", lambda path: Clocks())
    monkeypatch.setattr(worker, "settled_mhz", lambda samples, start, end: next(readings))

    def generate(
        prompt: Sequence[int], script: Sequence[int] | None, tokens: int
    ) -> Iterator[int]:
        yield from range(tokens)

    payload: worker.Payload = {
        "build": "unused",
        "config": {},
        "tree": None,
        "verify": [],
        "prompt": [1],
        "script": None,
        "runs": 1,
        "gated": True,
        "floor_mhz": 1300,
        "max_throttled_retries": 20,
        "out": "unused",
    }
    result = worker.measure(arm("fake", generate, tokens=2), payload)

    assert result["clocks"] == [1400]
    assert gate.waits == 3


def test_a_round_stops_after_the_configured_throttled_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gate:
        def wait(self) -> float:
            return 30.0

    class Clocks:
        samples: ClassVar[list[tuple[float, int, float]]] = []

        def stop(self) -> None:
            pass

    reads = 0

    def throttled(samples: object, start: float, end: float) -> int:
        nonlocal reads
        reads += 1
        if reads > 3:
            raise AssertionError("worker exceeded the configured retry limit")
        return 1000

    attempts = 0
    monkeypatch.setattr(worker, "find_macmon", lambda: "/fake/macmon")
    monkeypatch.setattr(worker, "Macmon", lambda path: path)
    monkeypatch.setattr(worker, "Cool", lambda sensor: Gate())
    monkeypatch.setattr(worker, "Clocks", lambda path: Clocks())
    monkeypatch.setattr(worker, "settled_mhz", throttled)

    def generate(
        prompt: Sequence[int], script: Sequence[int] | None, tokens: int
    ) -> Iterator[int]:
        nonlocal attempts
        attempts += 1
        yield from range(tokens)

    payload: worker.Payload = {
        "build": "unused",
        "config": {},
        "tree": None,
        "verify": [],
        "prompt": [1],
        "script": None,
        "runs": 1,
        "gated": True,
        "floor_mhz": 1300,
        "max_throttled_retries": 2,
        "out": "unused",
    }

    with pytest.raises(Throttled, match="throttled 3 times"):
        worker.measure(arm("fake", generate, tokens=2), payload)

    assert attempts == 4


def test_both_sides_use_the_current_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[Sequence[str]] = []
    payloads: list[dict[str, object]] = []

    def run(arguments: Sequence[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        payload = json.loads(Path(arguments[-1]).read_text())
        payloads.append(payload)
        Path(payload["out"]).write_text(
            json.dumps(
                {
                    "stream": [1, 2],
                    "samples": [
                        {"prompt_tokens": 1, "ttft": 0.1, "generated": 2, "decode_s": 0.1}
                    ],
                    "clocks": [1400],
                    "modules": {},
                }
            )
        )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", run)
    run_side(
        Side("baseline", build="engine:build"),
        [1],
        None,
        1,
        gated=True,
        floor_mhz=1300,
        max_throttled_retries=7,
        log=lambda message: None,
    )

    assert Path(commands[0][1]).resolve() == Path(worker.__file__).resolve()
    assert payloads[0]["max_throttled_retries"] == 7

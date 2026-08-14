"""The side that runs in its own interpreter, and the check that it ran the tree it was told
to. No model and no GPU: what is under test is the process boundary, not a decode."""

import json
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

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


def test_a_throttled_round_keeps_retrying_until_its_clock_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gate:
        waits = 0

        def wait(self) -> float:
            self.waits += 1
            return 30.0

    class Clocks:
        readings = iter((1000, 1100, 1400))

        def min_loaded_mhz(self, start: float, end: float) -> int:
            return next(self.readings)

        def stop(self) -> None:
            pass

    gate = Gate()
    clocks = Clocks()
    monkeypatch.setattr(worker, "find_macmon", lambda: "/fake/macmon")
    monkeypatch.setattr(worker, "Macmon", lambda path: path)
    monkeypatch.setattr(worker, "Cool", lambda sensor: gate)
    monkeypatch.setattr(worker, "Clocks", lambda path: clocks)

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
        readings = 0

        def min_loaded_mhz(self, start: float, end: float) -> int:
            self.readings += 1
            if self.readings > 3:
                raise AssertionError("worker exceeded the configured retry limit")
            return 1000

        def stop(self) -> None:
            pass

    attempts = 0
    monkeypatch.setattr(worker, "find_macmon", lambda: "/fake/macmon")
    monkeypatch.setattr(worker, "Macmon", lambda path: path)
    monkeypatch.setattr(worker, "Cool", lambda sensor: Gate())
    monkeypatch.setattr(worker, "Clocks", lambda path: Clocks())

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

"""The thermal gate, and the clock telemetry that rejects a round which throttled anyway.

The drift that reads as "a few percent over a battery" is throttle *spikes*, not smooth
decay: prefill throttles about 2x cool→hot, and sitting idle does not recover the cool state —
only cooling below a gate does. So every timed round starts from the same thermal state or it
is not comparable to the others.

A GPU that is hot and not trending down is a GPU something else is using, and waiting longer
does not fix that: past a bound the gate raises instead of measuring a loaded machine.

The sensor is injected. The policy here — plateau detection, the stall bound, the hard
ceiling, an implausible reading, an unreadable one — is the part that decides whether a
measurement happens at all, and it is testable only if reading a temperature is something the
test can hand over.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mlx_omnia.bench.report import stderr

_IMPLAUSIBLE_C = 5.0
"""Observed on macmon 0.7.2: a frozen ~3.7°C reading for tens of minutes. Not fatal — the
gate says so and goes on, because the alternative is refusing to run on a working machine."""

_CANDIDATES = ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon")


# `Macmon` and `Cool` implement these structurally rather than by inheritance: both are
# `slots=True` dataclasses, and a Protocol base is not slotted — inheriting one hands every
# instance back the `__dict__` that `slots=True` removes.
@runtime_checkable
class Sensor(Protocol):
    def temperature(self) -> float | None:
        """°C, or `None` when this reading did not come back."""


@runtime_checkable
class Gate(Protocol):
    def wait(self) -> float | None:
        """Blocks until a round may start; answers the temperature it started at."""


class Hot(RuntimeError):
    """The gate gave up: either something else is loading the GPU, or it never cooled."""


class Throttled(RuntimeError):
    """A round ran below the clock floor and did not recover on its retry. Both live here
    because both say the same thing — the machine was not in a state to be measured."""


def find_macmon(path: str | None = None) -> str | None:
    if path is not None:
        return path
    found = shutil.which("macmon")
    if found is not None:
        return found
    return next((candidate for candidate in _CANDIDATES if os.access(candidate, os.X_OK)), None)


@dataclass(frozen=True, slots=True)
class Macmon:
    """One reading per call, which is what a gate polling every ten seconds wants. The
    streaming form is `Clocks`, and it is deliberately not this: a sampler running at 100 ms
    during a timed round is load the round did not ask for."""

    path: str

    def temperature(self) -> float | None:
        try:
            done = subprocess.run(
                [self.path, "pipe", "-s1"], capture_output=True, timeout=30, check=True
            )
            reading = json.loads(done.stdout.splitlines()[0])["temp"]["gpu_temp_avg"]
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError):
            return None
        return float(reading) if isinstance(reading, int | float) else None


@dataclass(frozen=True, slots=True)
class Cool:
    """`sensor=None` runs ungated, which is for debugging: those numbers do not compare with
    gated ones and nothing downstream can tell them apart."""

    sensor: Sensor | None
    gate_c: float = 40.0
    poll_s: float = 10.0
    abort_s: float = 180.0
    """Minimum total wait before a stalled cool-down may abort."""
    stall_s: float = 90.0
    """Abort once no new minimum has been seen for this long."""
    max_s: float = 900.0
    """Hard ceiling even while still (slowly) cooling."""
    epsilon_c: float = 0.25
    """Sensor jitter around a plateau is not progress."""
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = stderr

    def wait(self) -> float | None:
        if self.sensor is None:
            return None
        waited = 0.0
        flaky = 0
        minimum: float | None = None
        progress_at = 0.0
        while True:
            temperature = self.sensor.temperature()
            if temperature is None:
                flaky += 1
                if flaky >= 3:
                    self.log("cool gate: no usable temperature sample, skipping")
                    return None
                self.sleep(2.0)
                continue
            flaky = 0
            if temperature <= _IMPLAUSIBLE_C:
                self.log(
                    f"cool gate: {temperature:.1f}°C reads implausible;"
                    " the gate may be decorative"
                )
            if temperature <= self.gate_c:
                if waited:
                    self.log(f"cool gate: passed at {temperature:.1f}°C after {waited:.0f}s")
                return temperature
            if minimum is None or temperature <= minimum - self.epsilon_c:
                minimum, progress_at = temperature, waited
            if waited >= self.abort_s and waited - progress_at >= self.stall_s:
                raise Hot(
                    f"cool gate: GPU hot and not cooling ({temperature:.1f}°C,"
                    f" min {minimum:.1f}°C, waited {waited:.0f}s) — something else is loading"
                    " it; free it up and rerun, or run ungated for a debug number"
                )
            if waited >= self.max_s:
                raise Hot(
                    f"cool gate: GPU did not reach {self.gate_c:.0f}°C within"
                    f" {self.max_s:.0f}s (current {temperature:.1f}°C)"
                )
            self.log(
                f"cool gate: {temperature:.1f}°C, waiting for"
                f" <={self.gate_c:.0f}°C ({waited:.0f}s)..."
            )
            self.sleep(self.poll_s)
            waited += self.poll_s


class Clocks:
    """macmon streamed on a thread: `(when, MHz, active ratio)` per sample, and nothing read
    off them. What a window of them says about a round is `worker.settled_mhz`, which says
    there why it is not a method here."""

    def __init__(self, path: str, interval_ms: int = 100) -> None:
        self.samples: list[tuple[float, int, float]] = []
        self.process = subprocess.Popen(
            [path, "pipe", "-i", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                sample = json.loads(line)
                self.samples.append(
                    (time.time(), int(sample["gpu_freq_mhz"]), float(sample["gpu_active_ratio"]))
                )
            except (ValueError, KeyError, TypeError):
                continue

    def stop(self) -> None:
        self.process.terminate()

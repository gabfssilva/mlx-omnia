"""The thermal gate and the page cache — what a round waits for before it may start."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from mlx_omnia.bench.gate import Macmon, find_macmon
from mlx_omnia.server.services.speed.protocols import Progress, Report, Task

_GATE_POLL_S = 5.0
_GATE_MAX_S = 900.0


def macmon() -> str | None:
    """The GPU temperature source the instrument gates on. Absent is not an error: the gate
    then does not run, and the row records no temperature rather than a made-up one."""
    if os.environ.get("OMNIA_COOL_GATE") == "0":
        return None
    return find_macmon()


def gpu_temperature(tool: str | None) -> float | None:
    return None if tool is None else Macmon(tool).temperature()


def wait_cool(
    task: Task, tool: str | None, gate_c: float | None, report: Report | None = None
) -> float | None:
    """Blocks until the GPU is at or below the gate, and answers with the temperature the round
    started at. The wait is bounded, and past the bound the measurement goes ahead with the
    temperature it found — recorded, so the row says what it was taken at."""
    temperature = gpu_temperature(tool)
    if gate_c is None or temperature is None:
        return temperature
    waited = 0.0
    while temperature is not None and temperature > gate_c and waited < _GATE_MAX_S:
        message = f"cooling: {temperature:.1f}°C, gate {gate_c:.0f}°C"
        task.report(Progress(message=message) if report is None else report.frame(message))
        time.sleep(_GATE_POLL_S)
        waited += _GATE_POLL_S
        temperature = gpu_temperature(tool)
    return temperature


def purge_page_cache() -> bool:
    """macOS's own `purge`. It needs root on a current system, so a failure refuses the shape
    instead of downgrading it: a cold measurement on a warm page cache is a warm measurement
    wearing the wrong label."""
    tool = shutil.which("purge") or "/usr/sbin/purge"
    try:
        subprocess.run([tool], capture_output=True, timeout=120, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True

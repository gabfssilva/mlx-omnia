"""What the machine is, and which of these numbers the machine actually answered.

The chip, the GPU core count, the memory and the free space of the catalog's volume are
discovered — `sysctl`, the IO registry, `statvfs`. The two bandwidth figures are not: 614 GB/s
is Apple's published number for this chip and 610 GB/s is what a serial chain of dependent
kernels sustains here, measured by `omnia-bench interleaved`. The sustained one is the
denominator of every "% of ceiling" the engine reports, which is why it must not travel
disguised as a reading: `constants` names the fields that were calibrated rather than read.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

from mlx_omnia.server.services import catalog

THEORETICAL_GBS = 614.0
SUSTAINED_GBS = 610.0

CONSTANTS = {
    "bandwidth_theoretical_gbs": (
        "Apple's published figure for this chip. Nothing on the machine reports it."
    ),
    "bandwidth_sustained_gbs": (
        "Measured, not published: what a serial chain of dependent kernels holds on this "
        "machine (~80% of theoretical), calibrated with omnia-bench interleaved. Every "
        "% of ceiling divides by it."
    ),
}


@dataclass(frozen=True)
class SystemInfo:
    """The machine, plus a note on each field that was calibrated instead of read."""

    chip: str
    gpu_cores: int
    memory_bytes: int
    bandwidth_theoretical_gbs: float
    bandwidth_sustained_gbs: float
    disk_free_bytes: int
    catalog: str
    version: str
    constants: dict[str, str]


class Hardware(NamedTuple):
    chip: str
    gpu_cores: int
    memory_bytes: int


@cache
def hardware() -> Hardware:
    """`sysctl` names the chip and the memory; the GPU core count is not a sysctl at all, it is
    a property of the accelerator in the IO registry. Cached because none of the three can
    change under a running process."""
    chip, memory = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    registry = subprocess.run(
        ["ioreg", "-rc", "AGXAccelerator", "-d1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cores = re.search(r'"gpu-core-count" = (\d+)', registry)
    assert cores is not None, "ioreg reported no gpu-core-count"
    return Hardware(chip, int(cores.group(1)), int(memory))


def free_bytes(directory: Path) -> int:
    """`disk_usage` needs a path that exists and the catalog directory does not, until the first
    download; any ancestor that does exist sits on the same volume."""
    volume = directory
    while not volume.exists():
        volume = volume.parent
    return shutil.disk_usage(volume).free


def system() -> SystemInfo:
    """Blocking discovery, so the route above it must keep it off the event loop."""
    machine = hardware()
    return SystemInfo(
        chip=machine.chip,
        gpu_cores=machine.gpu_cores,
        memory_bytes=machine.memory_bytes,
        bandwidth_theoretical_gbs=THEORETICAL_GBS,
        bandwidth_sustained_gbs=SUSTAINED_GBS,
        # The directory the scan actually walks, and not `huggingface_hub`'s constant: a test
        # points the first one somewhere else, and the free space this answers with has to be
        # the space the next download writes into.
        disk_free_bytes=free_bytes(catalog.HUB_CACHE),
        catalog=str(catalog.HUB_CACHE),
        version=version("mlx_omnia"),
        constants=CONSTANTS,
    )

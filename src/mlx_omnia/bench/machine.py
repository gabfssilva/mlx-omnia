"""The machine a number belongs to, derived and never declared.

Everything that can move a measurement is readable from the hardware itself — the chip is
binned (an "M5 Max" ships with 32 or 40 GPU cores, and the memory bandwidth follows the GPU
bin), so the brand string alone under-identifies the machine. What `sysctl` and IOKit cannot
report (disk, screen) also cannot move a token rate, so it stays out of the identity.
"""

import platform
import re
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Machine:
    chip: str
    hw_model: str
    memory_gb: int
    cpu_cores: dict[str, int]
    """Core counts by the level names the hardware itself reports — an M5 Max says "Super"
    and "Performance", not the P/E split the older chips did, so the names are not ours to
    normalize."""
    gpu_cores: int
    macos: str

    @property
    def slug(self) -> str:
        chip = self.chip.removeprefix("Apple ").lower().replace(" ", "-")
        return f"{chip}-{self.gpu_cores}gpu-{self.memory_gb}gb"

    def as_dict(self) -> dict[str, object]:
        return {"slug": self.slug, **asdict(self)}


def detect() -> Machine:
    return Machine(
        chip=_sysctl("machdep.cpu.brand_string"),
        hw_model=_sysctl("hw.model"),
        memory_gb=int(_sysctl("hw.memsize")) // 1024**3,
        cpu_cores={
            _sysctl(f"hw.perflevel{level}.name").lower(): int(
                _sysctl(f"hw.perflevel{level}.physicalcpu")
            )
            for level in range(int(_sysctl("hw.nperflevels")))
        },
        gpu_cores=_gpu_cores(),
        macos=platform.mac_ver()[0],
    )


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, check=True
    ).stdout.strip()


def _gpu_cores() -> int:
    listed = subprocess.run(
        ["ioreg", "-rd1", "-c", "AGXAccelerator"], capture_output=True, text=True, check=True
    ).stdout
    found = re.search(r'"gpu-core-count" = (\d+)', listed)
    if found is None:
        raise RuntimeError("ioreg did not report a gpu-core-count; is this Apple Silicon?")
    return int(found.group(1))

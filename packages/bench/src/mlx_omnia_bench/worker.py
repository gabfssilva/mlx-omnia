"""One side of a paired run, in its own interpreter.

Two checkouts of the same library never share a process, so a side is a subprocess with its
own `PYTHONPATH` — and the first thing it does is prove it: every module the caller named has
to have been imported from the tree this side was told to bench. Without that check, a side
that silently imported the installed copy measures the same code twice and the ratio comes out
at 1.000 with nothing to say it was a mistake.

The harness itself is *not* swapped. `build` is resolved in the current environment, so both
sides construct their arm the same way and only the code under test differs. The consequence
is a constraint worth stating: the baseline ref has to be source-compatible with this
environment, because `build` can only call an API that both sides have.

A round is rejected and retried once when the loaded clock dropped below the floor while it
ran. Twice is not a usable session — the machine is throttling under something this bench does
not control, and a number taken there is not comparable to one taken cool.
"""

import importlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from mlx_omnia_bench.arm import Arm
from mlx_omnia_bench.gate import Clocks, Cool, Macmon, Throttled, find_macmon
from mlx_omnia_bench.report import stderr
from mlx_omnia_bench.sample import Sample


class Payload(TypedDict):
    build: str
    config: dict[str, object]
    tree: str | None
    verify: list[str]
    prompt: list[int]
    script: list[int] | None
    runs: int
    gated: bool
    floor_mhz: int
    out: str


class Result(TypedDict):
    stream: list[int]
    samples: list[dict[str, float | int]]
    clocks: list[int | None]
    modules: dict[str, str]


def entry(build: str) -> object:
    module_name, _, attribute = build.partition(":")
    if not attribute:
        raise ValueError(f"{build!r} is not in 'module:function' form")
    return getattr(importlib.import_module(module_name), attribute)


def imported_from(names: Sequence[str], tree: str) -> dict[str, str]:
    """Imported here rather than left to `build`: a module the build function never touches
    would pass a check made afterwards, and that is exactly the module worth checking."""
    root = str(Path(tree).resolve())
    found: dict[str, str] = {}
    for name in names:
        file = importlib.import_module(name).__file__
        if file is None or not str(Path(file).resolve()).startswith(root):
            raise RuntimeError(f"{name} was imported from {file}, not from {root}")
        found[name] = file
    return found


def measure(arm: Arm, payload: Payload) -> Result:
    prompt = payload["prompt"]
    own = arm.stream(prompt)
    script = payload["script"] or own

    path = find_macmon() if payload["gated"] else None
    gate = Cool(Macmon(path)) if path is not None else Cool(None)
    clocks = Clocks(path) if path is not None else None
    samples: list[Sample] = []
    measured: list[int | None] = []
    try:
        for index in range(payload["runs"]):
            for attempt in (1, 2):
                gate.wait()
                start = time.time()
                sample = arm.timed(prompt, script)
                end = time.time()
                mhz = None if clocks is None else clocks.min_loaded_mhz(start, end)
                if mhz is None or mhz >= payload["floor_mhz"]:
                    samples.append(sample)
                    measured.append(mhz)
                    break
                if attempt == 2:
                    raise Throttled(
                        f"round {index + 1} throttled twice (min loaded clock {mhz} MHz <"
                        f" {payload['floor_mhz']}); not a usable session"
                    )
                stderr(f"round {index + 1} throttled ({mhz} MHz), one gated retry")
    finally:
        if clocks is not None:
            clocks.stop()
    return {
        "stream": own,
        "samples": [sample.as_dict() for sample in samples],
        "clocks": measured,
        "modules": {},
    }


def main(argv: Sequence[str]) -> None:
    payload: Payload = json.loads(Path(argv[0]).read_text())
    tree = payload["tree"]
    modules = imported_from(payload["verify"], tree) if tree is not None else {}
    build = entry(payload["build"])
    if not callable(build):
        raise TypeError(f"{payload['build']} is not callable")
    arm = build(**payload["config"])
    if not isinstance(arm, Arm):
        raise TypeError(f"{payload['build']} returned {type(arm).__name__}, not an Arm")
    result = measure(arm, payload)
    result["modules"] = modules
    Path(payload["out"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])

"""What a client is allowed to load to reach the daemon.

`lint-imports` cannot answer this one. The failure it misses is not the client importing
the engine — it is `mlx_omnia/__init__` doing it on the client's behalf, which the graph
records as an import by the roof and not by `mlx_omnia.cli`. An .app shipped that way: its
window loaded MLX to open, and in the bundle, where it shipped without MLX beside it, it did
not open at all. That window is Swift now; the CLI is what is left under this roof that must
not pay for the engine.

A fresh interpreter is the whole point. This one already has the engine resident.
"""

from __future__ import annotations

import subprocess
import sys

PROBE = """
import sys
import mlx_omnia.cli.main
loaded = sorted(m for m in ("mlx", "mlx.core", "numpy", "pyarrow") if m in sys.modules)
print(",".join(loaded))
"""


def test_reaching_the_daemon_does_not_load_the_engine() -> None:
    """The CLI speaks HTTP. MLX behind it means Metal starting for a process that will never
    ask it for anything, on a machine that may have no engine installed at all."""
    done = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "", (
        f"the client pulled in {done.stdout.strip()} — the engine is being imported on its "
        "behalf, most likely by a re-export in `mlx_omnia/__init__`"
    )


def test_the_public_api_still_resolves_through_the_roof() -> None:
    """The laziness that keeps the engine out of the client must not cost the re-export:
    `from mlx_omnia import load` is the documented entry point."""
    from mlx_omnia import load

    assert load.__module__ == "mlx_omnia.engine.task"


ROOF = """
import sys
import mlx_omnia
print("mlx" in sys.modules, len(mlx_omnia.__all__), "mlx" in sys.modules)
"""


def test_the_roof_promises_the_same_names_it_defers() -> None:
    """`mlx_omnia/__init__` says the engine's API twice — once in a `TYPE_CHECKING` branch an
    editor reads, once through the hook the interpreter runs. This is the runtime half: the
    star import has to take the 160 names the branch above promised, and importing the roof
    still has to cost nothing until a name is asked for.

    A fresh interpreter is the whole point. This one already has the engine resident.
    """
    done = subprocess.run(
        [sys.executable, "-c", ROOF], capture_output=True, text=True, check=True
    )
    before, names, after = done.stdout.split()
    assert before == "False", "importing the roof loaded the engine"
    assert int(names) > 100, "the star import would take nothing at runtime"
    assert after == "True", "reading __all__ off the roof did not reach the engine"

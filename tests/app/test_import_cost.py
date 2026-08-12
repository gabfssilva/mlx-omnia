"""What the window is allowed to load to draw itself.

`lint-imports` cannot answer this one. The failure it misses is not the window importing
the engine — it is `mlx_omnia/__init__` doing it on the window's behalf, which the graph
records as an import by the roof and not by `mlx_omnia.app`. An .app shipped that way: the
window loaded MLX to open, and in the bundle, where it ships without MLX beside it, it did
not open at all.

A fresh interpreter is the whole point. This one already has the engine resident.
"""

from __future__ import annotations

import subprocess
import sys

PROBE = """
import sys
import mlx_omnia.app.main
loaded = sorted(m for m in ("mlx", "mlx.core", "numpy", "pyarrow") if m in sys.modules)
print(",".join(loaded))
"""


def test_opening_the_window_does_not_load_the_engine() -> None:
    """The window speaks HTTP. MLX behind it means Metal starting to draw a window, and in
    the .app it means an interpreter that has no MLX reaching for one."""
    done = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "", (
        f"the window pulled in {done.stdout.strip()} — the engine is being imported on its "
        "behalf, most likely by a re-export in `mlx_omnia/__init__`"
    )


def test_the_public_api_still_resolves_through_the_roof() -> None:
    """The laziness that keeps the engine out of the window must not cost the re-export:
    `from mlx_omnia import load` is the documented entry point."""
    from mlx_omnia import load

    assert load.__module__ == "mlx_omnia.engine.task"

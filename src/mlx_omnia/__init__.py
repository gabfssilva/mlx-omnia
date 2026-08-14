"""Omnia: an LLM inference engine for Apple Silicon on MLX.

The engine is `mlx_omnia.engine`, re-exported here so `from mlx_omnia import load` keeps
working while the engine stays a sibling of `server`, `cli`, `bench` and `app` rather than
the package they all live inside.

The re-export is lazy, and that is not an optimization. Written as `from mlx_omnia.engine
import *`, every import under this roof pulls the engine in with it: `import mlx_omnia.app`
would load MLX and start Metal to open a window that speaks only HTTP, and in the .app —
where the window ships without MLX beside it — it does not open at all. Resolving the name
on first access instead keeps `from mlx_omnia import load` and costs the window nothing.

What lives directly under this roof is excluded by name so the import system, not this
hook, resolves it: `from mlx_omnia import app` and `from mlx_omnia import paths` must reach
their module without touching the engine on the way. A name missing from that set is not a
small oversight — it silently routes the import through the engine, which is what
`tests/app/test_import_cost.py` exists to catch.
"""

from typing import Any

_OWN = frozenset({"engine", "server", "cli", "bench", "app", "paths"})


def __getattr__(name: str) -> Any:
    if name in _OWN or name.startswith("__"):
        raise AttributeError(name)
    from mlx_omnia import engine

    return getattr(engine, name)


def __dir__() -> list[str]:
    from mlx_omnia import engine

    return [*engine.__all__, *_OWN]

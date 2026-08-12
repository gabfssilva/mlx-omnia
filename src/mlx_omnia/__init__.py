"""Omnia: an LLM inference engine for Apple Silicon on MLX.

The engine is `mlx_omnia.engine`, re-exported here so `from mlx_omnia import load` keeps
working while the engine stays a sibling of `server`, `cli`, `bench` and `app` rather than
the package they all live inside.

The re-export is lazy, and that is not an optimization. Written as `from mlx_omnia.engine
import *`, every import under this roof pulls the engine in with it: `import mlx_omnia.app`
would load MLX and start Metal to open a window that speaks only HTTP, and in the .app —
where the window ships without MLX beside it — it does not open at all. Resolving the name
on first access instead keeps `from mlx_omnia import load` and costs the window nothing.

Submodules are excluded by name so the import system, not this hook, resolves them:
`from mlx_omnia import app` must reach the subpackage without touching the engine on the
way.
"""

from typing import Any

_SUBPACKAGES = frozenset({"engine", "server", "cli", "bench", "app"})


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES or name.startswith("__"):
        raise AttributeError(name)
    from mlx_omnia import engine

    return getattr(engine, name)


def __dir__() -> list[str]:
    from mlx_omnia import engine

    return [*engine.__all__, *_SUBPACKAGES]

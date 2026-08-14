"""Omnia: an LLM inference engine for Apple Silicon on MLX.

The engine is `mlx_omnia.engine`, re-exported here so `from mlx_omnia import load` keeps
working while the engine stays a sibling of `server`, `cli` and `bench` rather than the
package they all live inside.

The re-export is lazy, and that is not an optimization. Written as `from mlx_omnia.engine
import *`, every import under this roof pulls the engine in with it: `import mlx_omnia.cli`
would load MLX and start Metal for a client that speaks only HTTP, and where it ships
without MLX beside it, it does not start at all. Resolving the name on first access instead
keeps `from mlx_omnia import load` and costs the client nothing.

Which is why there are two branches below, and they are the same API twice. The type checker
takes the first and never executes it: an editor completes `from mlx_omnia import ` with the
engine's 160 names, `load` carries its own signature, and a name that is not there is an
error instead of `Any` — a module-level `__getattr__` visible to a checker makes every
attribute legal, which is the opposite of what this package is. The interpreter takes the
second and imports nothing until something is actually asked for.

What lives directly under this roof is excluded by name in that second branch so the import
system, not the hook, resolves it: `from mlx_omnia import cli` and `from mlx_omnia import
paths` must reach their module without touching the engine on the way. A name missing from
that set is not a small oversight — it silently routes the import through the engine, which
is what `tests/test_import_cost.py` exists to catch.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_omnia.engine import *  # noqa: F403
    from mlx_omnia.engine import __all__ as __all__
else:
    from typing import Any

    _OWN = frozenset({"engine", "server", "cli", "bench", "paths"})

    def __getattr__(name: str) -> Any:
        # Answered rather than refused: `from mlx_omnia import *` reads `__all__` through
        # this hook, so refusing it would leave the star import taking nothing at runtime
        # while the branch above promises a checker all 160 names.
        if name == "__all__":
            from mlx_omnia import engine

            return engine.__all__
        if name in _OWN or name.startswith("__"):
            raise AttributeError(name)
        from mlx_omnia import engine

        return getattr(engine, name)

    def __dir__() -> list[str]:
        from mlx_omnia import engine

        return [*engine.__all__, *_OWN]

"""Scoped replacement of mlx entry points by kernels that cover a narrower case.

`mlx.nn` resolves `mx.quantized_matmul` at call time rather than binding it at import,
so replacing the module attribute reaches every caller that goes through Python without
touching a single call site. Calls made inside mlx's own C++ never see the replacement.

`original` is what a parity test has to compare against once `install` has run: reading
`mx.quantized_matmul` would hand it the replacement and the comparison would be the
kernel against itself.
"""

from collections.abc import Callable, Iterator
from contextlib import ContextDecorator, ExitStack, contextmanager
from dataclasses import dataclass
from functools import wraps
from types import ModuleType


@dataclass(frozen=True)
class Patch[**P, R]:
    """A replacement for `module.name`, used only where `applies` holds."""

    module: ModuleType
    name: str
    applies: Callable[P, bool]
    replacement: Callable[P, R]


def install(*patches: Patch[..., object]) -> None:
    """Install for the life of the process. Idempotent, so importing twice is harmless."""
    for patch in patches:
        key = (patch.module.__name__, patch.name)
        if key in _originals:
            continue
        _originals[key] = getattr(patch.module, patch.name)
        setattr(patch.module, patch.name, _dispatch(_originals[key], patch))


def original(module: ModuleType, name: str) -> Callable[..., object]:
    """What `module.name` was before `install` replaced it."""
    return _originals.get((module.__name__, name)) or getattr(module, name)


def uses[T](*patches: Patch[..., object]) -> Callable[[type[T]], type[T]]:
    """Class decorator: the patches hold for the duration of the model's `__call__`.

    Scoping to the forward is what keeps a replacement from reaching a model that has no
    kernel for its shapes, which a process-wide install cannot do. mlx's laziness does not
    leak across the boundary -- the patch decides which op enters the graph, and that
    happens while `__call__` runs, not when the result is evaluated.
    """

    def decorate(model: type[T]) -> type[T]:
        inner = model.__call__

        @wraps(inner)
        def call(self: T, *args: object, **kwargs: object) -> object:
            with patched(*patches):
                return inner(self, *args, **kwargs)

        model.__call__ = call
        return model

    return decorate


class patched(ContextDecorator):
    """Install patches for a block, restoring the originals on exit.

    Not usable as a decorator on a generator function: the wrapper would install and
    restore around the call that builds the generator, leaving nothing patched while it
    is iterated.
    """

    def __init__(self, *patches: Patch[..., object]) -> None:
        self.patches = patches
        self.stack = ExitStack()

    def __enter__(self) -> None:
        for patch in self.patches:
            self.stack.enter_context(_scoped(patch))

    def __exit__(self, *_: object) -> None:
        self.stack.close()


_originals: dict[tuple[str, str], Callable[..., object]] = {}


def _dispatch(
    original: Callable[..., object], patch: Patch[..., object]
) -> Callable[..., object]:
    def call(*args: object, **kwargs: object) -> object:
        if patch.applies(*args, **kwargs):
            return patch.replacement(*args, **kwargs)
        return original(*args, **kwargs)

    return call


@contextmanager
def _scoped(patch: Patch[..., object]) -> Iterator[None]:
    previous = getattr(patch.module, patch.name)
    setattr(patch.module, patch.name, _dispatch(previous, patch))
    try:
        yield
    finally:
        setattr(patch.module, patch.name, previous)

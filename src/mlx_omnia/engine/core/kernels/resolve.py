"""The one resolution rule every primitive's delegator shares.

A primitive's `_STRATEGIES` is an ordered tuple of strategy classes, and `resolve` walks
it handing each `build` the full declaration: the first strategy that returns an instance
wins. The classes themselves are the registry, so the classmethod is what a strategy has
to expose, and a tuple whose `build` signatures drift apart stops type-checking here
instead of at eighteen call sites.

Totality is still the last entry's job: every primitive ends its tuple with a default
that accepts everything, so the `StopIteration` a total tuple can never raise is left to
surface loudly if one day it can.
"""

from collections.abc import Iterable
from typing import Protocol


class Builds[**P, S](Protocol):
    """What a strategy class looks like from the outside: `build` bound to the class,
    returning an instance or declining with `None`."""

    def build(self, *args: P.args, **kwargs: P.kwargs) -> S | None: ...


RESOLVED: set[str] = set()
"""Modules of the strategies that built in this process. Every strategy module is imported
to form a `_STRATEGIES` tuple whether or not it ever builds, so imports alone cannot say
which kernels a run executed — this can, and the bench's results store reads it to decide
which files a stored measurement actually depends on."""


def resolve[**P, S](strategies: Iterable[Builds[P, S]], /, *args: P.args, **kwargs: P.kwargs) -> S:
    """Order is preference: the first strategy that builds wins."""
    built = next(
        built for strategy in strategies if (built := strategy.build(*args, **kwargs)) is not None
    )
    RESOLVED.add(type(built).__module__)
    return built

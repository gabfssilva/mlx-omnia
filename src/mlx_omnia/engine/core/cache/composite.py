"""A layer whose state is several named caches advancing together."""

from collections.abc import Callable, Mapping

import mlx.core as mx

from mlx_omnia.engine.core.cache.layer import LayerCache
from mlx_omnia.engine.core.cache.layout import Layout


class Composite(LayerCache):
    """A layer whose state is several named caches advancing together.

    A hybrid's attention beside its indexer, a compressor beside its local window: the layer
    presents one offset and one set of tensors, and the parts answer for their own. Written
    once here because the delegation is the same every time — the families that carry it
    differ in which parts they hold and in nothing else, and four hand-written copies of it
    is four places for a part to be forgotten out of exactly one of them.
    """


    @property
    def parts(self) -> Mapping[str, LayerCache]:
        raise NotImplementedError("a composite cache names its parts")

    @property
    def is_replayable(self) -> bool:
        return all(part.is_replayable for part in self.parts.values())

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for part in self.parts.values())

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(tensor for part in self.parts.values() for tensor in part.tensors)

    @property
    def signature(self) -> dict[str, object]:
        held: dict[str, object] = {}
        for name, part in self.parts.items():
            held |= {f"{name}.{key}": value for key, value in part.signature.items()}
        return held

    @property
    def layout(self) -> Mapping[str, Layout]:
        held: dict[str, Layout] = {}
        for name, part in self.parts.items():
            held |= {f"{name}.{key}": value for key, value in part.layout.items()}
        return held

    def checkpoint(self) -> Callable[[], None]:
        restores = [part.checkpoint() for part in self.parts.values()]

        def restore() -> None:
            for undo in restores:
                undo()

        return restore

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        held: dict[str, mx.array] = {}
        for name, part in self.parts.items():
            held |= {f"{name}.{key}": value for key, value in part.stored(start, stop).items()}
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        for name, part in self.parts.items():
            prefix = f"{name}."
            part.restore(
                offset,
                {
                    key.removeprefix(prefix): value
                    for key, value in tensors.items()
                    if key.startswith(prefix)
                },
            )


# ── ragged batch adapters ────────────────────────────────────────────────
#
# One layer of N sequences stepping together. They are caches themselves — the trunk reads
# them through the same `__call__` it reads a single sequence's through, which is what lets
# one forward serve prefill, decode and batch. What they are not is a cache of anything:
# they own no rows, and every write lands in the rows they were built from.

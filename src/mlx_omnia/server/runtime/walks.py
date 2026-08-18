"""What is under a model's facades, found by walking them."""

from collections.abc import Collection
from typing import Protocol, runtime_checkable

import mlx.nn as nn

from mlx_omnia.engine.core import api
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.model import Wrapping
from mlx_omnia.engine.quantizing import Quantizing


@runtime_checkable
class _Stopping(Protocol):
    stop: Collection[int]


def stop_ids(model: object) -> Collection[int]:
    """Every id that ends a turn for this model, off the facade that holds it — the walk
    `tokenizer_of` does, for the same reason: what resolved the set is the load, and out here
    there is a `LanguageModel[ModelInput]` and nothing else. It becomes the grammar's end, so a
    constrained run stops where a free one does."""
    while not isinstance(model, _Stopping):
        if not isinstance(model, Wrapping):
            return ()
        model = model.model
    return model.stop


def drafter(model: object) -> nn.Module | None:
    """The second checkpoint under this model, when one was paired with it. It is not in the
    model's own tree — two checkpoints are two trees — so nothing that walks `tree` ever weighs
    it, and residency has to ask for it by name."""
    while not isinstance(model, api.Drafting):
        if not isinstance(model, Wrapping):
            return None
        model = model.model
    return model.drafter


def tree(model: object) -> nn.Module | None:
    """The outermost `nn.Module` under the wrappers, or `None` when there is none — a test double
    is a `LanguageModel` and holds no tensors at all."""
    while not isinstance(model, nn.Module):
        if not isinstance(model, Wrapping):
            return None
        model = model.model
    return model


def trunk_of(model: object) -> tuple[Wrapping, api.LanguageModel[LayerCache]] | None:
    """The facade holding the checkpoint's own trunk, and that trunk with any policy already on
    it peeled off — the pair a KV policy is substituted through.

    The walk `tokenizer_of` and `stop_ids` do, stopping one level higher: what a policy replaces
    is the object the text facade hands to `stream_ids`, and the facade above it owns the
    tokenizer, the template and the prefix trie. Replacing in place is what keeps that trie: a
    facade rebuilt per request would prefill every turn from scratch.

    `None` when nothing under the wrappers answers `make_cache` — a test double, or a model that
    does not decode through a cache at all.
    """
    while isinstance(model, Wrapping):
        held = model.model
        if isinstance(held, Quantizing):
            return model, held.model
        if isinstance(held, api.LanguageModel):
            return model, held
        model = held
    return None

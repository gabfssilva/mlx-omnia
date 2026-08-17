"""Per-layer cache state, and everything the engine builds on it.

One module per kind of state rather than one per class: the promotion a class does, the
ragged adapter it hands back and the spans it cuts are the same subject, and a class and
its adapter that live apart is an import cycle written down. `KVCache.fixed` names
`RingKVCache`, `RingKVCache.promote` takes a `KVCache`, and `LayerCache.batched` names its
own adapter — split by class, every one of those becomes a deferred import.

- `layout` — how a tensor composes along the spans of a conversation
- `layer` — the contract, and the adapter for a layer that holds nothing
- `kv` — the growing buffer, the fixed one, and the ring a window keeps
- `recurrent` — a short conv's window and the DeltaNet's state, growing and fixed
- `shared` — a layer that reads another's rows
- `composite` — a layer whose state is several named caches at once
"""

from mlx_omnia.engine.core.cache.composite import Composite
from mlx_omnia.engine.core.cache.kv import (
    HEADROOM,
    BatchedKVCache,
    FixedKVCache,
    KVCache,
    RingKVCache,
    fit,
    regrow,
    reserve,
)
from mlx_omnia.engine.core.cache.layer import (
    BatchedLayerCache,
    LayerCache,
    batch,
    row_mask,
)
from mlx_omnia.engine.core.cache.layout import Layout, Rows, Snapshot
from mlx_omnia.engine.core.cache.recurrent import (
    BatchedConvCache,
    BatchedDeltaCache,
    ConvCache,
    DeltaCache,
    FixedConvCache,
    FixedDeltaCache,
)
from mlx_omnia.engine.core.cache.shared import BatchedSharedKVReader, SharedKVReader

__all__ = [
    "HEADROOM",
    "BatchedConvCache",
    "BatchedDeltaCache",
    "BatchedKVCache",
    "BatchedLayerCache",
    "BatchedSharedKVReader",
    "Composite",
    "ConvCache",
    "DeltaCache",
    "FixedConvCache",
    "FixedDeltaCache",
    "FixedKVCache",
    "KVCache",
    "LayerCache",
    "Layout",
    "RingKVCache",
    "Rows",
    "SharedKVReader",
    "Snapshot",
    "batch",
    "fit",
    "regrow",
    "reserve",
    "row_mask",
]

"""The models on disk: the Hugging Face cache and what `quantize=` leaves behind.

An entry's id is what `mlx_omnia.load` takes back — a hub repository id, or the directory of
a quantized entry. What decides that a directory is a model is the `weight_map` of
`model.safetensors.index.json`: an interrupted download keeps its config and some of its
shards, and listing it would turn a failed download into a load error at request time.

`bytes_per_token` is priced off the safetensors headers and never off the lazy tree, which is
built before `nn.quantize` and would price a 4-bit checkpoint at its dense shapes. A resident
entry does not use the estimate at all: the engine walked the real tree at load.
"""

from __future__ import annotations

from mlx_omnia.server.services.catalog.errors import (
    ImageSizeInvalid,
    ModelResident,
    NoModelCard,
    NoSuchAsset,
    NotTraceable,
    TakesNoImage,
    UnknownModel,
)
from mlx_omnia.server.services.catalog.headers import bytes_per_token as _bytes_per_token
from mlx_omnia.server.services.catalog.headers import complete as _complete
from mlx_omnia.server.services.catalog.headers import stored_carrier, weights_dtype
from mlx_omnia.server.services.catalog.headers import tensors_of as _tensors
from mlx_omnia.server.services.catalog.reads import (
    Forget,
    asset,
    blueprint,
    card,
    files,
    image_cost,
    model,
    models,
    remove,
    resident_bytes,
)
from mlx_omnia.server.services.catalog.scan import (
    HUB_CACHE,
    QUANTIZED_CACHE,
    CatalogEntry,
    CheckpointFile,
    context_of,
    defaults_of,
    entry_of,
    kv_head_width,
    scan,
    stamp_of,
)

__all__ = [
    "HUB_CACHE",
    "QUANTIZED_CACHE",
    "CatalogEntry",
    "CheckpointFile",
    "Forget",
    "ImageSizeInvalid",
    "ModelResident",
    "NoModelCard",
    "NoSuchAsset",
    "NotTraceable",
    "TakesNoImage",
    "UnknownModel",
    "_bytes_per_token",
    "_complete",
    "_tensors",
    "asset",
    "blueprint",
    "card",
    "context_of",
    "defaults_of",
    "entry_of",
    "files",
    "image_cost",
    "kv_head_width",
    "model",
    "models",
    "remove",
    "resident_bytes",
    "scan",
    "stamp_of",
    "stored_carrier",
    "weights_dtype",
]

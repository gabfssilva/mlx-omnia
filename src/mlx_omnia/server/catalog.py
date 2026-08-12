"""The models on disk: the Hugging Face cache and what `quantize=` leaves behind.

An entry's id is what `mlx_omnia.load` takes back. A hub repository is its repository id,
listed once at the revision `refs/main` names; a quantized entry has no repository name of
its own — `task.py` derives its folder from a fingerprint of the source — so its id is the
directory itself, which is the other thing `load` accepts.

What decides that a directory is a model is the `weight_map` of
`model.safetensors.index.json`. An interrupted download keeps its config and some of its
shards, and listing it turns a failed download into a load error at request time.

`bytes_per_token` is priced per entry, not cached: the arithmetic reads the safetensors
headers and never a shard, which on this machine's cache is 0.9 ms per entry warm (5.4 ms
cold on a 4-shard 17 GB checkpoint) against the 25 ms the walk already costs for 74
entries. The bytes have to come off the headers and not off the lazy tree, which is built
before `nn.quantize` and would price a 4-bit checkpoint at its dense shapes; what does come
off that tree is how many rows of an expert stack a step reads (`_slots`), because that is
the architecture's to say and no config key is a reliable name for it. Building one costs
3 ms on the median entry and 21 ms on the deepest, so it is cached per config.

Priced this way the number is still an estimate, and a resident entry does not use it: the
engine walked the real tree at load, and that is what the handlers answer with.

The handlers are sync on purpose: the scan stats a few hundred files and a delete can
unlink tens of gigabytes, so they run in the threadpool instead of stalling the loop —
and with it the token stream of whatever is generating.
"""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from typing import Annotated, NotRequired, TypedDict

import huggingface_hub.constants
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from mlx_omnia.engine.checkpoint import QuantizationJson, SamplingDefaults, sampling_defaults
from mlx_omnia.engine.footprint import expert_slots
from mlx_omnia.engine.task import source as checkpoint_source
from mlx_omnia.server.engine import Engine
from mlx_omnia.server.store import Store

HUB_CACHE = Path(huggingface_hub.constants.HF_HUB_CACHE)
QUANTIZED_CACHE = Path.home() / ".cache" / "mlx_omnia" / "quantized"


class _QuantizationJson(TypedDict):
    """Both shapes the block is written in: mlx's global one and the per-leaf plan
    `save_quantized` writes. Neither key is required — a dense config has no block at all,
    and other engines' per-path overrides sit next to the global keys under names we do not
    read."""

    bits: NotRequired[int]
    mode: NotRequired[str]
    leaves: NotRequired[dict[str, QuantizationJson]]


class _ShapeJson(TypedDict):
    """The keys the cache arithmetic reads, all of them transformers' own names and none of
    them any architecture's private vocabulary. They appear at the root or under
    `text_config`, which is where qwen3.5 keeps everything a text checkpoint declares."""

    max_position_embeddings: NotRequired[int]
    vocab_size: NotRequired[int]
    tie_word_embeddings: NotRequired[bool]
    hidden_size: NotRequired[int]
    num_hidden_layers: NotRequired[int]
    num_attention_heads: NotRequired[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    sliding_window: NotRequired[int | None]
    use_sliding_window: NotRequired[bool]
    layer_types: NotRequired[list[str]]
    full_attn_idxs: NotRequired[list[int]]
    layer_group_size: NotRequired[int]
    kv_lora_rank: NotRequired[int]
    qk_rope_head_dim: NotRequired[int]


class _TextConfigJson(_ShapeJson):
    pass


class _ConfigJson(_ShapeJson):
    model_type: NotRequired[str]
    dtype: NotRequired[str]
    torch_dtype: NotRequired[str]
    quantization: NotRequired[_QuantizationJson]
    text_config: NotRequired[_TextConfigJson]


class _TensorJson(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class _IndexJson(TypedDict):
    weight_map: dict[str, str]


_NO_TEXT: _TextConfigJson = {}

_EXPERT_SLICE = re.compile(r"(?:^|\.)experts\.\d+\.")
"""A conversion that ships one tensor per expert instead of the stack (LFM2, Laguna). The
loader stacks them into `[E, out, in]`; either way a step reads `k` of the `E` rows."""

_FLOATS = ("BF16", "F16", "F32", "F64")


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    directory: Path
    """The checkpoint: `config.json` and the shards."""
    store: Path
    """What the id owns on disk, and what DELETE removes. A hub repository owns its whole
    folder: the snapshot is symlinks into `blobs/`, and unlinking it alone frees nothing."""
    architecture: str
    quantization: str | None
    """The label the config declares — `None` when it declares none, which is dense."""
    dtype: str | None
    context: int | None
    bytes_on_disk: int
    defaults: SamplingDefaults = field(default_factory=SamplingDefaults)
    """How the checkpoint's own `generation_config.json` says it wants to be sampled. It is
    what a request that names no knob gets, and what the app fills its sliders with when a
    model is picked — an empty one is a checkpoint that says nothing, and there the dialect's
    defaults are all there is."""
    bytes_per_token: int | None = None
    """Bytes a decode step reads, and the denominator of every "% of the ceiling" the house
    reports. `None` when the shards' headers cannot be read."""
    kv_bytes_per_token: int | None = None
    """What the cache grows by per token, over every attending layer. The other half of the
    ceiling — weights amortise over the streams of a batch, this does not — and what decides
    whether a shape fits at all. `None` when the config does not answer."""
    attention_window: int | None = None
    """Where the cache stops growing. `None` for full attention, and also for a checkpoint
    that only slides on some of its layers: there the full ones keep growing, and a window
    reported for the whole model would price a saving it does not have."""
    vocab_size: int | None = None
    """What decides whether two checkpoints' logits can be subtracted at all."""
    shape: str | None = None
    """A print of the architecture — enough to tell a fine-tune's twin from a different
    model. It validates a fidelity pair somebody declared; it never invents kinship, since
    a fine-tune's print is identical to its base's."""
    resident: bool = False
    """Filled by the handler from the engine, not by the scan."""


def _label(bits: int, mode: str) -> str:
    return f"{bits}-bit" if mode == "affine" else mode


def _quantization(config: _ConfigJson) -> str | None:
    """What the config says it is, never what the tensors are. A per-leaf plan reports the
    width its leaves agree on, and `mixed` when they don't."""
    block = config.get("quantization")
    if block is None:
        return None
    if (leaves := block.get("leaves")) is not None:
        labels = {_label(leaf["bits"], leaf.get("mode", "affine")) for leaf in leaves.values()}
        return labels.pop() if len(labels) == 1 else "mixed"
    bits = block.get("bits")
    return None if bits is None else _label(bits, block.get("mode", "affine"))


def _tensors(directory: Path) -> list[tuple[str, _TensorJson, int]] | None:
    """Every tensor's name, header entry and physical bytes, read off the safetensors
    headers with no shard mapped: each file opens with a little-endian u64 holding the
    header's length, and `__metadata__` sits among the tensors carrying no offsets.

    `None` when a header does not parse. The scan lists what is on disk, and a file that is
    not safetensors costs the entry its numbers, not its listing. The length is checked
    against the file before it is read: on arbitrary bytes those first eight are an
    arbitrary u64, and reading it is a `MemoryError` rather than a bad JSON.
    """
    found: list[tuple[str, _TensorJson, int]] = []
    for shard in sorted(directory.glob("model*.safetensors")):
        try:
            with shard.open("rb") as file:
                length = int.from_bytes(file.read(8), "little")
                if length > shard.stat().st_size - 8:
                    return None
                header: dict[str, _TensorJson] = json.loads(file.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                begin, end = entry["data_offsets"]
                found.append((name, entry, end - begin))
        except (ValueError, KeyError):
            return None
    return found


def weights_dtype(directory: Path) -> str | None:
    """The dtype the shards' floating tensors carry, under the name safetensors gives it
    (`BF16`, `F16`, `F32`). It is what a quantization plan has to be priced in, and the
    config is not a reliable source for it: qwen3.5 nests `dtype` under `text_config` and
    some conversions declare none at all. The largest tensor decides — a checkpoint mixing
    float widths across its weights does not exist, and the odd float32 norm beside
    bfloat16 matrices must not."""
    tensors = _tensors(directory)
    if not tensors:
        return None
    floating = [(size, entry["dtype"]) for _, entry, size in tensors if entry["dtype"] in _FLOATS]
    return max(floating)[1] if floating else None


@lru_cache(maxsize=256)
def _slots(directory: Path, stamp: int) -> Mapping[int, int] | None:
    """How many rows of an expert stack a step reads, by the stack's depth, off the tree the
    architecture builds from its own config — no tensor is read and no key is guessed at.

    `None` when there is no such tree: an architecture this engine does not load, or a
    config it cannot parse. Then a checkpoint that stacks anything is not priced at all,
    because the alternative is what reading a key and defaulting to zero used to do
    silently — every expert of the stack charged on every token, and a decode rate reported
    against a ceiling an order of magnitude too low.

    Keyed by the config's mtime, which is what the tree is a function of. Every failure is
    the same answer, so the catch is total: a directory the catalog lists is not a directory
    the engine promised to load.
    """
    del stamp
    try:
        return expert_slots(checkpoint_source(directory, local_files_only=True).pending.model)
    except Exception:
        return None


def _bytes_per_token(directory: Path, config: _ConfigJson) -> int | None:
    """What one decode step reads, summed from the headers: `k` of the `E` rows of a stack
    and never the whole pile, the untied lm_head whole, the embedding table only when it is
    also the head — a step gathers one row of it, which is free — and never the vision
    tower's position table, which a text step does not reach.

    A stack is priced by `_slots`, keyed by its leading dimension; a checkpoint that ships
    one tensor per expert instead of the stack is grouped back into one and priced by how
    many of them there are. Something stacked that no tree answers for is not priced at all
    — that is the difference between an estimate and an invented number.

    Two facts stay with the config, and neither is any architecture's own. Whether the head
    is tied, which transformers defaults to true and gpt2 leaves unsaid. And the vocabulary,
    which is what tells the embedding table apart from the head: both are `[vocab, ·]`, and
    only one of them is named `lm_head`.
    """
    text = config.get("text_config", _NO_TEXT)
    vocab = text.get("vocab_size") or config.get("vocab_size") or 0
    nested = text.get("tie_word_embeddings")
    tied = config.get("tie_word_embeddings", True) if nested is None else nested
    tensors = _tensors(directory)
    if tensors is None:
        return None
    read = 0
    stacked: list[tuple[int, int]] = []
    sliced: dict[str, list[int]] = {}
    for name, entry, size in tensors:
        shape = entry["shape"]
        # gpt2's causal-mask buffer, which its loader drops before the tree ever sees it.
        if name.endswith(".attn.bias"):
            continue
        head = name.split(".")[-2:-1] == ["lm_head"]
        if tied and head:
            continue
        # The tables a step gathers rows of instead of reading: the tokens', when it is not
        # also the head, and the vision tower's positions — which a text step does not
        # reach at all, and which the tree holds as the `nn.Embedding` it is.
        if name.split(".")[-2:-1] == ["pos_embed"]:
            continue
        if not tied and not head and len(shape) >= 2 and shape[0] == vocab:
            continue
        if _EXPERT_SLICE.search(name):
            sliced.setdefault(_EXPERT_SLICE.sub(".experts.", name), []).append(size)
        elif len(shape) >= 3:
            stacked.append((shape[0], size))
        else:
            read += size
    if not stacked and not sliced:
        return read
    slots = _slots(directory, (directory / "config.json").stat().st_mtime_ns)
    if slots is None:
        return None
    for rows, size in stacked:
        # A depth no stack has is not a stack: a conv1d's weight is three-dimensional too,
        # and a step reads it whole.
        read += size * slots[rows] // rows if rows in slots else size
    for group in sliced.values():
        rows = len(group)
        if rows not in slots:
            return None
        read += sum(group) * slots[rows] // rows
    return read


_ATTENDING = frozenset({"full_attention", "sliding_attention", "attention"})
"""What transformers calls a layer that keeps a key-value cache. Everything else a
`layer_types` names — `linear_attention`, `conv`, `mamba`, `recurrent` — holds a state of
fixed size that does not grow with the context, so it is not in this arithmetic at all."""

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}


def _shape(config: _ConfigJson) -> _ShapeJson:
    """Where the architecture's numbers are, root or nested. A key present at both is the
    nested one's: `text_config` is the text tower's own declaration, and the root of a
    multimodal config describes the whole thing."""
    text = config.get("text_config")
    if text is None:
        return config
    merged: _ShapeJson = {**config, **text}
    return merged


def _attending_layers(shape: _ShapeJson) -> int | None:
    """How many layers keep a growing cache. A hybrid says so itself, in one of three forms —
    `layer_types` names every layer, `full_attn_idxs` lists the attending ones, and
    `layer_group_size` gives the stride of a repeating group — and counting all of them
    instead would charge a recurrent trunk for a cache it does not have.

    The stride form: one layer of each group attends, the one that closes it, and a trailing
    group the stride does not fill is attention throughout. So the count is the number of
    whole groups plus whatever is left over — `42` layers by `6` is `7`, not `42`.
    """
    layers = shape.get("num_hidden_layers")
    if layers is None:
        return None
    if (types := shape.get("layer_types")) is not None:
        return sum(1 for kind in types if kind in _ATTENDING)
    if (indices := shape.get("full_attn_idxs")) is not None:
        return len(indices)
    if (group := shape.get("layer_group_size")) is not None and group > 0:
        return layers // group + layers % group
    return layers


def _kv_head_width(shape: _ShapeJson) -> int | None:
    """How wide one row of one attending layer's cache is.

    Two cache shapes, both transformers' own declaration and neither any one port's: a
    latent cache (`kv_lora_rank`), whose row is one compressed vector plus the rotated key,
    and the ordinary one, whose row is one head — declared, or the hidden size over the heads
    that share it.
    """
    if (rank := shape.get("kv_lora_rank")) is not None:
        rope = shape.get("qk_rope_head_dim")
        return None if rope is None else rank + rope
    if (head_dim := shape.get("head_dim")) is not None:
        return head_dim
    heads = shape.get("num_attention_heads")
    hidden = shape.get("hidden_size")
    if heads is None or heads == 0 or hidden is None:
        return None
    return hidden // heads


def _kv_elements_per_layer(shape: _ShapeJson) -> int | None:
    """Elements one token adds to one attending layer's cache: one row for a latent cache,
    and a key plus a value per key-value head for the ordinary one."""
    width = _kv_head_width(shape)
    if width is None:
        return None
    if shape.get("kv_lora_rank") is not None:
        return width
    heads = shape.get("num_attention_heads")
    if heads is None:
        return None
    return 2 * shape.get("num_key_value_heads", heads) * width


def kv_head_width(model_id: str) -> int | None:
    """The width a compressed KV cache's formats have to close their groups on.

    A `QuantizedKVCache` packs along the head and never along tokens, so the axis the
    arithmetic is about is one row of one layer — the same figure `_kv_elements_per_layer`
    counts key-value heads of. `None` when the config does not answer: an id this daemon does
    not list, or a checkpoint declaring its shape under names that are not transformers' own.
    """
    entry = next((found for found in scan() if found.id == model_id), None)
    if entry is None:
        return None
    config: _ConfigJson = json.loads((entry.directory / "config.json").read_text())
    return _kv_head_width(_shape(config))


def _kv_bytes_per_token(directory: Path, config: _ConfigJson) -> int | None:
    """What one token costs the cache, over the whole model.

    The width comes off the config and the element size off the shards: the cache carries
    the activations' dtype, which is the checkpoint's float dtype — a 4-bit checkpoint
    computes and caches in the bfloat16 its norms are stored in, not in four bits. A config
    that does not answer gets `None`, the way `_bytes_per_token` already does with a stack
    no tree accounts for.
    """
    shape = _shape(config)
    layers = _attending_layers(shape)
    elements = _kv_elements_per_layer(shape)
    if layers is None or elements is None:
        return None
    width = _DTYPE_BYTES.get(weights_dtype(directory) or "")
    return None if width is None else layers * elements * width


def _attention_window(shape: _ShapeJson) -> int | None:
    """The context past which the cache stops growing, and `None` whenever it does not stop.

    Only when *every* attending layer slides. A checkpoint that alternates — gpt-oss keeps
    a full layer beside each sliding one — has a cache that still grows linearly, and
    reporting the window here would price a saving the model does not have.
    """
    window = shape.get("sliding_window")
    if window is None or not shape.get("use_sliding_window", True):
        return None
    types = shape.get("layer_types")
    if types is None:
        return window
    attending = [kind for kind in types if kind in _ATTENDING]
    return window if attending and all(kind == "sliding_attention" for kind in attending) else None


def _print(config: _ConfigJson) -> str | None:
    """`model_type` and the four numbers that decide whether two checkpoints are the same
    architecture. Not a fingerprint of the weights: a fine-tune prints the same as its base,
    which is exactly why this validates a declared pair and never proposes one."""
    shape = _shape(config)
    architecture = config.get("model_type")
    layers = shape.get("num_hidden_layers")
    hidden = shape.get("hidden_size")
    heads = shape.get("num_attention_heads")
    vocab = shape.get("vocab_size")
    if architecture is None or None in (layers, hidden, heads, vocab):
        return None
    return f"{architecture}/L{layers}/H{hidden}/A{heads}/V{vocab}"


def _complete(directory: Path) -> bool:
    """The `weight_map` is the list the loader will ask for, so it is the list that
    decides. `is_file` follows the link, which is what catches a snapshot pointing at a
    blob the download never finished."""
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return (directory / "model.safetensors").is_file()
    weights: _IndexJson = json.loads(index.read_text())
    return all((directory / shard).is_file() for shard in set(weights["weight_map"].values()))


_Stamp = tuple[tuple[str, int, int], ...]


def stamp_of(model_id: str) -> str | None:
    """One string that moves when the checkpoint under an id does, or `None` for an id the
    scan does not list. What it is for is keying anything derived from a checkpoint's own
    numbers: a file written under an id and read back after a shard was replaced is the wrong
    answer, arriving without an error."""
    entry = next((entry for entry in scan() if entry.id == model_id), None)
    if entry is None:
        return None
    digest = hashlib.sha256()
    for name, size, mtime in _stamp(Path(entry.directory)):
        digest.update(f"{name}\0{size}\0{mtime}\0".encode())
    return digest.hexdigest()


def _stamp(directory: Path) -> _Stamp:
    """Every file the entry is a function of, by name, size and mtime. A hub snapshot is
    links into `blobs/` and `stat` follows them, so a shard landing moves the stamp; a link
    whose blob is not there yet has no stat at all, and it moves when the blob appears.

    This is the walk `bytes_on_disk` already paid for, so pricing an entry costs the stat of
    its directory and nothing else once it is cached."""
    found: list[tuple[str, int, int]] = []
    for path in sorted(directory.iterdir()):
        try:
            info = path.stat()
        except OSError:
            continue
        if S_ISREG(info.st_mode):
            found.append((path.name, info.st_size, info.st_mtime_ns))
    return tuple(found)


@lru_cache(maxsize=256)
def _build(model_id: str, directory: Path, store: Path, stamp: _Stamp) -> CatalogEntry | None:
    """Keyed by the stamp, which is what every number below is read from: the headers of 77
    checkpoints are 470 ms of json and regex, and nothing in them moves while the files do
    not. `None` is cached too — an interrupted download is re-listed as often as a complete
    one, and it stops being `None` when its last shard moves the stamp."""
    config_path = directory / "config.json"
    if not config_path.is_file() or not _complete(directory):
        return None
    config: _ConfigJson = json.loads(config_path.read_text())
    architecture = config.get("model_type")
    if architecture is None:
        return None
    context = config.get("max_position_embeddings")
    if context is None and (text := config.get("text_config")) is not None:
        context = text.get("max_position_embeddings")
    return CatalogEntry(
        id=model_id,
        directory=directory,
        store=store,
        architecture=architecture,
        quantization=_quantization(config),
        dtype=config.get("dtype") or config.get("torch_dtype"),
        context=context,
        defaults=sampling_defaults(directory),
        bytes_on_disk=sum(size for _, size, _ in stamp),
        bytes_per_token=_bytes_per_token(directory, config),
        kv_bytes_per_token=_kv_bytes_per_token(directory, config),
        attention_window=_attention_window(_shape(config)),
        vocab_size=_shape(config).get("vocab_size"),
        shape=_print(config),
    )


def _entry(model_id: str, directory: Path, store: Path) -> CatalogEntry | None:
    return _build(model_id, directory, store, _stamp(directory))


def _head(repository: Path) -> Path | None:
    """The revision `load` would resolve. A snapshot fetched by sha has no `refs/main`, so
    the fallback is the most recent one rather than nothing."""
    snapshots = repository / "snapshots"
    if not snapshots.is_dir():
        return None
    reference = repository / "refs" / "main"
    sha = reference.read_text().strip() if reference.is_file() else ""
    if sha and (head := snapshots / sha).is_dir():
        return head
    revisions = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    return revisions[-1] if revisions else None


def _hub(root: Path) -> Iterator[CatalogEntry]:
    if not root.is_dir():
        return
    for repository in root.glob("models--*"):
        head = _head(repository)
        if head is None:
            continue
        model_id = repository.name.removeprefix("models--").replace("--", "/")
        if (entry := _entry(model_id, head, repository)) is not None:
            yield entry


def _quantized(root: Path) -> Iterator[CatalogEntry]:
    if not root.is_dir():
        return
    for source in root.iterdir():
        if not source.is_dir():
            continue
        for directory in source.iterdir():
            # `write_entry` stages under `.tmp-*` next to the entry it is about to become.
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            if (entry := _entry(str(directory), directory, directory)) is not None:
                yield entry


def scan() -> list[CatalogEntry]:
    return sorted([*_hub(HUB_CACHE), *_quantized(QUANTIZED_CACHE)], key=lambda entry: entry.id)


@lru_cache(maxsize=256)
def context_of(model_id: str) -> int | None:
    """The `max_position_embeddings` the checkpoint declares, by id — what a dialect caps a
    generation with. `None` for an id the disk does not answer for, which is every test
    double. Cached because it rides the request path and a checkpoint's context does not
    move; the scan behind it prices every entry, and the cache is what keeps that off every
    chat turn."""
    for entry in scan():
        if entry.id == model_id:
            return entry.context
    return None


@lru_cache(maxsize=256)
def defaults_of(model_id: str) -> SamplingDefaults:
    """The checkpoint's sampling defaults, by id, for the dialects to fill the knobs a
    request left out. Empty for an id the disk does not answer for — every test double, and
    a model deleted mid-request — which is the same thing as a checkpoint that declares
    nothing. Cached for the reason `context_of` is."""
    for entry in scan():
        if entry.id == model_id:
            return entry.defaults
    return SamplingDefaults()


async def resident_models(request: Request) -> Mapping[str, int | None]:
    """The ids the engine holds loaded, each with what its own tree says a decode step reads
    — the runtime facts a disk catalog does not have. Async so it reads the engine's dict on
    the loop that mutates it."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return {model_id: entry.active_bytes for model_id, entry in engine.residency.items()}


Resident = Annotated[Mapping[str, int | None], Depends(resident_models)]


def _store(request: Request) -> Store:
    """The daemon's own database, off the app. Declared here rather than imported from
    `profiles`: that module reads this one, and the cycle closes in either direction."""
    store = request.app.state.store
    assert isinstance(store, Store)
    return store


StoreDep = Annotated[Store, Depends(_store)]

router = APIRouter()


def _loaded(entry: CatalogEntry, resident: Mapping[str, int | None]) -> CatalogEntry:
    """A resident entry answers with the walk over the tree that is serving the requests,
    not with the estimate off the headers: a checkpoint ships blocks the loader drops
    (speculative-decoding heads) and tensors it fuses, and the headers cannot know which.
    `None` is a model with no tree under it, and there the estimate is all there is."""
    if entry.id not in resident:
        return entry
    active = resident[entry.id]
    return replace(
        entry,
        resident=True,
        bytes_per_token=entry.bytes_per_token if active is None else active,
    )


def _find(model_id: str) -> CatalogEntry:
    for entry in scan():
        if entry.id == model_id:
            return entry
    raise HTTPException(status_code=404, detail=f"{model_id!r} is not in the catalog")


@router.get("/admin/models")
def models(loaded: Resident, resident: bool = False) -> list[CatalogEntry]:
    entries = [_loaded(entry, loaded) for entry in scan()]
    return [entry for entry in entries if entry.resident or not resident]


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int
    """Through the symlink: a hub snapshot is links into `blobs/`, and the listing answers
    for the bytes, not the link."""


@router.get("/admin/models/{model_id:path}/card", response_class=PlainTextResponse)
def card(model_id: str) -> str:
    """The checkpoint's README, raw — rendering it is the client's job."""
    path = _find(model_id).directory / "README.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{model_id!r} has no model card")
    return path.read_text()


@router.get("/admin/models/{model_id:path}/files")
def files(model_id: str) -> list[CheckpointFile]:
    directory = _find(model_id).directory
    return sorted(
        (
            CheckpointFile(name=path.name, size=path.stat().st_size)
            for path in directory.iterdir()
            if path.is_file()
        ),
        key=lambda file: file.name,
    )


@router.get("/admin/models/{model_id:path}/assets/{asset:path}")
def asset(model_id: str, asset: str) -> FileResponse:
    """A file the card references relatively (its images). The name resolves inside the
    checkpoint and nowhere else — `..` and absolute paths are refused before the disk is
    asked."""
    target = _find(model_id).directory / asset
    if asset.startswith("/") or ".." in PurePosixPath(asset).parts or not target.is_file():
        raise HTTPException(status_code=404, detail=f"{model_id!r} has no file {asset!r}")
    return FileResponse(target)


@router.get("/admin/models/{model_id:path}")
def model(model_id: str, loaded: Resident) -> CatalogEntry:
    return _loaded(_find(model_id), loaded)


@router.delete("/admin/models/{model_id:path}", status_code=204)
def remove(model_id: str, loaded: Resident, store: StoreDep) -> None:
    from mlx_omnia.server.prefixes import forget

    entry = _find(model_id)
    if model_id in loaded:
        raise HTTPException(
            status_code=409,
            detail=f"{model_id!r} is resident: unload it before deleting it from disk",
        )
    shutil.rmtree(entry.store)
    # The conversations spilled under this id go with the weights. Unloading does not do this
    # — surviving an unload is the whole point of the disk tier — but nothing keyed to a
    # checkpoint outlives the checkpoint: what is left is bytes nobody can name.
    forget(store, model_id)

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, TypedDict, cast

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.checkpoint import MTP_PREFIX, Drafter, Pending, save_quantized
from mlx_omnia.core.cache import LayerCache
from mlx_omnia.footprint import checkpoint_bytes
from mlx_omnia.generate import CausalLM
from mlx_omnia.language import LanguageModel
from mlx_omnia.model import ModelInput
from mlx_omnia.models import (
    afmoe,
    apertus,
    bailing_hybrid,
    bailing_moe,
    bitnet,
    cohere,
    deepseek_v2,
    deepseek_v4,
    ernie4_5,
    ernie4_5_moe,
    exaone4,
    falcon_h1,
    gemma,
    gemma2,
    gemma3,
    gemma3n,
    gemma4,
    glm4,
    glm4_moe,
    gpt2,
    gpt_oss,
    granite,
    hunyuan_dense,
    hy3,
    jamba,
    laguna,
    lfm2,
    llama,
    llama4,
    longcat_flash_ngram,
    mamba2,
    mimo_v2,
    muse_glimmer,
    nemotron_h,
    olmo2,
    olmoe,
    phi3,
    qwen2,
    qwen2_moe,
    qwen3,
    qwen3_5,
    qwen3_next,
    seed_oss,
    smollm3,
    step3p7,
)
from mlx_omnia.models.glm4_moe import dsa as glm_moe_dsa
from mlx_omnia.quant.quantization import (
    ByPath,
    Quantization,
    QuantizationPlan,
    expand_plan,
    infer_quantization,
    inventory,
    quantize_weights,
)

__all__ = [
    "Source",
    "digest",
    "load",
    "load_drafter",
    "provenance",
    "source",
    "tree",
    "write_entry",
]


class _Config(TypedDict):
    model_type: str


def _download_config(
    repository: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    return Path(
        huggingface_hub.hf_hub_download(
            repository,
            "config.json",
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def _download_snapshot(
    repository: str,
    *,
    revision: str | None,
    local_files_only: bool,
    allow_patterns: list[str],
) -> Path:
    return Path(
        huggingface_hub.snapshot_download(
            repository,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    )


class _Packaged(Protocol):
    """What resolving a checkpoint needs of whoever declared it, which is less than being
    a model: the patterns an entry has to carry over, and the split where quantization
    happens. A `Checkpoint` satisfies it and so does a `Drafter`."""

    @property
    def patterns(self) -> tuple[str, ...]: ...

    @property
    def quantize(self) -> Callable[[Path, mx.Dtype | None], Pending[object]] | None: ...


class _Registered(_Packaged, Protocol):
    """What `load` and `tree` read off an architecture's `Checkpoint`. The tree loader
    is typed at `nn.Module`, not at the architecture's own class — a read-only callable
    return is covariant, and that erasure is what lets one dict hold declarations typed
    on nine different trees."""

    @property
    def load(self) -> Callable[[Path, mx.Dtype | None], nn.Module]: ...

    @property
    def task(self) -> Callable[[Path, mx.Dtype | None], LanguageModel[ModelInput]]: ...


_MODEL_SPECS: dict[str, _Registered] = {
    "afmoe": afmoe.CHECKPOINT,
    "apertus": apertus.CHECKPOINT,
    "bailing_hybrid": bailing_hybrid.CHECKPOINT,
    "bailing_moe": bailing_moe.CHECKPOINT,
    "deepseek_v2": deepseek_v2.CHECKPOINT,
    "deepseek_v4": deepseek_v4.CHECKPOINT,
    "bitnet": bitnet.CHECKPOINT,
    "ernie4_5": ernie4_5.CHECKPOINT,
    "ernie4_5_moe": ernie4_5_moe.CHECKPOINT,
    "exaone4": exaone4.CHECKPOINT,
    "falcon_h1": falcon_h1.CHECKPOINT,
    "gemma": gemma.CHECKPOINT,
    "gemma2": gemma2.CHECKPOINT,
    "gemma3_text": gemma3.CHECKPOINT,
    "gemma3n": gemma3n.CHECKPOINT,
    "gemma4": gemma4.CHECKPOINT,
    "gemma4_unified": gemma4.CHECKPOINT,
    "gemma4_assistant": gemma4.CHECKPOINT,
    "cohere": cohere.CHECKPOINT,
    "glm4": glm4.CHECKPOINT,
    "glm4_moe": glm4_moe.CHECKPOINT,
    "glm_moe_dsa": glm_moe_dsa.CHECKPOINT,
    "granite": granite.CHECKPOINT,
    "gpt2": gpt2.CHECKPOINT,
    "gpt_oss": gpt_oss.CHECKPOINT,
    "hunyuan_dense": hunyuan_dense.CHECKPOINT,
    "hunyuan_v1_dense": hunyuan_dense.CHECKPOINT,
    "hy_v3": hy3.CHECKPOINT,
    "jamba": jamba.CHECKPOINT,
    "laguna": laguna.CHECKPOINT,
    "lfm2": lfm2.dense.CHECKPOINT,
    "lfm2_moe": lfm2.moe.CHECKPOINT,
    "llama": llama.CHECKPOINT,
    "llama4": llama4.CHECKPOINT,
    "longcat_flash_ngram": longcat_flash_ngram.CHECKPOINT,
    "mamba2": mamba2.CHECKPOINT,
    "mimo_v2": mimo_v2.CHECKPOINT,
    "muse_glimmer": muse_glimmer.CHECKPOINT,
    "nemotron_h": nemotron_h.CHECKPOINT,
    "olmo2": olmo2.CHECKPOINT,
    "olmoe": olmoe.CHECKPOINT,
    "phi3": phi3.CHECKPOINT,
    "qwen2": qwen2.CHECKPOINT,
    "qwen2_moe": qwen2_moe.CHECKPOINT,
    "qwen3": qwen3.dense.CHECKPOINT,
    "qwen3_5": qwen3_5.CHECKPOINT,
    "qwen3_5_moe": qwen3_5.CHECKPOINT,
    "qwen3_moe": qwen3.moe.CHECKPOINT,
    "qwen3_next": qwen3_next.CHECKPOINT,
    "qwen3_vl": qwen3.vl.CHECKPOINT,
    "qwen3_vl_moe": qwen3.vl.MOE_CHECKPOINT,
    "seed_oss": seed_oss.CHECKPOINT,
    "smollm3": smollm3.CHECKPOINT,
    "step3p7": step3p7.CHECKPOINT,
}

_CACHE = Path.home() / ".cache" / "mlx_omnia" / "quantized"

# Bumped whenever the same plan over the same checkpoint would produce different bits.
_FORMAT_VERSION = 2
"""2: an entry now carries the source's MTP head under `MTP_PREFIX`. Same source, same plan,
different bits — and the difference is exactly the thing an old entry cannot be asked for,
since a head that was never written cannot be read back out."""


def _fingerprint(directory: Path, repository: str | None) -> tuple[str, dict[str, object]]:
    """What the entry was derived from. A repository resolves to the snapshot's commit sha
    (`snapshot_download` names the directory after it), which pins the bits whatever
    revision asked for them. A local directory has no such identity, so the fingerprint is
    a heuristic: name, size and mtime of each safetensors — a rewrite preserving all three
    is not detected. The first element only names the entry's folder for a human; two
    sources sharing it are still separated by the digest."""
    if repository is not None:
        return repository.replace("/", "--"), {
            "repository": repository,
            "commit": directory.name,
        }
    shards = sorted(directory.glob("model*.safetensors"))
    return directory.name, {
        "directory": str(directory.resolve()),
        "shards": [
            (shard.name, shard.stat().st_size, shard.stat().st_mtime_ns) for shard in shards
        ],
    }


def digest(
    fingerprint: Mapping[str, object],
    plan: QuantizationPlan,
    dtype: mx.Dtype | None,
) -> str:
    """The plan enters already expanded, so two selections that resolve to the same leaves
    share the entry; `sort_keys` makes the digest independent of the order the overrides
    were written in. `repr` of a frozen dataclass is total over the formats and needs no
    serializer of its own."""
    payload: dict[str, object] = {
        "format_version": _FORMAT_VERSION,
        "mlx": version("mlx"),
        "source": fingerprint,
        "dtype": str(dtype) if dtype is not None else None,
        "plan": {path: repr(format) for path, format in plan.items()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _auxiliary(source: Path, patterns: Sequence[str]) -> list[str]:
    """Everything the loader reads besides the weights and the config: the entry has to
    load on its own, and a symlink into the hub's snapshot dangles the day it is
    collected."""
    names = {path.name for pattern in patterns for path in source.glob(pattern)}
    return sorted(
        name
        for name in names
        if name != "config.json" and not fnmatch(name, "model*.safetensors")
    )


def write_entry(
    entry: Path,
    checkpoint: Path,
    patterns: Sequence[str],
    config: Mapping[str, object],
    weights: Mapping[str, mx.array],
    plan: QuantizationPlan,
    *,
    selection: ByPath | Quantization | None = None,
) -> None:
    """Staged next to the entry and renamed at the end: a process killed halfway leaves a
    `.tmp-*` that no lookup can take for an entry.

    An MTP head the source carries is packed and written here, under `MTP_PREFIX`, and that
    is deliberately *here* and not in either caller. There are two — `load(quantize=…)` and
    the daemon's own quantization job — and the first version of this put the merge in one of
    them, which produced entries that quantized fine and could not speculate. A caller cannot
    forget what it does not do.

    `selection` is what the head is packed under. `None` leaves it out, which is what a
    caller writing an entry for something that is not a model wants. The head is always
    round-to-nearest, even when the trunk was calibrated: the statistics a calibration
    gathers are the trunk's, and running the head through them would be a number derived
    from the wrong activations. It is 3.7% of the bytes it rides with.
    """
    entry.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=entry.parent))
    try:
        for name in _auxiliary(checkpoint, patterns):
            shutil.copy2(checkpoint / name, staging / name)
        head, head_plan = _packed_head(checkpoint, selection)
        save_quantized(staging, config, {**weights, **head}, plan | head_plan)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        staging.rename(entry)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if not entry.is_dir():
            raise


def _packed_head(
    checkpoint: Path, selection: ByPath | Quantization | None
) -> tuple[dict[str, mx.array], QuantizationPlan]:
    """The source's MTP head, packed, with its leaves named as the entry spells them.

    Empty for a checkpoint with no head and for a caller that named no selection. The plan is
    the head's own — `expand_plan` over the head's tree — because the trunk's is a map of
    leaves the head does not have.
    """
    spec = None if selection is None else _MTP_SPECS.get(_architecture(checkpoint))
    if spec is None or not has_mtp(checkpoint):
        return {}, {}
    pending = spec.quantize(checkpoint, None)
    plan = expand_plan(pending.model, selection)
    packed = _quantized(pending.model, pending.weights(), plan)
    return (
        {f"{MTP_PREFIX}{leaf}": value for leaf, value in packed.items()},
        {f"{MTP_PREFIX}{leaf}": format for leaf, format in plan.items()},
    )


def _architecture(directory: Path) -> str:
    config: _Config = json.loads((directory / "config.json").read_text())
    return config["model_type"]


def _quantized(
    model: nn.Module,
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
) -> dict[str, mx.array]:
    for leaf in inventory(model):
        if infer_quantization(weights, leaf.path, input_dims=leaf.shape[-1]) is not None:
            raise ValueError("quantize= cannot be applied to a quantized checkpoint")
    return quantize_weights(weights, plan)


def provenance(
    fingerprint: Mapping[str, object],
    *,
    revision: str | None,
    dtype: mx.Dtype | None,
    method: str,
) -> dict[str, object]:
    """What an entry records about its own making. In one place because it is written from
    two: the cache below and whoever writes an entry of its own."""
    return {
        "source": fingerprint,
        "revision": revision,
        "dtype": str(dtype) if dtype is not None else None,
        "method": method,
        "format_version": _FORMAT_VERSION,
        "mlx": version("mlx"),
    }


def _resolve[S: _Packaged](
    model: str | Path,
    revision: str | None,
    local_files_only: bool,
    registry: Mapping[str, S],
) -> tuple[Path, S, _Config, str | None]:
    """The config that says which architecture this is, the declaration that architecture
    registered, and the directory its weights are in.

    The registry is a parameter because not every door opens onto the same set: `load` and
    `tree` resolve against the models, and quantizing resolves against everything with
    weights to pack — a drafter is a checkpoint on disk without being a model that
    answers."""
    candidate = Path(model)
    repository = None if isinstance(model, Path) or candidate.is_dir() else model
    config_path = (
        candidate / "config.json"
        if repository is None
        else _download_config(
            repository,
            revision=revision,
            local_files_only=local_files_only,
        )
    )
    config: _Config = json.loads(config_path.read_text())
    model_type = config["model_type"]
    try:
        spec = registry[model_type]
    except KeyError:
        raise ValueError(f"unsupported model_type {model_type!r}") from None

    directory = (
        candidate
        if repository is None
        else _download_snapshot(
            repository,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=list(spec.patterns),
        )
    )
    return directory, spec, config, repository


def _pending(
    spec: _Packaged,
    model_type: str,
    directory: Path,
    dtype: mx.Dtype | None,
) -> Pending[object]:
    if spec.quantize is None:
        raise ValueError(f"quantize= is not supported for model_type {model_type!r}")
    return spec.quantize(directory, dtype)


@dataclass(frozen=True, slots=True)
class Source:
    """`load(quantize=…)` stopped where the weights are still unread: the lazy tree a plan
    resolves against, everything an entry has to carry besides them, and the fingerprint of
    what the bits came from — which is the entry's provenance whatever address it is
    written to."""

    directory: Path
    config: Mapping[str, object]
    patterns: tuple[str, ...]
    pending: Pending[object]
    fingerprint: Mapping[str, object]


def source(
    model: str | Path,
    *,
    dtype: mx.Dtype | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Source:
    """The resolution `load` does, handed over instead of consumed. What it buys is the
    plan before a weight is read: a caller that reports leaf by leaf, or that addresses the
    entry by something other than the cache's digest, needs both halves separately.

    It resolves against `_quantizable()` and not the models, so a drafter is addressable
    here: what it lacks is a task, and packing a checkpoint never asks for one.
    """
    directory, spec, config, repository = _resolve(
        model, revision, local_files_only, _quantizable()
    )
    return Source(
        directory=directory,
        config=config,
        patterns=spec.patterns,
        pending=_pending(spec, config["model_type"], directory, dtype),
        fingerprint=_fingerprint(directory, repository)[1],
    )


def load(
    model: str | Path,
    *,
    dtype: mx.Dtype | None = None,
    quantize: Quantization | ByPath | None = None,
    cache: bool = True,
    revision: str | None = None,
    local_files_only: bool = False,
) -> LanguageModel[ModelInput]:
    """Load a supported language model.

    Parameters
    ----------
    model : str | Path
        Hugging Face repository identifier or local checkpoint directory.
    dtype : mx.Dtype | None, optional
        Weight dtype override. Preserve the checkpoint dtype when omitted.
    quantize : Quantization | ByPath | None, optional
        Quantize a dense checkpoint on load, uniformly or leaf by leaf.
    cache : bool, optional
        Reuse (and write) the quantized checkpoint under ``~/.cache/mlx_omnia/quantized``.
        Disable it to quantize in memory and leave nothing behind.
    revision : str | None, optional
        Hugging Face revision used for repository identifiers.
    local_files_only : bool, optional
        Restrict repository resolution to the local Hugging Face cache.

    Returns
    -------
    LanguageModel[ModelInput]
        Language-model facade for the checkpoint's native modalities.

    Raises
    ------
    ValueError
        If the checkpoint architecture is unsupported, or if ``quantize`` is asked of an
        already quantized checkpoint.
    """
    directory, spec, config, repository = _resolve(model, revision, local_files_only, _MODEL_SPECS)
    if quantize is None:
        return spec.task(directory, dtype)

    pending = _pending(spec, config["model_type"], directory, dtype)
    plan = expand_plan(pending.model, quantize)
    if not cache:
        return pending.attach(_quantized(pending.model, pending.weights(), plan))

    name, fingerprint = _fingerprint(directory, repository)
    entry = _CACHE / name / digest(fingerprint, plan, dtype)
    if not entry.is_dir():
        # The miss writes a quantized checkpoint, so the hit is the ordinary load of one.
        recorded = provenance(fingerprint, revision=revision, dtype=dtype, method="rtn")
        write_entry(
            entry,
            directory,
            spec.patterns,
            {**config, "mlx_omnia": recorded},
            _quantized(pending.model, pending.weights(), plan),
            plan,
            selection=quantize,
        )
    return spec.task(entry, None)


_DRAFTER_SPECS: dict[str, Drafter[nn.Module]] = {
    "muse_glimmer_assistant": muse_glimmer.ASSISTANT,
}
"""The checkpoints that are not models: a drafter serves no task, has no tokenizer and no
head of its own, so it has no `Checkpoint` and does not answer to `load`. It is a tree
bound to a target — `speculative.Drafting.speculate_with` is where the two meet."""


def drafts(model_type: str) -> bool:
    """Whether an architecture is a drafter. A caller with a checkpoint in hand and no
    tree yet asks this: what a drafter cannot do is not a shape it fails at but a fact
    about the checkpoint — it has no tokenizer, so nothing that reads a corpus reaches it."""
    return model_type in _DRAFTER_SPECS


def _quantizable() -> dict[str, _Packaged]:
    """Everything with weights to pack. Wider than `_MODEL_SPECS` on purpose and read by
    `source` alone: quantizing a checkpoint asks it for a lazy tree and a weight dict, and
    neither is what makes something a model — while `load` and `tree` stay on the models,
    so a drafter named where a model belongs still fails at the door.

    Built per call and not once at import: the two registries are what a test registers a
    double into, and a snapshot would not see it."""
    return {**_MODEL_SPECS, **_DRAFTER_SPECS}


def load_drafter(directory: Path, dtype: mx.Dtype | None = None) -> nn.Module:
    """The drafter in `directory`, by the `model_type` its config declares.

    A directory and not an id: a drafter is something the caller already found on disk —
    nothing downloads one on demand, because nothing in a checkpoint says which drafter it
    wants.
    """
    config: _Config = json.loads((directory / "config.json").read_text())
    architecture = config["model_type"]
    spec = _DRAFTER_SPECS.get(architecture)
    if spec is None:
        raise ValueError(f"{architecture!r} is not a drafter architecture this engine loads")
    return spec.load(directory, dtype)


_MTP_SPECS: dict[str, Drafter[nn.Module]] = {
    "nemotron_h": nemotron_h.MTP,
}
"""Architectures whose *own* checkpoint carries a multi-token-prediction head.

A second table and not an entry in `_DRAFTER_SPECS`, because the question is different. That
one asks "is this directory a drafter and nothing else" — it has no tokenizer, serves no
task, answers to no `load`. Here the directory is the model; the head is a prefix inside its
shards that the trunk's loader drops (`_drop_mtp`, in nine families). Keyed the same way and
answering separately is what keeps `drafts()` from having to say two things at once about one
folder, which is the collision `64.3` names from the other side.

`prepare_weights` still drops `mtp.*` from the trunk's tree: the head is a second tree, out of
the model's own, so whoever accounts for memory can weigh it — `speculative.Drafting` says
why."""


def mtp_head(directory: Path, dtype: mx.Dtype | None = None) -> nn.Module:
    """The MTP head this checkpoint carries, as a tree of its own.

    Raises rather than returning `None` for a checkpoint without one: naming a head is what
    turns speculation on, and a switch that silently did nothing is a model answering at a
    speed the screen says it does not have.
    """
    config: _Config = json.loads((directory / "config.json").read_text())
    architecture = config["model_type"]
    spec = _MTP_SPECS.get(architecture)
    if spec is None:
        raise ValueError(f"{architecture!r} has no MTP head this engine loads")
    return spec.load(directory, dtype)


def has_mtp(directory: Path) -> bool:
    """Whether this checkpoint carries a head *and* this engine can build it.

    Both halves are needed and neither implies the other: a `nemotron_h` entry quantized
    before the head was kept has the architecture and none of the tensors, and a family whose
    checkpoints ship `mtp.*` has the tensors and no tree until somebody writes one. Read off
    the shard headers, so it costs no load — which is what the catalog and the screen need.
    """
    try:
        config: _Config = json.loads((directory / "config.json").read_text())
    except (OSError, ValueError):
        return False
    if config.get("model_type") not in _MTP_SPECS:
        return False
    return checkpoint_bytes(directory, MTP_PREFIX) > 0


def tree(
    model: str | Path,
    *,
    dtype: mx.Dtype | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> CausalLM[LayerCache]:
    """Load the bare model tree — the layer `stream_ids` and the bench consume,
    without the task-level facade `load` wraps around it.

    Parameters
    ----------
    model : str | Path
        Hugging Face repository identifier or local checkpoint directory.
    dtype : mx.Dtype | None, optional
        Weight dtype override. Preserve the checkpoint dtype when omitted.
    revision : str | None, optional
        Hugging Face revision used for repository identifiers.
    local_files_only : bool, optional
        Restrict repository resolution to the local Hugging Face cache.

    Returns
    -------
    CausalLM[LayerCache]
        The architecture's module tree, loaded and materialized.

    Raises
    ------
    ValueError
        If the checkpoint architecture is unsupported.
    """
    directory, spec, _, _ = _resolve(model, revision, local_files_only, _MODEL_SPECS)
    # The one erasure point. `CausalLM[C]` is invariant in C (the cache leaves through
    # `make_cache` and returns through `__call__`), so nine trees with nine concrete
    # caches share no honest supertype, and nothing in a `str | Path` argument can bind
    # a type var. The claim the cast makes is wider than the tree (it does not accept
    # arbitrary `LayerCache` rows) — safe because every consumer round-trips the cache
    # the model itself handed out, and none constructs a foreign one.
    return cast(CausalLM[LayerCache], spec.load(directory, dtype))

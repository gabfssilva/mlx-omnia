import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, TypedDict

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn

from sideros.checkpoint import Pending, save_quantized
from sideros.language import LanguageModel
from sideros.model import ModelInput
from sideros.models import (
    bitnet,
    falcon_h1,
    gemma3,
    gemma4,
    gpt2,
    gpt_oss,
    hy3,
    laguna,
    lfm2_moe,
    llama4,
    longcat_flash_ngram,
    mamba2,
    qwen2,
    qwen3,
    qwen3_5,
    qwen3_moe,
    step3p7,
)
from sideros.quant.quantization import (
    ByPath,
    Quantization,
    QuantizationPlan,
    expand_plan,
    infer_quantization,
    inventory,
    quantize_weights,
)

__all__ = ["Source", "digest", "load", "provenance", "source", "write_entry"]


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


class _Registered(Protocol):
    """What `load` reads off an architecture's `Checkpoint`. The tree loader is not part
    of it — this door only ever hands back a task-level model — and leaving it out is
    what lets one dict hold declarations typed on nine different trees."""

    @property
    def patterns(self) -> tuple[str, ...]: ...

    @property
    def task(self) -> Callable[[Path, mx.Dtype | None], LanguageModel[ModelInput]]: ...

    @property
    def quantize(self) -> Callable[[Path, mx.Dtype | None], Pending] | None: ...


_MODEL_SPECS: dict[str, _Registered] = {
    "bitnet": bitnet.CHECKPOINT,
    "falcon_h1": falcon_h1.CHECKPOINT,
    "gemma3_text": gemma3.CHECKPOINT,
    "gemma4": gemma4.CHECKPOINT,
    "gemma4_unified": gemma4.CHECKPOINT,
    "gemma4_assistant": gemma4.CHECKPOINT,
    "gpt2": gpt2.CHECKPOINT,
    "gpt_oss": gpt_oss.CHECKPOINT,
    "hy_v3": hy3.CHECKPOINT,
    "laguna": laguna.CHECKPOINT,
    "lfm2_moe": lfm2_moe.CHECKPOINT,
    "llama4": llama4.CHECKPOINT,
    "longcat_flash_ngram": longcat_flash_ngram.CHECKPOINT,
    "mamba2": mamba2.CHECKPOINT,
    "qwen2": qwen2.CHECKPOINT,
    "qwen3": qwen3.CHECKPOINT,
    "qwen3_5": qwen3_5.CHECKPOINT,
    "qwen3_5_moe": qwen3_5.CHECKPOINT,
    "qwen3_moe": qwen3_moe.CHECKPOINT,
    "step3p7": step3p7.CHECKPOINT,
}

_CACHE = Path.home() / ".cache" / "sideros" / "quantized"

# Bumped whenever the same plan over the same checkpoint would produce different bits.
_FORMAT_VERSION = 1


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
) -> None:
    """Staged next to the entry and renamed at the end: a process killed halfway leaves a
    `.tmp-*` that no lookup can take for an entry."""
    entry.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=entry.parent))
    try:
        for name in _auxiliary(checkpoint, patterns):
            shutil.copy2(checkpoint / name, staging / name)
        save_quantized(staging, config, weights, plan)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        staging.rename(entry)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if not entry.is_dir():
            raise


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


def _resolve(
    model: str | Path,
    revision: str | None,
    local_files_only: bool,
) -> tuple[Path, _Registered, _Config, str | None]:
    """The config that says which architecture this is, the declaration that architecture
    registered, and the directory its weights are in."""
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
        spec = _MODEL_SPECS[model_type]
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
    spec: _Registered,
    model_type: str,
    directory: Path,
    dtype: mx.Dtype | None,
) -> Pending:
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
    pending: Pending
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
    """
    directory, spec, config, repository = _resolve(model, revision, local_files_only)
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
        Reuse (and write) the quantized checkpoint under ``~/.cache/sideros/quantized``.
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
    directory, spec, config, repository = _resolve(model, revision, local_files_only)
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
            {**config, "sideros": recorded},
            _quantized(pending.model, pending.weights(), plan),
            plan,
        )
    return spec.task(entry, None)

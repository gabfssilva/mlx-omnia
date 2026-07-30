"""The load spine: what every architecture's checkpoint shares, and the declaration each
one hands to `sideros.load`.

Nothing here knows an architecture. The JSON shapes, the config parsing and the fusions
only one model uses live in that model's file, next to the tree they feed; what stays is
the part a second model would otherwise copy — the shard merge, the row-aligned fusions,
and the four-step tail (build → `nn.quantize` filtered by the tensors → `load_weights`
strict → `mx.eval`).
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, assert_never

import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import SegmentedQKV, SwitchLinear
from sideros.language import LanguageModel
from sideros.model import ModelInput
from sideros.quant.quantization import (
    MXFP,
    NVFP,
    Affine,
    Quantization,
    QuantizationPlan,
    infer_quantization,
)


class QuantizationJson(TypedDict):
    group_size: int
    bits: int
    mode: NotRequired[Literal["affine", "mxfp4", "mxfp8"]]


class _PlanJson(TypedDict):
    """The per-leaf shape of the same block. The formats sit one level down, under
    `leaves`, so the two shapes differ by a required key — which is what lets a reader
    tell them apart, and what makes a converter that only knows the global shape fail
    loudly instead of loading a quantized file as if it were dense."""

    leaves: dict[str, QuantizationJson]


class _GenerationJson(TypedDict):
    """Only the field read here. The file carries the sampling defaults too, and those are
    the client's to send — a daemon that took a checkpoint's `temperature` would answer a
    request that named none with a draw nobody asked for."""

    eos_token_id: NotRequired[int | list[int] | None]


class _DeclarationJson(TypedDict):
    """Only the block this module reads back. A config that carries the global shape
    instead has no `leaves` key, and the reader says so by returning nothing."""

    quantization: NotRequired[_PlanJson]


def leaf_json(format: Quantization) -> QuantizationJson:
    match format:
        case Affine(group_size=group_size, bits=bits):
            return {"group_size": group_size, "bits": bits}
        case MXFP(mode=mode, group_size=group_size, bits=bits):
            return {"group_size": group_size, "bits": bits, "mode": mode}
        case NVFP():
            raise ValueError("nvfp4 has no config block shape yet")


def leaf_format(raw: QuantizationJson) -> Quantization:
    mode = raw.get("mode", "affine")
    if mode == "affine":
        return Affine(group_size=raw["group_size"], bits=raw["bits"])
    return MXFP(mode=mode, group_size=raw["group_size"], bits=raw["bits"])


def stop_tokens(directory: Path, declared: tuple[int, ...]) -> tuple[int, ...]:
    """Every id that ends a turn: what `config.json` said, plus what `generation_config.json`
    adds. In load order, without repeats — the first is the one a checkpoint means by "the"
    eos, and the set is what the loop compares against.

    The second file is where transformers keeps the generation defaults, and it is the only
    place some checkpoints say the whole truth. openai/gpt-oss-20b declares
    `eos_token_id: 200002` (`<|return|>`) in its config and `[200002, 199999, 200012]` in its
    generation config: the one missing is `<|call|>`, which is how harmony ends a turn *that
    called a tool*. Reading only the config, a model offered a function writes the call, does
    not stop, and spends the rest of the budget inventing the result of its own call.

    A checkpoint without the file, or with an `eos_token_id` that is not ids, keeps what it
    declared: this widens a stop set and never narrows one.
    """
    path = directory / "generation_config.json"
    if not path.is_file():
        return declared
    raw: _GenerationJson = json.loads(path.read_text(encoding="utf-8"))
    found = raw.get("eos_token_id")
    extra = found if isinstance(found, list) else [found]
    ids = [*declared, *(token for token in extra if isinstance(token, int))]
    return tuple(dict.fromkeys(ids))


def declared_plan(path: Path) -> QuantizationPlan | None:
    """What the writer said it asked for, leaf by leaf, or nothing when the config carries
    no such block — a checkpoint that never declared one loads exactly as before. It is
    read to be confirmed against the tensors, never to decide anything: what a leaf *is*
    still comes off its own weight and scales."""
    raw: _DeclarationJson = json.loads(path.read_text())
    block = raw.get("quantization")
    if block is None or "leaves" not in block:
        return None
    return {leaf: leaf_format(entry) for leaf, entry in block["leaves"].items()}


def save_quantized(
    directory: Path,
    config: Mapping[str, object],
    weights: Mapping[str, mx.array],
    plan: QuantizationPlan,
) -> None:
    """One `model.safetensors`, no shard splitting. The config is the source's plus the
    plan already expanded, leaf by leaf and never the selection that produced it: the same
    selection over another checkpoint resolves to a different plan."""
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(directory / "model.safetensors"), dict(weights))
    block: _PlanJson = {"leaves": {path: leaf_json(format) for path, format in plan.items()}}
    (directory / "config.json").write_text(json.dumps({**config, "quantization": block}, indent=2))


_QKV = ("q", "k", "v")
_QKV_SUFFIXES = ("weight", "scales", "biases", "bias")


def _same_qkv_format(weights: Mapping[str, mx.array], prefix: str) -> bool:
    """Whether q/k/v carry the same format, read off the tensors alone. The three share
    the input, so the packed weight's trailing dims fix the bit width, the scales'
    trailing dims fix the group, and the scales dtype fixes the mode family (uint8 is an
    exponent, a float is affine) — shape and dtype already say everything `bits`,
    `group_size` and `mode` would, with no config and no input_dims to hand."""
    for suffix in _QKV_SUFFIXES:
        keys = [f"{prefix}{name}_proj.{suffix}" for name in _QKV]
        present = [key for key in keys if key in weights]
        if not present:
            continue
        if len(present) != len(keys):
            return False
        if suffix == "bias":
            continue
        if len({(weights[key].shape[1:], weights[key].dtype) for key in keys}) != 1:
            return False
    return True


def fuse_qkv(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """q/k/v concatenated on the output axis — holds for dense weight, packed u32 with
    its scales/biases, and Qwen2's projection `bias`, because all of them are
    row-aligned (the bias vector's only axis *is* the output axis).

    That alignment is what a mixed plan can break: q in 4 bits next to k in 8 has no
    common matrix. The three leaves then move under the fused name instead of into it,
    and `attach_weights` builds a `SegmentedQKV` over them. The checkpoint is untouched
    either way — the decision is the loader's, taken by comparing the tensors."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.self_attn."
        if not all(f"{prefix}{name}_proj.weight" in weights for name in _QKV):
            continue
        if not _same_qkv_format(weights, prefix):
            for name in _QKV:
                for suffix in _QKV_SUFFIXES:
                    key = f"{prefix}{name}_proj.{suffix}"
                    if key in weights:
                        weights[f"{prefix}qkv_proj.{name}_proj.{suffix}"] = weights.pop(key)
            continue
        for suffix in _QKV_SUFFIXES:
            keys = [f"{prefix}{name}_proj.{suffix}" for name in _QKV]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}qkv_proj.{suffix}"] = fused
    return weights


def concat_gate_up(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Dense sibling of `interleave_gate_up`: one matmul, split at the midpoint. The
    MoE variant interleaves row by row instead, because its decode kernel reads pairs."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def interleave_gate_up(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Gate and up row-interleaved into one stack ([g0,u0,g1,u1,…]) and the originals
    dropped — otherwise both copies stay resident."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.switch_mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            parts = [weights.pop(key) for key in keys]
            experts, rows, cols = parts[0].shape
            fused = mx.stack(parts, axis=2).reshape(experts, 2 * rows, cols)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def load_shards(directory: Path) -> dict[str, mx.array]:
    """A sharded checkpoint is `model-00001-of-000NN.safetensors`; the index.json only
    says which shard holds what, and merging every shard makes it redundant."""
    weights: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        weights.update(part)
    return weights


def drop_tied_head(weights: dict[str, mx.array]) -> None:
    """Tied checkpoints still serialize lm_head; transformers discards it."""
    for suffix in ("weight", "scales", "biases"):
        weights.pop(f"lm_head.{suffix}", None)


def reject_dtype_cast(dtype: mx.Dtype | None, weights: Mapping[str, mx.array]) -> None:
    """A quantized checkpoint carries its weights as packed uint32; casting them to a
    float dtype keeps the shape (so `update` accepts it) and destroys the numbers. What
    says it is quantized are the tensors, not the config's block: a converter that ships
    `.scales` without declaring them would otherwise be cast silently."""
    if dtype is not None and any(key.endswith(".scales") for key in weights):
        raise ValueError("dtype= cannot be applied to a quantized checkpoint")


type _Fusion = Callable[[dict[str, mx.array]], dict[str, mx.array]]


class _SpineConfig(Protocol):
    @property
    def tie_word_embeddings(self) -> bool: ...


def _reject_orphan_formats(weights: Mapping[str, mx.array]) -> None:
    """Every fusion moves `.scales`/`.biases` with the `.weight` they describe. A format
    tensor left without its weight means one of them moved and the other did not; without
    this, `load_weights` reports an unexpected name instead of the leaf that lost it."""
    for key in sorted(weights):
        for suffix in (".scales", ".biases"):
            if key.endswith(suffix) and f"{key.removesuffix(suffix)}.weight" not in weights:
                raise ValueError(f"{key.removesuffix(suffix)} carries {suffix[1:]} without weight")


def _modules_by_path(model: nn.Module) -> dict[str, nn.Module]:
    found: dict[str, nn.Module] = {}
    model.apply_to_modules(lambda path, module: found.__setitem__(path, module))
    return found


def _build_segments(model: nn.Module, weights: Mapping[str, mx.array]) -> None:
    """A leaf the checkpoint addresses by parts — `<leaf>.q_proj.weight` and siblings,
    with no `<leaf>.weight` — is a qkv the loader chose not to fuse. The tree declares one
    `nn.Linear` there, so the swap happens before `nn.quantize`: each segment is a leaf of
    its own from then on, and the per-leaf format comes off its own tensors."""
    modules = _modules_by_path(model)
    for path, module in sorted(modules.items()):
        if not isinstance(module, nn.Linear) or f"{path}.weight" in weights:
            continue
        segments = [f"{path}.{name}_proj.weight" for name in _QKV]
        if not all(key in weights for key in segments):
            continue
        queries, keys, values = (weights[key].shape[0] for key in segments)
        parent, _, attribute = path.rpartition(".")
        setattr(
            modules[parent],
            attribute,
            SegmentedQKV(
                module.weight.shape[-1],
                queries=queries,
                keys=keys,
                values=values,
                bias=f"{path}.q_proj.bias" in weights,
            ),
        )


def _confirm(path: str, declared: QuantizationPlan | None, inferred: Quantization | None) -> None:
    """The declaration is not a second shape table: it is checked against the one the
    tensors already gave. A leaf the config announces at another width — or announces at
    all while its tensors are dense — is an edited config over untouched weights, which
    would otherwise load quietly at whatever the tensors happen to say."""
    if declared is None:
        return
    requested = declared.get(path)
    if requested is None or requested == inferred:
        return
    raise ValueError(
        f"{path} is declared as {requested} but its tensors are "
        f"{'dense' if inferred is None else inferred}"
    )


def _quantization(
    weights: dict[str, mx.array],
    path: str,
    module: nn.Module,
    declared: QuantizationPlan | None = None,
) -> bool | dict[str, int | str]:
    """Which leaves are quantized, and with which parameters, read off the checkpoint
    itself: a packed `[out, in·bits/32]` next to `[out, in/group]` scales says both
    numbers. The 35B overrides `mlp.gate` and `shared_expert_gate` to 8 bits inside the
    config's `quantization` block, keyed by weight path — deriving the parameters from
    the tensors keeps that out of the config reader entirely."""
    if not isinstance(module, nn.Linear | nn.Embedding | SwitchLinear):
        if f"{path}.scales" in weights:
            raise TypeError(f"{path} carries scales but is a {type(module).__name__}")
        return False
    format = infer_quantization(weights, path, input_dims=module.weight.shape[-1])
    _confirm(path, declared, format)
    if format is None:
        return False
    match format:
        case Affine(group_size=group_size, bits=bits):
            return {"group_size": group_size, "bits": bits}
        case MXFP(mode=mode, group_size=group_size, bits=bits) | NVFP(
            mode=mode, group_size=group_size, bits=bits
        ):
            return {"group_size": group_size, "bits": bits, "mode": mode}
    assert_never(format)


def attach_weights[M: nn.Module](
    model: M,
    weights: dict[str, mx.array],
    *,
    declared: QuantizationPlan | None = None,
) -> M:
    """The tail every loader shares: which leaves are quantized comes off the tensors, and
    the materialization at the end is what keeps the first forward from reading a corrupted
    lazy view. `declared` only gets confirmed against that."""
    _reject_orphan_formats(weights)
    _build_segments(model, weights)
    nn.quantize(
        model,
        class_predicate=lambda path, module: _quantization(weights, path, module, declared),
    )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def prepare_weights(
    config: _SpineConfig,
    weights: dict[str, mx.array],
    fusions: Sequence[_Fusion],
    dtype: mx.Dtype | None,
) -> dict[str, mx.array]:
    """The dict side of the spine: reject a cast over packed tensors, drop the tied head,
    fuse. What `attach_weights` binds to the tree, and what the quantizing load transforms
    before any tree exists."""
    reject_dtype_cast(dtype, weights)
    if dtype is not None:
        weights = {key: value.astype(dtype) for key, value in weights.items()}

    if config.tie_word_embeddings:
        drop_tied_head(weights)

    for fuse in fusions:
        weights = fuse(weights)

    return weights


def load_checkpoint[M: nn.Module](
    model: M,
    config: _SpineConfig,
    weights: dict[str, mx.array],
    fusions: Sequence[_Fusion],
    dtype: mx.Dtype | None,
    *,
    declared: QuantizationPlan | None = None,
) -> M:
    """The spine every loader shares: `prepare_weights` on the dict side, then
    `attach_weights` — per-leaf format off the tensors, `load_weights(strict=True)`
    and `mx.eval`."""
    prepared = prepare_weights(config, weights, fusions, dtype)
    return attach_weights(model, prepared, declared=declared)


@dataclass(frozen=True, slots=True)
class Pending:
    """The same load split where quantization happens: the lazy tree (which is what the
    plan resolves against and costs nothing to build), the prepared weight dict, and the
    tail that binds one to the other. The split is what lets `cache=False` quantize
    without a file, and what keeps the hit from touching a single tensor."""

    model: nn.Module
    weights: Callable[[], dict[str, mx.array]]
    attach: Callable[[dict[str, mx.array]], LanguageModel[ModelInput]]


@dataclass(frozen=True, slots=True)
class Checkpoint[M: nn.Module]:
    """What one architecture declares about its own checkpoint, and the whole of what
    `sideros.load` knows about it.

    `load` is the tree alone — the parity suites and the bench consume it, and `task`
    is built on top of it. `task` is typed on the general model protocol, not on a
    language one: what an architecture produces is its own to say.
    """

    patterns: tuple[str, ...]
    load: Callable[[Path, mx.Dtype | None], M]
    task: Callable[[Path, mx.Dtype | None], LanguageModel[ModelInput]]
    quantize: Callable[[Path, mx.Dtype | None], Pending] | None = None


def checkpoint[M: nn.Module, C](
    patterns: tuple[str, ...],
    config: Callable[[Path], C],
    build: Callable[[C], M],
    weights: Callable[[Path, C, mx.Dtype | None], dict[str, mx.array]],
    composite: Callable[[Path, M], LanguageModel[ModelInput]],
) -> Checkpoint[M]:
    """`load`, `task` and `quantize` derived from the four parts an architecture actually
    owns: the config reader (handed `config.json`), the lazy tree, the checkpoint's tensors
    prepared into the tree's names and layout, and the facade over a loaded tree. Declaring
    them separately is what gives every architecture the quantizing load for free.

    A directory whose config carries the leaf-by-leaf `quantization` block was written by
    `save_quantized`: its tensors *are* a prepared dict, so the preparation does not run
    again — it is the one step allowed to be non-idempotent over values (gemma3 folds +1
    into its norm scales, gpt2 transposes Conv1D)."""

    def prepared(
        directory: Path, parsed: C, dtype: mx.Dtype | None, declared: QuantizationPlan | None
    ) -> dict[str, mx.array]:
        if declared is None:
            return weights(directory, parsed, dtype)
        tensors = load_shards(directory)
        reject_dtype_cast(dtype, tensors)
        return tensors

    def load(directory: Path, dtype: mx.Dtype | None) -> M:
        parsed = config(directory / "config.json")
        declared = declared_plan(directory / "config.json")
        return attach_weights(
            build(parsed),
            prepared(directory, parsed, dtype, declared),
            declared=declared,
        )

    def task(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
        return composite(directory, load(directory, dtype))

    def quantize(directory: Path, dtype: mx.Dtype | None) -> Pending:
        parsed = config(directory / "config.json")
        tree = build(parsed)
        return Pending(
            tree,
            lambda: weights(directory, parsed, dtype),
            lambda prepared: composite(directory, attach_weights(tree, prepared)),
        )

    return Checkpoint(patterns, load, task, quantize)

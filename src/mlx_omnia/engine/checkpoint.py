"""The load spine: what every architecture's checkpoint shares, and the declaration each
one hands to `mlx_omnia.load`.

Nothing here knows an architecture. The JSON shapes, the config parsing and the fusions
only one model uses live in that model's file, next to the tree they feed; what stays is
the part a second model would otherwise copy — the shard merge, the row-aligned fusions,
and the four-step tail (build → `nn.quantize` filtered by the tensors → `load_weights`
strict → `materialize`).
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, assert_never

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.config import load_config
from mlx_omnia.engine.core.layers import MultiLinear, SegmentedLinear, SegmentedQKV, SwitchLinear
from mlx_omnia.engine.language import LanguageModel
from mlx_omnia.engine.model import ModelInput
from mlx_omnia.engine.quant.quantization import (
    MXFP,
    NVFP,
    Affine,
    Quantization,
    QuantizationPlan,
    infer_quantization,
)

_DORMANT: ContextVar[bool] = ContextVar("dormant", default=False)


def materialize(*arrays: object) -> None:
    """Read what the load path has only described so far — the one door every fusion and
    the loader's tail go through.

    A checkpoint is mmapped and mlx is lazy, so a loaded tree is a plan until something
    asks for the numbers. That is normally exactly once, at the end, and the fusions in
    between stay lazy on purpose so that a concatenate does not resident a dense copy of
    what the quantizing load is about to pack. Routing them all here is what lets the door
    close.
    """
    if not _DORMANT.get():
        mx.eval(*arrays)


@contextmanager
def dormant() -> Iterator[None]:
    """Build a checkpoint's tree without reading a weight.

    Everything but the numbers is already free: the shard headers give shape and dtype,
    which is all `_build_segments` and `infer_quantization` read, and `nn.quantize` swaps
    a module's class rather than its values. So a tree built in here is the same tree the
    loader returns — same segments, same per-leaf format, and therefore the same kernel
    each `build()` picks — while nothing it points at has been faulted in.

    What it is for is asking a question *about* the model: which pieces there are, how they
    are wired, which strategy each operation resolved to. It is not a load. The tree comes
    back unwired and unread, and generating with it would fault the checkpoint in through
    the page cache one tensor at a time, which is the cost this exists to avoid.

    Scoped to the calling context rather than the process: a blueprint served on one
    request must not stop another request's load from materializing.
    """
    token = _DORMANT.set(True)
    try:
        yield
    finally:
        _DORMANT.reset(token)


class QuantizationJson(TypedDict):
    group_size: int
    bits: int
    mode: NotRequired[Literal["affine", "mxfp4", "mxfp8", "nvfp4"]]


class _PlanJson(TypedDict):
    """The per-leaf shape of the same block. The formats sit one level down, under
    `leaves`, so the two shapes differ by a required key — which is what lets a reader
    tell them apart, and what makes a converter that only knows the global shape fail
    loudly instead of loading a quantized file as if it were dense."""

    leaves: dict[str, QuantizationJson]


class _GenerationJson(TypedDict):
    """What transformers keeps beside the weights: the ids that end a turn, and how the
    people who trained the checkpoint meant it to be sampled. Both are read — the second
    only as a *default*, which is a value a request still overrides."""

    eos_token_id: NotRequired[int | list[int] | None]
    do_sample: NotRequired[bool]
    temperature: NotRequired[float | None]
    top_p: NotRequired[float | None]
    top_k: NotRequired[int | None]
    min_p: NotRequired[float | None]
    repetition_penalty: NotRequired[float | None]


@dataclass(frozen=True)
class SamplingDefaults:
    """The knobs a checkpoint declares, each `None` for one it does not — which is not the
    same as a knob it declares neutral. Qwen3 ships 0.6/0.95/20 and is visibly worse
    argmaxed; gpt-oss ships nothing and means whatever asks for it.

    `min_p` and `repetition_penalty` are here because some conversions do carry them, not
    because transformers samples with them by default."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None


def _positive(raw: object) -> float | None:
    """A knob is read only when it is a number that means something. `top_k: 0` and
    `top_p: 0` are how transformers spells *disabled*, and copied across as values they
    would be a cut that keeps nothing."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) if raw > 0 else None


def sampling_defaults(directory: Path) -> SamplingDefaults:
    """How the checkpoint says it wants to be sampled, or nothing when it does not say.

    `do_sample: false` is the whole answer when it appears: transformers ignores the
    temperature and the cuts under it and takes the argmax, so a file that carries 0.6 and
    `do_sample: false` means greedy — and `temperature: 0.0` is how every dialect here
    spells that. Reading the 0.6 instead would draw where the checkpoint says not to.
    """
    path = directory / "generation_config.json"
    if not path.is_file():
        return SamplingDefaults()
    raw: _GenerationJson = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("do_sample") is False:
        return SamplingDefaults(temperature=0.0)
    top_k = _positive(raw.get("top_k"))
    return SamplingDefaults(
        temperature=_positive(raw.get("temperature")),
        top_p=_positive(raw.get("top_p")),
        top_k=None if top_k is None else int(top_k),
        min_p=_positive(raw.get("min_p")),
        repetition_penalty=_positive(raw.get("repetition_penalty")),
    )


class _DeclarationJson(TypedDict):
    """Only the block this module reads back. A config that carries the global shape
    instead has no `leaves` key, and the reader says so by returning nothing."""

    quantization: NotRequired[_PlanJson]


def leaf_json(format: Quantization) -> QuantizationJson:
    match format:
        case Affine(group_size=group_size, bits=bits):
            return {"group_size": group_size, "bits": bits}
        case (
            MXFP(mode=mode, group_size=group_size, bits=bits)
            | NVFP(mode=mode, group_size=group_size, bits=bits)
        ):
            return {"group_size": group_size, "bits": bits, "mode": mode}


def leaf_format(raw: QuantizationJson) -> Quantization:
    mode = raw.get("mode", "affine")
    if mode == "affine":
        return Affine(group_size=raw["group_size"], bits=raw["bits"])
    if mode == "nvfp4":
        return NVFP(group_size=raw["group_size"], bits=raw["bits"])
    return MXFP(mode=mode, group_size=raw["group_size"], bits=raw["bits"])


class _TokenizerJson(TypedDict):
    """Only what the eos is looked up in. `added_tokens` carries the id of every special
    token by name, which is what turns the tokenizer's `eos_token` string into an id
    without building a tokenizer."""

    added_tokens: NotRequired[list[dict[str, object]]]


class _TokenizerConfigJson(TypedDict):
    eos_token: NotRequired[str | dict[str, object] | None]


def _named_eos(directory: Path) -> int | None:
    """The id of the token `tokenizer_config.json` calls the eos, or nothing when it names
    none — or names one `tokenizer.json` has no id for."""
    settings = directory / "tokenizer_config.json"
    tokens = directory / "tokenizer.json"
    if not settings.is_file() or not tokens.is_file():
        return None
    declared: _TokenizerConfigJson = json.loads(settings.read_text(encoding="utf-8"))
    named = declared.get("eos_token")
    # transformers writes it either as the string or as the whole AddedToken object.
    content = named.get("content") if isinstance(named, dict) else named
    if not isinstance(content, str):
        return None
    raw: _TokenizerJson = json.loads(tokens.read_text(encoding="utf-8"))
    for entry in raw.get("added_tokens", []):
        if entry.get("content") == content and isinstance(identifier := entry.get("id"), int):
            return identifier
    return None


def stop_tokens(directory: Path, declared: tuple[int, ...]) -> tuple[int, ...]:
    """Every id that ends a turn: what `config.json` said, plus what
    `generation_config.json` adds, plus the one the tokenizer itself calls the eos. In load
    order, without repeats — the first is the one a checkpoint means by "the" eos, and the
    set is what the loop compares against.

    The generation config is where transformers keeps the generation defaults, and it is the
    only place some checkpoints say the whole truth. openai/gpt-oss-20b declares
    `eos_token_id: 200002` (`<|return|>`) in its config and `[200002, 199999, 200012]` in its
    generation config: the one missing is `<|call|>`, which is how harmony ends a turn *that
    called a tool*. Reading only the config, a model offered a function writes the call, does
    not stop, and spends the rest of the budget inventing the result of its own call.

    The tokenizer is the third source because a conversion that drops the generation config
    can leave a config whose `eos_token_id` is not the token its own template ends turns
    with. `mlx-community/Qwen3.5-0.8B-bf16` ships no generation config, declares
    `eos_token_id: 248044` (`<|endoftext|>`, which is its *pad* token) and a template that
    ends every turn with `<|im_end|>` — so every answer it gives carries `<|im_end|>` in the
    text and runs on until it happens to write the pad token. What ends a turn is what the
    template writes, and the tokenizer is where the checkpoint names it.

    A file that is missing, or that names something the other cannot resolve, changes
    nothing: this widens a stop set and never narrows one.
    """
    ids = list(declared)
    path = directory / "generation_config.json"
    if path.is_file():
        raw: _GenerationJson = json.loads(path.read_text(encoding="utf-8"))
        found = raw.get("eos_token_id")
        extra = found if isinstance(found, list) else [found]
        ids += [token for token in extra if isinstance(token, int)]
    if (named := _named_eos(directory)) is not None:
        ids.append(named)
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


def fusible(weights: Mapping[str, mx.array], keys: Sequence[str]) -> bool:
    """Whether leaves sharing an input carry the same format, read off the tensors alone.
    The packed weight's trailing dims fix the bit width, the scales' trailing dims fix the
    group, and the scales dtype fixes the mode family (uint8 is an exponent, a float is
    affine) — shape and dtype already say everything `bits`, `group_size` and `mode`
    would, with no config and no input_dims to hand. A suffix only some of them carry is
    not a common format either."""
    present = [key for key in keys if key in weights]
    if not present:
        return True
    if len(present) != len(keys):
        return False
    return len({(weights[key].shape[1:], weights[key].dtype) for key in present}) == 1


def _same_qkv_format(weights: Mapping[str, mx.array], prefix: str) -> bool:
    return all(
        fusible(weights, [f"{prefix}{name}_proj.{suffix}" for name in _QKV])
        for suffix in _QKV_SUFFIXES
    )


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
            materialize(fused)
            weights[f"{prefix}qkv_proj.{suffix}"] = fused
    return weights


def concat_gate_up(
    weights: dict[str, mx.array], layers: int, *, prefix: str = "mlp"
) -> dict[str, mx.array]:
    """Dense sibling of `interleave_gate_up`: one matmul, split at the midpoint. The
    MoE variant interleaves row by row instead, because its decode kernel reads pairs."""
    for layer in range(layers):
        leaf = f"model.layers.{layer}.{prefix}."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{leaf}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            weights[f"{leaf}gate_up_proj.{suffix}"] = mx.concatenate(
                [weights.pop(key) for key in keys], axis=0
            )
    return weights


def stack_experts(
    weights: dict[str, mx.array], layers: int, experts: int, *, prefix: str = "mlp"
) -> dict[str, mx.array]:
    """Per-expert leaves (`{prefix}.experts.{e}.gate_proj.weight`) stacked into the
    `[experts, out, in]` tensors `SwitchLinear` declares, under `{prefix}.switch_mlp.*`.

    Two layouts ship for the same architecture: HuggingFace writes one leaf per expert,
    the mlx conversions write the stack. A checkpoint already in the second form has no
    per-expert key and passes through untouched, so both load through one path.

    Lazy on purpose, like every dict-side fusion: the quantizing load packs leaf by
    leaf, and an eval here materializes every dense expert of the checkpoint at once —
    the load path still evaluates everything through the loader's final `materialize`."""
    for layer in range(layers):
        source = f"model.layers.{layer}.{prefix}.experts."
        target = f"model.layers.{layer}.{prefix}.switch_mlp."
        for name in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                if f"{source}0.{name}.{suffix}" not in weights:
                    continue
                weights[f"{target}{name}.{suffix}"] = mx.stack(
                    [weights.pop(f"{source}{expert}.{name}.{suffix}") for expert in range(experts)]
                )
    return weights


def split_stacked_gate_up(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`experts.gate_up_proj` is `[E, in, 2·inner]` with gate and up in halves; the tree
    wants `[E, out, in]` per projection, which `interleave_gate_up` then pairs."""
    for layer in range(layers):
        source = f"model.layers.{layer}.mlp.experts."
        target = f"model.layers.{layer}.mlp.switch_mlp."
        fused = weights.pop(f"{source}gate_up_proj", None)
        if fused is None:
            continue
        middle = fused.shape[-1] // 2
        weights[f"{target}gate_proj.weight"] = fused[..., :middle].swapaxes(-2, -1)
        weights[f"{target}up_proj.weight"] = fused[..., middle:].swapaxes(-2, -1)
        weights[f"{target}down_proj.weight"] = weights.pop(f"{source}down_proj").swapaxes(-2, -1)
        materialize(
            weights[f"{target}gate_proj.weight"],
            weights[f"{target}up_proj.weight"],
            weights[f"{target}down_proj.weight"],
        )
    return weights


def fold_norm_scales(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Gemma stores the norm weight centred on zero (`scale = 1 + w`). Folding the sum
    on the dict side keeps the tree on plain `nn.RMSNorm`; left per call it is one extra
    kernel per norm per token (6 per block plus the final one)."""
    for key, value in weights.items():
        if key.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or (
            key == "model.norm.weight"
        ):
            weights[key] = value + 1
    materialize(list(weights.values()))
    return weights


def interleave_gate_up(
    weights: dict[str, mx.array], layers: int, *, prefix: str = "mlp"
) -> dict[str, mx.array]:
    """Gate and up row-interleaved into one stack ([g0,u0,g1,u1,…]) and the originals
    dropped — otherwise both copies stay resident."""
    for layer in range(layers):
        leaf = f"model.layers.{layer}.{prefix}.switch_mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{leaf}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            parts = [weights.pop(key) for key in keys]
            experts, rows, cols = parts[0].shape
            weights[f"{leaf}gate_up_proj.{suffix}"] = mx.stack(parts, axis=2).reshape(
                experts, 2 * rows, cols
            )
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


def _numbered_parts(weights: Mapping[str, mx.array], path: str) -> list[int]:
    """The output width of each `<leaf>.parts.<i>.weight`, in order, from 0 until one is
    missing — an empty list when the checkpoint addresses the leaf some other way."""
    outputs: list[int] = []
    while (key := f"{path}.parts.{len(outputs)}.weight") in weights:
        outputs.append(weights[key].shape[0])
    return outputs


def _build_segments(model: nn.Module, weights: Mapping[str, mx.array]) -> None:
    """A leaf the checkpoint addresses by parts — `<leaf>.q_proj.weight` and siblings, or
    `<leaf>.parts.<i>.weight`, with no `<leaf>.weight` — is a projection the loader chose
    not to fuse. The tree declares one `nn.Linear` there, so the swap happens before
    `nn.quantize`: each segment is a leaf of its own from then on, and the per-leaf format
    comes off its own tensors."""
    modules = _modules_by_path(model)
    for path, module in sorted(modules.items()):
        if not isinstance(module, nn.Linear) or f"{path}.weight" in weights:
            continue
        input_dims = module.weight.shape[-1]
        parent, _, attribute = path.rpartition(".")
        if outputs := _numbered_parts(weights, path):
            setattr(
                modules[parent],
                attribute,
                SegmentedLinear(input_dims, outputs, bias=f"{path}.parts.0.bias" in weights),
            )
            continue
        segments = [f"{path}.{name}_proj.weight" for name in _QKV]
        if not all(key in weights for key in segments):
            continue
        queries, keys, values = (weights[key].shape[0] for key in segments)
        setattr(
            modules[parent],
            attribute,
            SegmentedQKV(
                input_dims,
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
    if not isinstance(module, nn.Linear | nn.Embedding | SwitchLinear | MultiLinear):
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
        case (
            MXFP(mode=mode, group_size=group_size, bits=bits)
            | NVFP(mode=mode, group_size=group_size, bits=bits)
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
    materialize(model.parameters())
    if not _DORMANT.get():
        wire_resident()
    return model


def wire_resident() -> int:
    """Pin everything currently live into Metal's residency set, and return the limit set.

    mlx attaches an `MTLResidencySet` to its command queue but leaves it at capacity zero,
    so nothing is ever added and the driver re-establishes residency for the whole live
    footprint on *every* command buffer. Raising the limit migrates every tracked
    allocation in a single commit.

    The limit is deliberately the live size plus page slack and nothing more. mlx only
    pays a per-allocation commit for a buffer that still *fits* under the capacity, so
    headroom is the expensive part, not wiring: with the limit sitting exactly at the live
    bytes, every later transient fails the fit test and takes the commit-free path. Called
    on each load, so a second resident model widens the set through the one-commit resize
    rather than through per-allocation inserts.

    `clear_cache` first, so the target is measured over live buffers instead of mlx's free
    pool. The clamp is what keeps the call from raising: it throws above the device's
    recommended working set.
    """
    mx.clear_cache()
    recommended = mx.device_info()["max_recommended_working_set_size"]
    assert isinstance(recommended, int)
    return mx.set_wired_limit(min(mx.get_active_memory() + (64 << 20), recommended - (256 << 20)))


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
    and `materialize`."""
    prepared = prepare_weights(config, weights, fusions, dtype)
    return attach_weights(model, prepared, declared=declared)


@dataclass(frozen=True, slots=True)
class Pending[M]:
    """The same load split where quantization happens: the lazy tree (which is what the
    plan resolves against and costs nothing to build), the prepared weight dict, and the
    tail that binds one to the other. The split is what lets `cache=False` quantize
    without a file, and what keeps the hit from touching a single tensor.

    `M` is what the tail produces, because it is not always a model that answers: a
    drafter's is the tree itself."""

    model: nn.Module
    weights: Callable[[], dict[str, mx.array]]
    attach: Callable[[dict[str, mx.array]], M]


@dataclass(frozen=True, slots=True)
class ImageCost:
    """What one image of a given size costs a checkpoint: the size its tower would actually
    read, and the rows that reserves in the prompt.

    The two are separate because neither implies the other. Every family resizes before it
    looks — to a multiple of its patch block, under a cap on area or on rows — and how the
    resized pixels become rows is the family's own arithmetic again: a merge of 2x2 on one,
    a fixed grid on the next.
    """

    height: int
    width: int
    tokens: int


type Sight = Callable[[int, int], ImageCost]
"""`(height, width)` → what this checkpoint would do with an image that size."""


def _blind(directory: Path) -> Sight | None:
    return None


@dataclass(frozen=True, slots=True)
class Checkpoint[M: nn.Module]:
    """What one architecture declares about its own checkpoint, and the whole of what
    `mlx_omnia.load` knows about it.

    `load` is the tree alone — the parity suites and the bench consume it, and `task`
    is built on top of it. `task` is typed on the general model protocol, not on a
    language one: what an architecture produces is its own to say.
    """

    patterns: tuple[str, ...]
    load: Callable[[Path, mx.Dtype | None], M]
    task: Callable[[Path, mx.Dtype | None], LanguageModel[ModelInput]]
    quantize: Callable[[Path, mx.Dtype | None], Pending[LanguageModel[ModelInput]]] | None = None
    sight: Callable[[Path], Sight | None] = _blind
    """Whether this checkpoint takes images, and what one costs — answered off the config
    files, before a weight is read.

    It is declared rather than derived because the answer is a property of the *checkpoint*
    and not of the architecture: the same family ships text-only conversions, and a family
    whose name carries VL may have been ported without its tower. `None` is a checkpoint
    this engine takes no image for, and it must agree with what `task` builds — the
    capability that accepts a picture and this function read the same fields.
    """


@dataclass(frozen=True, slots=True)
class Drafter[M: nn.Module]:
    """`Checkpoint` without `task`, which is the whole difference: a drafter serves none.
    What it keeps is what quantizing a checkpoint needs — the patterns an entry has to
    carry over, the loader, and the same split `Pending` — so a drafter is quantized by
    the code that quantizes a model rather than by a second path."""

    patterns: tuple[str, ...]
    load: Callable[[Path, mx.Dtype | None], M]
    quantize: Callable[[Path, mx.Dtype | None], Pending[M]]


MTP_PREFIX = "mtp."
"""Where a multi-token-prediction head sits in a checkpoint that carries one, and where the
entry this engine writes puts it back.

A fact about the *file* and not about any architecture, which is why the spine owns it: nine
ported families ship a head under this prefix and every one of them drops it from the trunk's
tree, so the trunk's loader must not be handed those tensors — before quantization it never
was, because the family's own `weights` dropped them; after it, the entry holds both trees in
one file and `prepared` is where the second one gets out of the first one's way.

`task.mtp_head` is what builds it, `task.write_entry` is what packs it, and neither would be
enough on its own: an entry the trunk cannot load is not an entry."""

_CARRIED = ("generation_config.json",)
"""What every architecture reads and none has to declare. `stop_tokens` and
`sampling_defaults` both come off this file, and both of them *widen* what `config.json`
says: the ids that end a turn, and how the checkpoint asks to be sampled. A family that
leaves it out of its own patterns still loads on the hub snapshot, because the snapshot
has the file — it is the quantized entry, copied pattern by pattern, that arrives without
it and stops on the config's eos alone. Declared here, the omission cannot be made once
per family, which is how 47 of 49 of them had it."""


def checkpoint[M: nn.Module, C](
    patterns: tuple[str, ...],
    config: Callable[[Path], C] | type[C],
    build: Callable[[C], M],
    weights: Callable[[Path, C, mx.Dtype | None], dict[str, mx.array]],
    composite: Callable[[Path, M], LanguageModel[ModelInput]],
    *,
    model_types: tuple[str, ...] = (),
    sight: Callable[[Path], Sight | None] = _blind,
) -> Checkpoint[M]:
    """`load`, `task` and `quantize` derived from the four parts an architecture actually
    owns: the config reader (handed `config.json`), the lazy tree, the checkpoint's tensors
    prepared into the tree's names and layout, and the facade over a loaded tree. Declaring
    them separately is what gives every architecture the quantizing load for free.

    A directory whose config carries the leaf-by-leaf `quantization` block was written by
    `save_quantized`: its tensors *are* a prepared dict, so the preparation does not run
    again — it is the one step allowed to be non-idempotent over values (gemma3 folds +1
    into its norm scales, gpt2 transposes Conv1D).

    `config` is either a reader over `config.json`'s path or the mirror dataclass itself —
    the latter goes through `load_config`, gated on `model_types`.

    `patterns` is what the architecture declares, plus `_CARRIED` — see there for why the
    spine adds it rather than each family."""

    if isinstance(config, type):
        cls = config

        def parse(path: Path) -> C:
            return load_config(cls, path, allowed_model_types=model_types)
    else:
        parse = config

    def prepared(
        directory: Path, parsed: C, dtype: mx.Dtype | None, declared: QuantizationPlan | None
    ) -> dict[str, mx.array]:
        if declared is None:
            return weights(directory, parsed, dtype)
        tensors = {
            key: value
            for key, value in load_shards(directory).items()
            if not key.startswith(MTP_PREFIX)
        }
        reject_dtype_cast(dtype, tensors)
        return tensors

    def load(directory: Path, dtype: mx.Dtype | None) -> M:
        parsed = parse(directory / "config.json")
        declared = declared_plan(directory / "config.json")
        return attach_weights(
            build(parsed),
            prepared(directory, parsed, dtype, declared),
            declared=declared,
        )

    def task(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
        return composite(directory, load(directory, dtype))

    def quantize(directory: Path, dtype: mx.Dtype | None) -> Pending[LanguageModel[ModelInput]]:
        parsed = parse(directory / "config.json")
        tree = build(parsed)
        return Pending(
            tree,
            lambda: weights(directory, parsed, dtype),
            lambda prepared: composite(directory, attach_weights(tree, prepared)),
        )

    carried = patterns + tuple(name for name in _CARRIED if name not in patterns)
    return Checkpoint(carried, load, task, quantize, sight)

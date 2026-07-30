import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Literal, Protocol, TypeGuard

import mlx.core as mx
import mlx.nn as nn

type MXFPMode = Literal["mxfp4", "mxfp8"]


@dataclass(frozen=True)
class Affine:
    group_size: int
    bits: int

    def __post_init__(self) -> None:
        if self.group_size not in (32, 64, 128):
            raise ValueError(f"unsupported affine group size: {self.group_size}")
        if self.bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"unsupported affine bit width: {self.bits}")


@dataclass(frozen=True)
class MXFP:
    mode: MXFPMode
    group_size: int
    bits: int

    def __post_init__(self) -> None:
        expected_bits = 4 if self.mode == "mxfp4" else 8
        if self.group_size != 32 or self.bits != expected_bits:
            raise ValueError(
                f"{self.mode} requires group_size=32 and bits={expected_bits}"
            )


@dataclass(frozen=True)
class NVFP:
    """Its own type, not an MXFP with an extra field: the scale is e4m3 (not e8m0) and
    the group is 16 (not 32), so nothing an MXFP kernel assumes carries over."""

    group_size: int
    bits: int
    mode: Literal["nvfp4"] = "nvfp4"

    def __post_init__(self) -> None:
        if self.group_size != 16 or self.bits != 4:
            raise ValueError("nvfp4 requires group_size=16 and bits=4")


type Quantization = Affine | MXFP | NVFP


class Method[F: Quantization, W](Protocol):
    """The method (RTN, AWQ, …) is a property of the operation, not of the result: the
    same format loads identically whichever produced it."""

    def quantize(self, weight: mx.array, format: F) -> W: ...


def _input_dims(weight: mx.array, bits: int) -> int:
    if weight.dtype != mx.uint32:
        raise ValueError(f"weight must have dtype uint32, got {weight.dtype}")
    if weight.ndim < 2:
        raise ValueError(f"weight must have at least two dimensions, got {weight.ndim}")
    packed_bits = weight.shape[-1] * 32
    if packed_bits % bits:
        raise ValueError("packed weight shape is incompatible with its bit width")
    return packed_bits // bits


def _expected_scales_shape(
    weight: mx.array,
    *,
    group_size: int,
    bits: int,
) -> tuple[int, ...]:
    input_dims = _input_dims(weight, bits)
    if input_dims % group_size:
        raise ValueError(
            f"input dimension {input_dims} is not divisible by group size {group_size}"
        )
    return (*weight.shape[:-1], input_dims // group_size)


@dataclass(frozen=True)
class AffineWeight:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    format: Affine

    def __post_init__(self) -> None:
        expected = _expected_scales_shape(
            self.weight,
            group_size=self.format.group_size,
            bits=self.format.bits,
        )
        if self.scales.shape != expected:
            raise ValueError(f"expected scales shape {expected}, got {self.scales.shape}")
        if self.biases.shape != expected:
            raise ValueError(f"expected biases shape {expected}, got {self.biases.shape}")
        if self.scales.dtype != self.biases.dtype:
            raise ValueError("scales and biases must have the same dtype")
        if not mx.issubdtype(self.scales.dtype, mx.floating):
            raise ValueError("scales and biases must use a floating dtype")

    def tensors(self, path: str) -> dict[str, mx.array]:
        return {
            f"{path}.weight": self.weight,
            f"{path}.scales": self.scales,
            f"{path}.biases": self.biases,
        }

    def dequantize(self) -> mx.array:
        return mx.dequantize(
            self.weight,
            self.scales,
            self.biases,
            group_size=self.format.group_size,
            bits=self.format.bits,
        )


class AffineRTN:
    def quantize(
        self,
        weight: mx.array,
        format: Affine,
    ) -> AffineWeight:
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"RTN requires a floating-point weight, got {weight.dtype}")
        packed, scales, biases = mx.quantize(
            weight,
            group_size=format.group_size,
            bits=format.bits,
        )
        mx.eval(packed, scales, biases)
        return AffineWeight(packed, scales, biases, format)


def _check_exponent_scales(weight: mx.array, scales: mx.array, format: MXFP | NVFP) -> None:
    """What the two exponent-scaled formats share: one uint8 byte per group and no bias
    at all. The byte is an exponent field (e8m0 for mx, e4m3 for nv), so reading it as a
    float scale — the affine path — is a mode mix-up, not a rounding difference."""
    expected = _expected_scales_shape(
        weight, group_size=format.group_size, bits=format.bits
    )
    if scales.shape != expected:
        raise ValueError(f"expected scales shape {expected}, got {scales.shape}")
    if scales.dtype != mx.uint8:
        raise ValueError(
            f"{format.mode} scales must have dtype uint8, got {scales.dtype}"
        )


@dataclass(frozen=True)
class MXFPWeight:
    weight: mx.array
    scales: mx.array
    format: MXFP

    def __post_init__(self) -> None:
        _check_exponent_scales(self.weight, self.scales, self.format)

    def tensors(self, path: str) -> dict[str, mx.array]:
        return {
            f"{path}.weight": self.weight,
            f"{path}.scales": self.scales,
        }

    def dequantize(self) -> mx.array:
        return mx.dequantize(
            self.weight,
            self.scales,
            group_size=self.format.group_size,
            bits=self.format.bits,
            mode=self.format.mode,
        )


@dataclass(frozen=True)
class NVFPWeight:
    weight: mx.array
    scales: mx.array
    format: NVFP

    def __post_init__(self) -> None:
        _check_exponent_scales(self.weight, self.scales, self.format)

    def tensors(self, path: str) -> dict[str, mx.array]:
        return {
            f"{path}.weight": self.weight,
            f"{path}.scales": self.scales,
        }

    def dequantize(self) -> mx.array:
        return mx.dequantize(
            self.weight,
            self.scales,
            group_size=self.format.group_size,
            bits=self.format.bits,
            mode=self.format.mode,
        )


class MXFPRTN:
    def quantize(self, weight: mx.array, format: MXFP) -> MXFPWeight:
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"RTN requires a floating-point weight, got {weight.dtype}")
        packed, scales, *_ = mx.quantize(
            weight, group_size=format.group_size, bits=format.bits, mode=format.mode
        )
        mx.eval(packed, scales)
        return MXFPWeight(packed, scales, format)


class NVFPRTN:
    def quantize(self, weight: mx.array, format: NVFP) -> NVFPWeight:
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"RTN requires a floating-point weight, got {weight.dtype}")
        packed, scales, *_ = mx.quantize(
            weight, group_size=format.group_size, bits=format.bits, mode=format.mode
        )
        mx.eval(packed, scales)
        return NVFPWeight(packed, scales, format)


def _group_and_bits(
    weight: mx.array, scales: mx.array, path: str, input_dims: int
) -> tuple[int, int]:
    if scales.shape[-1] == 0 or input_dims % scales.shape[-1]:
        raise ValueError(f"{path} scales shape is incompatible with input dimension {input_dims}")
    packed_bits = weight.shape[-1] * 32
    if packed_bits % input_dims:
        raise ValueError(f"{path} weight shape is incompatible with input dimension {input_dims}")
    return input_dims // scales.shape[-1], packed_bits // input_dims


def exponent_scaled_format(
    weight: mx.array, scales: mx.array, path: str, input_dims: int
) -> MXFP | NVFP:
    """The mode read off the tensors alone: uint8 scales carry the group exponent, and the
    group width together with the packed bit width says which of the three it is. What a
    config declares can confirm this; it is never what decides it."""
    group_size, bits = _group_and_bits(weight, scales, path, input_dims)
    if group_size == 32 and bits in (4, 8):
        return MXFP(mode="mxfp4" if bits == 4 else "mxfp8", group_size=32, bits=bits)
    if group_size == 16 and bits == 4:
        return NVFP(group_size=16, bits=4)
    raise ValueError(
        f"{path} carries uint8 scales at group {group_size} and {bits} bits, "
        "which is neither an mxfp nor an nvfp layout"
    )


def infer_quantization(
    tensors: Mapping[str, mx.array],
    path: str,
    *,
    input_dims: int,
) -> Quantization | None:
    weight = tensors.get(f"{path}.weight")
    scales = tensors.get(f"{path}.scales")
    biases = tensors.get(f"{path}.biases")
    if scales is None:
        if biases is not None:
            raise ValueError(f"{path} carries biases without scales")
        return None
    if weight is None:
        raise ValueError(f"{path} carries scales without weight")

    if scales.dtype == mx.uint8:
        if biases is not None:
            raise ValueError(f"{path} carries exponent scales together with biases")
        format = exponent_scaled_format(weight, scales, path, input_dims)
        match format:
            case MXFP():
                MXFPWeight(weight, scales, format)
            case NVFP():
                NVFPWeight(weight, scales, format)
        return format

    if biases is None:
        raise ValueError(f"{path} carries affine scales without biases")
    group_size, bits = _group_and_bits(weight, scales, path, input_dims)
    affine = Affine(group_size=group_size, bits=bits)
    AffineWeight(weight, scales, biases, affine)
    return affine


@dataclass(frozen=True)
class ByPath:
    """Input type of the selection, never a per-leaf value. `None` — in the default or in
    an override — says dense explicitly."""

    default: Quantization | None
    overrides: Mapping[str | re.Pattern[str], Quantization | None]


type QuantizationPlan = dict[str, Quantization]


class Quantizable(Protocol):
    """`nn.Linear`, `nn.Embedding` and `SwitchLinear` are exactly the modules that define
    `to_quantized` — matching them structurally is what keeps this module below the
    model tree instead of importing from it."""

    weight: mx.array

    def to_quantized(self, group_size: int, bits: int, mode: str) -> nn.Module: ...


def _is_quantizable(module: nn.Module) -> TypeGuard[Quantizable]:
    """`isinstance` against a runtime_checkable Protocol reads attributes with
    `inspect.getattr_static`, which never reaches `nn.Module.__getattr__` — every leaf
    would come back unquantizable. The parameters live in the module's own dict."""
    return hasattr(module, "to_quantized") and hasattr(module, "weight")


@dataclass(frozen=True)
class Leaf:
    path: str
    shape: tuple[int, ...]
    kind: str
    dtype: mx.Dtype

    @property
    def weights(self) -> int:
        return prod(self.shape)


def inventory(model: nn.Module) -> list[Leaf]:
    """The quantizable leaves of an already built tree, ordered by path. The tree is lazy,
    so this reads shapes without materializing a single weight. A load-time fusion is
    already a single leaf here (fused qkv, interleaved gate‖up): the cost is of the leaf as
    the tree carries it, not of the three tensors the checkpoint had."""
    leaves: list[Leaf] = []

    def visit(path: str, module: nn.Module) -> None:
        if _is_quantizable(module):
            weight = module.weight
            leaves.append(Leaf(path, weight.shape, type(module).__name__, weight.dtype))

    model.apply_to_modules(visit)
    return sorted(leaves, key=lambda leaf: leaf.path)


@dataclass(frozen=True)
class Cost:
    """Physical bytes of one leaf in one format, split the way the tensors are: packed
    codes, scales, biases. Estimating does not serve — 4.0 against 4.6 effective bits per
    weight is exactly the scales and biases."""

    codes: int
    scales: int
    biases: int

    @property
    def total(self) -> int:
        return self.codes + self.scales + self.biases


def _packed_words(input_dims: int, bits: int) -> int:
    """mlx packs the codes bit-contiguously into uint32 words, so a row costs
    `input_dims * bits / 32` words and the widths that do not divide 32 (3, 5, 6) require
    the row to close on a word boundary."""
    packed_bits = input_dims * bits
    if packed_bits % 32:
        raise ValueError(f"{input_dims} weights at {bits} bits do not fill whole uint32 words")
    return packed_bits // 32


def format_cost(
    shape: tuple[int, ...], dtype: mx.Dtype, format: Quantization | None
) -> Cost:
    if len(shape) < 2:
        raise ValueError(f"a quantizable leaf has at least two dimensions, got {shape}")
    rows, input_dims = prod(shape[:-1]), shape[-1]
    if format is None:
        return Cost(rows * input_dims * dtype.size, 0, 0)
    if input_dims % format.group_size:
        raise ValueError(
            f"input dimension {input_dims} is not divisible by group size {format.group_size}"
        )
    codes = rows * _packed_words(input_dims, format.bits) * 4
    groups = rows * (input_dims // format.group_size)
    if isinstance(format, Affine):
        return Cost(codes, groups * dtype.size, groups * dtype.size)
    return Cost(codes, groups, 0)


def leaf_cost(leaf: Leaf, format: Quantization | None) -> Cost:
    return format_cost(leaf.shape, leaf.dtype, format)


AFFINE_CANDIDATES: tuple[Affine, ...] = tuple(
    Affine(group_size=group_size, bits=bits)
    for group_size in (32, 64, 128)
    for bits in (2, 3, 4, 5, 6, 8)
)

CANDIDATES: tuple[Quantization, ...] = (
    *AFFINE_CANDIDATES,
    MXFP(mode="mxfp4", group_size=32, bits=4),
)


def admits(shape: tuple[int, ...], format: Quantization) -> bool:
    """Whether the shape closes the format's groups and its rows fill whole uint32 words."""
    input_dims = shape[-1]
    return not (input_dims % format.group_size or (input_dims * format.bits) % 32)


def leaf_candidates(leaf: Leaf) -> dict[Quantization | None, Cost]:
    """Every format the leaf's own shape admits, dense included under `None`. A shape that
    does not close a group or a uint32 word is absent, not priced at an approximation."""
    costs: dict[Quantization | None, Cost] = {None: leaf_cost(leaf, None)}
    for format in CANDIDATES:
        if not admits(leaf.shape, format):
            continue
        costs[format] = leaf_cost(leaf, format)
    return costs


@dataclass(frozen=True)
class PlanCost:
    total_bytes: int
    weights: int

    @property
    def bits_per_weight(self) -> float:
        return 8 * self.total_bytes / self.weights


def plan_cost(leaves: Sequence[Leaf], plan: Mapping[str, Quantization]) -> PlanCost:
    """Effective bits per weight of a whole tree: physical bytes over weights counted, both
    summed leaf by leaf. A leaf outside the plan is dense and pays its dense bytes."""
    total = sum(leaf_cost(leaf, plan.get(leaf.path)).total for leaf in leaves)
    return PlanCost(total, sum(leaf.weights for leaf in leaves))


def _pattern(key: str | re.Pattern[str]) -> re.Pattern[str]:
    return key if isinstance(key, re.Pattern) else re.compile(key)


def claims(
    leaves: Sequence[Leaf], overrides: Mapping[str | re.Pattern[str], Quantization | None]
) -> dict[str, tuple[str, Quantization | None]]:
    claimed: dict[str, tuple[str, Quantization | None]] = {}
    for key, format in overrides.items():
        pattern = _pattern(key)
        matched = [leaf for leaf in leaves if pattern.fullmatch(leaf.path)]
        if not matched:
            raise ValueError(f'override "{pattern.pattern}" matches no leaf')
        for leaf in matched:
            previous = claimed.get(leaf.path)
            if previous is not None:
                first, second = sorted((previous[0], pattern.pattern))
                raise ValueError(
                    f'overrides "{first}" and "{second}" both match {leaf.path}'
                )
            claimed[leaf.path] = (pattern.pattern, format)
    return claimed


def expand_plan(model: nn.Module, quantize: Quantization | ByPath) -> QuantizationPlan:
    """The selection resolved leaf by leaf, without quantizing anything. Every key is a
    `fullmatch` pattern and there is no precedence: two keys over the same leaf raise, a
    key over no leaf raises. The result is ordered by path, so the cache digest is stable.

    The default only reaches the leaves whose shape admits it — a vision tower with a 4304
    intermediate closes no affine group, and it stays dense rather than failing the whole
    selection. An explicit override is a decision about one leaf and still raises."""
    leaves = inventory(model)
    if not isinstance(quantize, ByPath):
        return {
            leaf.path: quantize for leaf in leaves if admits(leaf.shape, quantize)
        }

    claimed = claims(leaves, quantize.overrides)
    plan: QuantizationPlan = {}
    for leaf in leaves:
        override = claimed.get(leaf.path)
        if override is not None:
            if override[1] is not None:
                plan[leaf.path] = override[1]
            continue
        if quantize.default is not None and admits(leaf.shape, quantize.default):
            plan[leaf.path] = quantize.default
    return plan


@dataclass(frozen=True)
class QuantizationIntent:
    """What an allocator is asked for: the base every free leaf starts at, the budget in
    effective bits per weight, and the user's fixed decisions. `ByPath` semantics for the
    overrides — `fullmatch`, no precedence, `None` says dense — because they are the same
    decisions, only reaching the allocator instead of the loader."""

    base: Quantization
    target_bpw: float | None = None
    hard_cap_bpw: float | None = None
    overrides: Mapping[str | re.Pattern[str], Quantization | None] = field(
        default_factory=dict[str | re.Pattern[str], "Quantization | None"]
    )

    def __post_init__(self) -> None:
        for name, value in (("target_bpw", self.target_bpw), ("hard_cap_bpw", self.hard_cap_bpw)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if (
            self.target_bpw is not None
            and self.hard_cap_bpw is not None
            and self.target_bpw > self.hard_cap_bpw
        ):
            raise ValueError(
                f"target_bpw {self.target_bpw} is above hard_cap_bpw {self.hard_cap_bpw}"
            )


@dataclass(frozen=True)
class Budget:
    """The intent resolved against one tree, which is what the allocator consumes: the
    fixed decisions already priced, the leaves still free, and the budget in bytes. The
    allocation itself is not here."""

    fixed: QuantizationPlan
    claimed: frozenset[str]
    fixed_bytes: int
    free: tuple[Leaf, ...]
    weights: int
    target_bytes: int | None
    hard_cap_bytes: int | None


def budget(model: nn.Module, intent: QuantizationIntent) -> Budget:
    leaves = inventory(model)
    claimed = claims(leaves, intent.overrides)
    fixed: QuantizationPlan = {}
    fixed_bytes = 0
    for leaf in leaves:
        claim = claimed.get(leaf.path)
        if claim is None:
            continue
        format = claim[1]
        if format is not None:
            fixed[leaf.path] = format
        fixed_bytes += leaf_cost(leaf, format).total
    weights = sum(leaf.weights for leaf in leaves)

    def bytes_of(bpw: float | None) -> int | None:
        return None if bpw is None else int(bpw * weights // 8)

    return Budget(
        fixed=fixed,
        claimed=frozenset(claimed),
        fixed_bytes=fixed_bytes,
        free=tuple(leaf for leaf in leaves if leaf.path not in claimed),
        weights=weights,
        target_bytes=bytes_of(intent.target_bpw),
        hard_cap_bytes=bytes_of(intent.hard_cap_bpw),
    )


_RTN = AffineRTN()


def quantize_weights(
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
    *,
    method: Method[Affine, AffineWeight] = _RTN,
) -> dict[str, mx.array]:
    """The transformation runs on the weight dict, tensor by tensor, on the same side as
    the load-time fusions: the dense weight leaves the dict as it is quantized, so the peak
    is one dense tensor and not the checkpoint. A leaf outside the plan stays dense."""
    missing = sorted(path for path in plan if f"{path}.weight" not in weights)
    if missing:
        raise ValueError(f"the plan names leaves the checkpoint lacks: {missing}")
    packed = sorted(
        path
        for path in plan
        if f"{path}.scales" in weights or weights[f"{path}.weight"].dtype == mx.uint32
    )
    if packed:
        raise ValueError(f"the plan names leaves that are already quantized: {packed}")
    for path, format in plan.items():
        if not isinstance(format, Affine):
            raise ValueError(
                f"{path}: {type(method).__name__} produces affine weights, not {format.mode}"
            )
        tensors = method.quantize(weights.pop(f"{path}.weight"), format).tensors(path)
        mx.eval(list(tensors.values()))
        weights.update(tensors)
    return weights

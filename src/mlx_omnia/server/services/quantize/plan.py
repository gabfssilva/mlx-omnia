"""The selection a quantizing job runs: what it is, what is impossible, and its formats.

The source is resolved with `local_files_only=True`: quantizing reads the disk the daemon
already owns.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import mlx.core as mx

from mlx_omnia.engine import task
from mlx_omnia.engine.quant.quantization import (
    MXFP,
    NVFP,
    Affine,
    ByPath,
    Quantization,
    QuantizationIntent,
)
from mlx_omnia.server.services import catalog

Mode = Literal["affine", "mxfp4", "mxfp8", "nvfp4"]
Method = Literal["rtn", "awq", "gptq", "oq", "oqe"]


class Invalid(Exception):
    """A selection that is impossible or meaningless, answered from the request alone."""


class Conflict(Exception):
    """The repo id is already being written, or already on disk."""


class Unknown(Exception):
    """The source is not on this disk."""


class Reporter(Protocol):
    """The job's progress door, which is also where the work learns it was cancelled."""

    def report(self, message: str, completed: float = 0.0, total: float | None = None) -> None: ...


@dataclass(frozen=True)
class Override:
    """One group of leaves against the rest of the plan. `group_size` absent is the plan's
    own."""

    bits: int
    group_size: int | None = None


@dataclass(frozen=True)
class Selection:
    """The selection alone — everything but where the result lands, which pricing does not
    need because nothing is written.

    `provided` is which fields the caller actually named; the refusals below distinguish a
    default from a caller who believes a calibration pass will run.
    """

    source: str
    mode: Mode = "affine"
    bits: int = 4
    group_size: int = 64
    overrides: Mapping[str, Override | None] = field(default_factory=dict)
    method: Method = "rtn"
    sequences: int = 16
    sequence_length: int = 256
    target_bpw: float | None = None
    hard_cap_bpw: float | None = None
    provided: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Request:
    """A selection plus the repo id the entry is served under."""

    selection: Selection
    repo: str


@dataclass(frozen=True)
class PlanLeaf:
    path: str
    kind: str
    shape: tuple[int, ...]
    bits: int | None
    group_size: int | None
    bytes: int


@dataclass(frozen=True)
class PricedPlan:
    leaves: list[PlanLeaf]
    total_bytes: int
    weights: int
    bits_per_weight: float
    entry_bytes: int


_CALIBRATION_FIELDS = ("sequences", "sequence_length")
_BUDGET_FIELDS = ("target_bpw", "hard_cap_bpw")

ALLOCATED = ("oq", "oqe")

_GPTQ_BITS = (2, 4, 8)
"""The widths whose uint32 packing is verified against `mx.quantize`."""

_HEADROOM = 1.0
"""Bits per weight the default oQ budget leaves above the RTN plan of the same request."""

_SHAPE: dict[str, tuple[int, int]] = {"mxfp4": (32, 4), "mxfp8": (32, 8), "nvfp4": (16, 4)}

SEED = 0

DTYPES: dict[str, mx.Dtype] = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


def slug(repository: str) -> str:
    """The folder the hub cache gives a repository."""
    return f"models--{repository.replace('/', '--')}"


def admissible(selection: Selection) -> None:
    """What is still impossible or meaningless, before the source is resolved and before a
    job exists."""
    set_fields = selection.provided
    if selection.method == "rtn":
        named = sorted(set_fields.intersection((*_CALIBRATION_FIELDS, *_BUDGET_FIELDS)))
        if named:
            raise Invalid(f"'rtn' reads no calibration, so it takes none of {named}")
    if selection.method not in ALLOCATED:
        named = sorted(set_fields.intersection(_BUDGET_FIELDS))
        if named:
            raise Invalid(
                f"{selection.method!r} keeps the plan the selection asked for, so it "
                f"allocates no budget: {named} is the allocator's alone"
            )
    if selection.mode != "affine":
        named = sorted(set_fields.intersection(("bits", "group_size")))
        if named:
            raise Invalid(
                f"{selection.mode!r} fixes {_SHAPE[selection.mode]}, so it takes no {named}"
            )
        if selection.method != "rtn":
            raise Invalid(
                f"{selection.method!r} searches a scale and a bias per group, which "
                f"{selection.mode!r} does not have: the exponent modes are rtn's"
            )
        named = sorted(
            pattern for pattern, override in selection.overrides.items() if override is not None
        )
        if named:
            raise Invalid(
                f"under {selection.mode!r} the format is the mode, so an override can only "
                f"be null (dense): {named}"
            )
    if selection.method == "gptq" and selection.bits not in _GPTQ_BITS:
        raise Invalid(
            f"gptq packs its own codes and only {list(_GPTQ_BITS)} have a layout "
            f"verified against mx.quantize, not {selection.bits}"
        )


def native_refusal(source: str, directory: Path) -> str | None:
    """Why a checkpoint quantized natively takes no plan, or `None` when the source is the
    dense file it claims to be.

    DeepSeek ships V4 as I8 codes with F8 scales and a sliver of bfloat16 norms, and nothing
    in its config says `quantization`, so a dense baseline would be priced at bfloat16: five
    hundred GB of fiction against 155 on disk.
    """
    carrier = catalog.stored_carrier(directory)
    if carrier is None or carrier in DTYPES:
        return None
    return (
        f"{source!r} keeps its weights as {carrier} codes — quantized natively, though "
        "its config does not say so — and a dense plan has no price against them"
    )


def drafter_refusal(source: str, config: Mapping[str, object], method: str) -> str | None:
    """Why a drafter takes `rtn` and nothing else: its input is the target's hidden states
    and not tokens, so there is no corpus to run through it."""
    model_type = config.get("model_type")
    if method == "rtn" or not isinstance(model_type, str) or not task.drafts(model_type):
        return None
    return (
        f"{source!r} is a drafter: its input is the target's hidden states and not tokens, "
        f"so there is no corpus to calibrate it against — {method!r} needs one, 'rtn' does not"
    )


def _format(selection: Selection, bits: int, group_size: int | None = None) -> Quantization:
    match selection.mode:
        case "affine":
            return Affine(group_size=group_size or selection.group_size, bits=bits)
        case "mxfp4" | "mxfp8":
            group_size, width = _SHAPE[selection.mode]
            return MXFP(mode=selection.mode, group_size=group_size, bits=width)
        case "nvfp4":
            group_size, width = _SHAPE[selection.mode]
            return NVFP(group_size=group_size, bits=width)


def formats_of(selection: Selection) -> ByPath:
    """The screen's controls as the engine's selection. The widths a format admits are the
    engine's table, not a second one here."""
    try:
        overrides: dict[str | re.Pattern[str], Quantization | None] = {
            pattern: None
            if override is None
            else _format(selection, override.bits, override.group_size)
            for pattern, override in selection.overrides.items()
        }
        return ByPath(_format(selection, selection.bits), overrides)
    except ValueError as error:
        raise Invalid(str(error)) from error


def intent_of(selection: Selection, formats: ByPath, floor: float) -> QuantizationIntent:
    target = selection.target_bpw if selection.target_bpw is not None else floor + _HEADROOM
    return QuantizationIntent(
        base=Affine(group_size=selection.group_size, bits=selection.bits),
        target_bpw=target,
        hard_cap_bpw=selection.hard_cap_bpw,
        overrides=formats.overrides,
    )

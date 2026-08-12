"""oQ sensitivity: how much a candidate format costs at the output of a block.

    sensitivity(block) = MSE(float output, quantized output) / mean(float output²)

Numerator and denominator are sums of squares accumulated in fp32 over the whole corpus,
so the element count cancels and the ratio is the corpus-wide mean, not a mean of means.

The measurement is per **block**, and the score is reused by every leaf inside it. q_proj,
k_proj and v_proj may end up with different widths, but by a rule on the tensor type in
the allocator — never by three independent measurements.

Nothing here perturbs the model: `calibration.collect` already runs the block, replaces
its quantizable leaves by quantize-dequantize replicas, runs it again and restores the
originals; a `BlockObservation` carries both outputs.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.checkpoint import (
    QuantizationJson,
    leaf_format,
    leaf_json,
    save_quantized,
)
from mlx_omnia.engine.quant.calibration import (
    BlockedForward,
    BlockObservation,
    CalibrationConfig,
    LeafObservation,
    collect,
    format_tag,
)
from mlx_omnia.engine.quant.quantization import (
    Leaf,
    PlanCost,
    Quantization,
    QuantizationIntent,
    QuantizationPlan,
    admits,
    claims,
    inventory,
    leaf_candidates,
    leaf_cost,
    plan_cost,
)

type _Key = tuple[int, str, Quantization]

_FLOOR = 1e-20


@dataclass(frozen=True)
class BlockScore:
    index: int
    path: str
    format: Quantization
    sensitivity: float


def _accumulate(totals: dict[_Key, mx.array], key: _Key, value: mx.array) -> None:
    previous = totals.get(key)
    totals[key] = value if previous is None else previous + value


class OqSensitivity:
    """Relative output error per (block, candidate format), the ordering key of the oQ
    allocator."""

    name = "oq"

    def __init__(self) -> None:
        self._error: dict[_Key, mx.array] = {}
        self._energy: dict[_Key, mx.array] = {}

    def observe_leaf(self, observation: LeafObservation) -> None:
        return

    def observe_block(self, observation: BlockObservation) -> None:
        reference = observation.output.astype(mx.float32)
        energy = (reference * reference).sum()
        for format, output in observation.perturbed.items():
            key = (observation.index, observation.path, format)
            difference = output.astype(mx.float32) - reference
            _accumulate(self._error, key, (difference * difference).sum())
            _accumulate(self._energy, key, energy)

    def flush(self) -> None:
        mx.eval(list(self._error.values()) + list(self._energy.values()))

    def _keys(self) -> list[_Key]:
        return sorted(self._error, key=lambda key: (key[0], key[1], format_tag(key[2])))

    def _ratio(self, key: _Key) -> mx.array:
        return self._error[key] / mx.maximum(self._energy[key], _FLOOR)

    def statistics(self) -> Mapping[str, mx.array]:
        return {
            f"{key[0]:04d}.{key[1]}.{format_tag(key[2])}.sensitivity": self._ratio(key)
            for key in self._keys()
        }

    def scores(self) -> tuple[BlockScore, ...]:
        keys = self._keys()
        ratios = [self._ratio(key) for key in keys]
        mx.eval(ratios)
        return tuple(
            BlockScore(index, path, format, ratio.item())
            for (index, path, format), ratio in zip(keys, ratios, strict=True)
        )


def measure_sensitivity(
    forward: BlockedForward,
    sequences: Sequence[Sequence[int]],
    formats: Sequence[Quantization],
) -> tuple[BlockScore, ...]:
    collector = OqSensitivity()
    collect(forward, sequences, [collector], perturbations=formats)
    return collector.scores()


type Reason = Literal[
    "override", "protection", "promotion", "base", "excluded", "inadmissible"
]


@dataclass(frozen=True)
class Rule:
    """A path pattern (`fullmatch`, as everywhere else) and the width it asks for. The
    width and not the format: the group size of every leaf is the request's, so a recipe
    that named one would be deciding, behind the caller, half of what was asked."""

    pattern: str
    bits: int


@dataclass(frozen=True)
class Recipe:
    """What `oQ4` names, spelled out and versioned. The rules of the reference allocator
    live in one named instance, never as constants of the allocator itself:

    - `protections` are mandatory — a leaf below them is raised to them before any
      allocation, and a cap that cannot pay for them fails;
    - `floors` are the same shape but best effort — applied while they fit;
    - `exclusions` keep a leaf at the base bits and out of the competition (routed
      experts, whose bank dominates the byte count);
    - `promotions` is the set of widths a free leaf may be promoted to.

    Every width here is read against the request's own base: what a recipe decides is how
    much precision a leaf deserves, never the group size it is measured in.
    """

    identifier: str
    version: int
    protections: tuple[Rule, ...] = ()
    floors: tuple[Rule, ...] = ()
    exclusions: tuple[str, ...] = ()
    promotions: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.identifier}.v{self.version}"


RECIPE_OQ4_V1 = Recipe(
    identifier="oQ4",
    version=1,
    protections=(
        Rule(r".*\blm_head", 8),
        Rule(r".*\bembed_tokens", 8),
    ),
    floors=(Rule(r".*\b(q_proj|k_proj|v_proj|qkv_proj)", 6),),
    exclusions=(r".*\bswitch_mlp\..*", r".*\bexperts\..*"),
    promotions=(5, 6, 8),
)


def widen(base: Quantization, bits: int) -> Quantization:
    """The base format at another width — the recipe's half of a format meeting the
    request's. Public because the calibration perturbs by the same set the allocator may
    promote to, and a candidate measured at another group size is not the candidate."""
    return replace(base, bits=bits)


@dataclass(frozen=True)
class Decision:
    path: str
    format: Quantization | None
    reason: Reason


@dataclass(frozen=True)
class Allocation:
    """The plan the loader consumes plus the audit of how it was reached. oQ is not a new
    method: `plan` is an ordinary `QuantizationPlan`."""

    plan: QuantizationPlan
    decisions: Mapping[str, Decision]
    cost: PlanCost
    recipe: str


class BudgetUnreachable(ValueError):
    pass


def _matches(patterns: Sequence[str], path: str) -> bool:
    return any(re.fullmatch(pattern, path) for pattern in patterns)


def _rule_for(rules: Sequence[Rule], path: str) -> int | None:
    for rule in rules:
        if re.fullmatch(rule.pattern, path):
            return rule.bits
    return None


def block_priorities(
    scores: Sequence[BlockScore], base: Quantization
) -> dict[str, float]:
    """One number per block: the relative error the base format caused there, or — if the
    base was not among the measured candidates — the worst any of them caused."""
    worst: dict[str, float] = {}
    at_base: dict[str, float] = {}
    for score in scores:
        worst[score.path] = max(worst.get(score.path, 0.0), score.sensitivity)
        if score.format == base:
            at_base[score.path] = score.sensitivity
    return {path: at_base.get(path, value) for path, value in worst.items()}


def _priority(path: str, priorities: Mapping[str, float]) -> float:
    """A leaf inherits the score of the deepest block whose path prefixes its own."""
    owner = max(
        (
            block
            for block in priorities
            if path == block or path.startswith(f"{block}.")
        ),
        key=len,
        default=None,
    )
    return 0.0 if owner is None else priorities[owner]


def _promotion_order(
    leaf: Leaf, promotions: Sequence[int], base: Quantization
) -> list[Quantization]:
    """The widths the leaf's own shape admits, most expensive first: the greedy takes the
    largest promotion that fits and falls back to the cheaper ones."""
    admitted = leaf_candidates(leaf)
    return sorted(
        (
            format
            for format in (widen(base, bits) for bits in promotions)
            if format in admitted
        ),
        key=lambda format: leaf_cost(leaf, format).total,
        reverse=True,
    )


class OQAllocator:
    """Greedy over the block sensitivity, priced in physical bytes at every step.

    The order is `(-sensitivity, path)`: the block that suffered most first, ties broken
    by path, so two candidates with the same score always resolve the same way.
    """

    def __init__(self, intent: QuantizationIntent, recipe: Recipe) -> None:
        self.intent = intent
        self.recipe = recipe

    def allocate(
        self, leaves: Sequence[Leaf], scores: Sequence[BlockScore]
    ) -> Allocation:
        claimed = claims(leaves, self.intent.overrides)
        weights = sum(leaf.weights for leaf in leaves)

        def bytes_of(bpw: float | None) -> int | None:
            return None if bpw is None else int(bpw * weights // 8)

        target_bytes = bytes_of(self.intent.target_bpw)
        cap_bytes = bytes_of(self.intent.hard_cap_bpw)
        if cap_bytes is None:
            cap_bytes = target_bytes

        chosen: dict[str, Quantization | None] = {}
        reasons: dict[str, Reason] = {}
        free: list[Leaf] = []
        for leaf in leaves:
            claim = claimed.get(leaf.path)
            if claim is not None:
                chosen[leaf.path] = claim[1]
                reasons[leaf.path] = "override"
                continue
            protected = _rule_for(self.recipe.protections, leaf.path)
            if protected is not None:
                protection = widen(self.intent.base, protected)
                self._admits(leaf, protection)
                chosen[leaf.path] = protection
                reasons[leaf.path] = "protection"
                continue
            if not admits(leaf.shape, self.intent.base):
                # A shape no group closes stays dense and out of the competition: every
                # promotion costs more than dense, so there is nothing to spend on it.
                chosen[leaf.path] = None
                reasons[leaf.path] = "inadmissible"
                continue
            chosen[leaf.path] = self.intent.base
            if _matches(self.recipe.exclusions, leaf.path):
                reasons[leaf.path] = "excluded"
            else:
                reasons[leaf.path] = "base"
                free.append(leaf)

        total = sum(leaf_cost(leaf, chosen[leaf.path]).total for leaf in leaves)
        if cap_bytes is not None and total > cap_bytes:
            raise BudgetUnreachable(
                f"{self.recipe.name}: the reserved decisions (overrides and protections) "
                f"already cost {8 * total / weights:.3f} bits per weight over {weights} "
                f"weights, above the cap of {8 * cap_bytes / weights:.3f}"
            )

        def spend(leaf: Leaf, format: Quantization, reason: Reason) -> bool:
            nonlocal total
            candidate = leaf_cost(leaf, format).total
            current = leaf_cost(leaf, chosen[leaf.path]).total
            if candidate <= current:
                return False
            if cap_bytes is not None and total - current + candidate > cap_bytes:
                return False
            total += candidate - current
            chosen[leaf.path] = format
            reasons[leaf.path] = reason
            return True

        for leaf in free:
            floored = _rule_for(self.recipe.floors, leaf.path)
            floor = None if floored is None else widen(self.intent.base, floored)
            if floor is not None and floor in leaf_candidates(leaf):
                spend(leaf, floor, "protection")

        priorities = block_priorities(scores, self.intent.base)
        ordered = sorted(
            free, key=lambda leaf: (-_priority(leaf.path, priorities), leaf.path)
        )
        for leaf in ordered:
            if target_bytes is not None and total >= target_bytes:
                break
            for format in _promotion_order(leaf, self.recipe.promotions, self.intent.base):
                if spend(leaf, format, "promotion"):
                    break

        plan: QuantizationPlan = {}
        for leaf in leaves:
            format = chosen[leaf.path]
            if format is not None:
                plan[leaf.path] = format
        decisions = {
            leaf.path: Decision(leaf.path, chosen[leaf.path], reasons[leaf.path])
            for leaf in leaves
        }
        return Allocation(plan, decisions, plan_cost(leaves, plan), self.recipe.name)

    def _admits(self, leaf: Leaf, format: Quantization) -> None:
        if format not in leaf_candidates(leaf):
            raise ValueError(
                f"{self.recipe.name}: {leaf.path} with shape {leaf.shape} does not admit "
                f"the protected format {format_tag(format)}"
            )


def allocate(
    model: nn.Module,
    scores: Sequence[BlockScore],
    intent: QuantizationIntent,
    recipe: Recipe,
) -> Allocation:
    return OQAllocator(intent, recipe).allocate(inventory(model), scores)


@dataclass(frozen=True)
class PlanProvenance:
    """Why a checkpoint's plan is the plan it is: the recipe that produced it, the
    calibration it was measured over, the budget asked for against the one obtained, and
    the reason behind every leaf.

    Audit, never contract. The load confirms the tensors against the `quantization` block
    and does not read this one — a checkpoint whose provenance was stripped loads the
    same, and nothing on the load side knows the word oQ.
    """

    recipe_identifier: str
    recipe_version: int
    calibration: CalibrationConfig
    target_bpw: float | None
    hard_cap_bpw: float | None
    bits_per_weight: float
    total_bytes: int
    weights: int
    decisions: Mapping[str, Decision]


def plan_provenance(
    allocation: Allocation,
    recipe: Recipe,
    intent: QuantizationIntent,
    calibration: CalibrationConfig,
) -> PlanProvenance:
    return PlanProvenance(
        recipe_identifier=recipe.identifier,
        recipe_version=recipe.version,
        calibration=calibration,
        target_bpw=intent.target_bpw,
        hard_cap_bpw=intent.hard_cap_bpw,
        bits_per_weight=allocation.cost.bits_per_weight,
        total_bytes=allocation.cost.total_bytes,
        weights=allocation.cost.weights,
        decisions=allocation.decisions,
    )


class _DecisionJson(TypedDict):
    reason: Reason
    format: QuantizationJson | None


class _ProvenanceJson(TypedDict):
    recipe_identifier: str
    recipe_version: int
    calibration: dict[str, object]
    target_bpw: float | None
    hard_cap_bpw: float | None
    bits_per_weight: float
    total_bytes: int
    weights: int
    decisions: dict[str, _DecisionJson]


class _OqConfigJson(TypedDict):
    oq: NotRequired[_ProvenanceJson]


def _decision_json(decision: Decision) -> _DecisionJson:
    format = decision.format
    return {"reason": decision.reason, "format": None if format is None else leaf_json(format)}


def provenance_json(provenance: PlanProvenance) -> _ProvenanceJson:
    """Public because `save_allocated` is not the only writer: an entry written by
    `task.write_entry` carries the same block, produced here rather than spelled twice."""
    # The calibration block is whatever `CalibrationConfig` already serializes (sorted
    # keys), so the two spellings of it cannot drift apart.
    calibration: dict[str, object] = json.loads(provenance.calibration.to_json())
    return {
        "recipe_identifier": provenance.recipe_identifier,
        "recipe_version": provenance.recipe_version,
        "calibration": calibration,
        "target_bpw": provenance.target_bpw,
        "hard_cap_bpw": provenance.hard_cap_bpw,
        "bits_per_weight": provenance.bits_per_weight,
        "total_bytes": provenance.total_bytes,
        "weights": provenance.weights,
        "decisions": {
            path: _decision_json(provenance.decisions[path])
            for path in sorted(provenance.decisions)
        },
    }


def save_allocated(
    directory: Path,
    config: Mapping[str, object],
    weights: Mapping[str, mx.array],
    allocation: Allocation,
    provenance: PlanProvenance,
) -> None:
    """`save_quantized` with one extra block. The plan travels where it always did — the
    leaf-by-leaf `quantization` block, byte for byte the same — and the provenance sits
    beside it under `oq`, ordered so that save → load → save reproduces the file."""
    save_quantized(
        directory,
        {**config, "oq": provenance_json(provenance)},
        weights,
        allocation.plan,
    )


def load_provenance(path: Path) -> PlanProvenance | None:
    """The `oq` block of a `config.json` back as a dataclass, or nothing when the
    checkpoint carries none. For auditing and tooling: no loader calls this."""
    raw: _OqConfigJson = json.loads(path.read_text())
    block = raw.get("oq")
    if block is None:
        return None
    return PlanProvenance(
        recipe_identifier=block["recipe_identifier"],
        recipe_version=block["recipe_version"],
        calibration=CalibrationConfig.from_json(json.dumps(block["calibration"])),
        target_bpw=block["target_bpw"],
        hard_cap_bpw=block["hard_cap_bpw"],
        bits_per_weight=block["bits_per_weight"],
        total_bytes=block["total_bytes"],
        weights=block["weights"],
        decisions={
            leaf: Decision(
                leaf,
                None if entry["format"] is None else leaf_format(entry["format"]),
                entry["reason"],
            )
            for leaf, entry in block["decisions"].items()
        },
    )

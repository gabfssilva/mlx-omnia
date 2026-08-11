import re

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.quant.oq import (
    RECIPE_OQ4_V1,
    Allocation,
    BlockScore,
    BudgetUnreachable,
    OQAllocator,
    Recipe,
    Rule,
    allocate,
    widen,
)
from mlx_omnia.quant.quantization import (
    Affine,
    Leaf,
    Quantization,
    QuantizationIntent,
    QuantizationPlan,
    inventory,
    leaf_cost,
    plan_cost,
)

_HIDDEN = 256
_VOCAB = 512
_LAYERS = 4
_BASE = Affine(group_size=64, bits=4)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.k_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.v_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.o_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.down_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.up_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.down_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.switch_mlp = _Experts()


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = [_Layer() for _ in range(_LAYERS)]
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)


def _leaves() -> list[Leaf]:
    return inventory(_Model())


def _scores(sensitivities: list[float]) -> list[BlockScore]:
    return [
        BlockScore(index=index, path=f"layers.{index}", format=_BASE, sensitivity=value)
        for index, value in enumerate(sensitivities)
    ]


def _protection(recipe: Recipe, path: str) -> Quantization | None:
    for rule in recipe.protections:
        if re.fullmatch(rule.pattern, path):
            return widen(_BASE, rule.bits)
    return None


def _baseline_bpw(leaves: list[Leaf], recipe: Recipe) -> float:
    plan = {leaf.path: _protection(recipe, leaf.path) or _BASE for leaf in leaves}
    return plan_cost(leaves, plan).bits_per_weight


def _allocate(
    leaves: list[Leaf],
    sensitivities: list[float],
    *,
    cap: float,
    target: float | None = None,
    recipe: Recipe = RECIPE_OQ4_V1,
    overrides: dict[str | re.Pattern[str], Quantization | None] | None = None,
) -> Allocation:
    intent = QuantizationIntent(
        base=_BASE,
        target_bpw=cap if target is None else target,
        hard_cap_bpw=cap,
        overrides={} if overrides is None else overrides,
    )
    return OQAllocator(intent, recipe).allocate(leaves, _scores(sensitivities))


@pytest.mark.parametrize(
    "sensitivities",
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.001, 0.001, 0.001],
        [0.1, 0.2, 0.4, 0.8],
    ],
    ids=["flat", "dominant", "monotonic"],
)
@pytest.mark.parametrize("margin", [0.05, 0.4, 3.0])
def test_no_plan_exceeds_the_hard_cap(sensitivities: list[float], margin: float) -> None:
    leaves = _leaves()
    cap = _baseline_bpw(leaves, RECIPE_OQ4_V1) + margin
    allocation = _allocate(leaves, sensitivities, cap=cap)
    assert allocation.cost.bits_per_weight <= cap
    assert plan_cost(leaves, allocation.plan).total_bytes == allocation.cost.total_bytes


def test_routed_experts_stay_at_base_out_of_the_competition() -> None:
    leaves = _leaves()
    allocation = _allocate(leaves, [1.0] * _LAYERS, cap=8.0)
    excluded = [
        decision
        for decision in allocation.decisions.values()
        if "switch_mlp" in decision.path
    ]
    assert len(excluded) == 2 * _LAYERS
    assert {decision.reason for decision in excluded} == {"excluded"}
    assert {decision.format for decision in excluded} == {_BASE}


def test_a_leaf_no_group_closes_stays_dense_and_out_of_the_competition() -> None:
    # 48 = 16·3: nenhum group size fecha, como o 4304 da torre de visão do Qwen3.6.
    odd = Leaf("visual.fc2", (64, 48), "Linear", mx.float32)
    allocation = _allocate([*_leaves(), odd], [1.0] * _LAYERS, cap=8.0)
    decision = allocation.decisions["visual.fc2"]
    assert decision.reason == "inadmissible"
    assert decision.format is None
    assert "visual.fc2" not in allocation.plan


def test_protections_are_reserved_before_the_allocation() -> None:
    leaves = _leaves()
    allocation = _allocate(leaves, [0.0] * _LAYERS, cap=8.0)
    for path in ("lm_head", "embed_tokens"):
        decision = allocation.decisions[path]
        assert decision.reason == "protection"
        assert decision.format == Affine(group_size=64, bits=8)


def test_every_width_the_recipe_names_is_read_at_the_group_size_that_was_asked_for() -> None:
    """The recipe names widths; the group size of the whole plan is the request's. A leaf
    protected or promoted at 64 under a base of 128 would be the allocator deciding, behind
    the caller, half of what the caller asked — and it would ship a checkpoint whose leaves
    disagree on a number nobody chose."""
    leaves = _leaves()
    intent = QuantizationIntent(
        base=Affine(group_size=128, bits=4),
        target_bpw=8.0,
        hard_cap_bpw=8.0,
        overrides={},
    )
    allocation = OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, _scores([1.0] * _LAYERS))

    assert allocation.decisions["lm_head"].format == Affine(group_size=128, bits=8)
    promoted = [
        decision.format
        for decision in allocation.decisions.values()
        if decision.reason == "promotion"
    ]
    assert promoted, "nothing was promoted, so the group size proves nothing"
    assert {format.group_size for format in promoted if format is not None} == {128}


def test_unreachable_target_raises_with_the_numbers() -> None:
    leaves = _leaves()
    cap = _baseline_bpw(leaves, RECIPE_OQ4_V1) - 0.5
    with pytest.raises(BudgetUnreachable) as excinfo:
        _allocate(leaves, [1.0] * _LAYERS, cap=cap)
    message = str(excinfo.value)
    assert RECIPE_OQ4_V1.name in message
    assert "bits per weight" in message
    assert "above the cap" in message


def test_override_survives_the_allocation() -> None:
    leaves = _leaves()
    dense = "layers.0.mlp.down_proj"
    overrides: dict[str | re.Pattern[str], Quantization | None] = {
        r"layers\.0\.self_attn\.q_proj": Affine(group_size=32, bits=3),
        dense: None,
    }
    allocation = _allocate(leaves, [1.0] * _LAYERS, cap=8.0, overrides=overrides)

    promoted = allocation.decisions["layers.0.self_attn.q_proj"]
    assert promoted.reason == "override"
    assert promoted.format == Affine(group_size=32, bits=3)
    assert allocation.plan["layers.0.self_attn.q_proj"] == Affine(group_size=32, bits=3)

    assert allocation.decisions[dense].reason == "override"
    assert allocation.decisions[dense].format is None
    assert dense not in allocation.plan


def test_a_tie_resolves_by_path_and_repeats() -> None:
    recipe = Recipe(
        identifier="test",
        version=1,
        protections=(Rule(r".*\blm_head", 8),),
        exclusions=(r".*\bswitch_mlp\..*", r"embed_tokens"),
        promotions=(6,),
    )
    leaves = _leaves()
    cap = _baseline_bpw(leaves, recipe) + 0.15
    first = _allocate(leaves, [1.0] * _LAYERS, cap=cap, recipe=recipe)
    second = _allocate(leaves, [1.0] * _LAYERS, cap=cap, recipe=recipe)

    assert first.plan == second.plan
    assert first.decisions == second.decisions

    free = sorted(
        decision.path
        for decision in first.decisions.values()
        if decision.reason in ("base", "promotion")
    )
    promoted = sorted(
        decision.path
        for decision in first.decisions.values()
        if decision.reason == "promotion"
    )
    assert 0 < len(promoted) < len(free)
    assert promoted == free[: len(promoted)]


def test_a_budget_for_one_block_goes_to_the_most_sensitive_one() -> None:
    # mutação: ordenar por sensibilidade crescente promove layers.0 e este teste falha.
    six = Affine(group_size=64, bits=6)
    recipe = Recipe(
        identifier="test",
        version=1,
        protections=(Rule(r".*\blm_head", 8),),
        exclusions=(r".*\bswitch_mlp\..*", r"embed_tokens"),
        promotions=(6,),
    )
    leaves = _leaves()
    free = [
        leaf
        for leaf in leaves
        if "switch_mlp" not in leaf.path and leaf.path not in ("embed_tokens", "lm_head")
    ]
    delta = leaf_cost(free[0], six).total - leaf_cost(free[0], _BASE).total
    baseline = plan_cost(
        leaves, {leaf.path: _protection(recipe, leaf.path) or _BASE for leaf in leaves}
    ).total_bytes
    weights = sum(leaf.weights for leaf in leaves)
    per_layer = len(free) // _LAYERS
    cap = 8 * (baseline + per_layer * delta + delta // 2) / weights

    allocation = _allocate(leaves, [0.1, 0.2, 0.4, 0.8], cap=cap, recipe=recipe)

    promoted = [
        decision.path
        for decision in allocation.decisions.values()
        if decision.reason == "promotion"
    ]
    assert len(promoted) == per_layer
    assert all(path.startswith("layers.3.") for path in promoted)


def test_allocate_over_a_model_matches_the_leaf_form() -> None:
    model = _Model()
    leaves = inventory(model)
    intent = QuantizationIntent(base=_BASE, target_bpw=6.0, hard_cap_bpw=6.0)
    scores = _scores([0.1, 0.2, 0.4, 0.8])
    assert allocate(model, scores, intent, RECIPE_OQ4_V1).plan == (
        OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, scores).plan
    )


_TRUNK_HIDDEN = 64
_TRUNK_VOCAB = 128
_TRUNK_SEQUENCES: list[list[int]] = [[(3 * step + 7) % _TRUNK_VOCAB for step in range(32)]]


class _TrunkBlock(nn.Module):
    def __init__(self, weight: mx.array) -> None:
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.proj.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class _Trunk(nn.Module):
    def __init__(self, weights: list[mx.array]) -> None:
        super().__init__()
        self.embed = nn.Embedding(_TRUNK_VOCAB, _TRUNK_HIDDEN)
        self.layers = [_TrunkBlock(weight) for weight in weights]


def _trunk_apply(block: nn.Module, hidden: mx.array) -> mx.array:
    assert isinstance(block, _TrunkBlock)
    return block(hidden)


def _trunk_weights() -> list[mx.array]:
    """Two nearly-constant blocks (insensitive at any width) interleaved with two
    gaussian ones: the budget uniform 6-bit spends on the flats, oQ moves to the
    blocks where it shows."""
    mx.random.seed(11)
    shape = (_TRUNK_HIDDEN, _TRUNK_HIDDEN)
    flats = [
        mx.full(shape, 1.0) + 1e-6 * mx.random.normal(shape),
        mx.full(shape, -0.5) + 1e-6 * mx.random.normal(shape),
    ]
    normals = [mx.random.normal(shape) for _ in range(2)]
    return [flats[0], normals[0], flats[1], normals[1]]


def _trunk_output(model: _Trunk, plan: QuantizationPlan) -> mx.array:
    hidden = model.embed(mx.array(_TRUNK_SEQUENCES))
    for index, block in enumerate(model.layers):
        weight = block.proj.weight
        format = plan.get(f"layers.{index}.proj")
        if format is not None:
            assert isinstance(format, Affine)
            packed, scales, biases = mx.quantize(
                weight, group_size=format.group_size, bits=format.bits
            )
            weight = mx.dequantize(
                packed, scales, biases, group_size=format.group_size, bits=format.bits
            )
        hidden = hidden @ weight.T
    return hidden.astype(mx.float32)


def test_the_oq_plan_beats_uniform_rtn_at_the_same_effective_size() -> None:
    """The point of the sensitivity-guided budget: against uniform 6-bit RTN of the same
    physical size, spending the same bytes as 8 bits on the gaussian blocks and 4 on the
    nearly-constant ones ends closer to the float output."""
    from mlx_omnia.quant.calibration import BlockedForward
    from mlx_omnia.quant.oq import measure_sensitivity

    model = _Trunk(_trunk_weights())
    mx.eval(model.parameters())
    leaves = inventory(model)
    projections = [leaf for leaf in leaves if leaf.path != "embed"]

    six = Affine(group_size=64, bits=6)
    uniform: QuantizationPlan = {leaf.path: six for leaf in projections}
    uniform_cost = plan_cost(leaves, uniform)

    forward = BlockedForward(
        embed=lambda tokens: model.embed(tokens),
        blocks=[(f"layers.{index}", block) for index, block in enumerate(model.layers)],
        apply=_trunk_apply,
    )
    scores = measure_sensitivity(forward, _TRUNK_SEQUENCES, [_BASE])

    intent = QuantizationIntent(
        base=_BASE,
        target_bpw=uniform_cost.bits_per_weight,
        hard_cap_bpw=uniform_cost.bits_per_weight,
        overrides={"embed": None},
    )
    recipe = Recipe(
        identifier="test",
        version=1,
        promotions=(5, 6, 8),
    )
    allocation = OQAllocator(intent, recipe).allocate(leaves, scores)
    assert allocation.cost.total_bytes <= uniform_cost.total_bytes
    assert allocation.plan != uniform

    float_output = _trunk_output(model, {})
    oq_error = _relative_error(_trunk_output(model, allocation.plan), float_output)
    rtn_error = _relative_error(_trunk_output(model, uniform), float_output)
    assert oq_error < rtn_error


def _relative_error(quantized: mx.array, reference: mx.array) -> float:
    return (
        mx.abs(quantized - reference).max() / mx.abs(reference).max()
    ).item()

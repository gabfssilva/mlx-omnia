"""Pricing a selection: the same plan a job would run, resolved and costed without writing."""

from dataclasses import replace

from mlx_omnia.engine import task
from mlx_omnia.engine.footprint import checkpoint_bytes
from mlx_omnia.engine.quant.oq import RECIPE_OQ4_V1, OQAllocator
from mlx_omnia.engine.quant.quantization import expand_plan, inventory, leaf_cost, plan_cost
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.quantize.plan import (
    ALLOCATED,
    DTYPES,
    Conflict,
    Invalid,
    PlanLeaf,
    PricedPlan,
    Selection,
    Unknown,
    admissible,
    drafter_refusal,
    formats_of,
    intent_of,
    native_refusal,
)


def price(selection: Selection) -> PricedPlan:
    """The same selection a job would run, resolved and costed without writing anything.

    The dtype comes off the shards and not off the tree: the tree is built before
    `nn.quantize`, so every leaf carries mlx's default float32 and a bfloat16 checkpoint would
    be priced at twice its scales.

    AWQ and GPTQ are priced by this code because neither moves a leaf's format. oQ and oQe
    are priced by the same allocator the job runs, handed no scores: the reserved decisions
    and the budget the greedy fills to are known before the pass, so the totals hold and only
    which free leaf takes the promotion is the job's to decide.
    """
    admissible(selection)
    chosen = formats_of(selection)
    try:
        resolved = task.source(selection.source, local_files_only=True)
    except ValueError as error:
        raise Invalid(str(error)) from error
    except OSError as error:
        raise Unknown(f"{selection.source!r} is not on this disk") from error
    if "quantization" in resolved.config:
        raise Conflict(f"{selection.source!r} is already quantized")
    refusal = native_refusal(selection.source, resolved.directory)
    if refusal is not None:
        raise Conflict(refusal)
    refusal = drafter_refusal(selection.source, resolved.config, selection.method)
    if refusal is not None:
        raise Conflict(refusal)
    stored = catalog.weights_dtype(resolved.directory)
    dtype = DTYPES.get(stored or "")
    if dtype is None:
        raise Conflict(f"{selection.source!r} is stored as {stored}, which has no price")
    try:
        leaves = [replace(leaf, dtype=dtype) for leaf in inventory(resolved.pending.model)]
        plan = expand_plan(resolved.pending.model, chosen)
        if selection.method in ALLOCATED:
            intent = intent_of(selection, chosen, plan_cost(leaves, plan).bits_per_weight)
            plan = OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, ()).plan
        costs = {leaf.path: leaf_cost(leaf, plan.get(leaf.path)) for leaf in leaves}
        dense = sum(leaf_cost(leaf, None).total for leaf in leaves)
    except ValueError as error:
        raise Invalid(str(error)) from error
    cost = plan_cost(leaves, plan)
    return PricedPlan(
        leaves=[
            PlanLeaf(
                path=leaf.path,
                kind=leaf.kind,
                shape=leaf.shape,
                bits=None if (format := plan.get(leaf.path)) is None else format.bits,
                group_size=None if format is None else format.group_size,
                bytes=costs[leaf.path].total,
            )
            for leaf in leaves
        ],
        total_bytes=cost.total_bytes,
        weights=cost.weights,
        bits_per_weight=cost.bits_per_weight,
        entry_bytes=checkpoint_bytes(resolved.directory) - dense + cost.total_bytes,
    )

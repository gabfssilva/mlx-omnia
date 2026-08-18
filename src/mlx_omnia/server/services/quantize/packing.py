"""Packing a plan's leaves, one loop per rounding the method asks for."""

from collections.abc import Mapping

import mlx.core as mx

from mlx_omnia.engine.quant.awq import Applied, Outcome
from mlx_omnia.engine.quant.gptq import gptq, to_affine
from mlx_omnia.engine.quant.oqe import ImportanceMatrixAffine
from mlx_omnia.engine.quant.quantization import Affine, QuantizationPlan, quantize_weights
from mlx_omnia.server.services.quantize.plan import Reporter


def outcome_json(outcome: Outcome) -> dict[str, object]:
    """The scale itself does not travel: it is one number per input channel of the target,
    and what an audit answers is whether the pair was taken and at which alpha."""
    common: dict[str, object] = {
        "target": outcome.pair.target,
        "absorber": outcome.pair.absorber,
    }
    if isinstance(outcome, Applied):
        return {
            **common,
            "applied": True,
            "alpha": outcome.search.alpha,
            "clip": outcome.search.clip,
            "rtn_error": outcome.rtn_error,
            "error": outcome.search.error,
            "improvement": outcome.improvement,
        }
    return {**common, "applied": False, "reason": outcome.reason}


def pack(reporter: Reporter, weights: dict[str, mx.array], plan: QuantizationPlan) -> None:
    total = len(plan)
    for index, (path, format) in enumerate(plan.items()):
        # Before the leaf and not after it: the report is also where the work finds out it
        # was cancelled, and a 30B has minutes of packing behind each one.
        reporter.report(path, completed=index, total=total)
        quantize_weights(weights, {path: format})


def pack_gptq(
    reporter: Reporter,
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
    statistics: Mapping[str, mx.array],
) -> dict[str, object]:
    """The same loop, with the rounding replaced where there is a second moment to round
    against. A leaf the pass never observed — the embedding and the head sit outside the
    trunk — is packed by RTN and named."""
    total = len(plan)
    fallback: list[str] = []
    errors: dict[str, float] = {}
    for index, (path, format) in enumerate(plan.items()):
        reporter.report(path, completed=index, total=total)
        moment = statistics.get(f"{path}.second_moment")
        if moment is None or not isinstance(format, Affine):
            fallback.append(path)
            quantize_weights(weights, {path: format})
            continue
        result = gptq(weights.pop(f"{path}.weight"), moment, format)
        tensors = to_affine(result.weight).tensors(path)
        mx.eval(list(tensors.values()))
        weights.update(tensors)
        errors[path] = result.error
    return {"reconstruction_error": errors, "rtn_fallback": fallback}


def pack_oqe(
    reporter: Reporter,
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
    statistics: Mapping[str, mx.array],
) -> dict[str, object]:
    """`pack`, with the grid of each group searched against the leaf's own imatrix instead
    of read off its extremes. The fallback rule is GPTQ's, and for the same reason."""
    total = len(plan)
    fallback: list[str] = []
    for index, (path, format) in enumerate(plan.items()):
        reporter.report(path, completed=index, total=total)
        mean_square = statistics.get(f"{path}.mean_square")
        if mean_square is None or not isinstance(format, Affine):
            fallback.append(path)
            quantize_weights(weights, {path: format})
            continue
        quantize_weights(weights, {path: format}, method=ImportanceMatrixAffine(mean_square))
    return {"rtn_fallback": fallback}

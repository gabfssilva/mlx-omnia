"""The calibration pass a method that reads a corpus runs before anything is packed."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine import task
from mlx_omnia.engine.language import Prefill, Tokenizer, tokenizer_of, trunk_of
from mlx_omnia.engine.quant.calibration import (
    CalibrationConfig,
    Collector,
    Encoder,
    ImportanceMatrix,
    SecondMoment,
    format_tag,
    intercepted_collect,
    load_corpus,
    sample_sequences,
)
from mlx_omnia.engine.quant.oq import (
    RECIPE_OQ4_V1,
    OQAllocator,
    OqSensitivity,
    plan_provenance,
    provenance_json,
    widen,
)
from mlx_omnia.engine.quant.quantization import (
    Affine,
    ByPath,
    Leaf,
    Quantization,
    QuantizationPlan,
    inventory,
    plan_cost,
)
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.quantize.plan import (
    DTYPES,
    SEED,
    Reporter,
    Selection,
    intent_of,
)


def _leaves(directory: Path, model: nn.Module) -> list[Leaf]:
    """The tree's leaves priced in the dtype the shards carry: the lazy tree is float32
    whatever is on disk."""
    dtype = DTYPES.get(catalog.weights_dtype(directory) or "")
    leaves = inventory(model)
    return leaves if dtype is None else [replace(leaf, dtype=dtype) for leaf in leaves]


class _Encoder(Encoder):
    """A `language.Tokenizer` as the corpus sampler's encoder: a prompt may still be
    arriving, so `encode` hands back an iterator, and the sampler slices a list."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))


def _observe(
    reporter: Reporter,
    source: str,
    selection: Selection,
    collectors: Sequence[Collector],
    perturbations: Sequence[Quantization],
) -> CalibrationConfig:
    """The pass: the checkpoint loaded dense, the built-in corpus sampled with the model's
    own tokenizer, and the blocks intercepted inside the model's real prefill.

    The report is per sequence because a block is one call inside a forward this side never
    sees — and it is where a cancelled job stops.
    """
    reporter.report(f"loading {source}")
    loaded = task.load(source, local_files_only=True)
    tree = trunk_of(loaded)
    tokenizer = tokenizer_of(loaded)
    if tree is None:
        raise ValueError(f"{source!r} exposes no tree a calibration pass can run")
    if tokenizer is None:
        raise ValueError(f"{source!r} carries no tokenizer to sample the corpus with")
    forward: Prefill = tree
    corpus = load_corpus(seed=SEED)
    sampled = sample_sequences(
        corpus,
        _Encoder(tokenizer),
        sequences=selection.sequences,
        length=selection.sequence_length,
    )
    total = len(sampled)
    observed = 0

    def prefill(ids: mx.array) -> mx.array:
        nonlocal observed
        reporter.report(f"calibrating {source}", completed=observed, total=total)
        observed += 1
        return forward(ids)

    intercepted_collect(tree, prefill, sampled, collectors, perturbations=perturbations)
    return CalibrationConfig(
        corpus=corpus.name,
        corpus_digest=corpus.digest,
        seed=corpus.seed,
        sequences=total,
        sequence_length=selection.sequence_length,
        perturbations=tuple(format_tag(format) for format in perturbations),
    )


def _candidates(base: Affine) -> tuple[Quantization, ...]:
    """The widths every block is perturbed by: the one asked for, plus the ones the recipe
    may promote a leaf to. A width nobody measured is a width the allocator would order
    blocks by without having seen it."""
    promotions = (widen(base, bits) for bits in RECIPE_OQ4_V1.promotions)
    return (base, *(format for format in promotions if format != base))


@dataclass(frozen=True)
class Calibrated:
    """What the pass leaves behind for the packing that follows it: the plan (oQ's is not the
    selection's), what the collector accumulated, and the blocks the entry records."""

    plan: QuantizationPlan
    statistics: Mapping[str, mx.array]
    mlx_omnia: dict[str, object]
    config: dict[str, object]


def calibrated_by(
    reporter: Reporter,
    source: str,
    selection: Selection,
    formats: ByPath,
    checkpoint: task.Source,
    plan: QuantizationPlan,
) -> Calibrated:
    match selection.method:
        case "oq" | "oqe":
            base = Affine(group_size=selection.group_size, bits=selection.bits)
            sensitivity = OqSensitivity()
            # One pass for both: the block sensitivities order the promotions and the
            # imatrix rounds the leaves, and neither is worth a second forward.
            imatrix = ImportanceMatrix() if selection.method == "oqe" else None
            collectors: list[Collector] = [sensitivity]
            if imatrix is not None:
                collectors.append(imatrix)
            calibration = _observe(reporter, source, selection, collectors, _candidates(base))
            leaves = _leaves(checkpoint.directory, checkpoint.pending.model)
            intent = intent_of(selection, formats, plan_cost(leaves, plan).bits_per_weight)
            allocation = OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, sensitivity.scores())
            provenance = plan_provenance(allocation, RECIPE_OQ4_V1, intent, calibration)
            return Calibrated(
                allocation.plan,
                {} if imatrix is None else imatrix.statistics(),
                {"calibration": json.loads(calibration.to_json())},
                {"oq": provenance_json(provenance)},
            )
        case "gptq":
            hessian = SecondMoment()
            calibration = _observe(reporter, source, selection, [hessian], ())
            return Calibrated(
                plan,
                hessian.statistics(),
                {"calibration": json.loads(calibration.to_json())},
                {},
            )
        case _:
            imatrix = ImportanceMatrix()
            calibration = _observe(reporter, source, selection, [imatrix], ())
            return Calibrated(
                plan,
                imatrix.statistics(),
                {"calibration": json.loads(calibration.to_json())},
                {},
            )

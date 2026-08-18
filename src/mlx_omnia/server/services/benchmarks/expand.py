"""The cartesian product, taken once so the plan and the job cannot disagree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mlx_omnia.server.db.models.benchmarks import FidelityResult, QualityResult
from mlx_omnia.server.services import speed
from mlx_omnia.server.services.benchmarks.specs import (
    FIDELITY_TODO,
    MAX_SHAPES,
    QUALITY_TODO,
    FidelitySpec,
    Invalid,
    Planned,
    QualitySpec,
    Spec,
    SpeedSpec,
    Unknown,
    fidelity_key,
    quality_key,
)
from mlx_omnia.server.services.speed import Entry, ModelFacts, Refusal, SpeedShape


def facts_by_id(entries: Sequence[Entry]) -> Mapping[str, ModelFacts]:
    return {entry.id: speed.facts_of(entry) for entry in entries}


def _closed(name: str, asked: Sequence[int], allowed: tuple[int, ...]) -> None:
    unknown = sorted(set(asked) - set(allowed))
    if unknown:
        raise Invalid(f"{name}: {unknown} is outside the fixed set {list(allowed)}")


def expand(spec: Spec, entries: Sequence[Entry], budget_bytes: int) -> list[Planned]:
    """Every row the batch would write, decided ones included. The one place the product is
    taken, so the plan and the job cannot disagree."""
    known = {entry.id: entry for entry in entries}
    match spec:
        case SpeedSpec():
            return _expand_speed(spec, known, facts_by_id(entries), budget_bytes)
        case QualitySpec():
            return _expand_quality(spec, known)
        case FidelitySpec():
            return _expand_fidelity(spec, known)


def _missing(model: str, known: Mapping[str, Entry]) -> None:
    if model not in known:
        raise Unknown(f"{model!r} is not in the catalog")


def _expand_speed(
    spec: SpeedSpec,
    known: Mapping[str, Entry],
    facts: Mapping[str, ModelFacts],
    budget_bytes: int,
) -> list[Planned]:
    _closed("contexts", spec.contexts, speed.CONTEXTS)
    _closed("generates", spec.generates, speed.GENERATES)
    _closed("concurrencies", spec.concurrencies, speed.CONCURRENCIES)
    planned: list[Planned] = []
    for model in spec.models:
        _missing(model, known)
        for context in sorted(set(spec.contexts)):
            for generate in sorted(set(spec.generates)):
                for concurrency in sorted(set(spec.concurrencies)):
                    shape = SpeedShape(
                        context=context,
                        generate=generate,
                        concurrency=concurrency,
                        rounds=spec.rounds,
                        page_cache=spec.page_cache,
                        gate_c=spec.thermal_gate_c,
                        stream_source=spec.stream_source,
                        sampling=spec.sampling.shape(),
                    )
                    refused = speed.refusal(shape, facts[model], budget_bytes)
                    planned.append(
                        Planned(
                            model=model,
                            key=shape.key,
                            kind="speed",
                            body=speed.empty_result(shape, facts[model], refused),
                            shape=shape,
                            refusal=refused,
                        )
                    )
    if len(planned) > MAX_SHAPES:
        raise Invalid(f"{len(planned)} shapes asked for, {MAX_SHAPES} is the ceiling")
    return planned


def _expand_quality(spec: QualitySpec, known: Mapping[str, Entry]) -> list[Planned]:
    planned: list[Planned] = []
    for model in spec.models:
        _missing(model, known)
        for dataset in spec.datasets:
            planned.append(
                Planned(
                    model=model,
                    key=quality_key(dataset, spec.items, spec.seed, spec.shots),
                    kind="quality",
                    body=QualityResult(
                        run_id="",
                        dataset=dataset,
                        items=spec.items,
                        seed=spec.seed,
                        shots=spec.shots,
                        scoring=spec.scoring,
                    ),
                    refusal=Refusal(reason="not_implemented", detail=QUALITY_TODO),
                )
            )
    return planned


def _expand_fidelity(spec: FidelitySpec, known: Mapping[str, Entry]) -> list[Planned]:
    planned: list[Planned] = []
    for pair in spec.pairs:
        _missing(pair.model, known)
        _missing(pair.reference, known)
        candidate, reference = known[pair.model], known[pair.reference]
        # Vocabulary is what decides whether the two vectors can be subtracted at all: a
        # mismatched pair costs nothing to refuse and would cost a load to discover.
        mismatch = (
            candidate.vocab_size is not None
            and reference.vocab_size is not None
            and candidate.vocab_size != reference.vocab_size
        )
        refused = (
            Refusal(
                reason="vocabulary_mismatch",
                detail=(
                    f"{pair.model!r} draws from {candidate.vocab_size} ids and"
                    f" {pair.reference!r} from {reference.vocab_size}: their logits are not"
                    " the same vector"
                ),
            )
            if mismatch
            else Refusal(reason="not_implemented", detail=FIDELITY_TODO)
        )
        # Same vocabulary is necessary and not sufficient, and it warns rather than refuses:
        # a fine-tune prints identically to its base, so blocking on a different print would
        # block only the pairs somebody deliberately declared.
        warning = (
            None
            if mismatch or candidate.shape is None or candidate.shape == reference.shape
            else (
                f"{pair.model!r} is {candidate.shape} and {pair.reference!r} is"
                f" {reference.shape}: the vocabularies match, the architectures do not"
            )
        )
        planned.append(
            Planned(
                model=pair.model,
                key=fidelity_key(spec.corpus, spec.tokens, spec.seed, pair.reference),
                kind="fidelity",
                body=FidelityResult(
                    run_id="",
                    reference=pair.reference,
                    corpus=spec.corpus,
                    tokens=spec.tokens,
                    seed=spec.seed,
                    topk=spec.topk,
                ),
                refusal=refused,
                warning=warning,
            )
        )
    return planned

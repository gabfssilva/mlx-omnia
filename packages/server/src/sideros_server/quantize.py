"""Quantizing a checkpoint here is creating a model, so it is `POST /admin/quantizations`
and a job — and what comes out is addressed by the repo id the screen asks for.

Where the entry lands is the whole of the second decision. `load(quantize=…)` writes under
`~/.cache/sideros/quantized/<name>/<digest>`: the digest of the plan over the source *is*
the address, which is what makes the second load with the same selection a hit and nobody's
job to name. An artifact somebody asked for is named by whoever asked, and the name the
screen offers is a repo id — so the entry goes into the hub cache under
`models--org--name/snapshots/<digest>`, the layout `catalog.scan` already lists by id and
`hf_hub_download` already resolves off `refs/main` when there is no Hub to ask. The digest
does not disappear with the path it used to be: it is written into the provenance next to
the fingerprint the source was already recorded by.

The `local/` prefix the screen suggests is worth keeping. A load by id asks the Hub first
and only falls back to `refs/main` when it gets nothing, so a name that also exists up
there is a name that resolves to somebody else's weights.

The entry takes its final name by a rename, for the reason `download` renames: a directory
under the name the catalog reads is a model the loader will open, and a half-written one is
a catalog entry that fails at request time. What is staged here is worth nothing to a
second attempt — a half-quantized checkpoint is not bytes anyone resumes from — so unlike a
download it is dropped on **either** ending, and cancelling mid-write leaves neither the
staging nor the entry.

The source is resolved with `local_files_only=True`: quantizing reads the disk the daemon
already owns, and a repository nobody downloaded is `POST /admin/models`, not this. That is
also what keeps the provenance a hub model deserves — repository plus commit sha, the
fingerprint `load` writes — instead of the anonymous directory heuristic a path would get.

All five methods run here. Four of them read a calibration pass, and that pass no longer
asks an architecture to describe its own trunk: `intercepted_collect` finds the blocks in
the tree and stands in for each one, so what supplies a block's arguments — the mask, the
rope pair, the cache — is the model's own prefill. What this route adds is the rest of the
selection: the corpus is the engine's `CORPUS_V1` at seed 0, and the request says how much
of it to read and, for oQ, the budget to allocate against.

The dense model is loaded a second time for the pass. The lazy tree a plan resolves against
carries no weights and the pass runs the real forward; the loaded model is dropped before
the weight dict is read, so the two copies are never resident together.

What each method does with what it measured is where the four part. oQ turns the block
sensitivities into a plan and the packing is the ordinary one. AWQ rewrites the dense
weights before that same packing, over the pairs the tree admits exactly. GPTQ replaces the
packing leaf by leaf, and a leaf the pass never observed — the embedding and the head sit
outside the trunk — falls back to RTN and is named in the provenance instead of dropped in
silence. oQe is oQ's plan with GPTQ's shape of packing: the same pass carries the imatrix
along with the sensitivities, the grid of every group is searched against it, and what was
never observed falls back to RTN by the same rule. Everything the four refused or measured
leaves as the entry's provenance, which is the only place a job has to say it: nothing here
writes to stdout.

`POST /admin/quantizations/plan` prices that same selection and writes nothing. It resolves
against the same lazy tree the job does, so a source the job would refuse is a source it
refuses too — today, any architecture whose `Checkpoint` declares no `quantize`. AWQ and
GPTQ are priced as RTN and by the same code, because their plan *is* RTN's: neither moves a
leaf's format, and the bytes of an entry are the formats alone. oQ and oQe are priced by the
allocator the job runs, handed no scores: the reserved decisions and the budget the greedy
fills to are known before the pass, so the totals hold and only which free leaf takes the
promotion is the job's to decide.
"""

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from sideros import task
from sideros.footprint import checkpoint_bytes
from sideros.language import Prefill, tokenizer_of, trunk_of
from sideros.quant.awq import Applied, Outcome, apply_awq, derive_pairs
from sideros.quant.calibration import (
    CalibrationConfig,
    Collector,
    ImportanceMatrix,
    SecondMoment,
    format_tag,
    intercepted_collect,
    load_corpus,
    sample_sequences,
)
from sideros.quant.gptq import gptq, to_affine
from sideros.quant.oq import (
    RECIPE_OQ4_V1,
    OQAllocator,
    OqSensitivity,
    plan_provenance,
    provenance_json,
    widen,
)
from sideros.quant.oqe import ImportanceMatrixAffine
from sideros.quant.quantization import (
    MXFP,
    NVFP,
    Affine,
    ByPath,
    Leaf,
    Quantization,
    QuantizationIntent,
    QuantizationPlan,
    expand_plan,
    inventory,
    leaf_cost,
    plan_cost,
    quantize_weights,
)
from sideros_server import catalog
from sideros_server.jobs import Job, Jobs, JobsDep, Progress, Quantize, Work, accepted

STAGING = ".quantizing"
"""Public because `download`'s staging window lists it: what is taking space and can be
dropped is one question, and a caller should not have to know which job wrote the bytes."""

_QUANTIZING: dict[str, str] = {}
"""The last job started for each output repo, which is how a second `POST` finds the first."""


def _slug(repository: str) -> str:
    """The folder the hub cache gives a repository."""
    return f"models--{repository.replace('/', '--')}"


def _running(registry: Jobs, repository: str) -> str | None:
    """The job writing it, if one still is. The registry has the last word rather than the
    dict: a job cancelled before its work ever ran leaves its entry behind, and a repo id
    that could never be produced again is a worse leak than a stale key."""
    job_id = _QUANTIZING.get(repository)
    return None if job_id is None or registry.live(job_id) is None else job_id


def _install(staged: Path, digest: str, final: Path) -> None:
    """The rename that gives the entry the id it was asked for. The reference is written
    here rather than left to a resolver: `refs/main` is how the catalog decides which
    snapshot is the entry, and how `sideros.load` finds it with no Hub to ask."""
    reference = staged / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(digest)
    final.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(final)


class Override(BaseModel):
    """One group of leaves against the rest of the plan. `group_size` absent is the plan's
    own — the group size is a decision about the whole checkpoint, and a leaf that departs
    from it says so."""

    model_config = ConfigDict(extra="forbid")

    bits: int
    group_size: int | None = None


class PlanRequest(BaseModel):
    """The selection alone — everything but where the result lands, which pricing does not
    need because nothing is written."""

    model_config = ConfigDict(extra="forbid")

    source: str
    """A catalog id: a repository already on disk, or the directory of a quantized entry."""
    mode: Literal["affine", "mxfp4", "mxfp8", "nvfp4"] = "affine"
    """The grid the codes are read against. Affine is a scale and a bias per group; the other
    three carry a power-of-two — or, in nvfp4, an E4M3 — exponent per group and no bias, so
    the mode alone fixes `group_size` and `bits`."""
    bits: int = 4
    group_size: int = 64
    overrides: dict[str, Override | None] = Field(default_factory=dict)
    """Format per group of leaves, keyed by the same `fullmatch` pattern the engine's
    selection takes; `null` says dense. A pattern matching no leaf fails the job, which is
    the engine's own rule about a selection that does not resolve."""
    method: Literal["rtn", "awq", "gptq", "oq", "oqe"] = "rtn"
    """The engine's five, and all five run. What separates them is the calibration pass the
    last four read; `rtn` reads none, which is why every field below is refused with it."""
    sequences: int = Field(default=16, gt=0)
    """How many windows of the corpus the pass runs. The corpus itself is not a field: it is
    the engine's `CORPUS_V1` at seed 0, so two identical requests observe identical rows."""
    sequence_length: int = Field(default=256, gt=0)
    target_bpw: float | None = None
    """The allocator's budget in effective bits per weight — physical bytes, scales and
    biases included. Absent, it is the RTN plan this same request would have produced plus
    `_HEADROOM`: a budget under what RTN costs is a request for a smaller checkpoint, and
    the allocator can only refuse it."""
    hard_cap_bpw: float | None = None
    """The width no promotion may cross. Absent, the target is also the cap."""


class QuantizeRequest(PlanRequest):
    repo: str = Field(pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    """Where it lands, and the id it is served under. Two segments, and nothing that could
    name a directory outside the hub cache."""


_CALIBRATION_FIELDS = ("sequences", "sequence_length")
_BUDGET_FIELDS = ("target_bpw", "hard_cap_bpw")

_ALLOCATED = ("oq", "oqe")
"""The two methods whose plan is the allocator's rather than the selection's: what oQe adds
to oQ is the rounding, not the widths."""

_GPTQ_BITS = (2, 4, 8)
"""The widths whose uint32 packing is verified against `mx.quantize`. GPTQ packs its own
codes — the scales a group was rounded against are fixed before the columns are compensated
— so 3, 5 and 6 have no layout here, and refusing them is not refusing the method."""

_HEADROOM = 1.0
"""Bits per weight the default oQ budget leaves above the RTN plan of the same request. The
promotions are 5, 6 and 8 bits over a 4-bit base: under a bit of headroom the greedy buys
nothing and oQ is an expensive RTN."""

_SHAPE: dict[str, tuple[int, int]] = {"mxfp4": (32, 4), "mxfp8": (32, 8), "nvfp4": (16, 4)}
"""Group size and width of each exponent-scaled mode — the one pair `mx.quantize` packs and
the engine's own `__post_init__` admits."""

_SEED = 0
"""The corpus draws its windows from a generator seeded with this, so the pass — and every
accumulator's summation order — is a function of the request alone."""

_DTYPES: dict[str, mx.Dtype] = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


def _admissible(request: PlanRequest) -> None:
    """What is still impossible or meaningless, answered from the request alone — before the
    source is resolved and before a job exists. A calibration selection under `rtn` is not a
    default this route can quietly ignore: it is a caller who believes a pass will run."""
    set_fields = request.model_fields_set
    if request.method == "rtn":
        named = sorted(set_fields.intersection((*_CALIBRATION_FIELDS, *_BUDGET_FIELDS)))
        if named:
            raise HTTPException(
                status_code=400,
                detail=f"'rtn' reads no calibration, so it takes none of {named}",
            )
    if request.method not in _ALLOCATED:
        named = sorted(set_fields.intersection(_BUDGET_FIELDS))
        if named:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{request.method!r} keeps the plan the selection asked for, so it "
                    f"allocates no budget: {named} is the allocator's alone"
                ),
            )
    if request.mode != "affine":
        named = sorted(set_fields.intersection(("bits", "group_size")))
        if named:
            raise HTTPException(
                status_code=400,
                detail=f"{request.mode!r} fixes {_SHAPE[request.mode]}, so it takes no {named}",
            )
        if request.method != "rtn":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{request.method!r} searches a scale and a bias per group, which "
                    f"{request.mode!r} does not have: the exponent modes are rtn's"
                ),
            )
        named = sorted(
            pattern for pattern, override in request.overrides.items() if override is not None
        )
        if named:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"under {request.mode!r} the format is the mode, so an override can only "
                    f"be null (dense): {named}"
                ),
            )
    if request.method == "gptq" and request.bits not in _GPTQ_BITS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"gptq packs its own codes and only {list(_GPTQ_BITS)} have a layout "
                f"verified against mx.quantize, not {request.bits}"
            ),
        )


def _drafter_refusal(source: str, config: Mapping[str, object], method: str) -> str | None:
    """Why a drafter takes `rtn` and nothing else, or `None` when the pair is fine.

    Nothing structural: `intercepted_collect` finds the blocks in any tree and the packing
    never knew what a model is. What a drafter has not got is a way to be *read* — its
    input is not tokens but the target's hidden states, so there is no tokenizer to run a
    corpus through and no forward that a sequence of ids alone can drive. Calibrating one
    means loading its target too, and nothing in a drafter's config names one.
    """
    model_type = config.get("model_type")
    if method == "rtn" or not isinstance(model_type, str) or not task.drafts(model_type):
        return None
    return (
        f"{source!r} is a drafter: its input is the target's hidden states and not tokens, "
        f"so there is no corpus to calibrate it against — {method!r} needs one, 'rtn' does not"
    )


def _format(request: PlanRequest, bits: int, group_size: int | None = None) -> Quantization:
    match request.mode:
        case "affine":
            return Affine(group_size=group_size or request.group_size, bits=bits)
        case "mxfp4" | "mxfp8":
            group_size, width = _SHAPE[request.mode]
            return MXFP(mode=request.mode, group_size=group_size, bits=width)
        case "nvfp4":
            group_size, width = _SHAPE[request.mode]
            return NVFP(group_size=group_size, bits=width)


def _selection(request: PlanRequest) -> ByPath:
    """The screen's controls as the engine's selection. The widths a format admits are the
    engine's table, not a second one here: an unsupported pair raises out of `Affine` and
    this route answers with what it said."""
    overrides: dict[str | re.Pattern[str], Quantization | None] = {
        pattern: None if override is None else _format(request, override.bits, override.group_size)
        for pattern, override in request.overrides.items()
    }
    return ByPath(_format(request, request.bits), overrides)


def _leaves(directory: Path, model: nn.Module) -> list[Leaf]:
    """The tree's leaves priced in the dtype the shards carry. The lazy tree is float32
    whatever is on disk, and a budget in bits per weight read off it would be the budget of
    a checkpoint twice this one's size."""
    dtype = _DTYPES.get(catalog.weights_dtype(directory) or "")
    leaves = inventory(model)
    return leaves if dtype is None else [replace(leaf, dtype=dtype) for leaf in leaves]


def _observe(
    job: Job,
    source: str,
    request: QuantizeRequest,
    collectors: Sequence[Collector],
    perturbations: Sequence[Quantization],
) -> CalibrationConfig:
    """The pass, and everything this route knows about running one: the checkpoint loaded
    dense, the built-in corpus sampled with the model's own tokenizer, and the blocks
    intercepted inside the model's real prefill.

    The report is per sequence because a block is one call inside a forward this side never
    sees — and it is where a cancelled job stops, which on a 30B is the only place it can.
    """
    job.report(Progress(message=f"loading {source}"))
    loaded = task.load(source, local_files_only=True)
    tree = trunk_of(loaded)
    tokenizer = tokenizer_of(loaded)
    if tree is None:
        raise ValueError(f"{source!r} exposes no tree a calibration pass can run")
    if tokenizer is None:
        raise ValueError(f"{source!r} carries no tokenizer to sample the corpus with")
    forward: Prefill = tree
    corpus = load_corpus(seed=_SEED)
    sampled = sample_sequences(
        corpus,
        tokenizer,
        sequences=request.sequences,
        length=request.sequence_length,
    )
    total = len(sampled)
    observed = 0

    def prefill(ids: mx.array) -> mx.array:
        nonlocal observed
        job.report(Progress(message=f"calibrating {source}", completed=observed, total=total))
        observed += 1
        return forward(ids)

    intercepted_collect(tree, prefill, sampled, collectors, perturbations=perturbations)
    return CalibrationConfig(
        corpus=corpus.name,
        corpus_digest=corpus.digest,
        seed=corpus.seed,
        sequences=total,
        sequence_length=request.sequence_length,
        perturbations=tuple(format_tag(format) for format in perturbations),
    )


def _candidates(base: Affine) -> tuple[Quantization, ...]:
    """The widths every block is perturbed by: the one the request asked for, plus the ones
    the recipe may promote a leaf to — all at the request's own group size, which is where
    the allocator will spend them. A width nobody measured is a width the allocator would be
    ordering blocks by without having seen it."""
    promotions = (widen(base, bits) for bits in RECIPE_OQ4_V1.promotions)
    return (base, *(format for format in promotions if format != base))


def _intent(request: PlanRequest, selection: ByPath, floor: float) -> QuantizationIntent:
    target = request.target_bpw if request.target_bpw is not None else floor + _HEADROOM
    return QuantizationIntent(
        base=Affine(group_size=request.group_size, bits=request.bits),
        target_bpw=target,
        hard_cap_bpw=request.hard_cap_bpw,
        overrides=selection.overrides,
    )


def _outcome_json(outcome: Outcome) -> dict[str, object]:
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


def _pack(job: Job, weights: dict[str, mx.array], plan: QuantizationPlan) -> None:
    total = len(plan)
    for index, (path, format) in enumerate(plan.items()):
        # Before the leaf and not after it: the report is also where the work finds out it
        # was cancelled, and a 30B has minutes of packing behind each one.
        job.report(Progress(message=path, completed=index, total=total))
        quantize_weights(weights, {path: format})


def _pack_gptq(
    job: Job,
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
    statistics: Mapping[str, mx.array],
) -> dict[str, object]:
    """The same loop, with the rounding replaced where there is a second moment to round
    against. A leaf the pass never observed — the embedding and the head are outside the
    trunk, and so is anything an architecture keeps there — is packed by RTN and named: a
    checkpoint half of whose leaves are not GPTQ's is a fact about the file."""
    total = len(plan)
    fallback: list[str] = []
    errors: dict[str, float] = {}
    for index, (path, format) in enumerate(plan.items()):
        job.report(Progress(message=path, completed=index, total=total))
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


def _pack_oqe(
    job: Job,
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
    statistics: Mapping[str, mx.array],
) -> dict[str, object]:
    """`_pack`, with the grid of each group searched against the leaf's own imatrix instead
    of read off its extremes. The fallback rule is GPTQ's, and for the same reason: a leaf
    the pass never observed has no statistic to round against, so it is packed by RTN and
    named — which leaves are oQe's is a fact about the file."""
    total = len(plan)
    fallback: list[str] = []
    for index, (path, format) in enumerate(plan.items()):
        job.report(Progress(message=path, completed=index, total=total))
        mean_square = statistics.get(f"{path}.mean_square")
        if mean_square is None or not isinstance(format, Affine):
            fallback.append(path)
            quantize_weights(weights, {path: format})
            continue
        quantize_weights(weights, {path: format}, method=ImportanceMatrixAffine(mean_square))
    return {"rtn_fallback": fallback}


@dataclass(frozen=True)
class _Calibrated:
    """What the pass leaves behind for the packing that follows it: the plan (oQ's is not
    the selection's), what the collector accumulated, and the blocks the entry records."""

    plan: QuantizationPlan
    statistics: Mapping[str, mx.array]
    sideros: dict[str, object]
    config: dict[str, object]


def _calibrated(
    job: Job,
    source: str,
    request: QuantizeRequest,
    selection: ByPath,
    checkpoint: task.Source,
    plan: QuantizationPlan,
) -> _Calibrated:
    match request.method:
        case "oq" | "oqe":
            base = Affine(group_size=request.group_size, bits=request.bits)
            sensitivity = OqSensitivity()
            # One pass for both: the block sensitivities order the promotions and the
            # imatrix rounds the leaves, and neither is worth a second forward.
            imatrix = ImportanceMatrix() if request.method == "oqe" else None
            collectors: list[Collector] = [sensitivity]
            if imatrix is not None:
                collectors.append(imatrix)
            calibration = _observe(job, source, request, collectors, _candidates(base))
            leaves = _leaves(checkpoint.directory, checkpoint.pending.model)
            intent = _intent(request, selection, plan_cost(leaves, plan).bits_per_weight)
            allocation = OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, sensitivity.scores())
            provenance = plan_provenance(allocation, RECIPE_OQ4_V1, intent, calibration)
            return _Calibrated(
                allocation.plan,
                {} if imatrix is None else imatrix.statistics(),
                {"calibration": json.loads(calibration.to_json())},
                {"oq": provenance_json(provenance)},
            )
        case "gptq":
            hessian = SecondMoment()
            calibration = _observe(job, source, request, [hessian], ())
            return _Calibrated(
                plan,
                hessian.statistics(),
                {"calibration": json.loads(calibration.to_json())},
                {},
            )
        case _:
            imatrix = ImportanceMatrix()
            calibration = _observe(job, source, request, [imatrix], ())
            return _Calibrated(
                plan,
                imatrix.statistics(),
                {"calibration": json.loads(calibration.to_json())},
                {},
            )


def _quantize(source: str, repo: str, selection: ByPath, request: QuantizeRequest) -> Work:
    def run(job: Job) -> None:
        # A report first: a job cancelled while it waited for a thread must not open the
        # checkpoint, let alone start packing it.
        job.report(Progress(message=f"reading {source}"))
        checkpoint = task.source(source, local_files_only=True)
        refusal = _drafter_refusal(source, checkpoint.config, request.method)
        if refusal is not None:
            raise ValueError(refusal)
        plan = expand_plan(checkpoint.pending.model, selection)
        calibrated = (
            _Calibrated(plan, {}, {}, {})
            if request.method == "rtn"
            else _calibrated(job, source, request, selection, checkpoint, plan)
        )
        plan = calibrated.plan
        # After the pass and not before it: the dense model it loaded is a second copy of
        # the checkpoint, and it is gone by the time these bytes are read.
        weights = checkpoint.pending.weights()
        total = len(plan)
        staging = catalog.HUB_CACHE / STAGING / _slug(repo)
        entry = staging / "snapshots" / task.digest(checkpoint.fingerprint, plan, None)
        recorded = {
            **task.provenance(
                checkpoint.fingerprint, revision=None, dtype=None, method=request.method
            ),
            **calibrated.sideros,
            "repo": repo,
            "digest": entry.name,
        }
        try:
            if request.method == "awq":
                pairs = derive_pairs(checkpoint.pending.model)
                outcomes = apply_awq(weights, plan, calibrated.statistics, pairs, prefix="")
                recorded["awq"] = [_outcome_json(outcome) for outcome in outcomes]
            if request.method == "gptq":
                recorded["gptq"] = _pack_gptq(job, weights, plan, calibrated.statistics)
            elif request.method == "oqe":
                recorded["oqe"] = _pack_oqe(job, weights, plan, calibrated.statistics)
            else:
                _pack(job, weights, plan)
            job.report(Progress(message=f"writing {repo}", completed=total, total=total))
            task.write_entry(
                entry,
                checkpoint.directory,
                checkpoint.patterns,
                {**checkpoint.config, **calibrated.config, "sideros": recorded},
                weights,
                plan,
            )
            # The last report before the entry takes the name the catalog reads: a
            # cancellation that arrived while the file was being written lands here.
            job.report(Progress(message=repo, completed=total, total=total))
            _install(staging, entry.name, catalog.HUB_CACHE / _slug(repo))
        except BaseException:
            # Nothing staged here is worth keeping: a half-quantized checkpoint is not a
            # resume, so a failure ends the way a cancellation does.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def work(job: Job) -> None:
        """`run` in a frame of its own, and the reason is the order: the packed weights are
        a whole checkpoint and they are only unreachable once that frame is gone, so a
        `clear_cache` from inside it would find them live and free nothing. Outside it, what
        the job read and packed goes back to the system instead of into MLX's own pool —
        an unload's reason, and the same one: the footprint is what the memory rail reads
        and what admission decides against, and a quantization is the largest thing this
        process ever allocates."""
        try:
            run(job)
        finally:
            mx.clear_cache()

    return work


router = APIRouter()


@router.post("/admin/quantizations", status_code=202)
async def create(request: QuantizeRequest, registry: JobsDep) -> JSONResponse:
    """On the loop, because `start` captures it to hand frames back from the worker thread —
    which is also what makes the two checks below a gate: a dict lookup and one `stat`, with
    no `await` between reading them and registering the job that answers them."""
    _admissible(request)
    try:
        selection = _selection(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    # Two jobs on one repo id are two jobs staging into one directory, and the first to
    # finish renames it out from under the second.
    if (job_id := _running(registry, request.repo)) is not None:
        raise HTTPException(
            status_code=409, detail=f"{request.repo!r} is already being written by job {job_id}"
        )
    # The folder rather than the catalog: a name already taken is found out by the rename
    # otherwise, which is after the whole checkpoint has been read, packed and written.
    if (catalog.HUB_CACHE / _slug(request.repo)).exists():
        raise HTTPException(status_code=409, detail=f"{request.repo!r} is already on disk")
    job = registry.start(
        Quantize(model=request.source, target=request.repo),
        _quantize(request.source, request.repo, selection, request),
    )
    _QUANTIZING[request.repo] = job.id
    return accepted(job)


class PlanLeaf(BaseModel):
    path: str
    kind: str
    """The leaf's own class in the tree — `Linear`, `Embedding`, `SwitchLinear`. A
    load-time fusion is one leaf here, not the two or three tensors the checkpoint has."""
    shape: tuple[int, ...]
    bits: int | None
    """`null` says the leaf stays dense, which is what the default or an override asked."""
    group_size: int | None
    """`null` for the same reason, and never the request's own field read back: under an
    exponent-scaled mode or an override of its own a leaf carries a group nobody typed."""
    bytes: int


class PricedPlan(BaseModel):
    leaves: list[PlanLeaf]
    total_bytes: int
    """Physical bytes of the leaves under the plan — codes, scales and biases, which is why
    4-bit costs 4.5 effective bits and not 4."""
    weights: int
    bits_per_weight: float
    entry_bytes: int
    """What the produced `model.safetensors` will weigh: the leaves at their new width plus
    everything no plan touches (norms, biases, conv taps), which stays as it is. A source
    that ties its head and still serializes `lm_head` counts that dropped copy here, the
    same upper bound `checkpoint_bytes` documents."""


@router.post("/admin/quantizations/plan")
def price(request: PlanRequest) -> PricedPlan:
    """The same selection `POST /admin/quantizations` would run, resolved and costed without
    writing anything: the source resolves the way the job resolves it, the lazy tree gives
    the leaves and their shapes, and no weight is read.

    The dtype comes off the shards and not off the tree, which is the one thing that would
    be wrong here: the tree is built before `nn.quantize`, so every leaf on it carries mlx's
    default float32, and a bfloat16 checkpoint would be priced at twice its scales and twice
    whatever the plan leaves dense.

    AWQ and GPTQ are priced here and by this code: neither moves a leaf's format — one
    rewrites the dense weights before the packing, the other replaces the rounding inside it
    — and what an entry weighs is the formats alone. oQ is priced by the same allocator the
    job runs, handed no scores: the sensitivity only orders the spending, and the reserved
    decisions (overrides, protections, floors) and the budget the greedy fills to are all
    known before the pass. The totals hold; which free leaf ends up promoted is the one
    thing the job decides differently, so a leaf's projected width is provisional where its
    reason would be "promotion".

    Sync like the catalog's own handlers — building the tree of a 35B takes 5 ms and reading
    the headers a few more, and neither is the loop's work.
    """
    _admissible(request)
    try:
        selection = _selection(request)
        resolved = task.source(request.source, local_files_only=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=404, detail=f"{request.source!r} is not on this disk"
        ) from error
    if "quantization" in resolved.config:
        raise HTTPException(status_code=409, detail=f"{request.source!r} is already quantized")
    refusal = _drafter_refusal(request.source, resolved.config, request.method)
    if refusal is not None:
        raise HTTPException(status_code=409, detail=refusal)
    stored = catalog.weights_dtype(resolved.directory)
    dtype = _DTYPES.get(stored or "")
    if dtype is None:
        raise HTTPException(
            status_code=409, detail=f"{request.source!r} is stored as {stored}, which has no price"
        )
    try:
        leaves = [replace(leaf, dtype=dtype) for leaf in inventory(resolved.pending.model)]
        plan = expand_plan(resolved.pending.model, selection)
        if request.method in _ALLOCATED:
            intent = _intent(request, selection, plan_cost(leaves, plan).bits_per_weight)
            plan = OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, ()).plan
        costs = {leaf.path: leaf_cost(leaf, plan.get(leaf.path)) for leaf in leaves}
        dense = sum(leaf_cost(leaf, None).total for leaf in leaves)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
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

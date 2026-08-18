"""The job body that writes the entry, and the reservation of the repo id it writes under.

The result lands in the hub cache under the repo id the caller asked for, staged beside it
and renamed into place — a half-written directory under the name the catalog reads is a model
the loader would open. Nothing staged here is worth a resume, so it is dropped on either
ending.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine import task
from mlx_omnia.engine.quant.awq import apply_awq, derive_pairs
from mlx_omnia.engine.quant.quantization import ByPath, expand_plan
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.quantize.calibrate import Calibrated, calibrated_by
from mlx_omnia.server.services.quantize.packing import (
    outcome_json,
    pack,
    pack_gptq,
    pack_oqe,
)
from mlx_omnia.server.services.quantize.plan import (
    Conflict,
    Reporter,
    Request,
    drafter_refusal,
    native_refusal,
    slug,
)

STAGING = ".quantizing"
"""Public because the download service's staging window lists it: what is taking space and
can be dropped is one question, and a caller should not have to know which job wrote it."""

_QUANTIZING: dict[str, str] = {}
"""The last job started for each output repo, which is how a second request finds the first."""


def _install(staged: Path, digest: str, final: Path) -> None:
    """`refs/main` is how the catalog decides which snapshot is the entry, and how
    `mlx_omnia.load` finds it with no Hub to ask."""
    reference = staged / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(digest)
    final.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(final)


def work(request: Request, formats: ByPath) -> Callable[[Reporter], None]:
    """The job body. It reports before every step it could be cancelled during."""
    selection = request.selection
    source = selection.source
    repo = request.repo

    def run(reporter: Reporter) -> None:
        # A report first: a job cancelled while it waited for a thread must not open the
        # checkpoint, let alone start packing it.
        reporter.report(f"reading {source}")
        checkpoint = task.source(source, local_files_only=True)
        refusal = native_refusal(source, checkpoint.directory) or drafter_refusal(
            source, checkpoint.config, selection.method
        )
        if refusal is not None:
            raise ValueError(refusal)
        plan = expand_plan(checkpoint.pending.model, formats)
        calibrated = (
            Calibrated(plan, {}, {}, {})
            if selection.method == "rtn"
            else calibrated_by(reporter, source, selection, formats, checkpoint, plan)
        )
        plan = calibrated.plan
        # After the pass and not before it: the dense model it loaded is a second copy of
        # the checkpoint, and it is gone by the time these bytes are read.
        weights = checkpoint.pending.weights()
        total = len(plan)
        staging = catalog.HUB_CACHE / STAGING / slug(repo)
        entry = staging / "snapshots" / task.digest(checkpoint.fingerprint, plan, None)
        recorded = {
            **task.provenance(
                checkpoint.fingerprint, revision=None, dtype=None, method=selection.method
            ),
            **calibrated.mlx_omnia,
            "repo": repo,
            "digest": entry.name,
        }
        try:
            if selection.method == "awq":
                pairs = derive_pairs(checkpoint.pending.model)
                outcomes = apply_awq(weights, plan, calibrated.statistics, pairs, prefix="")
                recorded["awq"] = [outcome_json(outcome) for outcome in outcomes]
            if selection.method == "gptq":
                recorded["gptq"] = pack_gptq(reporter, weights, plan, calibrated.statistics)
            elif selection.method == "oqe":
                recorded["oqe"] = pack_oqe(reporter, weights, plan, calibrated.statistics)
            else:
                pack(reporter, weights, plan)
            reporter.report(f"writing {repo}", completed=total, total=total)
            task.write_entry(
                entry,
                checkpoint.directory,
                checkpoint.patterns,
                {**checkpoint.config, **calibrated.config, "mlx_omnia": recorded},
                weights,
                plan,
                # The MTP head rides along, packed round-to-nearest whatever the trunk's
                # method was, which is what keeps a quantized entry able to speculate.
                selection=formats,
            )
            # The last report before the entry takes the name the catalog reads: a
            # cancellation that arrived while the file was being written lands here.
            reporter.report(repo, completed=total, total=total)
            _install(staging, entry.name, catalog.HUB_CACHE / slug(repo))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def body(reporter: Reporter) -> None:
        """`run` in a frame of its own: the packed weights are a whole checkpoint and they
        are only unreachable once that frame is gone, so a `clear_cache` from inside it would
        find them live and free nothing."""
        try:
            run(reporter)
        finally:
            mx.clear_cache()

    return body


def reserve(repo: str, running: Callable[[str], bool]) -> None:
    """That the repo id is free. Two jobs on one id are two jobs staging into one directory,
    and the first to finish renames it out from under the second.

    The registry has the last word rather than the dict: a job cancelled before its work ever
    ran leaves its entry behind, and a repo id that could never be produced again is a worse
    leak than a stale key.
    """
    job_id = _QUANTIZING.get(repo)
    if job_id is not None and running(job_id):
        raise Conflict(f"{repo!r} is already being written by job {job_id}")
    # The folder rather than the catalog: a name already taken is found out by the rename
    # otherwise, which is after the whole checkpoint has been read, packed and written.
    if (catalog.HUB_CACHE / slug(repo)).exists():
        raise Conflict(f"{repo!r} is already on disk")


def claim(repo: str, job_id: str) -> None:
    _QUANTIZING[repo] = job_id

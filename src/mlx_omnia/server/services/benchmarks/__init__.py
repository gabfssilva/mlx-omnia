"""The batch, its dry run, and the history it writes.

A benchmark is a cartesian product of shapes that runs for minutes, so the work holds the
engine's queue for as long as it measures and everything else waits at `submit` instead of
landing between two rounds and moving the median.

The expansion is written once and used twice: the plan answers what would run, what would be
skipped and why, and the job runs exactly that. A second copy of the arithmetic would be a
second answer to "does 128k by 16 fit", and one of the two would go stale the first time the
cache dtype moved.

Speed measures. Quality and fidelity validate their body, expand their shapes and write their
rows as `not_run`, then finish the job in `error` naming the task that fills the `# TODO`: a
job that ends `ok` with an invented number is the one outcome this section exists to not
produce.
"""

from mlx_omnia.server.services.benchmarks.expand import expand
from mlx_omnia.server.services.benchmarks.references import (
    build_reference,
    delete_reference,
    discard_reference,
    existing_reference,
    reference,
    reference_key,
    reference_path,
    references,
    save_reference,
)
from mlx_omnia.server.services.benchmarks.runs import (
    delete_run,
    insert_run,
    measured,
    record,
    run,
    runs,
    runs_by_key,
    runs_of,
)
from mlx_omnia.server.services.benchmarks.specs import (
    FIDELITY_TODO,
    QUALITY_TODO,
    REFERENCE_CACHE,
    TOPK,
    Body,
    FidelitySpec,
    Invalid,
    Measured,
    Pair,
    Planned,
    QualitySpec,
    Sampling,
    Spec,
    SpeedSpec,
    Unknown,
    fidelity_key,
    quality_key,
)
from mlx_omnia.server.services.benchmarks.work import Split, estimate, split, work

__all__ = [
    "FIDELITY_TODO",
    "QUALITY_TODO",
    "REFERENCE_CACHE",
    "TOPK",
    "Body",
    "FidelitySpec",
    "Invalid",
    "Measured",
    "Pair",
    "Planned",
    "QualitySpec",
    "Sampling",
    "Spec",
    "SpeedSpec",
    "Split",
    "Unknown",
    "build_reference",
    "delete_reference",
    "delete_run",
    "discard_reference",
    "estimate",
    "existing_reference",
    "expand",
    "fidelity_key",
    "insert_run",
    "measured",
    "quality_key",
    "record",
    "reference",
    "reference_key",
    "reference_path",
    "references",
    "run",
    "runs",
    "runs_by_key",
    "runs_of",
    "save_reference",
    "split",
    "work",
]

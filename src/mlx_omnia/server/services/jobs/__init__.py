"""Slow work as a state plus a stream of frames, and the threads it runs on.

Cancellation is cooperative and has to reach inside the blocking work: `cancel` sets a
`threading.Event` and `Job.report` — the same call the work already makes to publish
progress — raises `Cancelled` when it finds it set.

Every frame is persisted, so a job that outlived the process still reports where it
stopped. What stays in memory is only what cannot be a row: the cancellation flag and the
open streams.
"""

from mlx_omnia.server.services.jobs.registry import (
    Cancelled,
    Job,
    JobFinished,
    Jobs,
    NoSuchJob,
    Work,
)
from mlx_omnia.server.services.jobs.views import (
    TERMINAL,
    Bench,
    Benchmark,
    Download,
    JobView,
    Load,
    Progress,
    Quantize,
    Subject,
    abandon,
    recent,
    view,
)

__all__ = [
    "TERMINAL",
    "Bench",
    "Benchmark",
    "Cancelled",
    "Download",
    "Job",
    "JobFinished",
    "JobView",
    "Jobs",
    "Load",
    "NoSuchJob",
    "Progress",
    "Quantize",
    "Subject",
    "Work",
    "abandon",
    "recent",
    "view",
]

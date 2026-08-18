"""What each request cost, what the requests on a model add up to, and the stream of both.

The numbers are the `Meter`'s (33.1) and are read from outside the decode loop: the engine
publishes at the two boundaries of a request, both on the event loop, so being watched costs
the loop nothing — no `.item()`, no fanout per token.

What is kept: the last `_HISTORY` requests in a ring, and one counter set per model since
boot. Neither goes to sqlite. A row per request would be a table to migrate and a retention
policy to answer for a history nobody reads back; what survives a restart is the bench,
which is a measurement someone asked for rather than a side effect of chatting.

The routes over this register live in `api/management/state.py`; closing that stream cancels
nothing, because watching a generation is not taking part in it.
"""

from mlx_omnia.server.metrics.arithmetic import prefill_rate
from mlx_omnia.server.metrics.models import (
    Aggregate,
    RequestState,
    Sample,
    Snapshot,
    Speculation,
    Totals,
)
from mlx_omnia.server.metrics.register import Metrics

__all__ = [
    "Aggregate",
    "Metrics",
    "RequestState",
    "Sample",
    "Snapshot",
    "Speculation",
    "Totals",
    "prefill_rate",
]

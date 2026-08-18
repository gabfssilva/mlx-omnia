"""One shape of a speed measurement, run through the daemon's own queue.

The unit is the shape — `(context, generate, concurrency)` — and the result is four
measurements: how long the checkpoint takes to become ready, how fast it reads a prompt, how
long the first token takes, and how fast it decodes after that. The ceiling is recomputed for
every shape, which is what makes the percentage mean anything: weights read per step come off
the tree the engine loaded, the cache read per step comes off the catalog's
`kv_bytes_per_token`, and both are recorded beside the fraction so the division can be checked.

Nothing here touches the database. The budget, the facts and the task handle are given.
"""

from mlx_omnia import greedy
from mlx_omnia.server.services.speed.measure import (
    Measurement,
    empty_result,
    measure,
    percentile,
)
from mlx_omnia.server.services.speed.protocols import (
    Cancelled,
    Entry,
    Progress,
    Report,
    Task,
)
from mlx_omnia.server.services.speed.rounds import ConcurrentRound, Round, prompt_of
from mlx_omnia.server.services.speed.shapes import (
    BANDWIDTH_GBS,
    CONCURRENCIES,
    CONTEXTS,
    GENERATES,
    GREEDY,
    ModelFacts,
    Refusal,
    Sampling,
    SpeedShape,
    batch_weight_bytes,
    cached_tokens,
    ceiling_tps,
    facts_of,
    human,
    kv_peak_bytes,
    kv_step_bytes,
    refusal,
)
from mlx_omnia.server.services.speed.thermal import (
    gpu_temperature,
    macmon,
    purge_page_cache,
    wait_cool,
)

__all__ = [
    "BANDWIDTH_GBS",
    "CONCURRENCIES",
    "CONTEXTS",
    "GENERATES",
    "GREEDY",
    "Cancelled",
    "ConcurrentRound",
    "Entry",
    "Measurement",
    "ModelFacts",
    "Progress",
    "Refusal",
    "Report",
    "Round",
    "Sampling",
    "SpeedShape",
    "Task",
    "batch_weight_bytes",
    "cached_tokens",
    "ceiling_tps",
    "empty_result",
    "facts_of",
    "gpu_temperature",
    "greedy",
    "human",
    "kv_peak_bytes",
    "kv_step_bytes",
    "macmon",
    "measure",
    "percentile",
    "prompt_of",
    "purge_page_cache",
    "refusal",
    "wait_cool",
]

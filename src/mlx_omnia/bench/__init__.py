"""Thermally gated, teacher-forced A/B benchmarking for MLX on Apple Silicon.

The harness knows no engine. An arm is anything that generates ids, and the two batteries —
`interleaved` for two implementations in one process, `paired` for one implementation at two
revisions of itself — supply the conditions that make the numbers comparable: a thermal gate
before every round, a rotated order, one teacher-forcing script across the arms, throttle
rejection, and a verdict by dominance rather than a score.

`forced` is not re-exported here on purpose: it imports mlx, and importing this package
should not require it. `from mlx_omnia.bench.forcing import forced`.
"""

from mlx_omnia.bench.arm import Arm, Generate, NothingToTime, TokenId, arm
from mlx_omnia.bench.battery import Comparison, Incomparable, default_gate, interleaved
from mlx_omnia.bench.ceiling import ceiling
from mlx_omnia.bench.gate import (
    Clocks,
    Cool,
    Gate,
    Hot,
    Macmon,
    Sensor,
    Throttled,
    find_macmon,
)
from mlx_omnia.bench.paired import Paired, Side, git, git_root, paired, worktree
from mlx_omnia.bench.prompt import BENCH_PROMPT, tile
from mlx_omnia.bench.report import (
    Axis,
    Calibration,
    Drift,
    Outcome,
    Verdict,
    axis,
    stderr,
    verdict,
)
from mlx_omnia.bench.sample import Sample

__all__ = [
    "BENCH_PROMPT",
    "Arm",
    "Axis",
    "Calibration",
    "Clocks",
    "Comparison",
    "Cool",
    "Drift",
    "Gate",
    "Generate",
    "Hot",
    "Incomparable",
    "Macmon",
    "NothingToTime",
    "Outcome",
    "Paired",
    "Sample",
    "Sensor",
    "Side",
    "Throttled",
    "TokenId",
    "Verdict",
    "arm",
    "axis",
    "ceiling",
    "default_gate",
    "find_macmon",
    "git",
    "git_root",
    "interleaved",
    "paired",
    "stderr",
    "tile",
    "verdict",
    "worktree",
]

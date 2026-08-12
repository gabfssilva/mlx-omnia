"""The two byte counts a decode step is judged by: what it reads, and what it occupies.

The arithmetic is the house's: active experts and not the total, the whole untied
lm_head, and the embedding table only when it is also the head — a gather of one row is
free. It comes off the tensors themselves, never off the config, and the residency count
also comes off the safetensors headers, because admission has to size a checkpoint
before deciding to load it.

Nothing here knows an architecture. Two numbers the tensors cannot give come off the
routing module that owns the stack: how many of its experts a token reaches (`Routed`),
and where the candidates stop (`Candidates`) — a stack taller than its candidate count
carries slots no routing decides, and the step reads those whole.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, Protocol, TypedDict, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear

SUSTAINED_GBS = 610.0
"""M5 Max: 614 GB/s theoretical, of which a serial chain of dependent kernels holds ~95%."""


def ceiling(active_bytes: int) -> float:
    """The decode tok/s the bandwidth allows. Every measured number is reported as a
    percentage of this one — beating the baseline is not a conclusion."""
    return SUSTAINED_GBS * 1e9 / active_bytes


@runtime_checkable
class Routed(Protocol):
    """A module that routes each token to `k` of the experts stacked beneath it — the one
    number a stack's own tensors do not carry."""

    k: int


@runtime_checkable
class Candidates(Protocol):
    """A routing module that also says how many of the slots beneath it are candidates for
    routing. Every row past them is a slot the step reads whole: qwen3.5 stacks its shared
    expert as the row after the last expert, and the decode kernel gathers it every token.

    Declared `object` and read only when it is a number — three of the ports give the same
    name to the submodule that holds the stack.
    """

    experts: object


class _Routing(NamedTuple):
    """What the module above a stack knows about it. `candidates` is `None` when the
    module does not say, and then every row of the stack is one."""

    reached: int
    candidates: int | None

    def slots(self, rows: int) -> int:
        """The rows this step reads: the ones routing reaches, plus every row stacked past
        the candidates.

        Clamped at zero, and that is not defensive: one declaration covers every stack under
        the module, and the stacks are not the same depth. qwen3.5 says 256 candidates with
        `gate_up_proj` 257 rows deep and `down_proj` 256 — the first has one row past them and
        the second has none. Without the clamp the shorter stack subtracts a row it does not
        have and the step comes out reading *less* than routing reached.
        """
        extra = 0 if self.candidates is None else max(0, rows - self.candidates)
        return self.reached + extra


class _TensorEntry(TypedDict):
    data_offsets: list[int]


def _header(path: Path) -> dict[str, _TensorEntry]:
    """safetensors opens with a little-endian u64 holding the JSON header's length.
    `__metadata__` sits among the tensors and carries no offsets."""
    with path.open("rb") as file:
        size = int.from_bytes(file.read(8), "little")
        raw: dict[str, _TensorEntry] = json.loads(file.read(size))
    return {name: entry for name, entry in raw.items() if name != "__metadata__"}


def checkpoint_bytes(directory: Path, prefix: str = "") -> int:
    """The weights a checkpoint holds, summed from the headers of its shards with no
    tensor mapped. A tied checkpoint that still serializes `lm_head` is counted with it —
    the loader drops that copy, so this is an upper bound on what ends up resident.

    `prefix` narrows it to one part of the file, which is what an MTP head is: `mtp.*` in the
    same shards as the trunk, loaded as a tree of its own or not at all. Admission has to
    weigh it before the load and cannot open the tree to ask."""
    total = 0
    for shard in sorted(directory.glob("model*.safetensors")):
        for name, entry in _header(shard).items():
            if not name.startswith(prefix):
                continue
            begin, end = entry["data_offsets"]
            total += end - begin
    return total


def _modules(model: nn.Module) -> dict[str, nn.Module]:
    found: dict[str, nn.Module] = {}
    model.apply_to_modules(lambda path, module: found.__setitem__(path, module))
    return found


def _members(module: nn.Module) -> Mapping[str, object]:
    """An `nn.Module` *is* the dict of its own members; mlx leaves that dict untyped."""
    return module


def _own_bytes(module: nn.Module) -> int:
    """The module's own tensors and not its children's — the walk visits every node, so
    reaching down here would count them twice. The `_` prefix is mlx's own mark for a
    member that is not a parameter (rope frequency tables, and nothing a checkpoint
    names)."""
    return sum(
        value.nbytes
        for key, value in _members(module).items()
        if isinstance(value, mx.array) and not key.startswith("_")
    )


def _candidates(module: nn.Module) -> int | None:
    if not isinstance(module, Candidates):
        return None
    declared = module.experts
    return declared if isinstance(declared, int) else None


def _routing_above(path: str, routing: Mapping[str, _Routing]) -> _Routing:
    parts = path.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        found = routing.get(".".join(parts[:depth]))
        if found is not None:
            return found
    raise ValueError(f"{path}: expert stack with no routing module above it")


class _Stack(NamedTuple):
    """One stack of experts and what a step does with it: how deep it is, and how many of
    those rows are read."""

    rows: int
    slots: int


def _stacks(modules: Mapping[str, nn.Module]) -> dict[str, _Stack]:
    routing = {
        path: _Routing(module.k, _candidates(module))
        for path, module in modules.items()
        if isinstance(module, Routed)
    }
    found: dict[str, _Stack] = {}
    for path, module in modules.items():
        if isinstance(module, SwitchLinear | QuantizedSwitchLinear):
            rows = module.weight.shape[0]
            found[path] = _Stack(rows, _routing_above(path, routing).slots(rows))
    return found


def resident_bytes(model: nn.Module) -> int:
    """Every tensor the tree holds — what the model occupies once loaded."""
    return sum(_own_bytes(module) for module in _modules(model).values())


def expert_slots(model: nn.Module) -> dict[int, int]:
    """How many rows of a stack a decode step reads, by the stack's depth.

    What it is for is pricing a checkpoint nobody loaded: the bytes come off the
    safetensors headers, and the two numbers the tensors do not carry come from here —
    a stacked tensor finds its stack by its own leading dimension. The tree may be the
    lazily-initialized one, since nothing below reads a value.

    Two stacks of the same depth that are read differently keep the larger count: the
    alternative is pricing a stack that is read whole as if something routed between its
    rows.
    """
    slots: dict[int, int] = {}
    for stack in _stacks(_modules(model)).values():
        slots[stack.rows] = max(slots.get(stack.rows, 0), stack.slots)
    return slots


def active_bytes_per_token(model: nn.Module) -> int:
    """What one decode step reads.

    Three rules and no architecture among them: an expert stack is read `k` of its
    candidate rows plus every row past them, an embedding table is a one-row gather and
    does not count unless it is also the head (no `lm_head` in the tree says it is),
    everything else is read whole. Both embedding classes have to be named — mlx's
    quantized one does not subclass the dense one.
    """
    modules = _modules(model)
    stacks = _stacks(modules)
    tied = "lm_head" not in modules
    total = 0
    for path, module in modules.items():
        own = _own_bytes(module)
        if (stack := stacks.get(path)) is not None:
            total += own * stack.slots // stack.rows
        elif tied or not isinstance(module, nn.Embedding | nn.QuantizedEmbedding):
            total += own
    return total

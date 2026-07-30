"""What is resident, what it occupies, and how deep the queue is.

The residency figure is the **maximum** of three numbers and never one of them alone.
Once a model settles, both MLX's active memory and the process' resident size read
*below* what it actually occupies; oMLX admitted a second large model on that reading and
blew the ceiling (issue #1623), and the form that survived there is the one A6 adopts —
the accumulator summed off the trees is the floor neither live meter is allowed to
undershoot.

KV counts apart from the weights because it grows per request rather than per model, and
the limit has to keep holding with it inside. The live meters already have it inside,
which makes `resident_bytes + kv_bytes` an over-count rather than an under-count — the
direction a ceiling forgives.
"""

import ctypes
from dataclasses import dataclass
from typing import Annotated

import mlx.core as mx
from fastapi import APIRouter, Depends, Request

from sideros_server.engine import Engine

_MACH_TASK_BASIC_INFO = 20


class _TaskBasicInfo(ctypes.Structure):
    """`task_info` flavor 20. Only the first field is read; the rest is declared because
    the kernel checks the count it is given against the flavor's own size."""

    virtual_size: int
    resident_size: int
    resident_size_max: int

    _fields_ = (
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time_seconds", ctypes.c_int32),
        ("user_time_microseconds", ctypes.c_int32),
        ("system_time_seconds", ctypes.c_int32),
        ("system_time_microseconds", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    )


_libc = ctypes.CDLL(None)


def footprint_bytes() -> int:
    """The process' resident size now, not its peak: `getrusage` only answers the
    high-water mark, and a figure that never comes back down would make an eviction that
    did free the memory unobservable."""
    info = _TaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    status = _libc.task_info(
        _libc.mach_task_self(),
        _MACH_TASK_BASIC_INFO,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    assert status == 0, f"task_info returned {status}"
    return info.resident_size


@dataclass(frozen=True)
class Resident:
    id: str
    weights_bytes: int
    kv_bytes: int
    loaded_at: float
    last_used: float | None


@dataclass(frozen=True)
class Queue:
    running: int
    waiting: int


@dataclass(frozen=True)
class State:
    models: list[Resident]
    queue: Queue
    resident_bytes: int
    """`max(MLX's active memory, the process' resident size, the accumulator)`, the
    accumulator being the sum of `models[*].weights_bytes`."""
    kv_bytes: int


async def _engine(request: Request) -> Engine:
    """Async so it reads the engine's dicts on the loop that mutates them."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


@router.get("/admin/state")
async def state(engine: EngineDep) -> State:
    models = [
        Resident(
            id=model_id,
            weights_bytes=entry.weights_bytes,
            kv_bytes=entry.kv_bytes,
            loaded_at=entry.loaded_at,
            last_used=entry.last_used,
        )
        for model_id, entry in engine.residency.items()
    ]
    accumulator = sum(model.weights_bytes for model in models)
    return State(
        models=models,
        queue=Queue(running=engine.running, waiting=engine.waiting),
        resident_bytes=max(mx.get_active_memory(), footprint_bytes(), accumulator),
        kv_bytes=sum(model.kv_bytes for model in models),
    )

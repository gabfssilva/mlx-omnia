"""What the process holds right now, by the two live meters."""

import ctypes

import mlx.core as mx

_MACH_TASK_BASIC_INFO = 20


class _TaskBasicInfo(ctypes.Structure):
    """`task_info` flavor 20. Only the first fields are read; the rest is declared because
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
    """The process' resident size now, not its peak: `getrusage` only answers the high-water
    mark, and a figure that never comes back down would make an eviction that did free the
    memory unobservable."""
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


def occupied_bytes() -> int:
    """Both live meters, maxed: each reads *below* the real residency once a model has
    settled, which is why the accumulator is maxed in over them instead of checked against
    them."""
    return max(mx.get_active_memory(), footprint_bytes())

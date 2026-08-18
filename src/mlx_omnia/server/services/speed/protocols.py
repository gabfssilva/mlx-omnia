"""The handles a shape is measured through: what reports progress, what may cancel, and what a
catalog entry has to say."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Progress:
    """What every kind of long work reports the same way. `total` is absent while unknown."""

    message: str = ""
    completed: float = 0.0
    total: float | None = None


class Cancelled(Exception):
    """Raised inside the worker thread when the task was cancelled."""


class Task(Protocol):
    """The handle a long-running job hands its blocking work: the loop it can drive, the flag
    it must read between steps, and the door it reports through."""

    @property
    def id(self) -> str: ...

    @property
    def loop(self) -> asyncio.AbstractEventLoop: ...

    @property
    def cancelled(self) -> threading.Event: ...

    def report(self, progress: Progress) -> None: ...


class Entry(Protocol):
    """What a catalog entry has to say for a shape to be priced and validated."""

    @property
    def id(self) -> str: ...

    @property
    def directory(self) -> Path: ...

    @property
    def bytes_per_token(self) -> int | None: ...

    @property
    def kv_bytes_per_token(self) -> int | None: ...

    @property
    def attention_window(self) -> int | None: ...

    @property
    def vocab_size(self) -> int | None: ...

    @property
    def shape(self) -> str | None: ...


@dataclass(frozen=True)
class Report:
    """Where this shape sits in the batch. Without it a round's own frame overwrites the
    batch's index and total."""

    prefix: str = ""
    completed: float = 0.0
    total: float | None = None

    def frame(self, message: str) -> Progress:
        return Progress(
            message=f"{self.prefix} · {message}" if self.prefix else message,
            completed=self.completed,
            total=self.total,
        )

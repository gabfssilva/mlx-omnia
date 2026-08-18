"""The cold floor under the prefix store: a span the memory ceiling pushed out becomes a file,
and a miss in memory looks here before paying for a prefill.

Files live in `~/.cache/mlx_omnia/prefixes/<model-slug>/<digest>.safetensors`, one directory per
model so forgetting one model's conversations is an `rmtree` and not a scan that opens every
file to find out whose it is. The directory is created `0700`: unlike every other cache the
daemon keeps, what is written here is derived from the user's own conversation.

Spans are written behind the request; anchors are not, because a conversation supersedes its
anchor every turn. A clean restart therefore keeps a hybrid warm and a crash loses its anchors
and keeps its rows.

`FileVault` implements the engine's synchronous vault protocol while its index lives in an async
database, so every row it needs is reached by handing the coroutine to the daemon's loop from
whichever thread the generation is running on. It must not be called from the loop thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar

from mlx_omnia.engine.core.cache_file import UnreadableCache, dump, load
from mlx_omnia.engine.core.prefix import Payload, Slot, Vault
from mlx_omnia.server.db.models.prefixes import PrefixCacheFile

CACHE = Path.home() / ".cache" / "mlx_omnia" / "prefixes"

_MODE = 0o700

_QUEUE = 64
"""Spans a write-behind may fall behind by before `write` blocks. Full is backpressure and
never a drop: a dropped span is a conversation that prefills whole on every resume."""

T = TypeVar("T")


def directory(model_id: str) -> Path:
    return CACHE / model_id.replace("/", "--")


async def rows(model: str | None = None) -> list[PrefixCacheFile]:
    """What is on disk, for one model or for all of them, newest use first."""
    query = PrefixCacheFile.objects.order_by(["-used_at", "key"])
    if model is not None:
        query = query.filter(model=model)
    return await query.all()


async def rows_by_use() -> list[PrefixCacheFile]:
    """Least recently used first, which is the order a ceiling evicts in."""
    return await PrefixCacheFile.objects.order_by(["used_at", "key"]).all()


async def total_bytes() -> int:
    return sum(entry.bytes for entry in await PrefixCacheFile.objects.all())


async def save(entry: PrefixCacheFile) -> None:
    await drop(entry.key, entry.kind)
    await entry.save()


async def touch(key: str, kind: str, used_at: float) -> None:
    """A hit is a use. Without this the LRU would evict by age and drop the conversation
    somebody is in the middle of."""
    await PrefixCacheFile.objects.filter(key=key, kind=kind).update(used_at=used_at)


async def drop(key: str, kind: str) -> bool:
    return await PrefixCacheFile.objects.delete(key=key, kind=kind) == 1


async def forget(model_id: str) -> int:
    """Everything stored for one model, row and file. What an unload does not do, and what
    deleting a checkpoint must."""
    entries = await rows(model_id)
    for entry in entries:
        with contextlib.suppress(OSError):
            os.unlink(entry.path)
        await drop(entry.key, entry.kind)
    with contextlib.suppress(OSError):
        directory(model_id).rmdir()
    return len(entries)


async def sweep() -> int:
    """Rows whose file is gone, dropped at boot. The opposite — a file with no row — is
    answered by the directory being one per model and disposable."""
    dropped = 0
    for entry in await rows():
        if not Path(entry.path).exists():
            await drop(entry.key, entry.kind)
            dropped += 1
    return dropped


class FileVault(Vault):
    """One model's disk tier, addressed by the prefix store's own keys.

    It holds the model's identity rather than being told it per call: the store's business is
    keys and payloads, and which model wrote them is already inside the key.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, model_id: str, *, ceiling: int) -> None:
        self._loop = loop
        self._model = model_id
        self._ceiling = ceiling
        self._known: dict[Slot, Path] = {}
        """What this vault has on disk, read from the index once and kept: a lookup per span of
        a 62k conversation would be 242 database round trips on the path of a hit."""
        self._loaded = False
        self._queue: queue.Queue[tuple[Slot, Payload, int] | None] = queue.Queue(_QUEUE)
        self._writer: threading.Thread | None = None
        self._lock = threading.Lock()

    def holds(self, key: Slot) -> bool:
        self._index()
        return key in self._known

    def read(self, key: Slot) -> Payload | None:
        self._index()
        path = self._known.get(key)
        if path is None:
            return None
        try:
            payload = load(path)
        except (UnreadableCache, OSError):
            # A truncated or vanished file is a miss and a prefill. The row goes with it, or
            # every lookup would pay for the same failure and the ceiling would count bytes
            # nobody holds.
            self.forget(key)
            return None
        self._await(touch(key[1], key[0], time.time()))
        return payload

    def write(self, key: Slot, payload: Payload, nbytes: int) -> bool:
        """Queue the write; block when the queue is full. The caller is on the request's own
        thread, so the wait is real, and it is the honest price of never dropping a span.

        The tensors were evaluated where they were cut (`core.prefix`), which is the only
        thread that could.
        """
        if self._ceiling <= 0:
            return False
        self._index()
        if key in self._known:
            return True
        with self._lock:
            if self._writer is None or not self._writer.is_alive():
                self._writer = threading.Thread(
                    target=self._drain, daemon=True, name="prefix-vault"
                )
                self._writer.start()
        self._queue.put((key, payload, nbytes))
        return True

    def forget(self, key: Slot) -> None:
        path = self._known.pop(key, None)
        if path is not None:
            with contextlib.suppress(OSError):
                os.unlink(path)
        self._await(drop(key[1], key[0]))

    def flush(self, timeout: float | None = None) -> None:
        """Wait for the writes in flight. Never called on the request path — that is the point
        of writing behind — but a caller that wants to read back what it stored has to, and so
        does a shutdown that would otherwise leave a staging file behind."""
        self._queue.put(None)
        writer = self._writer
        if writer is not None:
            writer.join(timeout)

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            key, payload, nbytes = item
            path = directory(self._model) / f"{_digest(key)}.safetensors"
            try:
                path.parent.mkdir(parents=True, exist_ok=True, mode=_MODE)
                dump(payload, path)
            except OSError:
                # A full disk is a cache that does not exist. No row is written, so nothing
                # will look for the file.
                continue
            self._known[key] = path
            now = time.time()
            self._await(
                save(
                    PrefixCacheFile(
                        key=key[1],
                        kind=key[0],
                        model=self._model,
                        path=str(path),
                        bytes=nbytes,
                        created_at=now,
                        used_at=now,
                    )
                )
            )
            self._enforce()

    def _index(self) -> None:
        """The rows for this model, once. Rebuilt from the database and not from the directory:
        `stat` on every file of a full cache is the cost that table exists to avoid."""
        if self._loaded:
            return
        self._loaded = True
        for entry in self._await(rows(self._model)):
            slot = _slot(entry)
            if slot is not None:
                self._known[slot] = Path(entry.path)

    def _enforce(self) -> None:
        """Least recently used out until the total is under the ceiling."""
        total = self._await(total_bytes())
        if total <= self._ceiling:
            return
        for entry in self._await(rows_by_use()):
            if total <= self._ceiling:
                return
            total -= entry.bytes
            slot = _slot(entry)
            if slot is not None:
                self.forget(slot)

    def _await(self, work: Coroutine[object, object, T]) -> T:
        return asyncio.run_coroutine_threadsafe(work, self._loop).result()


def _slot(entry: PrefixCacheFile) -> Slot | None:
    """A row back as the key it was written under, or `None` for a kind this release does not
    know: the column is text, and a row a later version wrote is a file this one cannot read."""
    match entry.kind:
        case "rows":
            return ("rows", entry.key)
        case "anchor":
            return ("anchor", entry.key)
        case _:
            return None


def _digest(key: Slot) -> str:
    """A slot as one name. The kind prefixes the file name rather than being dropped, because a
    span and the state that stood on its far boundary are two payloads of one key."""
    kind, digest = key
    return digest if kind == "rows" else f"{kind}-{digest}"

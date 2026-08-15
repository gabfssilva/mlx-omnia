"""The cold floor under the prefix store: a span the ceiling pushed out becomes a file, and a
miss in memory looks here before paying for a prefill.

Why it is worth a file at all is arithmetic. On `Ling-3.0-flash` a 62k-token conversation is
~573 MB of cache; read once at NVMe speed that is a fraction of a second, against tens of
seconds of prefill. What it is emphatically *not* is streaming the live cache: attention reads
the whole KV per token, so serving decode off disk would be ~80 ms a token against the
610 GB/s of unified memory — the decode would fall from ~65 tok/s to ~11.

Where: `~/.cache/mlx_omnia/prefixes/<model-slug>/<digest>.safetensors`, following the two
neighbours that already exist (`catalog.QUANTIZED_CACHE`, `fidelity.CACHE`) rather than the
store's own `~/.config`. The split those two draw is the right one here: what does not
regenerate lives beside the database a user backs up, and what is heavy and regenerable lives
under the cache. Like both of them this fixes `Path.home() / ".cache"` and does not honour
`XDG_CACHE_HOME` — honouring it here alone would be a third behaviour, worse than the
inconsistency; if it is to be honoured, all three change together.

Unlike both of them, what is written here is derived from the user's own conversation, so the
directory is created `0700`. It is the first state of that kind the daemon puts on disk.

Spans are written behind the request: a span closes, the write is queued, and the thread that
generated it goes on. Anchors are not. A conversation supersedes its anchor every turn, so a
write-behind of one would be hundreds of megabytes per turn for a file that dies with the next
one; an anchor reaches disk only when the ceiling evicts it or when the daemon drains on its
way out. What that costs is stated where it is paid: a clean restart keeps a hybrid warm, and
a crash loses its anchors and keeps its rows.

Which model a span belongs to is not in the key — the key is a digest, and the digest already
folds the checkpoint — so the directory carries it instead, and "forget this model's
conversations" stays one `rmtree` rather than a scan that opens every file to find out whose
it is.
"""

import contextlib
import os
import queue
import threading
import time
from pathlib import Path

from mlx_omnia.engine.core.cache_file import UnreadableCache, dump, load
from mlx_omnia.engine.core.prefix import Payload, Slot, Vault
from mlx_omnia.server.store import PrefixCacheFile, Store

CACHE = Path.home() / ".cache" / "mlx_omnia" / "prefixes"

_MODE = 0o700
"""The content is derived from what the user said to the model. Every other cache directory
the daemon keeps holds weights, which are already on disk in the open."""

_QUEUE = 64
"""Spans a write-behind may fall behind by. Bounded rather than unbounded: a disk slower than
the prefill it is saving would otherwise grow a queue holding every span of every request
alive in memory, which is the ceiling this tier exists under, inverted."""


def directory(model_id: str) -> Path:
    """One directory per model, so "forget this model's conversations" is an `rmtree` and not
    a scan that has to open every file to find out whose it is. The slug is the id with its
    slashes flattened, the shape `catalog.QUANTIZED_CACHE` already uses."""
    return CACHE / model_id.replace("/", "--")


class FileVault(Vault):
    """One model's disk tier, addressed by the store's own keys.

    It holds the model's identity rather than being told it per call, because the store knows
    nothing about checkpoints and must not learn: its business is keys and payloads, and which
    model wrote them is already inside the key it was handed.
    """

    def __init__(self, store: Store, model_id: str, *, ceiling: int) -> None:
        self._store = store
        self._model = model_id
        self._ceiling = ceiling
        self._known: dict[Slot, Path] = {}
        """What this vault has on disk, read from the index once and kept: a lookup per span
        of a 62k conversation is 242 sqlite round trips on the path of a hit."""
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
            # A truncated or vanished file is a miss and a prefill, which is always correct.
            # The row goes with it: an index that points at nothing would make every lookup
            # pay for the same failure, and the ceiling would count bytes nobody holds.
            self.forget(key)
            return None
        self._store.touch_prefix_file(key[1], key[0], time.time())
        return payload

    def write(self, key: Slot, payload: Payload, nbytes: int) -> None:
        """Queue the write and return. The caller is on the request's own thread — a span
        closes inside a prefill block or between two decode steps — and half a gigabyte
        written there is half a gigabyte the next token waits for.

        The tensors were evaluated where they were cut (`core.prefix`), which is the only
        thread that could: an array still carrying the decode's queued work meets
        `no Stream(gpu, 1) in current thread` anywhere else.
        """
        if self._ceiling <= 0:
            return
        self._index()
        if key in self._known:
            return
        with self._lock:
            if self._writer is None or not self._writer.is_alive():
                self._writer = threading.Thread(
                    target=self._drain, daemon=True, name="prefix-vault"
                )
                self._writer.start()
        try:
            self._queue.put_nowait((key, payload, nbytes))
        except queue.Full:
            # A disk that cannot keep up is a cache that does not have this span, which is
            # the state the daemon was in before the feature. Dropping is the only answer
            # that does not make a request wait on it.
            return

    def forget(self, key: Slot) -> None:
        path = self._known.pop(key, None)
        if path is not None:
            with contextlib.suppress(OSError):
                os.unlink(path)
        self._store.delete_prefix_file(key[1], key[0])

    def flush(self, timeout: float | None = None) -> None:
        """Wait for the writes in flight. The daemon never calls it on the request path —
        that is the whole point of writing behind — but a caller that wants to read back what
        it stored has to, and so does a shutdown that would otherwise leave a staging file
        behind."""
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
                # A full disk or a directory that cannot be made is a cache that does not
                # exist. No row is written, so nothing will look for the file.
                continue
            self._known[key] = path
            self._store.save_prefix_file(
                PrefixCacheFile(
                    key=key[1],
                    kind=key[0],
                    model=self._model,
                    path=str(path),
                    bytes=nbytes,
                    created_at=time.time(),
                    used_at=time.time(),
                )
            )
            self._enforce()

    def _index(self) -> None:
        """The rows for this model, once. Rebuilt from sqlite and not from the directory:
        `stat` on every file of a full cache is the cost that table exists to avoid."""
        if self._loaded:
            return
        self._loaded = True
        for entry in self._store.prefix_files(self._model):
            slot = _slot(entry)
            if slot is not None:
                self._known[slot] = Path(entry.path)

    def _enforce(self) -> None:
        """Least recently used out until the total is under the ceiling."""
        total = self._store.prefix_bytes()
        if total <= self._ceiling:
            return
        for entry in self._store.prefix_files_by_use():
            if total <= self._ceiling:
                return
            total -= entry.bytes
            slot = _slot(entry)
            if slot is not None:
                self.forget(slot)


def forget(store: Store, model_id: str) -> int:
    """Everything stored for one model, row and file. What an unload does not do — a cache
    that dies with the model would be a cache that never survives a restart — and what
    deleting a checkpoint must."""
    entries = store.prefix_files(model_id)
    for entry in entries:
        with contextlib.suppress(OSError):
            os.unlink(entry.path)
        store.delete_prefix_file(entry.key, entry.kind)
    with contextlib.suppress(OSError):
        directory(model_id).rmdir()
    return len(entries)


def sweep(store: Store) -> int:
    """Rows whose file is gone, dropped at boot. A daemon that was killed between the rename
    and the insert leaves the opposite — a file with no row — which the ceiling would then
    never count; that one is answered by the directory being one per model and disposable."""
    dropped = 0
    for entry in store.prefix_files():
        if not Path(entry.path).exists():
            store.delete_prefix_file(entry.key, entry.kind)
            dropped += 1
    return dropped


def _slot(entry: PrefixCacheFile) -> Slot | None:
    """A row back as the key it was written under, or `None` for a kind this release does not
    know. Unknown rather than an error: the column is text, and a row a later version wrote
    is a file this one has no reader for."""
    match entry.kind:
        case "rows" | "anchor":
            return (entry.kind, entry.key)
        case _:
            return None


def _digest(key: Slot) -> str:
    """A slot as one name. The kind is a prefix and not a column of the file name because a
    span and the state that stood on its far boundary are two payloads of one key, and a
    directory listing should show which is which."""
    kind, digest = key
    return digest if kind == "rows" else f"{kind}-{digest}"


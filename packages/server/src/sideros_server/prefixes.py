"""The disk tier under the prefix trie: an evicted conversation becomes a file, and a miss
in memory looks here before paying for a prefill.

Why it is worth a file at all is arithmetic. On `Ling-3.0-flash` a 62k-token conversation is
~573 MB of cache; read once at NVMe speed that is a fraction of a second, against tens of
seconds of prefill. What it is emphatically *not* is streaming the live cache: attention
reads the whole KV per token, so serving decode off disk would be ~80 ms a token against the
610 GB/s of unified memory — the decode would fall from ~65 tok/s to ~11.

Where: `~/.cache/sideros/prefixes/<model-slug>/<digest>.safetensors`, following the two
neighbours that already exist (`catalog.QUANTIZED_CACHE`, `fidelity.CACHE`) rather than the
store's own `~/.config`. The split those two draw is the right one here: what does not
regenerate lives beside the database a user backs up, and what is heavy and regenerable lives
under the cache. Like both of them this fixes `Path.home() / ".cache"` and does not honour
`XDG_CACHE_HOME` — honouring it here alone would be a third behaviour, worse than the
inconsistency; if it is to be honoured, all three change together.

Unlike both of them, what is written here is derived from the user's own conversation, so the
directory is created `0700`. It is the first state of that kind the daemon puts on disk.

Writing happens on eviction and off the request thread: the queue is serialized at depth one,
so half a gigabyte written inside a request is half a gigabyte the next request waits for.
"""

import contextlib
import os
import threading
import time
from array import array
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from sideros.core.cache import LayerCache
from sideros.core.cache_file import (
    UnreadableCache,
    dump,
    key,
    policy,
    restore,
    storable,
    written,
)
from sideros.core.prompt_cache import Role
from sideros_server.store import PrefixCacheFile, Store

CACHE = Path.home() / ".cache" / "sideros" / "prefixes"

_MODE = 0o700
"""The content is derived from what the user said to the model. Every other cache directory
the daemon keeps holds weights, which are already on disk in the open."""


def directory(model_id: str) -> Path:
    """One directory per model, so "forget this model's conversations" is an `rmtree` and not
    a scan that has to open every file to find out whose it is. The slug is the id with its
    slashes flattened, the shape `catalog.QUANTIZED_CACHE` already uses."""
    return CACHE / model_id.replace("/", "--")


class DiskSpill:
    """One model's disk tier: the trie hands it evictions and asks it about misses.

    It holds the model's identity rather than being told it per call, because a `PromptCache`
    knows nothing about checkpoints and must not learn: the trie's business is tokens and
    caches, and which model wrote them is what makes the key.
    """

    def __init__(
        self,
        store: Store,
        model_id: str,
        *,
        stamp: str,
        ceiling: int,
        floor: int,
    ) -> None:
        self._store = store
        self._model = model_id
        self._stamp = stamp
        self._ceiling = ceiling
        self._floor = floor
        """Below this many bytes an entry is not written: a short prefix costs more in a file
        than the prefill it saves, and every one of them is a row in an index that a lookup
        walks."""
        self._writing = threading.Lock()
        self._pending: list[threading.Thread] = []

    def flush(self, timeout: float | None = None) -> None:
        """Wait for the writes in flight. The daemon never calls it — a spill is fire and
        forget, and what it is for is exactly not making a request wait on an SSD — but a
        caller that wants to read back what it evicted has to, and so does a shutdown that
        would otherwise leave a staging file behind."""
        for thread in list(self._pending):
            thread.join(timeout)
        self._pending = [thread for thread in self._pending if thread.is_alive()]

    def keep(self, tokens: Sequence[int], caches: Sequence[LayerCache], *, role: Role) -> None:
        """Not on the caller's thread. The eviction that triggers this happens inside a
        request — `insert` is what overruns the budget — and the queue behind it is
        serialized, so writing here would be the next request waiting on an SSD."""
        if self._ceiling <= 0 or not storable(caches):
            return
        size = written(caches)
        if size < self._floor:
            return
        # Materialized here and not in the thread below, because a stream belongs to the
        # thread that made it: an array still carrying the decode thread's work cannot be
        # evaluated anywhere else, and at shutdown that thread is already gone —
        # `save_safetensors` then raises `no Stream(gpu, 1) in current thread` and the
        # conversation is lost with no request to report it to.
        mx.eval([tensor for layer in caches for tensor in layer.tensors])
        digest = key(self._model, self._stamp, policy(caches), tokens)
        target = directory(self._model) / f"{digest}.safetensors"
        entry = PrefixCacheFile(
            key=digest,
            model=self._model,
            path=str(target),
            ids=_packed(tokens),
            tokens=len(tokens),
            bytes=size,
            created_at=time.time(),
            used_at=time.time(),
        )
        thread = threading.Thread(
            target=self._write, args=(entry, list(caches)), daemon=True, name="prefix-spill"
        )
        self._pending = [held for held in self._pending if held.is_alive()]
        self._pending.append(thread)
        thread.start()

    def _write(self, entry: PrefixCacheFile, caches: list[LayerCache]) -> None:
        with self._writing:
            try:
                target = Path(entry.path)
                target.parent.mkdir(parents=True, exist_ok=True, mode=_MODE)
                dump(caches, target)
            except OSError:
                # A disk that is full or a directory that cannot be made is a cache that does
                # not exist, which is the state the daemon was in before this feature. The
                # row is not written, so nothing will look for the file.
                return
            self._store.save_prefix_file(entry)
            self._enforce()

    def recall(self, tokens: Sequence[int], into: Sequence[LayerCache]) -> int | None:
        """The most of `tokens` any stored file covers, read into `into`.

        The same rule the trie's `take` follows, and for the same reason: a stored cache
        *longer* than the match is rewound to it rather than thrown away, and a trunk whose
        layers keep no history to rewind to is skipped instead. Which is why the ids are in
        the index — a digest answers "is this the same" and never "how much of this is the
        same", and a conversation whose last turn re-renders one token differently would go
        from a hit to a whole prefill.
        """
        if self._ceiling <= 0 or not storable(into):
            return None
        # The policy of the trunk about to read, which is the one that decides: a file written
        # by a cache of another format holds bytes this one would take for its own.
        reading = policy(into)
        rewinds = all(layer.is_trimmable for layer in into)
        limit = len(tokens) - 1
        best: tuple[int, PrefixCacheFile] | None = None
        for candidate in self._store.prefix_files(self._model):
            ids = _unpacked(candidate.ids)
            # The stamp and the policy are checked through the key rather than beside it:
            # they are what the digest of these very ids was taken over, so one comparison
            # answers both.
            if candidate.key != key(self._model, self._stamp, reading, ids):
                continue
            # One row is always left to prefill: a forward needs one, and the logits of the
            # last position are what the sampler reads.
            usable = min(_common(ids, tokens), limit)
            if usable == 0 or (usable < candidate.tokens and not rewinds):
                continue
            # Same reuse for less read: the shorter file is the one to open.
            if best is None or (usable, -candidate.tokens) > (best[0], -best[1].tokens):
                best = (usable, candidate)
        if best is None:
            return None
        covered, entry = best
        try:
            restore(into, Path(entry.path))
        except (UnreadableCache, OSError):
            # A truncated or vanished file is a miss and a prefill, which is always correct.
            # The row goes with it: an index that points at nothing would make every lookup
            # pay for the same failure, and the ceiling would count bytes nobody holds.
            self._forget(entry)
            return None
        if entry.tokens > covered:
            for layer in into:
                layer.trim(covered)
        self._store.touch_prefix_file(entry.key, time.time())
        return covered

    def _enforce(self) -> None:
        """Least recently used out until the total is under the ceiling. Read from the index
        rather than by walking the directory: `stat` on every file of a full cache is the
        cost this table exists to avoid."""
        total = self._store.prefix_bytes()
        if total <= self._ceiling:
            return
        for entry in self._store.prefix_files_by_use():
            if total <= self._ceiling:
                return
            total -= entry.bytes
            self._forget(entry)

    def _forget(self, entry: PrefixCacheFile) -> None:
        with contextlib.suppress(OSError):
            os.unlink(entry.path)
        self._store.delete_prefix_file(entry.key)


def _packed(tokens: Sequence[int]) -> bytes:
    """Ids as int32, which is four bytes a token against the 8 KB a token the file beside them
    weighs. A JSON list would be three times that and would still have to be parsed."""
    return array("i", tokens).tobytes()


def _unpacked(raw: bytes) -> list[int]:
    held = array("i")
    held.frombytes(raw)
    return held.tolist()


def _common(one: Sequence[int], other: Sequence[int]) -> int:
    """How much of a prefix the two share."""
    length = 0
    for left, right in zip(one, other, strict=False):
        if left != right:
            break
        length += 1
    return length


def forget(store: Store, model_id: str) -> int:
    """Everything stored for one model, row and file. What an unload does not do — a cache
    that dies with the model would be a cache that never survives a restart — and what
    deleting a checkpoint must."""
    entries = store.prefix_files(model_id)
    for entry in entries:
        with contextlib.suppress(OSError):
            os.unlink(entry.path)
        store.delete_prefix_file(entry.key)
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
            store.delete_prefix_file(entry.key)
            dropped += 1
    return dropped

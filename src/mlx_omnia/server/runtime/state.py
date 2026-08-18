"""Everything one engine holds, and the readings that are a plain look at it."""

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.engine.core.prefix import PrefixStore
from mlx_omnia.server.runtime.environment import (
    Compression,
    DiskVault,
    Environment,
    MetricsSink,
    NoMetrics,
    Settings,
)
from mlx_omnia.server.runtime.jobs import Job, Release
from mlx_omnia.server.runtime.residency import Residency

Loader = Callable[[str], LanguageModel[ModelInput]]

SPILL_JOIN_SECONDS = 30.0
"""How long a shutdown waits on a write in flight. Long enough for half a gigabyte on any disk
that is not failing, and bounded because a daemon that cannot exit is worse than a cache entry
that was not written."""


def _nothing() -> None:
    pass


class EngineState:
    def __init__(
        self,
        loader: Loader,
        environment: Environment | None = None,
        metrics: MetricsSink | None = None,
    ) -> None:
        self._loader = loader
        self._environment = environment
        """Where the ceiling, the TTL and the concurrency come from, read per decision. An engine
        built without one admits everything and expires nothing."""
        self._models: dict[str, LanguageModel[ModelInput]] = {}
        self._loading: dict[str, asyncio.Task[LanguageModel[ModelInput]]] = {}
        self._admission = asyncio.Lock()
        """One cold load at a time — see `_load`. Built here rather than lazily because a `Lock`
        binds to the loop that first awaits it, and this engine has exactly one."""
        self._compiling = asyncio.Lock()
        """One grammar built at a time — see `constrain`. Two requests racing the same cold
        vocabulary would each pay the 0.27 s, and the loser's table would stay alive for as long
        as the grammar it compiled sits in the cache."""
        self._residency: dict[str, Residency] = {}
        self._vaults: dict[str, tuple[int, DiskVault | None]] = {}
        """One disk tier per model, and the ceiling it was built with. Kept rather than made per
        request because it holds the index it read at construction and the thread that writes
        behind: a new one per request would re-read the index every turn."""
        self._prefixes: tuple[tuple[int, int], PrefixStore] | None = None
        """The one prefix store every resident model reads and writes, with the ceiling it was
        built for. Here and not on the model for the reason the residency table is here: what it
        arbitrates is two models against each other, and neither can see the other. Which
        checkpoint a span came from is inside its key, so a model unloaded and loaded again finds
        its spans where it left them."""
        self._quantizable: dict[tuple[str, Compression], str | None] = {}
        """What a `(model, KV policy)` pair was found to be — `None` for one that holds, the
        refusal in words for one that does not. It outlives residency on purpose: the answer is a
        fact about a checkpoint's shape and its family."""
        self._loads = 0
        """Counts completed cold loads. Read from the decode thread to find out whether a model
        landed while this request was running — a plain int, written on the loop."""
        self._queue: asyncio.Queue[Job | Release] = asyncio.Queue()
        self._model_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnia-model")
        """One thread for every step of every generation. Pinned because MLX's default stream is
        the *thread's* — two consecutive ticks on two threads are two streams, and a batch's cache
        would be waited on across them once per token."""
        self._load_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnia-load")
        """And one more for reading checkpoints. What separates them is not how long either takes
        — a load is mmap plus a lazy tree, seconds even for the largest checkpoint — it is that
        one thread makes them wait for each other in a way neither can be blamed for.

        Reading queued behind decode is a cycle rather than a delay: a load finishes when the
        thread frees, the thread frees when the generation on it produces its last token, and a
        generation can be waiting on something that only happens once the load is in. Nothing
        about that resolves by being quick.

        Decode queued behind reading is the other direction, and it charges the wrong request:
        while a cold model reads, every *resident* model is unreachable too — a generation that
        needs no load at all, only a thread to decode on.

        Serialized among themselves, because `_admission` already allows one cold load at a time —
        so this is a pin and not a pool. What a checkpoint read here owes the thread that decodes
        it is that nothing in it is still lazy, and `_read` is where that is paid: an MLX stream
        belongs to the thread that made it, so an array this thread queued and did not evaluate is
        an array the other one cannot.
        """
        self._pending: deque[Job | Release] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._current: list[Job] = []
        self._changed = asyncio.Event()
        """Set when a model lands and when a lease is let go — the two ways the next expiry can
        move earlier than the one the sweep went to sleep on."""
        self._metrics: MetricsSink = NoMetrics() if metrics is None else metrics
        self._reserving = asyncio.Lock()
        """One reservation at a time — two benchmarks measuring at once would each be the other's
        noise."""
        self._reserved: object | None = None
        """Who holds the queue, when somebody does. A token and not a flag: the holder's own
        submissions have to pass, and comparing the token is what tells them apart."""
        self._free = asyncio.Event()
        self._free.set()
        self.on_change: Callable[[], None] = _nothing
        """Raised at every transition the state route reports — a model landing or leaving, the
        queue moving, the reservation changing hands."""

    @property
    def resident(self) -> list[str]:
        return list(self._models)

    @property
    def metrics(self) -> MetricsSink:
        """What every request through this engine cost. The boundaries of a request are the
        worker's, which is why the register is reached through here."""
        return self._metrics

    @property
    def residency(self) -> Mapping[str, Residency]:
        """In load order, which is the order the daemon took the memory in."""
        return MappingProxyType(self._residency)

    @property
    def running(self) -> int:
        return len(self._current)

    @property
    def waiting(self) -> int:
        return self._queue.qsize() + len(self._pending)

    @property
    def reserved(self) -> bool:
        """Whether somebody is holding the queue exclusively. Out on the state route so a client
        can stop polling the expensive routes while a measurement runs."""
        return self._reserved is not None

    def _settings(self) -> Settings | None:
        return None if self._environment is None else self._environment.settings()

"""Resident models plus the global FCFS generation queue.

MLX enqueues on a single GPU stream and the decode loop is CPU-hot: requests must
not call generate() concurrently. The gate serializes generation — one worker task
consumes the queue and runs each job to completion (or cancellation) before the
next. Model loading happens outside the gate: a cold load of a 30B is seconds, and
holding the gate through it would make every resident model wait on it.

Nothing is resident at boot. A request names its model and that is what loads it, and
nothing but the memory limit takes it away again: a load that would cross the ceiling
evicts the least recently used model first, and one that has been idle past its TTL leaves
on its own. Both figures come from `/admin/config`, read per decision.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable, Collection, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import NotRequired, Protocol, TypedDict, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from sideros import GenerationOptions, LanguageModel, ModelInput, UnsupportedInput
from sideros.footprint import active_bytes_per_token, checkpoint_bytes, resident_bytes
from sideros.generate import Constraint, Meter
from sideros.grammar import Grammar, Vocabulary
from sideros.language import tokenizer_of
from sideros.model import Wrapping
from sideros.suppress import Segment
from sideros_server.metrics import Metrics, RequestState
from sideros_server.store import Store

Loader = Callable[[str], LanguageModel[ModelInput]]

# Defined with the record it ends up in: what a request became is read from `/admin/metrics`.
JobState = RequestState


class ModelTooLarge(Exception):
    """A checkpoint that does not fit the memory limit with nothing else resident. Named, and
    raised before anything is evicted: no sequence of evictions makes room for it, so a loop
    chasing that space would empty the daemon of every other model and still fail."""


class NotResident(Exception):
    """A model a request named while `/admin/config` says `not_resident: "fail"` — decision 3.
    A cold load is seconds of the queue for whoever is behind it, and a daemon told to fail
    fast says so instead of paying it. Loading on purpose is still
    `PUT /admin/models/{id}/residency`, which is an order and not a request."""


class NotConstrainable(Exception):
    """A resident model no grammar can be compiled against: nothing under the facades holds a
    tokenizer, the catalog has no `config.json` to read the head's width out of, or the load
    resolved no stop id for a document to end on.

    Named rather than left as whatever fails first, because the way out belongs to the client:
    the same schema without `strict` is checked after the answer and needs none of the three.
    """


# `catalog`, `config` and `state` each import `Engine`; the three imports below sit inside the
# functions that need them because at module level the cycle closes in either direction.


@dataclass(frozen=True)
class _Settings:
    """The four fields of `/admin/config` this file decides with, read together because they
    come out of one row set. Named rather than a tuple: three of the four are integers, and a
    positional read is how the ceiling ends up being compared against a TTL."""

    limit: int
    ttl: int | None
    prefix_budget: int
    not_resident: str


def _config(store: Store) -> _Settings:
    """What `/admin/config` has at this instant. Nothing caches it — that is what makes the
    four fields `applied` — so this is one read per cold load, one per sweep and one per
    decision a request takes, and a PATCH is in force for the next of each."""
    from sideros_server.config import current

    config = current(store)
    return _Settings(
        limit=config.memory_limit_bytes,
        ttl=config.idle_ttl_seconds,
        prefix_budget=config.prefix_cache_bytes,
        not_resident=config.not_resident,
    )


def _measured() -> int:
    """What the process holds now, by the two live meters. Both read *below* the real
    residency once a model has settled (oMLX #1623), which is why the accumulator is maxed
    in over them instead of checked against them."""
    from sideros_server.state import footprint_bytes

    return max(mx.get_active_memory(), footprint_bytes())


def _checkpoint_size(model_id: str) -> int:
    """What the model about to be loaded weighs, off the safetensors headers — the only
    figure that exists before the load. A model the catalog does not list weighs zero here:
    the two caches are what the daemon's own loader can open, and an id outside them has no
    header to sum and nothing for admission to decide against."""
    from sideros_server.catalog import scan

    entry = next((entry for entry in scan() if entry.id == model_id), None)
    return 0 if entry is None else checkpoint_bytes(entry.directory)


class _TextConfigJson(TypedDict):
    vocab_size: NotRequired[int]


class _ConfigJson(TypedDict):
    """Only the field a mask needs. A multimodal checkpoint declares the trunk one level
    down, and the head that produces the row is the trunk's."""

    vocab_size: NotRequired[int]
    text_config: NotRequired[_TextConfigJson]


def _head_width(model_id: str) -> int | None:
    """How wide the row the model produces is, off the checkpoint's own `config.json`.

    Not the tokenizer's count: a head can be padded past it — Qwen3's is 151936 over 151669
    ids — and a mask one column short of the row is a mask with unmasked ids in it. The
    catalog is what turns an id into a directory, the same way admission sizes a checkpoint.
    """
    from sideros_server.catalog import scan

    entry = next((entry for entry in scan() if entry.id == model_id), None)
    if entry is None:
        return None
    config: _ConfigJson = json.loads((entry.directory / "config.json").read_text())
    width = config.get("vocab_size")
    text = config.get("text_config")
    return text.get("vocab_size") if width is None and text is not None else width


@runtime_checkable
class _Stopping(Protocol):
    stop: Collection[int]


def _stop_ids(model: object) -> Collection[int]:
    """Every id that ends a turn for this model, off the facade that holds it — the walk
    `tokenizer_of` does, for the same reason: what resolved the set is the load (`stop_tokens`
    over the config and the generation config), and out here there is a
    `LanguageModel[ModelInput]` and nothing else. It becomes the grammar's end, so a
    constrained run stops where a free one does."""
    while not isinstance(model, _Stopping):
        if not isinstance(model, Wrapping):
            return ()
        model = model.model
    return model.stop


def _vocabulary(model_id: str, model: LanguageModel[ModelInput]) -> Vocabulary:
    """The token table this model's grammars compile against. Off the loop: it decodes every
    id the head can draw and hands the table to Rust, which is 0.27 s over 150k of them."""
    tokenizer = tokenizer_of(model)
    stop = _stop_ids(model)
    size = _head_width(model_id)
    if tokenizer is None or not size or not stop:
        missing = [
            name
            for name, found in (("tokenizer", tokenizer), ("vocab_size", size), ("stop id", stop))
            if not found
        ]
        raise NotConstrainable(
            f"{model_id!r} has no {' and no '.join(missing)}: a strict schema is compiled "
            "against the checkpoint's own token table, and there is none to compile against. "
            "The same schema without strict is checked after the answer and needs neither."
        )
    return Vocabulary(tokenizer, size=size, stop=stop)


def tree(model: object) -> nn.Module | None:
    """The outermost `nn.Module` under the wrappers, or `None` when there is none — a test
    double is a `LanguageModel` and holds no tensors at all."""
    while not isinstance(model, nn.Module):
        if not isinstance(model, Wrapping):
            return None
        model = model.model
    return model


@dataclass
class Residency:
    """What one resident model costs and when it was last worth its space."""

    weights_bytes: int
    """Every tensor the tree holds, summed once at load: the accumulator A6 calls the floor
    that the live meters undershoot once the model has settled."""
    loaded_at: float
    last_used: float | None = None
    """When a request last ran on it, `None` while it has only been loaded. Stamped when
    the request is accepted and again when it ends, so a model half an hour into a
    generation does not read as half an hour idle."""
    kv_bytes: int = 0
    """What the last request on this model added on top of the settled weights — the KV
    cache and the activations around it. The last request's peak rather than a live
    reading, because the cache lives inside the generator `stream` returns and dies with
    it; what survives is what the next request will need room for."""
    active_bytes: int | None = None
    """What one decode step reads, summed off the same tree at load. `None` when there is no
    tree; it is the denominator of every "% of ceiling" the metrics report, and a model
    whose bytes nobody counted reports no percentage rather than an invented one."""
    leases: int = 0
    """How many requests are holding this model — queued or running, from `submit` until the
    worker is done with the job. Eviction reads this and never the scheduler's state: a job
    that is queued is as much in flight as the one decoding, and the two lines that take a
    lease run with no await between them and the resolve that found the entry."""
    vocabulary: Vocabulary | None = None
    """The token table this model's grammars compile against, built the first time a strict
    schema names it. It hangs off this record rather than off a table keyed by schema for the
    reason the prefix trie hangs off the model: two resident models are two vocabularies, and
    an unload drops the record with this inside it. Built on demand and not at load, because
    it costs 0.27 s and 150k entries that most models are never asked for."""
    grammars: dict[str, Grammar] = field(default_factory=dict)
    """The schemas already compiled against that table, keyed by the schema's canonical text.
    Compiling is 0.19 ms, so what this saves is small; what it is for is that a grammar
    outlives no model — everything a walk changes lives in the constraint it opens, which is
    per request and shared with nobody."""

    @property
    def idle_since(self) -> float:
        """When this model last had a reason to be resident. A model loaded and never asked
        for is idle since it landed, which is what makes it the first to go."""
        return self.loaded_at if self.last_used is None else self.last_used


@dataclass
class Job:
    model_id: str
    model: LanguageModel[ModelInput]
    input: ModelInput
    options: GenerationOptions
    loop: asyncio.AbstractEventLoop
    lease: Residency | None = None
    """The record this job took its lease on, held as the object and not as a key. An unload
    followed by a cold reload of the same id puts a *different* record under that key: giving
    the lease back by name would take it from the new one — which then reads as idle while a
    request is queued on it, and goes negative when this job's own successor ends."""
    chunks: asyncio.Queue[Segment | None] = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    meter: Meter = field(default_factory=Meter)
    """The numbers of this generation, filled by the model as it runs — which is the only
    place they exist. Complete by the time the sentinel reaches the consumer."""
    state: JobState = "queued"
    """What became of the request. Only the worker moves it to a terminal state, so the
    `cancel()` every response ends in — the `finally` of the SSE generator — cannot rewrite
    a job that already finished."""
    error: str | None = None

    def cancel(self) -> None:
        self.cancelled.set()


@dataclass
class _Release:
    """The last reference to an unloaded model, riding the queue to the one place where
    nothing is decoding. The field is emptied rather than the object dropped, because whoever
    waits on `done` is holding this dataclass."""

    model: LanguageModel[ModelInput] | None
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Engine:
    def __init__(self, loader: Loader, store: Store | None = None) -> None:
        self._loader = loader
        self._store = store
        """Where the ceiling and the TTL come from, read per decision. An engine built
        without one admits everything and expires nothing — which is the engine of every
        test that is not about residency, and never the daemon's."""
        self._models: dict[str, LanguageModel[ModelInput]] = {}
        self._loading: dict[str, asyncio.Task[LanguageModel[ModelInput]]] = {}
        self._admission = asyncio.Lock()
        """One cold load at a time — see `_load`. Built here rather than lazily because a
        `Lock` binds to the loop that first awaits it, and this engine has exactly one."""
        self._compiling = asyncio.Lock()
        """One grammar built at a time — see `constrain`. Two requests racing the same cold
        vocabulary would each pay the 0.27 s, and the loser's table would stay alive for as
        long as the grammar it compiled sits in the cache."""
        self._residency: dict[str, Residency] = {}
        self._loads = 0
        """Counts completed cold loads. Read from the decode thread to find out whether a
        model landed while this request was running — a plain int, written on the loop."""
        self._queue: asyncio.Queue[Job | _Release] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._current: Job | None = None
        self._changed = asyncio.Event()
        """Set when a model lands and when a lease is let go — the two ways the next expiry
        can move earlier than the one the sweep went to sleep on."""
        self._metrics = Metrics()

    @property
    def resident(self) -> list[str]:
        return list(self._models)

    @property
    def metrics(self) -> Metrics:
        """What every request through this engine cost. It lives here because the boundaries
        of a request are the worker's, and it outlives the models: a metric is about a
        generation that happened, not about what is resident now."""
        return self._metrics

    @property
    def residency(self) -> Mapping[str, Residency]:
        """In load order, which is the order the daemon took the memory in."""
        return MappingProxyType(self._residency)

    @property
    def running(self) -> int:
        """0 or 1: decision 4 fixes the effective depth at one, because two models
        generating at once contend for the same GPU and the same bandwidth."""
        return 0 if self._current is None else 1

    @property
    def waiting(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._worker = loop.create_task(self._run())
        self._sweeper = loop.create_task(self._sweep())

    def stop(self) -> None:
        if self._sweeper is not None:
            # Before the worker, so nothing new reaches a queue that is about to be drained.
            self._sweeper.cancel()
            self._sweeper = None
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        if self._current is not None:
            self._current.cancel()
            self._current.chunks.put_nowait(None)
            self._current = None
        while not self._queue.empty():
            queued = self._queue.get_nowait()
            if isinstance(queued, _Release):
                # The models are cleared below anyway; what this line is for is the `unload`
                # waiting on the other side, which would otherwise wait for ever.
                queued.model = None
                queued.done.set()
                continue
            queued.state = "cancelled"
            queued.chunks.put_nowait(None)
        for task in self._loading.values():
            task.cancel()
        self._loading.clear()
        self._models.clear()
        self._residency.clear()

    async def resolve(self, model_id: str) -> LanguageModel[ModelInput]:
        """Loads the model if it is not resident. Concurrent callers share one load.

        A model that was already resident keeps the entry it had: a second request on it
        neither reloads it nor duplicates its bytes in the accounting.
        """
        if (resident := self._models.get(model_id)) is not None:
            return resident
        task = self._loading.get(model_id)
        if task is None:
            task = asyncio.create_task(self._load(model_id))
            self._loading[model_id] = task
        try:
            model = await asyncio.shield(task)
        finally:
            self._loading.pop(model_id, None)
        if model_id not in self._models:
            # Every caller that shared the load reaches this line, and only the first may
            # write the entry: the others would overwrite `last_used` with `None` for a
            # model that is already generating, and pay the walk over the tree again.
            found = tree(model)
            self._models[model_id] = model
            self._residency[model_id] = Residency(
                weights_bytes=0 if found is None else resident_bytes(found),
                active_bytes=None if found is None else active_bytes_per_token(found),
                loaded_at=time.time(),
            )
            self._loads += 1
            self._changed.set()
        return model

    async def _load(self, model_id: str) -> LanguageModel[ModelInput]:
        """Admission and the load in the same task — the one `_loading` holds — so a second
        caller for the same id joins it instead of admitting the model twice and evicting for
        space the first admission already made.

        One cold load at a time across ids, which is what the lock buys and it is not about
        the loader: admission decides against what the process holds, and a model that has
        been admitted but not yet read holds nothing. Two of them admitted in parallel each
        decide as if the other did not exist — the way to cross a ceiling twice with two
        requests. Serialized, the second one measures the first, because by then the weights
        are allocated whether or not its entry has been written yet. Generation is not held
        up by this: the gate is the queue's, and this lock is only ever taken off it.
        """
        async with self._admission:
            await self._admit(model_id)
            return await asyncio.to_thread(self._loader, model_id)

    async def _admit(self, model_id: str) -> None:
        """Makes room for what is about to be loaded, or says the room cannot exist.

        What it decides against is `max(both live meters, the accumulator) + KV`, never a
        meter on its own: once a model settles both meters read below what it holds, and that
        reading is what let oMLX admit a second large model over its own ceiling. The KV
        counts on top because it grows per request, and the limit has to keep holding with the
        next request's cache inside it.

        The loop subtracts what it evicts from its own figure rather than measuring again.
        `unload` does not return until the worker has dropped the reference and given the
        buffers back, so a fresh reading would in fact have moved; what the subtraction buys
        is a count of evictions that is a function of the accounting alone, instead of one
        steered by whichever meter happened to be the maximum and by when the allocator
        chose to return the pages.

        Eviction stops at the leases. When what is left is all in flight the load goes through
        over the limit: refusing a request because another one is running is a worse answer
        than a ceiling briefly crossed, and the request in flight is the one already paid for.
        """
        if self._store is None:
            return
        limit = _config(self._store).limit
        # Off the loop: the scan stats every file of every checkpoint in the two caches, and
        # the loop it would otherwise sit on is carrying whatever is decoding.
        incoming = await asyncio.to_thread(_checkpoint_size, model_id)
        if incoming > limit:
            raise ModelTooLarge(
                f"{model_id!r} weighs {incoming} bytes and the whole memory limit is {limit}"
            )
        occupied = self._occupied()
        while occupied + incoming > limit:
            victim = self._victim()
            if victim is None:
                return
            entry = self._residency[victim]
            occupied -= entry.weights_bytes + entry.kv_bytes
            await self.unload(victim)

    def _occupied(self) -> int:
        weights = sum(entry.weights_bytes for entry in self._residency.values())
        kv = sum(entry.kv_bytes for entry in self._residency.values())
        return max(_measured(), weights) + kv

    def _victim(self) -> str | None:
        """The least recently used model no request is holding, or `None` when every resident
        one is leased. Ties break on the id so two models loaded in the same clock tick still
        evict in one order rather than in whichever the dict happened to give."""
        free = [
            (entry.idle_since, model_id)
            for model_id, entry in self._residency.items()
            if entry.leases == 0
        ]
        return min(free)[1] if free else None

    async def _sweep(self) -> None:
        """Expiry by deadline and not by interval: the sweep sleeps exactly until the next
        model falls due, and is woken when one lands or a lease is let go. An interval is what
        loses resolution while everything is idle, which is the whole of when a TTL decides
        anything.

        A TTL patched shorter reaches a sweep already asleep only at its next wake — the
        config is read once per iteration, not watched.
        """
        while True:
            self._changed.clear()
            ttl = None if self._store is None else _config(self._store).ttl
            if ttl is not None:
                await self._expire(ttl)
            with suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), self._due(ttl))

    def _due(self, ttl: int | None) -> float | None:
        """Seconds until the first model expires, `None` when none can: no TTL, nothing
        resident, or everything held by a request — and a lease let go sets the event this
        wait is on, so the last case is not a wait that never ends."""
        if ttl is None:
            return None
        idle = [entry.idle_since for entry in self._residency.values() if entry.leases == 0]
        return None if not idle else max(0.0, min(idle) + ttl - time.time())

    async def _expire(self, ttl: int) -> None:
        """A snapshot of the ids, because `unload` waits for the gate and the dict moves under
        it — including this sweep's own previous eviction."""
        now = time.time()
        for model_id in list(self._residency):
            entry = self._residency.get(model_id)
            if entry is None or entry.leases > 0 or entry.idle_since + ttl > now:
                continue
            await self.unload(model_id)

    async def unload(self, model_id: str) -> bool:
        """Lets go of a resident model, answering whether there was one. It interrupts no
        generation and returns only once the reference is gone.

        The two dict entries go first, here on the loop: from this point nothing new resolves
        to this model, so a request that arrives while the unload is pending loads it cold
        instead of joining one that is being let go. The last reference travels the queue
        rather than being dropped here — the worker takes it between two jobs, which is the
        one moment where no decode thread is running and nothing else is allocating.

        What was already queued for this model still runs: those jobs captured it at `submit`
        and keep it alive on their own, so turning one client's unload into another client's
        failed request would buy nothing. Their bytes come back when the last of them ends.

        `_loading` is not searched: `resolve` pops it and writes `_models` with no await in
        between, so no id is ever in both.
        """
        if model_id not in self._models:
            return False
        release = _Release(self._models.pop(model_id))
        del self._residency[model_id]
        await self._queue.put(release)
        await release.done.wait()
        return True

    async def _reachable(self, model_id: str) -> LanguageModel[ModelInput]:
        """The model a request named, loaded if this daemon loads on demand.

        `NotResident` is a decision about the request and not about the model, so it is read
        here rather than in `resolve` — `PUT /admin/models/{id}/residency` goes through the
        same load and is an order. `constrain` comes through this door too: a grammar needs
        the checkpoint's own token table, and wanting one is not a reason to load a model the
        daemon was told never to load.
        """
        not_resident = "load" if self._store is None else _config(self._store).not_resident
        if not_resident == "fail" and model_id not in self._models:
            raise NotResident(f"{model_id!r} is not loaded and this daemon does not load on demand")
        return await self.resolve(model_id)

    async def constrain(self, model_id: str, schema: Mapping[str, object]) -> Constraint:
        """One request's walk over `schema`, compiled against this model's own token table.

        Three lifetimes, and the whole of what this method is: the table is per resident model
        and dies with its record, the compiled grammar is per (model, schema) and shared, and
        the walk is per request and shared with nobody.

        No lease is taken. Between here and `submit` an eviction can take the model and the
        next request load it again — what comes back is the same table, so the walk stays
        valid and what the race costs is a reload, not a wrong mask.
        """
        model = await self._reachable(model_id)
        entry = self._residency[model_id]
        key = json.dumps(schema, sort_keys=True)
        async with self._compiling:
            grammar = entry.grammars.get(key)
            if grammar is None:
                vocabulary = entry.vocabulary
                if vocabulary is None:
                    vocabulary = await asyncio.to_thread(_vocabulary, model_id, model)
                    entry.vocabulary = vocabulary
                grammar = vocabulary.compile(schema)
                entry.grammars[key] = grammar
        return grammar.constrain()

    async def submit(self, model_id: str, input: ModelInput, options: GenerationOptions) -> Job:
        """Raises `UnsupportedInput` before queueing: a model that cannot take this input
        never becomes a job, so the caller answers with a client error instead of the
        worker failing mid-generation. `NotResident` comes before even that, when the config
        says so — see `_reachable`.
        """
        model = await self._reachable(model_id)
        if not model.accepts(input):
            raise UnsupportedInput(input)
        entry = self._residency[model_id]
        entry.last_used = time.time()
        # The lease is taken here and nowhere else. `resolve` returns without suspending for a
        # model that is already resident, and the load it awaits ends with the entry written,
        # so no admission runs between the entry being found and being held.
        entry.leases += 1
        # How much of what it reads this model may keep for the next request. It rides in the
        # options because out here there is no prompt yet — only a conversation — and it is a
        # number because the cache's element type is the trunk's own, which nothing holding a
        # `LanguageModel[ModelInput]` can name. An engine with no store passes 0, which is the
        # cold path every suite that is not about reuse runs on.
        budget = 0 if self._store is None else _config(self._store).prefix_budget
        options = replace(options, prefix_budget=budget)
        job = Job(model_id, model, input, options, asyncio.get_running_loop(), lease=entry)
        await self._queue.put(job)
        return job

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if isinstance(item, _Release):
                item.model = None
                # Back to the system rather than into MLX's own buffer cache: what an unload
                # is for is a footprint that came down, and the footprint is what the memory
                # rail reads and what admission decides against.
                mx.clear_cache()
                item.done.set()
                continue
            job = item
            self._current = job
            entry = self._residency.get(job.model_id)
            # Both ends of the record are taken here rather than in `_decode`: the meter is
            # written from the decode thread, but nothing about publishing it belongs there.
            self._metrics.begin(
                job.model_id, job.meter, None if entry is None else entry.active_bytes
            )
            try:
                if job.cancelled.is_set():
                    job.state = "cancelled"
                    job.chunks.put_nowait(None)
                else:
                    await asyncio.to_thread(self._decode, job)
            except Exception as error:
                job.error = f"{type(error).__name__}: {error}"
                job.state = "error"
                job.chunks.put_nowait(None)
            finally:
                self._current = None
                self._metrics.end(job.state)
                if job.lease is not None:
                    # The record `submit` took it from, not a fresh lookup by id: see `Job`.
                    # A record an unload has already dropped is nobody's now, and giving the
                    # lease back to it costs nothing.
                    job.lease.leases -= 1
                self._changed.set()
                # The loop suspends on the next `get` with this frame alive, and a local
                # still naming the finished job keeps its model reachable past an unload —
                # the `_Release` riding the queue rebinds `item`, never `job`.
                del item, job

    def _decode(self, job: Job) -> None:
        """The meter rides in the options because the counts live behind `stream`: the
        conversation is rendered by the checkpoint's own template and tokenized inside the
        model, so out here there is text and nothing else to count."""
        job.state = "running"
        # Nothing has run yet: `stream` returns a generator, so this is the memory the
        # weights settled at, with none of this request's own allocation in it.
        stream = job.model.stream(job.input, replace(job.options, meter=job.meter))
        settled = mx.get_active_memory()
        loads = self._loads
        mx.reset_peak_memory()
        try:
            for piece in stream:
                if job.cancelled.is_set():
                    break
                job.loop.call_soon_threadsafe(job.chunks.put_nowait, piece)
        finally:
            self._account(job.model_id, settled, loads)
        # Before the sentinel: what the consumer reads after it is a terminal state.
        job.state = "cancelled" if job.cancelled.is_set() else "completed"
        job.loop.call_soon_threadsafe(job.chunks.put_nowait, None)

    def _account(self, model_id: str, settled: int, loads: int) -> None:
        """The gate serializes generation, so between `settled` and here this job is the
        only thing allocating: MLX's own peak, minus what the weights had settled at, is
        what the request added on top of them. The entry is gone when `stop()` or an `unload`
        dropped the models while this thread was still decoding.

        Loading is the one thing the gate does *not* serialize (see the module docstring),
        and a model that landed during this decode put its whole weight into the same peak.
        Rather than charge one model's weights to another's KV, that request leaves the
        previous figure standing: stale beats attributed to the wrong model.
        """
        entry = self._residency.get(model_id)
        if entry is None:
            return
        if self._loads == loads:
            entry.kv_bytes = max(0, mx.get_peak_memory() - settled)
        entry.last_used = time.time()

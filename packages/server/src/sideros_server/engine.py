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
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import NoReturn, NotRequired, Protocol, TypedDict, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from sideros import GenerationOptions, LanguageModel, ModelInput, UnsupportedInput
from sideros.core.cache import LayerCache
from sideros.core.prompt_cache import Budget
from sideros.footprint import active_bytes_per_token, checkpoint_bytes, resident_bytes
from sideros.generate import Constraint, Meter
from sideros.grammar import Grammar, Vocabulary
from sideros.language import tokenizer_of
from sideros.model import Wrapping
from sideros.parsers import Segment
from sideros.quant.quantization import Quantization
from sideros.quantizing import Quantizing, admits
from sideros_server.metrics import Metrics, RequestState
from sideros_server.prefixes import DiskSpill
from sideros_server.store import Store

Loader = Callable[[str], LanguageModel[ModelInput]]

_SPILL_JOIN_SECONDS = 30.0
"""How long a shutdown waits on a write in flight. Long enough for half a gigabyte on any
disk that is not failing, and bounded because a daemon that cannot exit is worse than a cache
entry that was not written."""

_SPILL_FLOOR = 16 * 1024**2
"""Below this an evicted prefix is not written out. A file is a row in an index every lookup
walks and an inode the ceiling has to account for, and what it buys is the prefill it saves:
at 8 KB a token this is a two-thousand-token conversation, which prefills in well under the
time the write costs."""

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


class NotQuantizable(Exception):
    """A model whose settings ask for a compressed KV cache it cannot hold.

    Three ways, and the message says which: a head whose width closes none of the format's
    groups, a format no cache can be written under at all, or a family that reaches its cache
    through `update_and_fetch` and so never enters a compressed layer's read path.

    Named and raised in `submit`, before the request becomes a job, because the alternative is
    the one thing a compression switch must never do — generate densely while the screen says
    the cache is compressed. A client told the policy is on has no way to find out it was not,
    and every figure it reports about the saving would be about a policy nobody applied.
    """


@dataclass(frozen=True)
class Compression:
    """A KV policy as the engine applies it: a format per side and where the dense head ends.

    Frozen, and made of frozen formats, because it is half the key the probe memoizes on — the
    other half being the model. Its JSON spelling is `features.KvCache`; what travels past this
    line is objects, so nothing downstream re-parses a string it could parse differently.
    """

    k_format: Quantization
    v_format: Quantization
    start_tokens: int


@dataclass(frozen=True)
class KvCompression:
    """The verdict on a model's KV policy: whether it is in force, and why not when it is not.

    Published on the residency record and out on `/admin/state`, because "off" is the answer a
    reader cannot derive from anything else. The policy is stored per model and refused per
    trunk, so a row that is set and a cache that is dense look identical from the settings
    route; the reason is the only thing that tells the two apart.
    """

    applied: bool
    reason: str | None = None


# `catalog`, `config`, `features` and `state` each import `Engine`; the imports below sit inside
# the functions that need them because at module level the cycle closes in either direction.


@dataclass(frozen=True)
class _Settings:
    """The fields of `/admin/config` this file decides with, read together because they come
    out of one row set. Named rather than a tuple: four of the five are integers, and a
    positional read is how the ceiling ends up being compared against a TTL."""

    limit: int
    ttl: int | None
    prefix_budget: int
    disk_budget: int
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
        disk_budget=config.prefix_disk_bytes,
        not_resident=config.not_resident,
    )


def _measured() -> int:
    """What the process holds now, by the two live meters. Both read *below* the real
    residency once a model has settled, which is why the accumulator is maxed
    in over them instead of checked against them."""
    from sideros_server.state import footprint_bytes

    return max(mx.get_active_memory(), footprint_bytes())


def _checkpoint_size(model_id: str) -> int:
    """What the model about to be loaded weighs, off the safetensors headers — the only
    figure that exists before the load. A model the catalog does not list weighs zero here:
    the two caches are what the daemon's own loader can open, and an id outside them has no
    header to sum and nothing for admission to decide against.

    The MTP head comes off, because the trunk's loader drops it (`_drop_mtp`, in nine
    families): what a checkpoint carrying one weighs *as a model* is the file minus `mtp.*`.
    `drafter_bytes` puts it back when the settings turn speculation on, which is the only
    time it is resident — counting it here as well would charge it twice.
    """
    from sideros.task import MTP_PREFIX
    from sideros_server.catalog import scan

    entry = next((entry for entry in scan() if entry.id == model_id), None)
    if entry is None:
        return 0
    return checkpoint_bytes(entry.directory) - checkpoint_bytes(entry.directory, MTP_PREFIX)


def _incoming_size(model_id: str, store: Store) -> int:
    """Everything the load is about to put in memory: the checkpoint, plus the draft that
    lands with it when the model's settings name one."""
    from sideros_server.features import drafter_bytes

    return _checkpoint_size(model_id) + drafter_bytes(model_id, store)


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


def drafter(model: object) -> nn.Module | None:
    """The second checkpoint under this model, when one was paired with it. It is not in
    the model's own tree — two checkpoints are two trees — so nothing that walks `tree`
    ever weighs it, and residency has to ask for it by name."""
    from sideros_server.features import drafting

    facade = drafting(model)
    return None if facade is None else facade.drafter


@runtime_checkable
class _CausalLM(Protocol):
    """`generate.CausalLM` with a runtime check on it.

    Declared here rather than made `runtime_checkable` upstream: what needs the check is this
    file, which has to *find* the trunk under the facades before it can substitute one, and the
    protocol every model is written against should not grow a decorator for a walk that lives
    in a daemon.
    """

    def make_cache(self) -> list[LayerCache]: ...

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array: ...


def _trunk(model: object) -> tuple[Wrapping, _CausalLM] | None:
    """The facade holding the checkpoint's own trunk, and that trunk with any policy already on
    it peeled off — the pair a KV policy is substituted through.

    The walk `tokenizer_of` and `_stop_ids` do, stopping one level higher: what a policy
    replaces is the object `TextLanguageModel.stream` hands to `stream_ids`, and the facade
    above it owns the tokenizer, the template and the prefix trie — none of which the policy
    has an opinion about. Replacing in place is what keeps that trie: it is an attribute of the
    facade, and a facade rebuilt per request would prefill every turn from scratch.

    `None` when nothing under the wrappers answers `make_cache` — a test double, or a model
    that does not decode through a cache at all.
    """
    while isinstance(model, Wrapping):
        held = model.model
        if isinstance(held, Quantizing):
            return model, held.model
        if isinstance(held, _CausalLM):
            return model, held
        model = held
    return None


def _probe(trunk: _CausalLM, policy: Compression) -> str | None:
    """One short prefill and one decode step through the wrapped trunk, `None` when it ran.

    The half of the gate arithmetic cannot answer. A family that reaches its cache with
    `update_and_fetch` never enters `QuantizedKVCache.attend`, and the first thing that says so
    is the attribute that is not there — no shape and no config predicts it. Two ids and then
    one, because a policy that survives the prefill and dies on the step appending to what the
    prefill wrote is the failure a single forward misses.

    Built with `start_tokens=0` whatever the policy says: what has to be reached is the
    compressed region, and a dense head longer than the probe's three ids would let every
    policy pass by never entering it.
    """
    wrapped = Quantizing(trunk, policy.k_format, policy.v_format)
    try:
        cache = wrapped.make_cache()
        mx.eval(wrapped(mx.array([[0, 1]]), cache))
        mx.eval(wrapped(mx.array([[1]]), cache))
    except Exception as failure:
        return f"{type(failure).__name__}: {failure}"
    return None


def _verdict(model_id: str, trunk: _CausalLM, policy: Compression) -> str | None:
    """Why this policy cannot hold on this trunk, `None` when it can.

    Run off the loop, for the reason the loader is: the probe is a forward, and the catalog
    scan behind the width stats every file of every checkpoint.
    """
    from sideros_server.catalog import kv_head_width

    width = kv_head_width(model_id)
    if width is None:
        return (
            "the catalog cannot read this checkpoint's KV head width, and a format whose "
            "groups nobody checked would be quantizing a shape it may not describe"
        )
    if not admits(width, policy.k_format, policy.v_format):
        return f"a KV head of {width} does not close the groups both formats pack in"
    return _probe(trunk, policy)


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
    kv_cache: KvCompression | None = None
    """What became of this model's compressed-KV policy, `None` while its settings ask for
    none. Written at every `submit` that resolves one, so a policy saved and then refused is
    visible on `/admin/state` from the first request that met the refusal rather than only in
    the error that request got back."""
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


def _refuse(model_id: str, entry: Residency, reason: str) -> NoReturn:
    """Publishes the verdict and raises it. One place, because the two have to agree: a reason
    on `/admin/state` that the refused request did not get back is two accounts of one
    decision."""
    entry.kv_cache = KvCompression(applied=False, reason=reason)
    raise NotQuantizable(f"{model_id!r} cannot decode under its KV cache policy: {reason}")


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
    load_seconds: float | None = None
    """What this request paid to put its model in memory, and `None` when it found it there.
    It is the request's number and not the model's: the same checkpoint is cold once and warm
    for every request after it, and a reader told `load 12 s` on the second turn would be
    reading somebody else's wait."""
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


def _nothing() -> None:
    pass


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
        self._spills: dict[str, tuple[int, DiskSpill | None]] = {}
        """One disk tier per model, and the ceiling it was built with. Kept rather than made
        per request because it is what the trie is constructed against: a new object every
        time would be a new trie every time, and the reuse would never survive one turn."""
        self._prefixes: Budget | None = None
        """The one memory ceiling every resident model's trie answers to. Here and not on the
        model for the reason the residency table is here: what it arbitrates is two models
        against each other, and neither of them can see the other. It holds no cache — the
        tries it weighs do, by weak reference — so an unload gives its bytes back on its own."""
        self._quantizable: dict[tuple[str, Compression], str | None] = {}
        """What a `(model, KV policy)` pair was found to be — `None` for one that holds, the
        refusal in words for one that does not. It outlives residency on purpose: the answer is
        a fact about a checkpoint's shape and its family, and an eviction followed by a reload
        of the same id would otherwise pay the probe's forward again for a verdict that cannot
        have moved."""
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
        self._reserving = asyncio.Lock()
        """One reservation at a time — two benchmarks measuring at once would each be the
        other's noise."""
        self._reserved: object | None = None
        """Who holds the queue, when somebody does. A token and not a flag: the holder's own
        submissions have to pass, and comparing the token is what tells them apart from
        everybody else's."""
        self._free = asyncio.Event()
        self._free.set()
        self.on_change: Callable[[], None] = _nothing
        """Raised at every transition `/admin/state` reports — a model landing or leaving,
        the queue moving, the reservation changing hands. What listens is the event stream;
        an engine nobody wired one to announces into the default and costs a call."""

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

    @property
    def reserved(self) -> bool:
        """Whether somebody is holding the queue exclusively. Out on `/admin/state` so a
        client can stop polling the expensive routes while a measurement runs: a catalog
        walk every two seconds is disk the benchmark is also asking for."""
        return self._reserved is not None

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
        # Before the models go, not after: the tries hang off them and the budget holds them
        # weakly, so a cleared table is a budget with nothing left to drain.
        if self._prefixes is not None:
            self._prefixes.drain()
        self._models.clear()
        self._residency.clear()
        # A spill in flight is a file half written under a staging name and a row that will
        # never be inserted. Nothing else waits for it — that is the whole point of writing
        # off the request thread — so shutdown is where it is waited for.
        for _, spill in self._spills.values():
            if spill is not None:
                spill.flush(_SPILL_JOIN_SECONDS)
        self._spills.clear()

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
            drafted = drafter(model)
            self._models[model_id] = model
            self._residency[model_id] = Residency(
                # The drafter's weights are as resident as the model's. They are not in
                # `active_bytes`: what that denominator prices is one decode step, and a
                # drafter is read once per round and not once per token.
                weights_bytes=(0 if found is None else resident_bytes(found))
                + (0 if drafted is None else resident_bytes(drafted)),
                active_bytes=None if found is None else active_bytes_per_token(found),
                loaded_at=time.time(),
            )
            self._loads += 1
            self._changed.set()
            self.on_change()
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
        reading is what once let another MLX server admit a second large model over its
        own ceiling. The KV
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
        incoming = await asyncio.to_thread(_incoming_size, model_id, self._store)
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
        self.on_change()
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

    async def acquire_queue(self) -> object:
        """The queue, exclusively, until the token comes back.

        A measurement is only worth what nothing else running beside it is worth, and the
        gate alone does not give that: it serialises generation, so a chat request lands
        *between* two rounds and moves the median without appearing anywhere. Holding the
        queue is what closes it. Everybody else waits in `submit`, before being queued at
        all, so nothing is dropped and the ordering stays the queue's own.

        The token is what the holder passes back to `submit` to get through its own
        reservation, and what `release_queue` is checked against.
        """
        await self._reserving.acquire()
        token = object()
        self._reserved = token
        self._free.clear()
        self.on_change()
        # What was already queued when the reservation was taken is still somebody's
        # request, and it is measured against nothing: let the worker finish it before the
        # first round starts.
        while self._current is not None or not self._queue.empty():
            await asyncio.sleep(0.01)
        return token

    async def release_queue(self, token: object) -> None:
        """Idempotent, and a no-op for a token that is not the holder's: the release rides a
        `finally`, and a benchmark that raised after the daemon was already restarted must
        not hand somebody else's reservation back."""
        if self._reserved is not token:
            return
        self._reserved = None
        self._free.set()
        self._reserving.release()
        self.on_change()

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[object]:
        """`acquire_queue` and `release_queue` as a block, for a caller that lives on the
        loop. The work of a job does not — it runs in a thread and drives the loop through
        `run_coroutine_threadsafe` — which is why the two halves exist separately."""
        token = await self.acquire_queue()
        try:
            yield token
        finally:
            await self.release_queue(token)

    async def submit(
        self,
        model_id: str,
        input: ModelInput,
        options: GenerationOptions,
        reservation: object | None = None,
    ) -> Job:
        """Raises `UnsupportedInput` before queueing: a model that cannot take this input
        never becomes a job, so the caller answers with a client error instead of the
        worker failing mid-generation. `NotResident` comes before even that, when the config
        says so — see `_reachable`.
        """
        # Before anything else, including the load: a request that waits here has not taken
        # a lease, has not moved a model's `last_used`, and has not put a byte on the GPU
        # while somebody is measuring.
        while self._reserved is not None and reservation is not self._reserved:
            await self._free.wait()
        # Cold decided before the await, because after it the model is resident either way.
        # What is timed is the whole wait — admission and the eviction it may order included:
        # they are the request's seconds too, and a load timed from the loader alone reports
        # less than the reader sat through.
        cold = model_id not in self._models
        started = time.perf_counter()
        model = await self._reachable(model_id)
        loaded = time.perf_counter() - started if cold else None
        if not model.accepts(input):
            raise UnsupportedInput(input)
        entry = self._residency[model_id]
        entry.last_used = time.time()
        # The lease is taken here and nowhere else. `resolve` returns without suspending for a
        # model that is already resident, and the load it awaits ends with the entry written,
        # so no admission runs between the entry being found and being held.
        entry.leases += 1
        try:
            model = await self._compress(model_id, model, entry)
        except NotQuantizable:
            # The lease is taken before the policy is applied because `_compress` awaits, and an
            # await between the record being found and being held is one an eviction can use to
            # drop it. A refusal queues no job, so nothing else will ever give this lease back.
            entry.leases -= 1
            self._changed.set()
            raise
        # How much of what it reads the resident models may keep for the next request. It
        # rides in the options because out here there is no prompt yet — only a conversation —
        # and it is a ceiling rather than the cache because the cache's element type is the
        # trunk's own, which nothing holding a `LanguageModel[ModelInput]` can name. An engine
        # with no store passes 0, the cold path every suite that is not about reuse runs on.
        options = replace(options, prefix_budget=self._budget(), prefix_spill=self._spill(model_id))
        job = Job(
            model_id,
            model,
            input,
            options,
            asyncio.get_running_loop(),
            lease=entry,
            load_seconds=loaded,
        )
        await self._queue.put(job)
        self.on_change()
        return job

    async def _compress(
        self, model_id: str, model: LanguageModel[ModelInput], entry: Residency
    ) -> LanguageModel[ModelInput]:
        """The model this request generates through, under whatever KV policy its settings ask
        for — with the verdict on that policy written to the record either way.

        The gate has two halves because they cost different things. `admits` is arithmetic over
        the catalog's config, the head's width against the groups both formats pack in, and it
        answers with no forward at all. The probe runs the trunk, and only exists for what no
        config predicts (see `_probe`). Both answers are memoized on `(model, policy)`, so a
        checkpoint pays the forward once however many requests arrive under the policy.

        Nothing is cached about the *wrapping*: `Quantizing` holds a reference and no weights,
        so building one per request costs a `list` comprehension the first time the request
        makes a cache. What that buys is that a PATCHed policy is in force for the next request
        rather than for the next load — the switch is `applied`, and the model that would have
        to be reloaded for it is the one thing this daemon takes seconds to do.
        """
        from sideros_server.features import compressing

        policy = None if self._store is None else compressing(self._store, model_id)
        found = _trunk(model)
        if policy is None:
            entry.kv_cache = None
            if found is not None:
                # A policy withdrawn while the model is resident. Put back rather than left
                # standing: the row is what the daemon answers with, and a trunk still wrapped
                # would go on compressing for a switch the screen shows as off.
                found[0].model = found[1]
            return model
        if found is None:
            _refuse(
                model_id,
                entry,
                "nothing under this model's facades answers make_cache, so there is no trunk "
                "for a cache policy to be about",
            )
        holder, trunk = found
        key = (model_id, policy)
        if key not in self._quantizable:
            self._quantizable[key] = await asyncio.to_thread(_verdict, model_id, trunk, policy)
        refusal = self._quantizable[key]
        if refusal is not None:
            _refuse(model_id, entry, refusal)
        holder.model = Quantizing(
            trunk, policy.k_format, policy.v_format, start_tokens=policy.start_tokens
        )
        entry.kv_cache = KvCompression(applied=True)
        return model

    def _budget(self) -> Budget | int:
        """The shared ceiling, rebuilt when the config moved it. Rebuilt and not adjusted:
        `prefix_cache` reads identity to decide whether the trie it holds is still the right
        one, so a new object is how a PATCH reaches models that are already resident."""
        if self._store is None:
            return 0
        total = _config(self._store).prefix_budget
        if self._prefixes is None or self._prefixes.total != total:
            self._prefixes = Budget(total)
        return self._prefixes

    def _spill(self, model_id: str) -> DiskSpill | None:
        """This model's disk tier, or `None` when there is nowhere to put one: no store, the
        ceiling at zero, or an id the catalog does not list — a checkpoint with no stamp has
        no key, and a key that cannot name the checkpoint is a file that outlives the weights
        it was written from.

        The catalog import is inside for the reason `_config`'s and `_checkpoint_size`'s are:
        that module reads this one. `prefixes` does not, so it comes in at the top.
        """
        from sideros_server.catalog import stamp_of

        if self._store is None:
            return None
        ceiling = _config(self._store).disk_budget
        held = self._spills.get(model_id)
        if held is not None and held[0] == ceiling:
            return held[1] if ceiling > 0 else None
        if ceiling <= 0:
            self._spills[model_id] = (ceiling, None)
            return None
        stamp = stamp_of(model_id)
        if stamp is None:
            return None
        spill = DiskSpill(
            self._store,
            model_id,
            stamp=stamp,
            ceiling=ceiling,
            floor=_SPILL_FLOOR,
        )
        self._spills[model_id] = (ceiling, spill)
        return spill

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
            self.on_change()
            entry = self._residency.get(job.model_id)
            # Both ends of the record are taken here rather than in `_decode`: the meter is
            # written from the decode thread, but nothing about publishing it belongs there.
            self._metrics.begin(
                job.model_id,
                job.meter,
                None if entry is None else entry.active_bytes,
                job.load_seconds,
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
                self.on_change()
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
                if piece.channel == "header":
                    # Routing, nobody's prose: `<|start|>assistant to=user<|message|>` names
                    # the channel the next text rides, and no API dialect has a field for
                    # it. Dropped here once, so the four dialects never see one.
                    continue
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

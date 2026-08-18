"""What puts a model in memory, what takes it out again, and the deadline that expires it."""

import asyncio
import time
from contextlib import suppress

import mlx.core as mx

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.engine.footprint import active_bytes_per_token, resident_bytes
from mlx_omnia.server.runtime.errors import ModelTooLarge, NotResident
from mlx_omnia.server.runtime.footprint import occupied_bytes
from mlx_omnia.server.runtime.jobs import Release
from mlx_omnia.server.runtime.residency import Residency
from mlx_omnia.server.runtime.state import EngineState
from mlx_omnia.server.runtime.walks import drafter, tree


class Admitting(EngineState):
    async def resolve(self, model_id: str) -> LanguageModel[ModelInput]:
        """Loads the model if it is not resident. Concurrent callers share one load.

        A model that was already resident keeps the entry it had: a second request on it neither
        reloads it nor duplicates its bytes in the accounting.
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
            # Every caller that shared the load reaches this line, and only the first may write
            # the entry: the others would overwrite `last_used` with `None` for a model that is
            # already generating, and pay the walk over the tree again.
            found = tree(model)
            drafted = drafter(model)
            self._models[model_id] = model
            self._residency[model_id] = Residency(
                # The drafter's weights are as resident as the model's. They are not in
                # `active_bytes`: what that denominator prices is one decode step, and a drafter
                # is read once per round and not once per token.
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
        """Admission and the load in the same task — the one `_loading` holds — so a second caller
        for the same id joins it instead of admitting the model twice and evicting for space the
        first admission already made.

        One cold load at a time across ids, which is what the lock buys and it is not about the
        loader: admission decides against what the process holds, and a model that has been
        admitted but not yet read holds nothing. Two of them admitted in parallel each decide as
        if the other did not exist. Generation is not held up by this: the gate is the queue's,
        and this lock is only ever taken off it.
        """
        async with self._admission:
            await self._admit(model_id)
            return await asyncio.get_running_loop().run_in_executor(
                self._load_thread, self._read, model_id
            )

    def _read(self, model_id: str) -> LanguageModel[ModelInput]:
        """The loader, and the one thing that has to be true before its result can leave this
        thread: every tensor in it evaluated.

        An array nobody has evaluated belongs to the stream of the thread that queued it, and only
        that thread can evaluate it — a weight still lazy when the decode thread reads it raises
        `no Stream(gpu, N) in current thread` on the first forward. Every loader is meant to end at
        `mx.eval(model.parameters())` for its own reason (a mmapped transpose can come back wrong on
        the first read), and doing it again here costs a walk of a tree that has already settled.
        What it buys is that the hand-off is this engine's guarantee rather than each loader's
        promise: the two threads exist so that reading and decoding do not wait for each other, and
        the price of that is paid here, once per load, instead of by a class of failure that
        appears only when some loader forgets.
        """
        model = self._loader(model_id)
        for found in (tree(model), drafter(model)):
            if found is not None:
                mx.eval(found.parameters())
        return model

    async def _admit(self, model_id: str) -> None:
        """Makes room for what is about to be loaded, or says the room cannot exist.

        What it decides against is `max(both live meters, the accumulator) + KV`, never a meter on
        its own: once a model settles both meters read below what it holds, and that reading is
        what once let another MLX server admit a second large model over its own ceiling. The KV
        counts on top because it grows per request, and the limit has to keep holding with the
        next request's cache inside it.

        The loop subtracts what it evicts from its own figure rather than measuring again. What
        the subtraction buys is a count of evictions that is a function of the accounting alone,
        instead of one steered by whichever meter happened to be the maximum.

        Eviction stops at the leases. When what is left is all in flight the load goes through
        over the limit: refusing a request because another one is running is a worse answer than a
        ceiling briefly crossed.
        """
        environment = self._environment
        if environment is None:
            return
        limit = environment.settings().limit
        # Off the loop: the scan stats every file of every checkpoint in the two caches, and the
        # loop it would otherwise sit on is carrying whatever is decoding.
        incoming = await asyncio.to_thread(environment.incoming_bytes, model_id)
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
        return max(occupied_bytes(), weights) + kv

    def _victim(self) -> str | None:
        """The least recently used model no request is holding, or `None` when every resident one
        is leased. Ties break on the id so two models loaded in the same clock tick still evict in
        one order rather than in whichever the dict happened to give."""
        free = [
            (entry.idle_since, model_id)
            for model_id, entry in self._residency.items()
            if entry.leases == 0
        ]
        return min(free)[1] if free else None

    async def _sweep(self) -> None:
        """Expiry by deadline and not by interval: the sweep sleeps exactly until the next model
        falls due, and is woken when one lands or a lease is let go. An interval is what loses
        resolution while everything is idle, which is the whole of when a TTL decides anything.

        A TTL patched shorter reaches a sweep already asleep only at its next wake — the config is
        read once per iteration, not watched.
        """
        while True:
            self._changed.clear()
            settings = self._settings()
            ttl = None if settings is None else settings.ttl
            if ttl is not None:
                await self._expire(ttl)
            with suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), self._due(ttl))

    def _due(self, ttl: int | None) -> float | None:
        """Seconds until the first model expires, `None` when none can: no TTL, nothing resident,
        or everything held by a request — and a lease let go sets the event this wait is on, so
        the last case is not a wait that never ends."""
        if ttl is None:
            return None
        idle = [entry.idle_since for entry in self._residency.values() if entry.leases == 0]
        return None if not idle else max(0.0, min(idle) + ttl - time.time())

    async def _expire(self, ttl: int) -> None:
        """A snapshot of the ids, because `unload` waits for the gate and the dict moves under it
        — including this sweep's own previous eviction."""
        now = time.time()
        for model_id in list(self._residency):
            entry = self._residency.get(model_id)
            if entry is None or entry.leases > 0 or entry.idle_since + ttl > now:
                continue
            await self.unload(model_id)

    async def unload(self, model_id: str) -> bool:
        """Lets go of a resident model, answering whether there was one. It interrupts no
        generation and returns only once the reference is gone.

        The two dict entries go first, here on the loop: from this point nothing new resolves to
        this model, so a request that arrives while the unload is pending loads it cold instead of
        joining one that is being let go. The last reference travels the queue rather than being
        dropped here — the worker takes it between two jobs, which is the one moment where no
        decode thread is running and nothing else is allocating.

        What was already queued for this model still runs: those jobs captured it at `submit` and
        keep it alive on their own. Their bytes come back when the last of them ends.

        `_loading` is not searched: `resolve` pops it and writes `_models` with no await in
        between, so no id is ever in both.
        """
        if model_id not in self._models:
            return False
        release = Release(self._models.pop(model_id))
        del self._residency[model_id]
        self.on_change()
        await self._queue.put(release)
        await release.done.wait()
        return True

    async def reachable(self, model_id: str) -> LanguageModel[ModelInput]:
        """The loaded model, for a caller that has to read something off the checkpoint before the
        job is submitted — which is the call envelope a forced `tool_choice` constrains to. No
        lease, for `constrain`'s reason: what a race costs is a reload."""
        return await self._reachable(model_id)

    async def _reachable(self, model_id: str) -> LanguageModel[ModelInput]:
        """The model a request named, loaded if this daemon loads on demand.

        `NotResident` is a decision about the request and not about the model, so it is read here
        rather than in `resolve` — an explicit residency order goes through the same load.
        `constrain` comes through this door too: a grammar needs the checkpoint's own token table,
        and wanting one is not a reason to load a model the daemon was told never to load.
        """
        settings = self._settings()
        not_resident = "load" if settings is None else settings.not_resident
        if not_resident == "fail" and model_id not in self._models:
            raise NotResident(f"{model_id!r} is not loaded and this daemon does not load on demand")
        return await self.resolve(model_id)

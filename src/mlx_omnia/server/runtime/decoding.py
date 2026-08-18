"""How a job enters the clock, and what its generation cost once it leaves it."""

import asyncio
import time
from dataclasses import replace
from functools import partial

import mlx.core as mx

from mlx_omnia.engine.language import ContinuousLanguageModel
from mlx_omnia.engine.model import CompositeModel
from mlx_omnia.server.runtime.flow import Member, Outlet
from mlx_omnia.server.runtime.jobs import (
    Generation,
    GenerationMember,
    Job,
    Phase,
    Prefilling,
    Streaming,
)
from mlx_omnia.server.runtime.prefixing import Prefixing


class Phases(Prefixing):
    def _begin(self, model: ContinuousLanguageModel | None, job: Job) -> Phase:
        """The phase this job enters the clock in — on the model thread, because both branches
        touch MLX: one reads the memory the weights settled at, the other makes the request's
        cache and takes the trie's match."""
        if model is None:
            return self._streaming(job)
        assert isinstance(job.model, CompositeModel)
        prefill = model.begin_batch(
            job.model.prepare(job.input), replace(job.options, meter=job.meter)
        )
        if prefill is not None:
            return Prefilling(prefill)
        # `can_batch` answers on the options, before the conversation has been rendered; what the
        # prompt turned out to be is only known here, and a picture is a prompt no `step_batch`
        # takes. Refused this late the request is not a failure — it is a request that does not
        # batch, which is what the streaming phase is for.
        return self._streaming(job)

    def _streaming(self, job: Job) -> Streaming:
        # Nothing has run yet: `stream` returns a generator, so this is the memory the weights
        # settled at, with none of this request's own allocation in it.
        pieces = job.model.stream(job.input, replace(job.options, meter=job.meter))
        settled = mx.get_active_memory()
        loads = self._loads
        mx.reset_peak_memory()
        return Streaming(pieces, settled, loads)

    async def _enlist(self, model: ContinuousLanguageModel | None, job: Job) -> GenerationMember:
        phase = await asyncio.get_running_loop().run_in_executor(
            self._model_thread, partial(self._begin, model, job)
        )
        job.state = "running"
        return Member(Generation(job, phase), Outlet(job.loop, job.chunks), job.cancelled)

    def _account(self, model_id: str, settled: int, loads: int) -> None:
        """The gate serializes generation, so between `settled` and here this job is the only
        thing allocating: MLX's own peak, minus what the weights had settled at, is what the
        request added on top of them. The entry is gone when `stop()` or an `unload` dropped the
        models while this thread was still decoding.

        Loading is the one thing the gate does *not* serialize, and a model that landed during
        this decode put its whole weight into the same peak. Rather than charge one model's
        weights to another's KV, that request leaves the previous figure standing: stale beats
        attributed to the wrong model.
        """
        entry = self._residency.get(model_id)
        if entry is None:
            return
        if self._loads == loads:
            entry.kv_bytes = max(0, mx.get_peak_memory() - settled)
        entry.last_used = time.time()

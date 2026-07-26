"""Global FCFS generation queue.

MLX enqueues on a single GPU stream and the decode loop is CPU-hot: requests must
not call generate() concurrently. The gate serializes generation — one worker task
consumes the queue and runs each job to completion (or cancellation) before the
next. Model loading happens outside the gate.
"""

import asyncio
import threading
from dataclasses import dataclass, field

from sideros import GPT2, GPT2Tokenizer, stream_generate


@dataclass
class Job:
    prompt: str
    max_tokens: int
    loop: asyncio.AbstractEventLoop
    chunks: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: str | None = None

    def cancel(self) -> None:
        self.cancelled.set()


class Engine:
    def __init__(self, model: GPT2, tokenizer: GPT2Tokenizer, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._current: Job | None = None

    def start(self) -> None:
        self._worker = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        if self._current is not None:
            self._current.cancel()
            self._current.chunks.put_nowait(None)
            self._current = None
        while not self._queue.empty():
            self._queue.get_nowait().chunks.put_nowait(None)

    async def submit(self, prompt: str, max_tokens: int) -> Job:
        job = Job(prompt, max_tokens, asyncio.get_running_loop())
        await self._queue.put(job)
        return job

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            self._current = job
            try:
                if job.cancelled.is_set():
                    job.chunks.put_nowait(None)
                else:
                    await asyncio.to_thread(self._decode, job)
            except Exception as error:
                job.error = f"{type(error).__name__}: {error}"
                job.chunks.put_nowait(None)
            finally:
                self._current = None

    def _decode(self, job: Job) -> None:
        stream = stream_generate(
            self.model, self.tokenizer, job.prompt, max_tokens=job.max_tokens
        )
        for piece in stream:
            if job.cancelled.is_set():
                break
            job.loop.call_soon_threadsafe(job.chunks.put_nowait, piece)
        job.loop.call_soon_threadsafe(job.chunks.put_nowait, None)

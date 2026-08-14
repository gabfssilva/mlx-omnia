# 09 — Serving

What changes when a model stops being a function you call and becomes a process that many requests share.

## The constraint that shapes everything

MLX enqueues on a single GPU stream, and the decode loop is CPU-hot. Two requests calling `generate()` at once do not run twice as fast; they interleave badly and both get slower.

So generation is **serialized behind a global FCFS gate**: one worker task consumes a queue and runs each job to completion (or cancellation) before starting the next (`packages/mlx_omnia-server/src/mlx_omnia_server/engine.py`).

That is a deliberate simplification with a cost — a long generation blocks short ones behind it — and it is the honest starting point. Continuous batching (running several sequences through one forward pass, admitting and retiring them independently) is the standard answer and is a goal here, not a solved problem.

One thing is explicitly kept *outside* the gate: **model loading**. A cold load of a large model takes seconds, and holding the generation gate through it would make every resident model wait on it.

## Residency

Nothing is resident at boot. A request names a model and that is what loads it. What removes a model again:

- **The memory ceiling.** A load that would cross it evicts the least recently used model first.
- **An idle TTL.** A model untouched past it unloads on its own.

Both figures come from configuration, read per decision rather than cached — so changing them takes effect on the next admission, not on the next restart.

Admission needs to price a checkpoint *before* loading it, which is why `footprint.py` computes `checkpoint_bytes` straight from safetensors headers with nothing mapped (chapter 07). Deciding whether something fits by loading it and seeing is not a decision.

Loading is exposed as a **job** rather than a synchronous call, because a request that waits on a cold load is a request that times out. A model already resident answers the same way over a job that completes immediately — the button is idempotent, not a second load. Unloading is *not* a job: what it costs is waiting for the gate, and what the caller wants back is whether the memory returned, which a status code says and a job id does not.

## Prefix reuse

Consecutive requests in a conversation share a long prefix: the system prompt, then the whole history. Re-prefilling it every turn is the largest avoidable cost in a chat workload.

`core/prompt_cache.py` stores materialized caches in a **per-token trie**, under a byte budget.

Per token, not a compressed radix, and the docstring says why: every node is then a legal cut point. Which makes the case that matters cheap — a stored cache *longer* than the common prefix is **rewound** to it with `LayerCache.trim` instead of being thrown away.

Three rules that generalize to any such cache:

- **A non-trimmable cache is skipped, not rewound.** Recurrent state and conv windows cannot be reconstructed backwards (chapter 05), and a wrong cache is exactly what survives a greedy decode.
- **Eviction is by role, not FIFO**: `assistant` drains before `user` before `system`. The system prompt is the prefix every request shares, so it is the last thing to give up.
- **`take` hands the cache over rather than lending it.** Whoever generates from it writes past the prefix; the trie's record of it is no longer true. The caller re-inserts it, keyed by the full sequence, when the request finishes.

One subtlety worth keeping: `take` always leaves at least one token to prefill. A forward pass needs a row, and the logits the sampler reads are the ones of the last prompt position.

## Jobs

Loading, downloading, quantizing and benchmarking are long, cancellable, and interesting while they run. The shape used for all of them (`packages/mlx_omnia-server/src/mlx_omnia_server/jobs.py`):

**A job is a state plus a stream of events, and every frame is the whole state.** Never a delta. The stream opens with the state as it is now, and the subscription is registered *before* that first state is read — so a transition landing in between is delivered twice rather than lost. A repeated frame costs nothing when the frame is the whole state; a lost transition costs a client stuck at 40%.

**Cancellation is cooperative and must reach inside the blocking work.** The body runs in a thread, where cancelling an async task interrupts neither an MLX load nor a socket read. So `DELETE` sets a `threading.Event`, and `report` — the same call the work already makes to publish progress — raises when it finds it set. Cancellation rides the progress channel, which is the only channel guaranteed to be reached.

**Every frame is persisted.** `GET` is answered from storage, so a job that outlived the process still reports where it stopped. What stays in memory is only what cannot be persisted: the cancellation flag and the open streams.

## Dialects

The server speaks OpenAI, Anthropic and Gemini request/response shapes over the same engine. Each is a translation layer: message shapes, streaming event formats, tool-call encodings and error bodies differ; the generation underneath does not.

Tool calling is where the abstraction leaks. Models emit tool calls as *text* in a family-specific format — an XML-ish envelope, a JSON block, a special token sequence — which a parser (`parsers/`) must recognize incrementally, in a stream, without having buffered the whole output. That is why parsing is a streaming segmenter with a fallback rather than a regex over a finished string.

## Constrained decoding

When the caller demands JSON matching a schema, the constraint is enforced at *sampling* time, not by retrying: at each step, ids that cannot continue a valid document have their logits set to `-inf`.

It is not a stateless logit filter. The mask at step `n+1` depends on the id drawn at step `n`, so the object has state and a per-token side effect. The `Constraint` protocol in `generate.py` says exactly that, with `mlx_omnia.grammar` as the implementation. Its `remaining` argument — how many steps the run still has, this one included — lets the grammar force what *closes* the output before the budget runs out, instead of letting a valid document be truncated into an invalid one.

## The decode loop, once more

The per-token loop in `generate.py` is worth reading for one property. The sampler returns the token as an `mx.array` and does not call `.item()` before the next step is queued:

> step n+1 is async-evaluated before step n's sync, so the GPU never idles between steps.

Reading a value back to the CPU is a synchronization point. Doing it before enqueueing the next step means the GPU sits idle for the round trip, every token. Keeping the token on-device and enqueueing first turns that into overlap. It is a small change with a shape worth recognizing: on a latency-bound loop, the cost is often not the work but the waiting for an answer nobody needed yet.

## Where it lives

- `engine.py` — residency, eviction, the FCFS gate, the worker.
- `jobs.py` — the job/state/SSE machinery and cooperative cancellation.
- `residency.py` — the two model-screen buttons, and a routing-order note worth reading before adding a path under `/admin/models/{model_id:path}`.
- `store.py` — persistence.
- `app.py`, `anthropic.py`, `gemini.py`, `responses.py` — the dialects.
- `core/prompt_cache.py` — prefix reuse (in the engine package, not the server).
- `metrics.py` — per-request state and counters.

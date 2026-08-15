# 09: Serving

What changes when a model stops being a function you call and becomes a process that many requests share.

## The constraint that shapes everything

MLX enqueues on a single GPU stream, and the decode loop is CPU-hot. Two requests calling `generate()` at once do not run twice as fast; they interleave badly and both get slower.

So generation runs on **one model thread**, and requests share it through **continuous batching**: a single clock ticks a group of sequences through one forward pass per token, admitting and retiring members independently (`engine.py`, `flow.py`). Every model family batches; a request the batch cannot answer for — a prompt still arriving as an iterator, an image prompt — streams alone on the same thread. `max_concurrent_requests` bounds the group.

One thing is explicitly kept *outside* the gate: **model loading**. A cold load of a large model takes seconds, and holding the generation gate through it would make every resident model wait on it.

## Residency

Nothing is resident at boot. A request names a model and that is what loads it. What removes a model again:

- **The memory ceiling.** A load that would cross it evicts the least recently used model first.
- **An idle TTL.** A model untouched past it unloads on its own.

Both figures come from configuration and are read for every decision. Changes take effect on the next admission without a restart.

Admission prices a checkpoint *before* loading it. `src/mlx_omnia/engine/footprint.py` computes `checkpoint_bytes` directly from safetensors headers without mapping the tensors (chapter 07). Loading first would consume the memory that admission is supposed to protect.

Loading is exposed as a **job** because cold loads can outlive an HTTP request. An already resident model returns an immediately completed job, which keeps the operation idempotent. Unloading waits for the generation gate and returns whether memory was released; a synchronous status code carries that result directly.

## Prefix reuse

Consecutive requests in a conversation share a long prefix: the system prompt, then the whole history. Re-prefilling it every turn is the largest avoidable cost in a chat workload.

`src/mlx_omnia/engine/core/prompt_cache.py` stores materialized caches in a **per-token trie**, under a byte budget.

Each token gets a node rather than using a compressed radix, which makes every node a legal cut point. If a stored cache extends past the common prefix, `LayerCache.trim` rewinds it to that point instead of discarding it.

Three rules that generalize to any such cache:

- **Non-trimmable caches are skipped.** Recurrent state and convolution windows cannot be reconstructed backwards (chapter 05), and a wrong cache can survive greedy decoding.
- **Role controls eviction order.** `assistant` drains before `user`, then `system`. The system prompt is shared by every request, so it is evicted last.
- **`take` transfers ownership of the cache.** Generation writes past the prefix and invalidates the trie's record. The caller re-inserts it under the full sequence when the request finishes.

One subtlety worth keeping: `take` always leaves at least one token to prefill. A forward pass needs a row, and the logits the sampler reads are the ones of the last prompt position.

## Jobs

Loading, downloading, quantizing and benchmarking are long, cancellable, and interesting while they run. The shape used for all of them (`jobs.py`):

**A job combines state with an event stream; every frame contains the full state.** The subscription is registered before reading the initial state. A transition in that interval may be delivered twice but cannot be lost. Duplicate full-state frames are harmless; a missing transition can leave a client stuck at 40%.

**Cancellation is cooperative and must reach inside blocking work.** The body runs in a thread, so cancelling an async task does not interrupt an MLX load or socket read. `DELETE` sets a `threading.Event`. The existing `report` call checks it while publishing progress and raises when cancellation is set.

**Every frame is persisted.** `GET` is answered from storage, so a job that outlived the process still reports where it stopped. What stays in memory is only what cannot be persisted: the cancellation flag and the open streams.

## Dialects

The server speaks OpenAI, Anthropic and Gemini request/response shapes over the same engine. Each is a translation layer: message shapes, streaming event formats, tool-call encodings and error bodies differ; the generation underneath does not.

Tool calling exposes a model-specific detail. Models emit calls as *text* using an XML-like envelope, a JSON block or special tokens. A parser in `parsers/` must recognize that format incrementally before the full output exists, so parsing uses a streaming segmenter with a fallback.

## Constrained decoding

When the caller demands JSON matching a schema, the constraint is enforced at *sampling* time, not by retrying: at each step, ids that cannot continue a valid document have their logits set to `-inf`.

The logit filter is stateful. Its mask at step `n+1` depends on the id drawn at step `n`, which creates a per-token side effect. The `Constraint` protocol in `src/mlx_omnia/engine/generate.py` captures that behavior; `src/mlx_omnia/engine/grammar.py` implements it. The `remaining` argument counts available steps including the current one, allowing the grammar to force closing tokens before the budget ends.

## The decode loop, once more

The per-token loop in `src/mlx_omnia/engine/generate.py` is worth reading for one property. The sampler returns the token as an `mx.array` and does not call `.item()` before the next step is queued:

> step n+1 is async-evaluated before step n's sync, so the GPU never idles between steps.

Reading a value back to the CPU is a synchronization point. Doing it before enqueueing the next step means the GPU sits idle for the round trip, every token. Keeping the token on-device and enqueueing first turns that into overlap. It is a small change with a shape worth recognizing: on a latency-bound loop, the cost is often not the work but the waiting for an answer nobody needed yet.

## Where it lives

- `engine.py`: residency, eviction, the FCFS gate, the worker.
- `jobs.py`: the job/state/SSE machinery and cooperative cancellation.
- `residency.py`: the two model-screen buttons, and a routing-order note worth reading before adding a path under `/admin/models/{model_id:path}`.
- `store.py`: persistence.
- `app.py`, `anthropic.py`, `gemini.py`, `responses.py`: the dialects.
- `src/mlx_omnia/engine/core/prompt_cache.py`: prefix reuse, in the engine rather than the server.
- `metrics.py`: per-request state and counters.

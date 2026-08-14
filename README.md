<p align="center">
  <img src="logo.svg" alt="mlx-omnia" width="400">
</p>

> **Work in progress.** APIs, model coverage, and internals change without notice.

mlx-omnia is an open-source inference engine for Apple Silicon, written in Python and [Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), on top of [MLX](https://github.com/ml-explore/mlx). It runs models locally and exposes them through APIs compatible with OpenAI, Anthropic and Gemini, and it can also be used as a Python library, from the command line, or from a macOS menu bar app.

mlx-omnia depends on MLX alone to run a model and supports around 45 model architecture families, and it downloads and quantizes checkpoints itself. Its goal is to get as close as possible to the physical limit of the machine while respecting numerical accuracy and maintainable engineering.

## Installation

Everything below needs an Apple Silicon Mac. One distribution ships all the parts, and the extras decide which of them you install.

### Engine

```sh
pip install mlx-omnia
```

`load` is the only entry point, and it takes a Hugging Face repository or a local checkout:

```python
from mlx_omnia import Chat, ChatMessage, GenerationOptions, load

model = load("Qwen/Qwen3-4B")
question: ChatMessage = {"role": "user", "content": "Explain KV caching in two sentences."}

for piece in model.stream(Chat((question,)), GenerationOptions(max_tokens=512)):
    print(piece.text, end="", flush=True)
```

### Server

```sh
pip install "mlx-omnia[server]"
omnia-server
```

The server listens on `127.0.0.1:8642`, and `--host` and `--port` move it. Point an OpenAI SDK at `http://127.0.0.1:8642/api/openai/v1`, an Anthropic client at `/api/anthropic/v1` and a Gemini client at `/api/gemini/v1beta`.

### CLI

```sh
pip install "mlx-omnia[cli]"
omnia chat
```

The CLI speaks HTTP and nothing else, so it needs a server to talk to, and `--url` points it at one that is not local. Besides `chat`, it has `omnia run` for a single prompt on stdout, `omnia models list` for what is on disk and what is loaded, and `omnia status` for the daemon and the machine under it.

Installing `mlx-omnia[all]` gives you the server and the CLI at once.

### macOS app

The app is not published yet, so it is built from the checkout with [mise](https://mise.jdx.dev) and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/gabfssilva/mlx-omnia
cd mlx-omnia
mise run app
```

It is a SwiftUI panel hanging off the menu bar that does chat and model management over the same HTTP API, and it starts the server itself when nothing answers on the port. From that checkout, `mise run sync` installs the extras that a bare `uv sync` leaves out.

## How it works

Every layer of the engine declares what it needs and leaves the layer below it to decide how that gets served. The snippets that follow are the load-bearing places where that happens.

### A model is a signature

```python
class Model[I: ModelInput, O, Options](Protocol):
    def accepts(self, input: ModelInput) -> TypeIs[I]: ...
    def stream(self, input: I, options: Options) -> Iterator[O]: ...
```

The intuition is that a model is a function, and its signature tells you what the function reads and what it writes. The signature is two sets of content types, $\mathsf{Input}$ and $\mathsf{Output}$, where a content type is one of six modalities (text, image, audio, video, document, vector) paired with a media type. The whole contract fits in one arrow:

$$\mathsf{model} : \mathsf{Input} \times \mathsf{Options} \longrightarrow \mathsf{Stream}(\mathsf{Output})$$

In words: give the model data of the types in $\mathsf{Input}$, plus generation options, and it gives back a stream of data of the types in $\mathsf{Output}$. That arrow is `stream` in the snippet, and `accepts` checks whether a piece of data belongs to the left side.

A capability is a smaller function that ends where the model begins. It takes one type $a$ that the model does not read and turns it into something the model does:

$$\mathsf{capability} : a \longrightarrow \mathsf{Input}, \qquad a \notin \mathsf{Input}$$

Placing it in front of the model widens the left side of the arrow and changes nothing on the right:

$$\mathsf{model} \circ \mathsf{capability} : (\mathsf{Input} \cup \lbrace a \rbrace) \times \mathsf{Options} \longrightarrow \mathsf{Stream}(\mathsf{Output})$$

The one capability shipping today is an image tower placed ahead of a text trunk, converting pixels into rows the trunk reads like any other, and the trunk never learns images exist. Audio, video and document front-ends will come as the same construction: each one is a new $a$ composed in front, with the model untouched.

A language model is one way of filling the arrow in: both sets hold only text, and the function inside is next-token prediction, called once per token to produce the stream. An embedding model is the same arrow with a vector on the right and a stream of length one. Everything the engine does above the arrow, loading, scheduling, serving, works on the arrow and never on the filling.

### The model declares the arithmetic it needs

```python
route = Route(self.gate.weight, experts=self.experts, k=self.k, normalize=self.norm_topk)
gate_up = GateUp(switch.gate_up_proj, hidden=self.hidden, inner=switch.inner)
down = DownCombine(switch.down_proj, hidden=self.hidden, inner=switch.inner)
```

That is `qwen3_moe` asking for its routing, its two expert gemvs and its routed sum. No Metal kernel is named, no quantization format is mentioned, and the same three lines appear in families with nothing else in common. Resolution happens once, at the first single-token step, when the weights are loaded and the leaves' formats are final.

### Choosing the best kernel

<p align="center">
  <img src="docs/kernel-strategy.svg" alt="A Qmv call over an NVFP4 leaf with a gate epilogue binds GatedNvfp4Qmv, the one strategy in core/kernels/qmv/ that computes exactly that projection" width="740">
</p>

A model calls a primitive such as `Qmv` handing it everything that affects the result: the weight with its quantization format, the logical shape, and the epilogue that follows the projection. Every strategy under the primitive implements that same operation for one specialization, and its `build` reads the declaration and returns an instance only when it computes exactly what was declared, down to each knob; an NVFP4 weight with a gate epilogue binds `GatedNvfp4Qmv`, and the same weight without the epilogue would bind `Nvfp4Qmv` instead. Candidates are tried in order of preference and `DefaultQmv` accepts everything, so the choice is made once, at construction, and always lands on something correct. The strictness is the point: a kernel that almost matches would silently run another model's arithmetic and still produce plausible text, and declining is how a strategy avoids that.

### Prefill and decode are different programs

```python
def step(self, h: mx.array, residual: mx.array) -> mx.array:
    """T=1: routing, both expert gemvs, the routed sum and the residual join in
    three dispatches."""
    route, gate_up, down = self._kernels()
    chosen, weights = route(h.reshape(-1), logits=self.gate(h).reshape(-1))
    act = gate_up(h.reshape(-1), chosen)
    return down(act, chosen, weights, residual.reshape(-1)).reshape(1, 1, self.hidden)

def __call__(self, x: mx.array) -> mx.array:
    chosen, weights = self.route(x)
    if x.shape[-2] * self.k >= SORTED_GATHER_MIN:
        routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
```

Decode reads one row per step and is limited by memory bandwidth, so the whole MoE block collapses into three dispatches and nothing on that path syncs with the host or grows with the context. Prefill is limited by compute instead, so above a token threshold the same block sorts tokens by expert and gathers them, which is worth its cost only in bulk. The ceiling for decode comes from the checkpoint's own active bytes per token, and results are reported against it: gpt-oss 20B in MXFP4 decodes at 118.8 tok/s, 89.9% of that ceiling.

### Weights are reorganized once, at load

```python
return prepare_weights(
    config,
    load_shards(directory),
    [
        lambda weights: fuse_qkv(weights, layers),
        lambda weights: interleave_gate_up(weights, layers),
    ],
    dtype,
)
```

qkv is concatenated, gate and up are interleaved by row, and the experts are stacked for gather matmuls. The cost is paid once so that every step afterwards reads memory in fewer, larger pieces.

### A speedup is a number, and it has to survive the numerics

```python
assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")
```

`floor` is three times the noise the fixture itself measured for that tensor, so the checkpoint decides the tolerance and the implementation lives with it. The comparison runs over the full logits, because a cache bug can preserve the greedy pick while already producing a different distribution. On top of that, `omnia-bench` runs A and B interleaved in one process behind a thermal gate, and an optimization only lands when the gain survives that comparison and the parity tests stay green.

## Contributing

The distribution is one package with four siblings inside it. `mlx_omnia` itself only re-exports the engine's public API, so none of the four is the package the others live in.

| module | what it does |
| --- | --- |
| `mlx_omnia.engine` | Holds the model packages, the checkpoint loading, the generation pipeline, the Metal kernels and the quantization. |
| `mlx_omnia.server` | A FastAPI server that speaks the OpenAI, Anthropic and Gemini APIs, streaming included, behind a global FCFS queue. |
| `mlx_omnia.cli` | An HTTP client for the server. It depends only on httpx. |
| `mlx_omnia.bench` | The measurement instrument (`omnia-bench`). It runs a thermal gate, teacher forcing and interleaved rounds, and prints a dominance verdict. It is engine-agnostic, and omnia and mlx-lm are optional adapters under it. |

Keeping them as siblings is what makes the boundaries checkable, because `lint-imports` can then forbid the harness a single name, `mlx_omnia.engine`, which covers whatever the engine grows next.

The app is not one of the four. It lives in `app/` as a SwiftUI menu bar panel with its own SwiftPM package, and it reaches the daemon over HTTP like any other client, which is what lets it be written in another language.

Inside the engine:

```
src/mlx_omnia/engine/
  model.py              the contract: signature, content types, composed capabilities
  models/<family>/      one self-contained package per architecture family
  checkpoint.py         the load spine; each architecture declares a CHECKPOINT
  task.py               `load` is the only entry point; it dispatches on model_type
  language.py           the language task: prompt, tokenizer, generation options
  generate.py           model-agnostic decode pipeline
  bpe.py                byte-level BPE tokenizer
  quant/                quantization: formats, plans, calibration
  core/                 architecture-agnostic infrastructure (cache, rope, masks, kernels/)
```

These are the design rules that keep it in this shape:

- **Each architecture family gets one self-contained, checkpoint-shaped package.** The property names are the checkpoint's own, so the module tree is the shape table and strict loading is the totality contract. Nothing renames anything along the way.
- **Families share no modeling layer.** It is fine for two of them to repeat an attention shape, because a shared abstraction couples architectures that will diverge later. Code only moves to `core/` on the second byte-identical use.
- **Loading has a single door.** `mlx_omnia.load` reads `model_type` and dispatches to that architecture's `CHECKPOINT`, and there is no public per-architecture loader.
- **Protocols sit at the boundary.** Model-agnostic code depends on a `Protocol` sized to what it actually calls, and that protocol is defined where it is consumed. Models satisfy it structurally and never import the consumer.
- **Layering is strict and one-directional**, and import-linter contracts and `uv tree` enforce it. The server knows only the engine's public API, while the CLI and the app speak HTTP and nothing else.
- **Kernels are named after operations and never after models.** They live in `core/kernels/` and export a cheap `*_applies(...)` predicate, so the model decides when a kernel applies to it and the kernel never knows the model exists.
- **Typing is strict and has no escape hatches.** pyright runs in strict mode at zero errors, and stale upstream stubs get corrected in `core/mxcompat.py` instead of silenced with `# type: ignore`.

You run the suite and the rest of the gate like this:

```sh
uv run pytest -q                                             # suite
uv run ruff check && uv run pyright && uv run lint-imports   # rest of the gate
```

Fixtures are generated from reference implementations (`tests/fixtures/generate_*.py`) and are not checked in, while `SHA256SUMS` is. Parity tests compare full logits against those fixtures, with tolerances derived from measured noise floors.

Benchmarks run interleaved A/B in the same process, behind a thermal gate. `omnia-bench interleaved` compares against a baseline engine, and `omnia-bench paired` compares the working tree against a git ref.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) is the array framework everything here runs on.
- [mlx-lm](https://github.com/ml-explore/mlx-lm) is the numerical reference and the benchmark baseline for large checkpoints.
- [PyTorch](https://github.com/pytorch/pytorch) and [transformers](https://github.com/huggingface/transformers) are the authoritative reference implementations that every port is validated against.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [oMLX](https://github.com/jundot/omlx) and [LM Studio](https://lmstudio.ai)'s mlx-engine were studied for ports and server-side ideas.

## License

MIT

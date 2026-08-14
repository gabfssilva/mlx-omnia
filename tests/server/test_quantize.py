"""The quantize job: what it writes, under which id, which methods it offers, and what a
`DELETE` leaves behind.

The model is built here — four leaves and a config — because what this covers is the job
around the quantization and not the arithmetic inside it: the engine's own suite holds
`quantize_weights` against the formats. The other half is exactly what no double could
show, so it is real: the entry is written by `mlx_omnia.engine.task`, listed by `catalog.scan` and
opened by `mlx_omnia.load` under the repo id that was asked for.

Both caches move to `tmp_path` — `catalog.HUB_CACHE` and `huggingface_hub`'s own — because
the entry lands in the hub cache layout and the load by id resolves through it. Nothing
here may write into, or read out of, the machine's real cache, and no test reaches the
network: every source is on the disk the fixtures built.
"""

import gc
import json
import math
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, TypeIs

import huggingface_hub.constants
import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlx.utils import tree_flatten

import mlx_omnia
import mlx_omnia.engine.task as task
from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.checkpoint import Checkpoint, Drafter, Pending, attach_weights, save_quantized
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.engine.quant.quantization import (
    MXFP,
    NVFP,
    Affine,
    QuantizationPlan,
    infer_quantization,
    inventory,
    quantize_weights,
)
from mlx_omnia.server import catalog, jobs, quantize
from mlx_omnia.server.store import Store

REPO = "local/tiny-4bit"
SOURCE_REPO = "tiny/dense"
SOURCE_SHA = "0123456789abcdef0123456789abcdef01234567"

_DEADLINE = 10.0
"""Every wait in this module is bounded by it — a job that never arrives fails here instead
of hanging the suite."""

_IDS = mx.array([[3, 1, 4, 1, 5]])
_LEAVES = ("attn", "embed", "head", "mlp")
"""Every quantizable leaf of the model below, in the order `inventory` sorts them."""

_CALIBRATED = ("awq", "gptq", "oq", "oqe")
"""The four that read a calibration pass. They run here — over the blocked model below,
which is the one with a trunk to intercept."""

_HIDDEN = 64
_INNER = 128
_VOCAB = 64
_BLOCKS = 2
_CALIBRATION: dict[str, object] = {"sequences": 2, "sequence_length": 64}
"""Enough of the corpus for every statistic to exist and be positive, and small enough that
the pass is part of a unit suite: what is covered here is the job around the pass."""


class _Tiny(nn.Module):
    """`norm` is the one tensor no plan touches: `inventory` does not list it, the entry
    carries it unchanged, and it is the difference between what a plan costs and what the
    file it produces weighs."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 64)
        self.attn = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Linear(64, 64, bias=False)
        self.norm = nn.RMSNorm(64)
        self.head = nn.Linear(64, 32, bias=False)

    def __call__(self, ids: mx.array) -> mx.array:
        return self.head(self.norm(nn.silu(self.mlp(self.attn(self.embed(ids))))))


class _Backend:
    def __init__(self, model: _Tiny) -> None:
        self.model = model
        self.tokenizer = _Encoder()
        """A checkpoint has one, and a calibrated method over this model has to fail on what
        it really lacks — a trunk — and not on a tokenizer no double happened to carry."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def _tensors(directory: Path, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    loaded = mx.load(str(directory / "model.safetensors"))
    assert isinstance(loaded, dict)
    if dtype is None:
        return loaded
    return {name: value.astype(dtype) for name, value in loaded.items()}


def _tree(directory: Path, dtype: mx.Dtype | None) -> _Tiny:
    return attach_weights(_Tiny(), _tensors(directory, dtype))


def _model(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
    return CompositeModel(_Backend(_tree(directory, dtype)), [])


def _pending(directory: Path, dtype: mx.Dtype | None) -> Pending[LanguageModel[ModelInput]]:
    tree = _Tiny()
    return Pending(
        tree,
        lambda: _tensors(directory, dtype),
        lambda packed: CompositeModel(_Backend(attach_weights(tree, packed)), []),
    )


CHECKPOINT = Checkpoint(
    ("config.json", "model.safetensors", "tokenizer.json"),
    _tree,
    _model,
    _pending,
)


def _checkpoint(directory: Path, dtype: mx.Dtype = mx.float32) -> Path:
    mx.random.seed(0)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "tiny", "hidden_size": 64}))
    (directory / "tokenizer.json").write_text("{}")
    weights = {
        **{f"{leaf.path}.weight": mx.random.normal(leaf.shape) for leaf in inventory(_Tiny())},
        "norm.weight": mx.ones((64,)),
    }
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        {name: value.astype(dtype) for name, value in weights.items()},
    )
    return directory


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.k_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.v_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.o_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        length = x.shape[-2]
        scores = (self.q_proj(x) @ self.k_proj(x).swapaxes(-1, -2)) / math.sqrt(_HIDDEN)
        causal = mx.triu(mx.full((length, length), float("-inf")), k=1)
        return self.o_proj(mx.softmax(scores + causal, axis=-1) @ self.v_proj(x))


class _Gated(nn.Module):
    """The three names AWQ's derivation looks for: the input of `down_proj` is
    `silu(gate) ⊙ up`, so a per-channel scale on `up_proj` passes through it exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(_HIDDEN, _INNER, bias=False)
        self.up_proj = nn.Linear(_HIDDEN, _INNER, bias=False)
        self.down_proj = nn.Linear(_INNER, _HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(_HIDDEN)
        self.self_attn = _Attention()
        self.post_attention_layernorm = nn.RMSNorm(_HIDDEN)
        self.mlp = _Gated()

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class _Blocked(nn.Module):
    """A trunk, which is the whole difference from `_Tiny`: `discover_blocks` finds
    `layers`, the pass intercepts each element of it, and `embed_tokens` and `lm_head` sit
    outside — which is exactly the leaf GPTQ has no statistic for and oQ protects."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = [_Layer() for _ in range(_BLOCKS)]
        self.norm = nn.RMSNorm(_HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def __call__(self, ids: mx.array) -> mx.array:
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


class _Encoder:
    """The corpus is sampled with the model's own tokenizer, so the test model needs one.
    Bytes modulo the vocabulary: every id the pass draws indexes a row that exists."""

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([byte % _VOCAB for byte in whole.encode()])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(id % 256 for id in ids)


class _TrunkBackend:
    def __init__(self, model: _Blocked) -> None:
        self.model = model
        self.tokenizer = _Encoder()

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def _blocked_tree(directory: Path, dtype: mx.Dtype | None) -> _Blocked:
    return attach_weights(_Blocked(), _tensors(directory, dtype))


def _blocked_model(directory: Path, dtype: mx.Dtype | None) -> LanguageModel[ModelInput]:
    return CompositeModel(_TrunkBackend(_blocked_tree(directory, dtype)), [])


def _blocked_pending(directory: Path, dtype: mx.Dtype | None) -> Pending[LanguageModel[ModelInput]]:
    tree = _Blocked()
    return Pending(
        tree,
        lambda: _tensors(directory, dtype),
        lambda packed: CompositeModel(_TrunkBackend(attach_weights(tree, packed)), []),
    )


BLOCKED = Checkpoint(
    ("config.json", "model.safetensors", "tokenizer.json"),
    _blocked_tree,
    _blocked_model,
    _blocked_pending,
)

class _Draft(nn.Module):
    """A drafter's shape and the whole of what makes it one: weights, and no way in from a
    token. No embedding, no head, no tokenizer beside it on disk."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.out = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.out(nn.silu(self.fc(hidden)))


def _draft_tree(directory: Path, dtype: mx.Dtype | None) -> _Draft:
    return attach_weights(_Draft(), _tensors(directory, dtype))


def _draft_pending(directory: Path, dtype: mx.Dtype | None) -> Pending[_Draft]:
    tree = _Draft()
    return Pending(
        tree, lambda: _tensors(directory, dtype), lambda packed: attach_weights(tree, packed)
    )


DRAFTER = Drafter(("config.json", "model.safetensors"), _draft_tree, _draft_pending)

_DRAFT_LEAVES = ("fc", "out")


def _draft_checkpoint(directory: Path) -> Path:
    mx.random.seed(0)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "drafting"}))
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        {f"{leaf}.weight": mx.random.normal((_HIDDEN, _HIDDEN)) for leaf in _DRAFT_LEAVES},
    )
    return directory


@pytest.fixture
def drafter(tmp_path: Path, caches: Path) -> Path:
    return _draft_checkpoint(tmp_path / "drafter")


_BLOCK_LEAVES = tuple(
    sorted(
        f"layers.{index}.{leaf}"
        for index in range(_BLOCKS)
        for leaf in (
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        )
    )
)
"""The leaves a pass over the trunk observes — and, for GPTQ, exactly the ones that do not
fall back to RTN."""

_OUTSIDE = ("embed_tokens", "lm_head")


def _blocked_checkpoint(directory: Path) -> Path:
    mx.random.seed(0)
    model = _Blocked()
    mx.eval(model.parameters())
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "blocked", "hidden_size": _HIDDEN})
    )
    (directory / "tokenizer.json").write_text("{}")
    mx.save_safetensors(
        str(directory / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    return directory


@pytest.fixture
def blocked(tmp_path: Path, caches: Path) -> Path:
    """The source every calibrated method runs over. Float32, like the dense fixture above:
    what these tests measure is the job, and a bfloat16 pass would put the engine's own
    rounding floors inside a suite that is not about them."""
    return _blocked_checkpoint(tmp_path / "blocked")


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hub = tmp_path / "hub"
    monkeypatch.setattr(catalog, "HUB_CACHE", hub)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    # What `mlx_omnia.load` resolves a repo id through, which is the same cache the entry is
    # written into — and the reason no test of this module can touch the real one.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setitem(task._MODEL_SPECS, "tiny", CHECKPOINT)
    monkeypatch.setitem(task._MODEL_SPECS, "blocked", BLOCKED)
    monkeypatch.setitem(task._DRAFTER_SPECS, "drafting", DRAFTER)
    return hub


@pytest.fixture
def source(tmp_path: Path, caches: Path) -> Path:
    """The dense checkpoint as a directory: what the catalog reports for a quantized entry,
    and what `mlx_omnia.load` takes back."""
    return _checkpoint(tmp_path / "checkpoint")


@pytest.fixture
def bf16_source(tmp_path: Path, caches: Path) -> Path:
    """The same checkpoint in bfloat16. The lazy tree a plan resolves against is float32
    whatever the shards carry, so this is the source that tells the two apart."""
    return _checkpoint(tmp_path / "bf16", mx.bfloat16)


@pytest.fixture
def hub_source(caches: Path) -> str:
    """The same checkpoint as a downloaded repository. Resolving it by id is what gives the
    provenance a repository and a commit instead of a directory and three mtimes."""
    repository = caches / f"models--{SOURCE_REPO.replace('/', '--')}"
    _checkpoint(repository / "snapshots" / SOURCE_SHA)
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text(SOURCE_SHA)
    return SOURCE_REPO


@pytest.fixture
def client(tmp_path: Path, caches: Path) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(quantize.router)
    app.include_router(jobs.router)
    app.state.jobs = jobs.Jobs(Store(tmp_path / "server.db"))
    with TestClient(app) as running:
        yield running


@dataclass
class Packer:
    """`quantize_weights` with a gate around it: the seam every leaf is packed through, and
    therefore where a test stops a job with leaves still dense. It delegates — what lands on
    disk is what the engine wrote."""

    at: int
    """The call to stop inside, before it packs anything."""
    calls: int = 0
    reached: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=threading.Event)

    def __call__(self, weights: dict[str, mx.array], plan: QuantizationPlan) -> dict[str, mx.array]:
        self.calls += 1
        if self.calls == self.at:
            self.reached.set()
            assert self.proceed.wait(_DEADLINE), "the job was never released"
        return quantize_weights(weights, plan)


@dataclass
class Writer:
    """`save_quantized` with a gate around it: the one point inside `task.write_entry` where
    the staging directory exists under the `.tmp-` name a lookup ignores."""

    staged: list[Path] = field(default_factory=list)
    reached: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=threading.Event)

    def __call__(
        self,
        directory: Path,
        config: Mapping[str, object],
        weights: Mapping[str, mx.array],
        plan: QuantizationPlan,
    ) -> None:
        save_quantized(directory, config, weights, plan)
        self.staged.append(directory)
        self.reached.set()
        assert self.proceed.wait(_DEADLINE), "the write was never released"


def start(client: TestClient, **body: object) -> str:
    response = client.post("/admin/quantizations", json=body)
    assert response.status_code == 202, response.text
    assert response.headers["location"].endswith(response.json()["id"])
    # The jobs screen filters by kind, so the string is a contract and not a label.
    assert response.json()["kind"] == "quantize"
    job_id = response.json()["id"]
    assert isinstance(job_id, str)
    return job_id


class _PricedJson(TypedDict):
    leaves: list[dict[str, object]]
    total_bytes: int
    weights: int
    bits_per_weight: float
    entry_bytes: int


def price(client: TestClient, **body: object) -> _PricedJson:
    response = client.post("/admin/quantizations/plan", json=body)
    assert response.status_code == 200, response.text
    priced: _PricedJson = response.json()
    return priced


def view(client: TestClient, job_id: str) -> dict[str, object]:
    response = client.get(f"/admin/jobs/{job_id}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def wait_for(client: TestClient, job_id: str, state: str) -> dict[str, object]:
    deadline = time.monotonic() + _DEADLINE
    while True:
        current = view(client, job_id)
        if current["state"] == state:
            return current
        assert time.monotonic() < deadline, (
            f"job stayed in {current['state']!r} ({current['error']!r}), wanted {state!r}"
        )
        time.sleep(0.01)


def progress(current: Mapping[str, object]) -> tuple[float, float]:
    frame = current["progress"]
    assert isinstance(frame, dict)
    completed, total = frame["completed"], frame["total"]
    assert isinstance(completed, int | float) and isinstance(total, int | float)
    return completed, total


def _logits(loaded: LanguageModel[ModelInput]) -> mx.array:
    assert isinstance(loaded, CompositeModel)
    backend = loaded.model
    assert isinstance(backend, _Backend)
    return backend.model(_IDS)


def _difference(ours: mx.array, reference: mx.array) -> float:
    value = mx.abs(ours.astype(mx.float32) - reference.astype(mx.float32)).max().item()
    assert isinstance(value, float)
    return value


def test_a_drafter_is_quantized_by_the_route_that_quantizes_a_model(
    client: TestClient, drafter: Path, caches: Path
) -> None:
    """It has no task, no tokenizer and no head, and the job never asks for one: what it
    reads off a source is a lazy tree and a weight dict. The entry lands in the catalog
    under the id that was asked for, which is the id `dflash.drafter` names."""
    finished = wait_for(
        client,
        start(client, source=str(drafter), repo="local/draft-4bit", bits=4, method="rtn"),
        "ok",
    )

    assert progress(finished) == (len(_DRAFT_LEAVES), len(_DRAFT_LEAVES))
    entries = catalog.scan()
    assert [(entry.id, entry.quantization) for entry in entries] == [("local/draft-4bit", "4-bit")]

    packed = task.load_drafter(entries[0].directory)
    assert {
        path for path, module in packed.named_modules() if isinstance(module, nn.QuantizedLinear)
    } == set(_DRAFT_LEAVES)


@pytest.mark.parametrize("method", _CALIBRATED)
def test_a_drafter_takes_rtn_and_refuses_every_method_that_reads_a_corpus(
    client: TestClient, drafter: Path, method: str
) -> None:
    """Not a shape it fails at — `intercepted_collect` would find its blocks. It is that a
    drafter is not read from tokens: its input is the target's hidden states, so there is
    no corpus to run through it and nothing in its config names the target that could. The
    pricing refuses it up front, and the job refuses it before it opens the checkpoint."""
    body: dict[str, object] = {"source": str(drafter), "method": method}
    priced = client.post("/admin/quantizations/plan", json=body)

    assert priced.status_code == 409, priced.text
    assert "is a drafter" in priced.json()["detail"]

    failed = wait_for(client, start(client, **body, repo="local/draft-4bit"), "error")

    assert "is a drafter" in str(failed["error"])
    assert catalog.scan() == []


def test_a_job_writes_an_entry_the_catalog_offers_by_the_repo_id_that_was_asked_for(
    client: TestClient, source: Path, caches: Path
) -> None:
    """The whole of the first two deliveries in one run: the job reports leaf by leaf, the
    catalog offers what it wrote under the id that was asked for — not under the digest the
    load cache addresses by — and what `load` opens by that id is, tensor for tensor, the
    quantization the engine produces in memory over the same checkpoint."""
    finished = wait_for(
        client, start(client, source=str(source), repo=REPO, bits=4, method="rtn"), "ok"
    )

    assert progress(finished) == (len(_LEAVES), len(_LEAVES))
    entries = catalog.scan()
    assert [entry.id for entry in entries] == [REPO]
    assert entries[0].quantization == "4-bit"
    assert (entries[0].directory / "tokenizer.json").is_file(), "the entry cannot load alone"

    loaded = mlx_omnia.load(REPO, local_files_only=True)
    memory = mlx_omnia.load(source, quantize=Affine(group_size=64, bits=4), cache=False)

    assert _difference(_logits(loaded), _logits(memory)) == 0.0
    assert list((caches / quantize.STAGING).iterdir()) == [], "the staging outlived the job"


def test_the_job_reports_the_leaf_it_is_packing_and_how_many_are_left(
    client: TestClient, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the screen has to draw while it waits. Leaves and not bytes: the plan is what
    the job already holds, and it is the unit the packing advances by."""
    packer = Packer(at=2)
    monkeypatch.setattr(quantize, "quantize_weights", packer)
    job_id = start(client, source=str(source), repo=REPO)
    assert packer.reached.wait(_DEADLINE), "the job never reached the second leaf"

    frame = view(client, job_id)["progress"]

    assert isinstance(frame, dict)
    assert (frame["message"], frame["completed"], frame["total"]) == ("embed", 1, len(_LEAVES))

    packer.proceed.set()

    assert progress(wait_for(client, job_id, "ok")) == (len(_LEAVES), len(_LEAVES))


def test_the_entry_carries_the_width_the_plan_asked_for_leaf_by_leaf(
    client: TestClient, source: Path
) -> None:
    """Read off the tensors and not off the block the config declares: what a leaf *is* is
    its own weight next to its own scales, and a plan that reached `save_quantized` without
    reaching the packing would declare 8 bits over a 4-bit tensor. `null` is the other half
    of the selection — a leaf left dense on purpose — and the entry still has to load."""
    wait_for(
        client,
        start(
            client,
            source=str(source),
            repo=REPO,
            bits=4,
            group_size=64,
            overrides={"head": {"bits": 8, "group_size": 32}, "embed": None},
        ),
        "ok",
    )

    (entry,) = catalog.scan()
    tensors = mx.load(str(entry.directory / "model.safetensors"))
    assert isinstance(tensors, dict)

    assert {leaf: infer_quantization(tensors, leaf, input_dims=64) for leaf in _LEAVES} == {
        "attn": Affine(group_size=64, bits=4),
        "embed": None,
        "head": Affine(group_size=32, bits=8),
        "mlp": Affine(group_size=64, bits=4),
    }
    declared = json.loads((entry.directory / "config.json").read_text())["quantization"]
    assert declared["leaves"] == {
        "attn": {"group_size": 64, "bits": 4},
        "head": {"group_size": 32, "bits": 8},
        "mlp": {"group_size": 64, "bits": 4},
    }
    assert entry.quantization == "mixed"
    assert _logits(mlx_omnia.load(REPO, local_files_only=True)).shape == (1, 5, 32)


def test_a_group_size_the_plan_does_not_share_travels_with_the_leaf_that_asked_for_it(
    client: TestClient, source: Path
) -> None:
    """The priced plan answers per leaf, and a group size is half of what a leaf is: an
    override that names one has to reach the price the same way the width does, or the
    screen shows a plan that is not the one the job would write."""
    priced = price(
        client,
        source=str(source),
        bits=4,
        group_size=64,
        overrides={"head": {"bits": 8, "group_size": 32}, "embed": None},
    )

    assert {leaf["path"]: (leaf["bits"], leaf["group_size"]) for leaf in priced["leaves"]} == {
        "attn": (4, 64),
        "embed": (None, None),
        "head": (8, 32),
        "mlp": (4, 64),
    }


@pytest.mark.parametrize(
    ("mode", "format"),
    [
        ("mxfp4", MXFP(mode="mxfp4", group_size=32, bits=4)),
        ("mxfp8", MXFP(mode="mxfp8", group_size=32, bits=8)),
        ("nvfp4", NVFP(group_size=16, bits=4)),
    ],
)
def test_an_exponent_scaled_mode_packs_its_own_shape_and_the_entry_loads_by_its_id(
    client: TestClient, source: Path, mode: str, format: object
) -> None:
    """The mode is the whole selection: neither width nor group size is a control under it,
    and what comes out is read off the tensors — uint8 scales at the mode's own group. A
    leaf left dense is still the caller's, so the `null` half of the selection stays."""
    wait_for(
        client,
        start(client, source=str(source), repo=REPO, mode=mode, overrides={"embed": None}),
        "ok",
    )

    (entry,) = catalog.scan()
    tensors = mx.load(str(entry.directory / "model.safetensors"))
    assert isinstance(tensors, dict)

    assert {leaf: infer_quantization(tensors, leaf, input_dims=64) for leaf in _LEAVES} == {
        "attn": format,
        "embed": None,
        "head": format,
        "mlp": format,
    }
    assert entry.quantization == mode
    assert _logits(mlx_omnia.load(REPO, local_files_only=True)).shape == (1, 5, 32)


def test_an_exponent_scaled_mode_refuses_what_it_does_not_decide(
    client: TestClient, blocked: Path
) -> None:
    """Three requests the mode already answered: a width and a group size it fixes, a method
    that searches a scale and a bias per group it does not have, and an override naming a
    width where the width is the mode. Refused from the request alone, before a job exists —
    and `null` is still an override, because dense is not a width."""
    refusals: list[dict[str, object]] = [
        {"source": str(blocked), "repo": REPO, "mode": "nvfp4", "bits": 4},
        {"source": str(blocked), "repo": REPO, "mode": "nvfp4", "group_size": 16},
        {"source": str(blocked), "repo": REPO, "mode": "nvfp4", "method": "awq"},
        {
            "source": str(blocked),
            "repo": REPO,
            "mode": "nvfp4",
            "overrides": {"lm_head": {"bits": 8}},
        },
    ]
    for body in refusals:
        response = client.post("/admin/quantizations", json=body)
        assert response.status_code == 400, response.text

    accepted = client.post(
        "/admin/quantizations/plan",
        json={"source": str(blocked), "mode": "nvfp4", "overrides": {"lm_head": None}},
    )

    assert accepted.status_code == 200, accepted.text
    assert client.get("/admin/jobs").json() == [], "a refused request started a job"


def test_the_provenance_records_the_source_and_the_digest_the_path_stopped_carrying(
    client: TestClient, hub_source: str
) -> None:
    """Addressing by repo id takes the digest out of the path, so it goes into the config:
    what a load cache separates by digest, an entry somebody named has to be able to say.
    The rest of the block is what `load(quantize=…)` already wrote, and it is written from
    the same place — a repository resolves to its commit, which is what pins the bits
    whatever revision asked for them, and it only does so because the job resolves the
    source by its id instead of by the directory the catalog points at."""
    wait_for(client, start(client, source=hub_source, repo=REPO), "ok")

    assert [entry.id for entry in catalog.scan()] == [REPO, SOURCE_REPO]
    entry = catalog.scan()[0]
    recorded = json.loads((entry.directory / "config.json").read_text())["mlx_omnia"]

    assert recorded["source"] == {"repository": SOURCE_REPO, "commit": SOURCE_SHA}
    assert recorded["digest"] == entry.directory.name
    assert recorded["repo"] == REPO
    assert recorded["method"] == "rtn"
    assert recorded["format_version"] == task._FORMAT_VERSION


def _recorded(entry_directory: Path) -> dict[str, object]:
    config = json.loads((entry_directory / "config.json").read_text())
    block = config["mlx_omnia"]
    assert isinstance(block, dict)
    return block


def _calibration_of(entry_directory: Path) -> dict[str, object]:
    block = _recorded(entry_directory)["calibration"]
    assert isinstance(block, dict)
    return block


def _logits_of(loaded: LanguageModel[ModelInput]) -> mx.array:
    assert isinstance(loaded, CompositeModel)
    backend = loaded.model
    assert isinstance(backend, _TrunkBackend)
    return backend.model(_IDS)


def _finish(client: TestClient, **body: object) -> Path:
    """A job run to its end, answering with the entry it wrote — by its repo id, because a
    test that compares two methods has two entries in the cache."""
    wait_for(client, start(client, **body), "ok")
    (entry,) = [found for found in catalog.scan() if found.id == body["repo"]]
    return entry.directory


def test_every_calibrated_method_records_the_pass_it_read_and_the_entry_still_loads(
    client: TestClient, blocked: Path
) -> None:
    """The three that were refused by name until the pass stopped needing an architecture to
    describe its own trunk. What is common to them is here: the job ends, the entry lands
    under the id that was asked for, the provenance names the method and the calibration —
    the corpus by its digest, because the sampled ids depend on the tokenizer and the file is
    what identifies what was read — and what `load` opens by that id is a model that runs."""
    for method in _CALIBRATED:
        repo = f"local/{method}"
        directory = _finish(client, source=str(blocked), repo=repo, method=method, **_CALIBRATION)
        recorded = _recorded(directory)
        calibration = _calibration_of(directory)

        assert recorded["method"] == method
        assert calibration["corpus"] == "calibration-v1.txt"
        assert len(str(calibration["corpus_digest"])) == 64
        assert (calibration["sequences"], calibration["sequence_length"]) == (2, 64)
        assert _logits_of(mlx_omnia.load(repo, local_files_only=True)).shape == (1, 5, _VOCAB)


def test_awq_takes_the_pairs_the_tree_admits_and_writes_no_trace_of_having_run(
    client: TestClient, blocked: Path
) -> None:
    """The derivation is structural and exact: `down_proj ← up_proj` because the gate is an
    elementwise factor, `o_proj ← v_proj` because attention is linear in v's output channels.
    Both per block, and nothing else — the tensors that come out are an RTN checkpoint's,
    which is the whole point of the method being a change of variables."""
    directory = _finish(
        client, source=str(blocked), repo=REPO, method="awq", bits=4, **_CALIBRATION
    )
    pairs = _recorded(directory)["awq"]

    assert isinstance(pairs, list)
    assert [(pair["target"], pair["absorber"]) for pair in pairs] == [
        (f"layers.{index}.{parent}.{target}", f"layers.{index}.{parent}.{absorber}")
        for index in range(_BLOCKS)
        for parent, target, absorber in (
            ("mlp", "down_proj", "up_proj"),
            ("self_attn", "o_proj", "v_proj"),
        )
    ], "the pairs are not the two the structure makes exact, per block"
    assert all(pair["applied"] for pair in pairs), pairs
    assert any(pair["alpha"] > 0.0 for pair in pairs), "every pair fell back to alpha zero"
    assert all(pair["improvement"] >= 0.0 for pair in pairs)

    written = _written(directory)
    rtn = _finish(client, source=str(blocked), repo="local/rtn", method="rtn", bits=4)
    assert set(written) == set(_written(rtn))
    assert any(
        not mx.array_equal(written[name], _written(rtn)[name]).item()
        for name in written
        if name.startswith("layers.0.mlp.down_proj")
    ), "AWQ packed exactly what RTN packs: the scale never reached the weights"


def test_gptq_rounds_against_the_second_moment_and_names_what_fell_back_to_rtn(
    client: TestClient, blocked: Path
) -> None:
    """A leaf the pass never observed is packed by RTN — the embedding and the head are
    outside the trunk — and the entry says which ones, because a checkpoint whose head is not
    GPTQ's is a fact about the file. The leaves that *were* observed are packed from their own
    second moment, and the tensors prove it: identical to RTN outside the trunk, different
    inside it."""
    directory = _finish(
        client, source=str(blocked), repo=REPO, method="gptq", bits=4, **_CALIBRATION
    )
    block = _recorded(directory)["gptq"]
    assert isinstance(block, dict)

    assert block["rtn_fallback"] == list(_OUTSIDE)
    errors = block["reconstruction_error"]
    assert isinstance(errors, dict)
    assert sorted(errors) == list(_BLOCK_LEAVES)
    assert all(value >= 0.0 for value in errors.values())

    written = _written(directory)
    rtn = _written(_finish(client, source=str(blocked), repo="local/rtn", method="rtn", bits=4))
    assert set(written) == set(rtn)
    for path in _OUTSIDE:
        assert mx.array_equal(written[f"{path}.weight"], rtn[f"{path}.weight"]).item()
    assert any(
        not mx.array_equal(written[f"{path}.weight"], rtn[f"{path}.weight"]).item()
        for path in _BLOCK_LEAVES
    ), "the second moment never reached the rounding: every leaf came out RTN's"


def test_oq_allocates_by_the_sensitivity_it_measured_and_says_why_leaf_by_leaf(
    client: TestClient, blocked: Path
) -> None:
    """oQ's plan is not the selection's: the request asks for 4 bits and what lands is a
    mixture the block sensitivities ordered, inside the budget. The `oq` block sits next to
    the `quantization` block the loader reads — audit, never contract — and it carries the
    reason of every leaf, which is where the recipe's protections become visible."""
    directory = _finish(
        client,
        source=str(blocked),
        repo=REPO,
        method="oq",
        bits=4,
        group_size=64,
        **_CALIBRATION,
    )
    config = json.loads((directory / "config.json").read_text())
    provenance = config["oq"]
    declared = config["quantization"]["leaves"]

    assert provenance["recipe_identifier"] == "oQ4"
    assert provenance["calibration"]["perturbations"] == [
        "affine-g64-b4",
        "affine-g64-b5",
        "affine-g64-b6",
        "affine-g64-b8",
    ], "the allocator ordered blocks by widths the pass never measured"
    assert provenance["bits_per_weight"] <= provenance["target_bpw"]
    reasons = {path: entry["reason"] for path, entry in provenance["decisions"].items()}
    assert {reasons[path] for path in _OUTSIDE} == {"protection"}
    assert {declared[path]["bits"] for path in _OUTSIDE} == {8}
    assert "promotion" in reasons.values(), "the budget bought nothing"
    assert {declared[path]["bits"] for path in _BLOCK_LEAVES} != {4}, "the plan is uniform"


def test_oqe_keeps_oq_s_plan_and_changes_only_what_the_imatrix_rounds(
    client: TestClient, blocked: Path
) -> None:
    """The two halves are independent: the widths are the same allocator's over the same
    sensitivities — the `quantization` block of the two entries is identical leaf by leaf —
    and what oQe replaces is the grid each group is rounded against. Outside the trunk there
    is no imatrix to round against, so those leaves are RTN's in both and the entry says so.
    """
    request: dict[str, object] = {
        "source": str(blocked),
        "bits": 4,
        "group_size": 64,
        **_CALIBRATION,
    }
    directory = _finish(client, repo=REPO, method="oqe", **request)
    plain = _finish(client, repo="local/oq", method="oq", **request)

    block = _recorded(directory)["oqe"]
    assert isinstance(block, dict)
    assert block["rtn_fallback"] == list(_OUTSIDE)

    config = json.loads((directory / "config.json").read_text())
    assert (
        config["quantization"] == json.loads((plain / "config.json").read_text())["quantization"]
    ), "the imatrix moved a width: oQe's plan is oQ's"
    assert config["oq"]["recipe_identifier"] == "oQ4"

    written, reference = _written(directory), _written(plain)
    for path in _OUTSIDE:
        assert mx.array_equal(written[f"{path}.weight"], reference[f"{path}.weight"]).item()
    assert any(
        not mx.array_equal(written[f"{path}.weight"], reference[f"{path}.weight"]).item()
        for path in _BLOCK_LEAVES
    ), "the imatrix never reached the rounding: every leaf came out RTN's"


def test_a_source_without_a_trunk_fails_the_calibrated_job_and_writes_nothing(
    client: TestClient, source: Path, caches: Path
) -> None:
    """The one thing the pass still demands of an architecture: a list of blocks to find.
    `_Tiny` has four leaves and no trunk, and the failure is the engine's own message."""
    failed = wait_for(
        client,
        start(client, source=str(source), repo=REPO, method="gptq", **_CALIBRATION),
        "error",
    )

    error = failed["error"]
    assert isinstance(error, str) and "no list of blocks" in error
    assert catalog.scan() == []
    assert not (caches / quantize._slug(REPO)).exists()


def test_a_calibration_selection_is_refused_where_no_pass_reads_it(
    client: TestClient, blocked: Path
) -> None:
    """Fields that name a pass under `rtn`, and a budget under a method that allocates none,
    are a caller who believes something will run. Refused from the request alone — before the
    source is resolved and before a job exists — and so is the width GPTQ has no packing for:
    it packs its own codes, and 3, 5 and 6 have no layout verified against `mx.quantize`."""
    refusals: list[dict[str, object]] = [
        {"source": str(blocked), "repo": REPO, "sequences": 4},
        {"source": str(blocked), "repo": REPO, "method": "awq", "target_bpw": 5.0},
        {"source": str(blocked), "repo": REPO, "method": "gptq", "bits": 3},
    ]
    for body in refusals:
        response = client.post("/admin/quantizations", json=body)
        assert response.status_code == 400, response.text

    unknown = client.post(
        "/admin/quantizations", json={"source": str(blocked), "repo": REPO, "method": "awq-lite"}
    )

    assert unknown.status_code == 422, unknown.text
    assert client.get("/admin/jobs").json() == [], "a refused request started a job"


def test_awq_and_gptq_are_priced_as_rtn_and_oq_by_the_scoreless_allocator(
    client: TestClient, blocked: Path
) -> None:
    """Neither AWQ nor GPTQ moves a leaf's format — one rewrites the dense weights before the
    packing, the other replaces the rounding inside it — so the bytes are RTN's and the price
    is the same number. oQ is priced by the allocator with no scores: the sensitivity only
    orders the spending, so the reserved decisions and the budget the greedy fills to are the
    same numbers the job reaches — what moves is which free leaf gets the promotion. oQe is
    that same price: its plan is oQ's, and a searched grid weighs what a rounded one weighs.
    """
    baseline = price(client, source=str(blocked), bits=4)

    for method in ("awq", "gptq"):
        assert price(client, source=str(blocked), bits=4, method=method) == baseline

    projected = price(client, source=str(blocked), bits=4, method="oq")

    assert price(client, source=str(blocked), bits=4, method="oqe") == projected, (
        "oQe was priced as something other than the plan it shares with oQ"
    )
    assert projected["entry_bytes"] > baseline["entry_bytes"]
    assert (
        baseline["bits_per_weight"]
        < projected["bits_per_weight"]
        <= (baseline["bits_per_weight"] + 1.0)
    )
    protected = {
        str(leaf["path"]): leaf["bits"]
        for leaf in projected["leaves"]
        if leaf["path"] in ("embed_tokens", "lm_head")
    }
    assert protected == {"embed_tokens": 8, "lm_head": 8}
    assert client.get("/admin/jobs").json() == [], "pricing started a job"


def test_a_delete_while_the_entry_is_being_written_leaves_neither_the_tmp_nor_the_entry(
    client: TestClient, source: Path, caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation inside the write is the ordinary case for a checkpoint of any size.
    `write_entry` stages under a `.tmp-` so that a process killed there leaves nothing a
    lookup can take for an entry; a job that is merely cancelled has to go further and take
    the staging with it, because what is staged is a run nobody can resume — a
    half-quantized checkpoint is not bytes to pick up from."""
    writer = Writer()
    monkeypatch.setattr(task, "save_quantized", writer)
    job_id = start(client, source=str(source), repo=REPO)
    assert writer.reached.wait(_DEADLINE), "the write never began"
    (staged,) = writer.staged
    assert staged.name.startswith(".tmp-"), "the entry was written under its final name"
    assert (staged / "model.safetensors").is_file(), "the cancellation came before the write"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    writer.proceed.set()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None
    assert not staged.exists(), "the `.tmp-` outlived the job"
    assert not (caches / quantize._slug(REPO)).exists(), "the entry took its final name"
    assert catalog.scan() == []
    # What the two above cannot see: the entry the write did finish, still sitting under the
    # staging name it was renamed to.
    assert list((caches / quantize.STAGING).iterdir()) == []


def test_a_delete_between_two_leaves_stops_the_job_with_leaves_still_to_pack(
    client: TestClient, source: Path, caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`asyncio.Task.cancel()` reaches neither the thread the packing runs in nor the MLX
    call inside it. What stops it is the flag read on the way past every leaf — and the
    report before each leaf, rather than after, is what makes that flag worth reading on a
    model where one leaf is minutes."""
    packer = Packer(at=2)
    monkeypatch.setattr(quantize, "quantize_weights", packer)
    job_id = start(client, source=str(source), repo=REPO)
    assert packer.reached.wait(_DEADLINE), "the job never reached the second leaf"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    packer.proceed.set()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None
    assert packer.calls < len(_LEAVES), "the job packed every leaf and only then noticed"
    assert catalog.scan() == []
    assert not (caches / quantize._slug(REPO)).exists()


def test_a_second_job_for_the_same_repo_id_is_refused_while_the_first_runs(
    client: TestClient, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two jobs on one repo id are two jobs staging into one directory, and the first to
    finish renames it out from under the second. The refusal goes out while the first job is
    demonstrably inside the packing — a guard tested before the job starts is a guard tested
    against nothing."""
    packer = Packer(at=1)
    monkeypatch.setattr(quantize, "quantize_weights", packer)
    first = start(client, source=str(source), repo=REPO)
    assert packer.reached.wait(_DEADLINE), "the job never began packing"

    response = client.post("/admin/quantizations", json={"source": str(source), "repo": REPO})

    assert response.status_code == 409, response.text
    assert first in response.json()["detail"], "the refusal must name the job to watch"
    assert len(client.get("/admin/jobs").json()) == 1, "a second job was started anyway"

    packer.proceed.set()
    wait_for(client, first, "ok")

    assert [entry.id for entry in catalog.scan()] == [REPO]


def test_a_repo_id_already_in_the_hub_cache_is_refused_before_a_job_exists(
    client: TestClient, source: Path
) -> None:
    """The entry takes its name by a rename, so a name already taken is a failure at the very
    end — after the whole checkpoint has been read, packed and written."""
    wait_for(client, start(client, source=str(source), repo=REPO), "ok")

    response = client.post("/admin/quantizations", json={"source": str(source), "repo": REPO})

    assert response.status_code == 409, response.text
    assert REPO in response.json()["detail"]
    assert len(client.get("/admin/jobs").json()) == 1, "a job was started for a doomed write"


def test_a_width_the_format_does_not_admit_is_refused_before_a_job_exists(
    client: TestClient, source: Path
) -> None:
    """Which widths exist is the engine's table and nothing here repeats it; what this route
    owns is answering with what the table said, on the way in. A job accepted at 7 bits would
    fail after reading the checkpoint, which on a 70 GB source is minutes of disk spent on
    something the request already contained."""
    response = client.post(
        "/admin/quantizations", json={"source": str(source), "repo": REPO, "bits": 7}
    )

    assert response.status_code == 400, response.text
    assert "7" in response.json()["detail"]
    assert client.get("/admin/jobs").json() == [], "a job was started for a refused width"


def test_an_override_that_matches_no_leaf_fails_the_job_and_writes_nothing(
    client: TestClient, source: Path, caches: Path
) -> None:
    """The selection resolves against a tree the request cannot see, so a pattern that
    matches nothing is a typo only the job can catch — and it catches it before a weight is
    read, which is why the ending is an error with the engine's own message and a hub cache
    exactly as empty as it was."""
    failed = wait_for(
        client,
        start(client, source=str(source), repo=REPO, overrides={"lm_head": {"bits": 8}}),
        "error",
    )

    error = failed["error"]
    assert isinstance(error, str) and "lm_head" in error
    assert catalog.scan() == []
    assert not (caches / quantize._slug(REPO)).exists()


def _written(entry_directory: Path) -> dict[str, mx.array]:
    tensors = mx.load(str(entry_directory / "model.safetensors"))
    assert isinstance(tensors, dict)
    return tensors


def test_the_priced_plan_is_the_bytes_the_job_then_writes(client: TestClient, source: Path) -> None:
    """What the screen draws before anybody commits to minutes of packing. `total_bytes` is
    the leaves the plan touches — codes plus scales plus biases, never the four bits alone —
    and `entry_bytes` adds what no plan touches and the entry carries unchanged. Both are
    checked against the file the job actually produces, over the same selection.

    Five bits per weight and not four and a half: this source is float32, so a group of 64
    pays two float32 numbers instead of two bfloat16 ones."""
    priced = price(client, source=str(source), bits=4, group_size=64)

    wait_for(client, start(client, source=str(source), repo=REPO, bits=4, group_size=64), "ok")

    (entry,) = catalog.scan()
    written = _written(entry.directory)
    assert priced["entry_bytes"] == sum(tensor.nbytes for tensor in written.values())
    assert priced["total_bytes"] == sum(
        tensor.nbytes for name, tensor in written.items() if not name.startswith("norm.")
    )
    assert priced["bits_per_weight"] == 5.0
    assert [leaf["path"] for leaf in priced["leaves"]] == list(_LEAVES)


def test_a_bfloat16_source_is_priced_in_bfloat16_and_not_in_the_lazy_tree_s_float32(
    client: TestClient, bf16_source: Path
) -> None:
    """The tree the plan resolves against is built before `nn.quantize`, so every leaf on it
    carries mlx's default float32 however the shards are stored. Priced off that, a bfloat16
    checkpoint comes back with float32 scales and — for whatever the selection leaves dense —
    twice the leaf itself. `embed` is left dense here so both halves are in the number."""
    selection: dict[str, object] = {"bits": 4, "overrides": {"embed": None}}
    priced = price(client, source=str(bf16_source), **selection)

    wait_for(client, start(client, source=str(bf16_source), repo=REPO, **selection), "ok")

    (entry,) = catalog.scan()
    written = _written(entry.directory)
    assert priced["entry_bytes"] == sum(tensor.nbytes for tensor in written.values())
    assert {leaf["path"]: leaf["bits"] for leaf in priced["leaves"]} == {
        "attn": 4,
        "embed": None,
        "head": 4,
        "mlp": 4,
    }


def test_pricing_refuses_what_the_job_refuses_and_starts_nothing(
    client: TestClient, source: Path
) -> None:
    """The route resolves the source and the selection exactly where the job does, so the
    three ways a request can be wrong answer here instead of minutes later inside a worker —
    and none of them leaves a job behind."""
    refusals: dict[int, list[dict[str, object]]] = {
        400: [
            {"source": str(source), "bits": 7},
            {"source": str(source), "overrides": {"x": {"bits": 8}}},
        ],
        404: [{"source": "nobody/nothing"}],
    }
    for status, bodies in refusals.items():
        for body in bodies:
            response = client.post("/admin/quantizations/plan", json=body)
            assert response.status_code == status, response.text

    assert client.get("/admin/jobs").json() == [], "pricing a plan started a job"


def test_a_checkpoint_quantized_natively_is_refused_with_its_codes_named(
    client: TestClient, tmp_path: Path, caches: Path
) -> None:
    """DeepSeek ships V4 as I8 codes beside a sliver of bfloat16 norms, and nothing in its
    config says `quantization` — so it passes the already-quantized gate, `weights_dtype`
    truthfully answers BF16 for the sliver, and a dense baseline priced at bfloat16
    outweighs the file itself: `entry_bytes` lands below zero. The majority of the bytes is
    what tells this checkpoint apart, and it refuses the plan and the job alike."""
    directory = tmp_path / "native"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "tiny", "hidden_size": 64}))
    (directory / "tokenizer.json").write_text("{}")
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        {
            **{
                f"{leaf.path}.weight": mx.zeros(leaf.shape, dtype=mx.int8)
                for leaf in inventory(_Tiny())
            },
            "norm.weight": mx.ones((64,), dtype=mx.bfloat16),
        },
    )

    priced = client.post("/admin/quantizations/plan", json={"source": str(directory)})
    assert priced.status_code == 409, priced.text
    assert "I8" in priced.json()["detail"]

    ended = wait_for(client, start(client, source=str(directory), repo=REPO), "error")
    assert isinstance(ended["error"], str) and "I8" in ended["error"]
    assert not (catalog.HUB_CACHE / "models--local--tiny-4bit").exists()


def test_an_entry_that_is_already_quantized_is_not_priced_again(
    client: TestClient, source: Path
) -> None:
    """The lazy tree is dense whatever the shards hold, so pricing a quantized checkpoint
    would answer with what quantizing its dense original would have cost — a number for a job
    that cannot run at all."""
    wait_for(client, start(client, source=str(source), repo=REPO), "ok")

    response = client.post("/admin/quantizations/plan", json={"source": REPO})

    assert response.status_code == 409, response.text
    assert REPO in response.json()["detail"]


def test_a_finished_job_gives_the_buffers_back_to_the_system(
    client: TestClient, blocked: Path
) -> None:
    """What a quantization allocates is a whole checkpoint twice over — the dense model the
    pass runs and the packed dict it writes — and dropping the last reference to them only
    returns them to MLX's buffer pool, which is inside the process footprint the memory rail
    reads and admission decides against. The reading is the cache and not the active memory:
    the references die either way, and what this covers is the pool they die into.

    `ok` is enough of a barrier: the terminal frame is published after the job body returns,
    and the body is where the cache is emptied.
    """
    gc.collect()
    mx.clear_cache()

    wait_for(
        client,
        start(client, source=str(blocked), repo=REPO, method="oqe", **_CALIBRATION),
        "ok",
    )

    assert mx.get_cache_memory() == 0, "the job left its checkpoint in MLX's buffer pool"

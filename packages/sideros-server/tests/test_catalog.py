"""The disk catalog: what the scan accepts as a model, and how the routes name it.

The fake caches are built file by file rather than downloaded: the two cases that matter
— a snapshot missing a shard of its `weight_map`, and an id carrying a `/` — are exactly
the ones a real download never produces on demand. The shards are real safetensors all the
same, because `bytes_per_token` is read out of their headers.

That number is checked where it can be: against `footprint.active_bytes_per_token` over a
tree built and loaded here, which is the same walk the engine does once a model is
resident, and against the 1.711 GB the house measured for the 30B MoE when that checkpoint
happens to be on the machine.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlx.utils import tree_flatten

from sideros.footprint import active_bytes_per_token, ceiling
from sideros.models.glm4_moe import CHECKPOINT as GLM4_MOE
from sideros.models.qwen3.moe import CHECKPOINT as QWEN3_MOE
from sideros.models.qwen3.moe import Qwen3MoE, Qwen3MoEConfig
from sideros.task import source
from sideros_server import catalog

QUANTIZED = "mlx-community/Qwen3-0.6B-4bit"
DENSE = "Qwen/Qwen3-0.6B"
MOE = "mlx-community/Qwen3-30B-A3B-4bit"
MOE_SHARED = "mlx-community/Qwen3.6-35B-A3B-4bit"
MOE_TINY = "local/tiny-moe"

TINY = Qwen3MoEConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=96,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    moe_intermediate_size=32,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    eos_token_id=(0,),
)
"""Small enough to build, quantize and load inside a unit test, and shaped so that no
projection's output width collides with the vocabulary — the only 2-D leaves of that height
are the embedding table and the head, which is what tells them apart from each other."""

GLM_LAYERS = 2
GLM_CONFIG: dict[str, object] = {
    "model_type": "glm4_moe",
    "hidden_size": 64,
    "num_hidden_layers": GLM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 96,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "intermediate_size": 48,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": True,
    "first_k_dense_replace": 0,
    "n_shared_experts": 1,
    "eos_token_id": 0,
    "max_position_embeddings": 128,
}
"""A MoE that names its expert count `n_routed_experts` — no key the scan reads — and that
ships an MTP block one layer past the trunk, which the loader drops. The two things a
config alone cannot price."""

CONFIG: dict[str, object] = {
    "model_type": "qwen3",
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
    "quantization": {"group_size": 64, "bits": 4},
}
DENSE_CONFIG: dict[str, object] = {
    "model_type": "qwen3",
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    hub = tmp_path / "hub"
    quantized = tmp_path / "quantized"
    monkeypatch.setattr(catalog, "HUB_CACHE", hub)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", quantized)
    return hub, quantized


def _repository(hub: Path, model_id: str) -> Path:
    return hub / f"models--{model_id.replace('/', '--')}"


def _main(hub: Path, model_id: str, sha: str) -> None:
    reference = _repository(hub, model_id) / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(sha)


SHARD_BYTES = 2048
"""The payload of every fake shard below, and therefore what the scan prices each of them
at: one bfloat16 vector that no rule excludes."""


type Tensor = tuple[str, list[int], int]
"""A tensor as a fake shard declares it: name, shape and the bytes it occupies."""


def _shard(path: Path, extra: Sequence[Tensor] = ()) -> None:
    """safetensors as the scan reads it: a little-endian u64 with the header's length, the
    JSON header, then the payload the offsets describe. `extra` is written after the vector
    every shard carries."""
    tensors: dict[str, object] = {
        "model.norm.weight": {"dtype": "BF16", "shape": [1024], "data_offsets": [0, SHARD_BYTES]}
    }
    offset = SHARD_BYTES
    for name, shape, size in extra:
        tensors[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header = json.dumps(tensors).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * offset)


def _checkpoint(
    directory: Path,
    config: Mapping[str, object],
    shards: Sequence[str] = ("model.safetensors",),
    missing: Sequence[str] = (),
    extra: Sequence[Tensor] = (),
) -> Path:
    """The three things the scan reads: the config, the index naming every shard, and the
    shards that were actually written — `missing` names the ones the download never got."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    weight_map = {f"weight.{i}": name for i, name in enumerate([*shards, *missing])}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in shards:
        _shard(directory / name, extra)
    return directory


def _moe_checkpoint(directory: Path, *, bits: int | None) -> Path:
    """The tiny MoE on disk, written under the tree's own parameter names. Both of this
    architecture's load-time fusions are no-ops when the pre-fusion tensors are absent, so
    what is flattened out of the tree here is exactly what `load_weights(strict=True)` reads
    back — no second name table, and the loaded tree is the one that was saved."""
    mx.random.seed(0)
    model = Qwen3MoE(TINY)
    if bits is not None:
        nn.quantize(model, group_size=32, bits=bits)
    mx.eval(model.parameters())
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({**asdict(TINY), "model_type": "qwen3_moe", "max_position_embeddings": 128})
    )
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return directory


def _glm_checkpoint(directory: Path) -> int:
    """The tiny glm4_moe on disk, plus the MTP block GLM ships one layer past the trunk and
    the loader drops. Answers with the bytes of that block, which is what the headers price
    and the tree does not.

    The tree it saves is the one the config builds — `source` stops the load exactly there —
    so the names on disk are the ones `load_weights(strict=True)` asks for; this
    architecture's fusions are all no-ops when the pre-fusion tensors are absent."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(GLM_CONFIG))
    mx.random.seed(0)
    model = source(directory, local_files_only=True).pending.model
    mx.eval(model.parameters())
    extra = {
        f"model.layers.{GLM_LAYERS}.mlp.gate_proj.weight": mx.zeros((32, 64)),
        f"model.layers.{GLM_LAYERS}.input_layernorm.weight": mx.zeros((64,)),
    }
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(directory / "model.safetensors"), {**weights, **extra})
    return sum(value.nbytes for value in extra.values())


def _installed(hub: Path, model_id: str, config: Mapping[str, object] = CONFIG) -> Path:
    """A repository at a single revision, which is what `refs/main` points at."""
    snapshot = _checkpoint(_repository(hub, model_id) / "snapshots" / "head", config)
    _main(hub, model_id, "head")
    return snapshot


def _client(*resident: str, active: int | None = None) -> TestClient:
    """`active` is what the engine's own walk over the loaded tree says each of them reads —
    `None` is a resident model with no tree under it, which is what a test double is."""
    app = FastAPI()
    app.include_router(catalog.router)
    app.dependency_overrides[catalog.resident_models] = lambda: dict.fromkeys(resident, active)
    return TestClient(app)


def test_a_repository_is_one_entry_at_the_revision_refs_main_names(
    caches: tuple[Path, Path],
) -> None:
    hub, _ = caches
    _checkpoint(
        _repository(hub, QUANTIZED) / "snapshots" / "stale",
        {**CONFIG, "max_position_embeddings": 1},
    )
    head = _checkpoint(_repository(hub, QUANTIZED) / "snapshots" / "head", CONFIG)
    _main(hub, QUANTIZED, "head")

    entries = catalog.scan()
    assert [entry.id for entry in entries] == [QUANTIZED]
    entry = entries[0]
    assert entry.directory == head
    assert entry.store == _repository(hub, QUANTIZED)
    assert entry.architecture == "qwen3"
    assert entry.quantization == "4-bit"
    assert entry.dtype == "bfloat16"
    assert entry.context == 40960
    assert entry.bytes_on_disk == sum(path.stat().st_size for path in head.iterdir())
    assert entry.bytes_per_token == SHARD_BYTES


def test_a_snapshot_missing_a_shard_of_the_weight_map_is_not_an_entry(
    caches: tuple[Path, Path],
) -> None:
    hub, _ = caches
    snapshot = _checkpoint(
        _repository(hub, QUANTIZED) / "snapshots" / "head",
        CONFIG,
        shards=("model-00001-of-00002.safetensors",),
        missing=("model-00002-of-00002.safetensors",),
    )
    _main(hub, QUANTIZED, "head")
    assert catalog.scan() == []

    # And it is the shard that decides, not something else about this snapshot.
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 2048)
    assert [entry.id for entry in catalog.scan()] == [QUANTIZED]


def test_a_shard_whose_blob_never_landed_is_not_an_entry(caches: tuple[Path, Path]) -> None:
    """The hub stores a snapshot as symlinks into `blobs/`, so an incomplete download shows
    up as a link with nothing behind it rather than as a missing name."""
    hub, _ = caches
    snapshot = _installed(hub, QUANTIZED)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(_repository(hub, QUANTIZED) / "blobs" / "deadbeef")
    assert catalog.scan() == []


def test_a_quantized_entry_is_named_by_its_directory_and_staging_is_not_listed(
    caches: tuple[Path, Path],
) -> None:
    _, quantized = caches
    source = quantized / "mlx-community--Qwen3-0.6B"
    entry_directory = _checkpoint(source / "0123456789abcdef", CONFIG)
    _checkpoint(source / ".tmp-halfway", CONFIG)

    entries = catalog.scan()
    assert [entry.id for entry in entries] == [str(entry_directory)]
    assert entries[0].store == entry_directory


@pytest.mark.parametrize(
    ("leaves", "expected"),
    [
        ({"a": {"group_size": 64, "bits": 4}, "b": {"group_size": 64, "bits": 4}}, "4-bit"),
        ({"a": {"group_size": 64, "bits": 4}, "b": {"group_size": 64, "bits": 6}}, "mixed"),
        ({"a": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}, "mxfp4"),
    ],
)
def test_a_per_leaf_plan_reports_the_width_its_leaves_agree_on(
    caches: tuple[Path, Path], leaves: dict[str, object], expected: str
) -> None:
    hub, _ = caches
    _installed(hub, QUANTIZED, {**CONFIG, "quantization": {"leaves": leaves}})
    assert catalog.scan()[0].quantization == expected


def test_a_checkpoint_that_declares_no_quantization_is_dense(caches: tuple[Path, Path]) -> None:
    hub, _ = caches
    _installed(hub, DENSE, DENSE_CONFIG)
    entry = catalog.scan()[0]
    assert entry.quantization is None
    assert entry.dtype == "bfloat16"


def test_an_id_with_a_slash_is_routed_raw_and_percent_encoded(
    caches: tuple[Path, Path],
) -> None:
    hub, _ = caches
    _installed(hub, QUANTIZED)
    client = _client()

    raw = client.get(f"/admin/models/{QUANTIZED}")
    encoded = client.get(f"/admin/models/{quote(QUANTIZED, safe='')}")
    assert raw.status_code == 200, raw.text
    assert raw.json()["id"] == QUANTIZED
    assert encoded.status_code == 200, encoded.text
    assert encoded.json() == raw.json()


def test_the_catalog_says_which_entry_is_resident_and_can_list_only_those(
    caches: tuple[Path, Path],
) -> None:
    hub, _ = caches
    _installed(hub, QUANTIZED)
    _installed(hub, DENSE, DENSE_CONFIG)
    client = _client(QUANTIZED)

    everything = client.get("/admin/models").json()
    assert {entry["id"]: entry["resident"] for entry in everything} == {
        QUANTIZED: True,
        DENSE: False,
    }
    only = client.get("/admin/models", params={"resident": "true"}).json()
    assert [entry["id"] for entry in only] == [QUANTIZED]


def test_deleting_a_resident_model_is_refused_and_says_why(caches: tuple[Path, Path]) -> None:
    hub, _ = caches
    _installed(hub, QUANTIZED)
    response = _client(QUANTIZED).delete(f"/admin/models/{QUANTIZED}")
    assert response.status_code == 409, response.text
    assert "resident" in response.json()["detail"]
    assert _repository(hub, QUANTIZED).is_dir()


def test_deleting_a_model_that_is_not_resident_takes_its_blobs_with_it(
    caches: tuple[Path, Path],
) -> None:
    """Removing the snapshot alone frees nothing: it is symlinks, and the bytes are in
    `blobs/`."""
    hub, _ = caches
    _installed(hub, QUANTIZED)
    blob = _repository(hub, QUANTIZED) / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\0" * 2048)

    response = _client().delete(f"/admin/models/{QUANTIZED}")
    assert response.status_code == 204, response.text
    assert not _repository(hub, QUANTIZED).exists()
    assert catalog.scan() == []


def test_a_model_that_is_not_on_disk_is_a_404(caches: tuple[Path, Path]) -> None:
    client = _client()
    assert client.get("/admin/models/nope").status_code == 404
    assert client.delete("/admin/models/nope").status_code == 404


def test_the_card_is_the_readme_raw_and_absent_is_a_404(caches: tuple[Path, Path]) -> None:
    hub, _ = caches
    snapshot = _installed(hub, QUANTIZED)
    _installed(hub, DENSE)
    (snapshot / "README.md").write_text("---\nlicense: mit\n---\n# hello\n")
    client = _client()
    answer = client.get(f"/admin/models/{quote(QUANTIZED, safe='')}/card")
    assert answer.status_code == 200
    assert answer.text == "---\nlicense: mit\n---\n# hello\n"
    assert client.get(f"/admin/models/{quote(DENSE, safe='')}/card").status_code == 404


def test_the_files_listing_prices_each_name_through_the_symlink(
    caches: tuple[Path, Path],
) -> None:
    hub, _ = caches
    snapshot = _installed(hub, QUANTIZED)
    blob = snapshot.parent.parent / "blobs" / "cafe"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x89PNG" + b"\0" * 96)
    (snapshot / "ladder.png").symlink_to(blob)
    client = _client()
    answer = client.get(f"/admin/models/{quote(QUANTIZED, safe='')}/files")
    assert answer.status_code == 200
    listed = {entry["name"]: entry["size"] for entry in answer.json()}
    assert listed["ladder.png"] == 100
    assert set(listed) == {
        "config.json",
        "model.safetensors.index.json",
        "model.safetensors",
        "ladder.png",
    }
    assert [entry["name"] for entry in answer.json()] == sorted(listed)


def test_an_asset_is_served_from_the_checkpoint_and_nowhere_else(
    caches: tuple[Path, Path], tmp_path: Path
) -> None:
    hub, _ = caches
    snapshot = _installed(hub, QUANTIZED)
    (snapshot / "ladder.png").write_bytes(b"\x89PNG-bytes")
    outside = tmp_path / "secret.txt"
    outside.write_text("not yours")
    client = _client()
    base = f"/admin/models/{quote(QUANTIZED, safe='')}/assets"
    assert client.get(f"{base}/ladder.png").content == b"\x89PNG-bytes"
    assert client.get(f"{base}/missing.png").status_code == 404
    assert client.get(f"{base}/{quote('../secret.txt', safe='')}").status_code == 404
    assert client.get(f"{base}/{quote(str(outside), safe='')}").status_code == 404


def test_a_shard_that_is_not_safetensors_leaves_the_entry_listed_without_a_price(
    caches: tuple[Path, Path],
) -> None:
    """A directory the loader will fail on is still a directory the catalog has to show —
    its id, its architecture and its bytes on disk are all true. What it loses is the one
    number that comes out of the shard itself."""
    hub, _ = caches
    snapshot = _installed(hub, QUANTIZED)
    (snapshot / "model.safetensors").write_bytes(b"not safetensors at all")

    (entry,) = catalog.scan()

    assert entry.id == QUANTIZED
    assert entry.architecture == "qwen3"
    assert entry.bytes_on_disk > 0
    assert entry.bytes_per_token is None


@pytest.mark.parametrize("bits", [None, 4])
def test_the_scan_prices_a_step_at_what_the_loaded_tree_reads(
    caches: tuple[Path, Path], bits: int | None
) -> None:
    """The number reported for a checkpoint nobody loaded is the number the engine computes
    once it is: the same three rules, taken off the headers instead of off the tree. Both
    widths, because a quantized step reads packed codes plus scales plus biases and the
    config's `bits` alone would not add up to that.

    The routed part is most of a MoE, so the price also has to be a fraction of the
    checkpoint — 2 of 8 experts here, and reporting the whole stack would put it above.
    """
    hub, _ = caches
    directory = _moe_checkpoint(_repository(hub, MOE_TINY) / "snapshots" / "head", bits=bits)
    _main(hub, MOE_TINY, "head")

    (entry,) = catalog.scan()

    assert entry.bytes_per_token == active_bytes_per_token(QWEN3_MOE.load(directory, None))
    assert entry.bytes_per_token is not None
    assert entry.bytes_per_token * 2 < (directory / "model.safetensors").stat().st_size


def test_the_rows_a_step_reads_come_off_the_tree_and_not_off_a_config_key(
    caches: tuple[Path, Path],
) -> None:
    """The regression this pricing exists for. This checkpoint's expert count is under
    `n_routed_experts`, a name the scan does not read and never will: the count comes from
    the architecture's own tree, so the family that spells it differently is priced like
    every other one. Reading a key and falling back to zero when it is absent is what
    charged all 256 experts of a 2.4-bit DeepSeek-V4 on every token — 92 GB against 8, and
    a chat that reported 372% of its own ceiling.

    What the headers cannot know is on the other side of the equality: the MTP block ships
    in the shard and the loader drops it, so the estimate is over the tree's number by
    exactly that block and by nothing else.
    """
    hub, _ = caches
    directory = _repository(hub, MOE_TINY) / "snapshots" / "head"
    dropped = _glm_checkpoint(directory)
    _main(hub, MOE_TINY, "head")

    (entry,) = catalog.scan()

    assert entry.bytes_per_token == active_bytes_per_token(GLM4_MOE.load(directory, None)) + dropped
    assert entry.bytes_per_token * 2 < (directory / "model.safetensors").stat().st_size


def test_a_resident_entry_is_priced_by_the_tree_that_is_answering_the_requests(
    caches: tuple[Path, Path],
) -> None:
    """The estimate off the headers is what an entry carries until it is loaded; from then
    on the number is the engine's walk over the real tree, which is the one that knows what
    the loader dropped. A resident model with no tree under it keeps the estimate."""
    hub, _ = caches
    directory = _repository(hub, MOE_TINY) / "snapshots" / "head"
    dropped = _glm_checkpoint(directory)
    _main(hub, MOE_TINY, "head")
    estimate = active_bytes_per_token(GLM4_MOE.load(directory, None)) + dropped

    listed = _client(MOE_TINY, active=estimate - dropped).get("/admin/models").json()
    assert [entry["bytes_per_token"] for entry in listed] == [estimate - dropped]

    untreed = _client(MOE_TINY).get(f"/admin/models/{quote(MOE_TINY, safe='')}").json()
    assert untreed["bytes_per_token"] == estimate
    assert catalog.scan()[0].bytes_per_token == estimate


def test_an_architecture_with_no_tree_leaves_a_stacked_checkpoint_unpriced(
    caches: tuple[Path, Path],
) -> None:
    """A checkpoint the engine cannot load can still be listed, and its stacks are still
    stacks — but how many of their rows a step reads is the tree's to say, and there is no
    tree. The entry reports no number rather than an invented one."""
    hub, _ = caches
    directory = _repository(hub, MOE_TINY) / "snapshots" / "head"
    _glm_checkpoint(directory)
    (directory / "config.json").write_text(json.dumps({**GLM_CONFIG, "model_type": "glm9_moe"}))
    _main(hub, MOE_TINY, "head")

    (entry,) = catalog.scan()

    assert entry.architecture == "glm9_moe"
    assert entry.bytes_on_disk > 0
    assert entry.bytes_per_token is None


def test_the_thirty_billion_moe_prices_at_the_gigabyte_the_house_measured() -> None:
    """1.711 GB/token and a 286 tok/s ceiling — 8 of 128 experts at 4 bits plus the whole
    untied lm_head, with the 0.175 GB embedding table out because a step gathers one row of
    it. Read off the headers of a checkpoint this suite never opens, let alone loads;
    skipped rather than downloading 17 GB."""
    if not _repository(catalog.HUB_CACHE, MOE).is_dir():
        pytest.skip(f"{MOE} is not in the hub cache")

    entry = {found.id: found for found in catalog.scan()}[MOE]

    assert entry.bytes_per_token is not None
    assert round(entry.bytes_per_token / 1e9, 3) == 1.711
    assert round(ceiling(entry.bytes_per_token)) == 286


@pytest.mark.parametrize(
    ("name", "priced"),
    [("vision_tower.pos_embed.weight", False), ("vision_tower.merger.norm.weight", True)],
)
def test_the_vision_towers_position_table_is_not_priced_into_a_text_step(
    caches: tuple[Path, Path], name: str, priced: bool
) -> None:
    """The tower gathers rows of `pos_embed` and a text step never reaches it at all, so the
    tree — which holds it as the `nn.Embedding` it is — leaves it out. Off the headers it is
    a `[2304, 1152]` matrix like any other and its height is nobody's vocabulary, so the name
    is what tells it apart; the tensor beside it, from the same tower, is still priced."""
    hub, _ = caches
    table = 2304 * 1152 * 2
    _checkpoint(
        _repository(hub, QUANTIZED) / "snapshots" / "head",
        CONFIG,
        extra=((name, [2304, 1152], table),),
    )
    _main(hub, QUANTIZED, "head")

    (entry,) = catalog.scan()

    assert entry.bytes_per_token == SHARD_BYTES + (table if priced else 0)


def test_the_thirty_five_billion_moe_prices_the_shared_expert_whole() -> None:
    """8 of 256 experts at 4 bits, the shared expert — which the checkpoint ships as plain
    2-D tensors and the tree stacks as slot 256 of the routed pile — read whole, and the
    tower's position table out: 2,557,394,656 bytes, the same integer `test_footprint`
    asserts over this checkpoint once loaded. That equality is what delta 0 means for the
    sixth family, and it is what this test is for; the number itself still carries the
    vision tower and still misses the recurrent state, which `test_footprint` spells out."""
    if not _repository(catalog.HUB_CACHE, MOE_SHARED).is_dir():
        pytest.skip(f"{MOE_SHARED} is not in the hub cache")

    entry = {found.id: found for found in catalog.scan()}[MOE_SHARED]

    assert entry.bytes_per_token == 2_557_394_656


def test_a_real_repository_from_the_hub_cache_reads_its_own_numbers() -> None:
    """The scan against the machine's cache, over the checkpoint the API gate already
    uses. Skipped rather than downloading: this suite is not what pulls 350 MB."""
    if not _repository(catalog.HUB_CACHE, QUANTIZED).is_dir():
        pytest.skip(f"{QUANTIZED} is not in the hub cache")

    entries = {entry.id: entry for entry in catalog.scan()}
    assert QUANTIZED in entries
    entry = entries[QUANTIZED]
    assert entry.architecture == "qwen3"
    assert entry.quantization == "4-bit"
    assert entry.context == 40960
    assert (entry.directory / "model.safetensors").is_file()
    assert 0.30 < entry.bytes_on_disk / 1e9 < 0.45

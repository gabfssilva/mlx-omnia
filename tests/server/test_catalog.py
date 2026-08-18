"""The disk catalog: what the scan accepts as a model, and what it prices a step at.

`bytes_per_token` is checked where it can be: against `footprint.active_bytes_per_token` over
a tree built and loaded here, which is the same walk the engine does once a model is resident,
and against the 1.711 GB the house measured for the 30B MoE when that checkpoint happens to be
on the machine.

The routes over these entries are `test_catalog_routes.py`; the cache arithmetic per family
shape is `test_catalog_kv.py`.
"""

import json
from pathlib import Path

import pytest

from mlx_omnia.engine.footprint import active_bytes_per_token, ceiling
from mlx_omnia.engine.models.glm4_moe import CHECKPOINT as GLM4_MOE
from mlx_omnia.engine.models.qwen3.moe import CHECKPOINT as QWEN3_MOE
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.catalog.config import TensorJson

from .catalog_stand import (
    CONFIG,
    DENSE,
    DENSE_CONFIG,
    GLM_CONFIG,
    HEADERS,
    MOE,
    MOE_SHARED,
    MOE_TINY,
    QUANTIZED,
    SHARD_BYTES,
    checkpoint,
    glm_checkpoint,
    installed,
    main,
    moe_checkpoint,
    repository,
    shard,
    use_caches,
)


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    return use_caches(tmp_path, monkeypatch)


def test_a_repository_is_one_entry_at_the_revision_refs_main_names(
    caches: tuple[Path, Path],
) -> None:
    hub, _ = caches
    checkpoint(
        repository(hub, QUANTIZED) / "snapshots" / "stale",
        {**CONFIG, "max_position_embeddings": 1},
    )
    head = checkpoint(repository(hub, QUANTIZED) / "snapshots" / "head", CONFIG)
    main(hub, QUANTIZED, "head")

    entries = catalog.scan()
    assert [entry.id for entry in entries] == [QUANTIZED]
    entry = entries[0]
    assert entry.directory == head
    assert entry.store == repository(hub, QUANTIZED)
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
    snapshot = checkpoint(
        repository(hub, QUANTIZED) / "snapshots" / "head",
        CONFIG,
        shards=("model-00001-of-00002.safetensors",),
        missing=("model-00002-of-00002.safetensors",),
    )
    main(hub, QUANTIZED, "head")
    assert catalog.scan() == []

    # And it is the shard that decides, not something else about this snapshot.
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 2048)
    assert [entry.id for entry in catalog.scan()] == [QUANTIZED]


def test_a_shard_whose_blob_never_landed_is_not_an_entry(caches: tuple[Path, Path]) -> None:
    """The hub stores a snapshot as symlinks into `blobs/`, so an incomplete download shows
    up as a link with nothing behind it rather than as a missing name."""
    hub, _ = caches
    snapshot = installed(hub, QUANTIZED)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(repository(hub, QUANTIZED) / "blobs" / "deadbeef")
    assert catalog.scan() == []


def test_a_quantized_entry_is_named_by_its_directory_and_staging_is_not_listed(
    caches: tuple[Path, Path],
) -> None:
    _, quantized = caches
    source = quantized / "mlx-community--Qwen3-0.6B"
    entry_directory = checkpoint(source / "0123456789abcdef", CONFIG)
    checkpoint(source / ".tmp-halfway", CONFIG)

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
    installed(hub, QUANTIZED, {**CONFIG, "quantization": {"leaves": leaves}})
    assert catalog.scan()[0].quantization == expected


def test_a_checkpoint_that_declares_no_quantization_is_dense(caches: tuple[Path, Path]) -> None:
    hub, _ = caches
    installed(hub, DENSE, DENSE_CONFIG)
    entry = catalog.scan()[0]
    assert entry.quantization is None
    assert entry.dtype == "bfloat16"


def test_a_shard_that_is_not_safetensors_leaves_the_entry_listed_without_a_price(
    caches: tuple[Path, Path],
) -> None:
    """A directory the loader will fail on is still a directory the catalog has to show —
    its id, its architecture and its bytes on disk are all true. What it loses is the one
    number that comes out of the shard itself."""
    hub, _ = caches
    snapshot = installed(hub, QUANTIZED)
    (snapshot / "model.safetensors").write_bytes(b"not safetensors at all")

    (entry,) = catalog.scan()

    assert entry.id == QUANTIZED
    assert entry.architecture == "qwen3"
    assert entry.bytes_on_disk > 0
    assert entry.bytes_per_token is None


def test_a_second_scan_reads_no_header_and_a_shard_that_moves_is_repriced(
    caches: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scan is on the app's poll and on every event the watcher raises, so the headers
    are read once per state of the disk and not once per caller. What decides that state is
    the stamp: rewriting a shard reprices the entry even though its name never changed."""
    hub, _ = caches
    snapshot = installed(hub, DENSE)
    reads: list[Path] = []
    inner = catalog._tensors

    def counted(path: Path) -> list[tuple[str, TensorJson, int]] | None:
        reads.append(path)
        return inner(path)

    monkeypatch.setattr(HEADERS, "tensors_of", counted)

    first = catalog.scan()
    assert reads
    assert set(reads) == {snapshot}
    once = list(reads)

    assert catalog.scan() == first
    assert reads == once

    shard(snapshot / "model.safetensors", extra=[("model.extra.weight", [512], SHARD_BYTES)])
    (second,) = catalog.scan()

    assert reads == once + once
    assert second.bytes_per_token == 2 * SHARD_BYTES

    # The same bytes at the same length, written again: only the clock moved, and a stamp
    # that cannot see it would answer a checkpoint that was rebuilt in place with the
    # numbers of the one it replaced.
    shard(snapshot / "model.safetensors", extra=[("model.extra.weight", [512], SHARD_BYTES)])

    assert catalog.scan() == [second]
    assert reads == once + once + once


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
    directory = moe_checkpoint(repository(hub, MOE_TINY) / "snapshots" / "head", bits=bits)
    main(hub, MOE_TINY, "head")

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
    directory = repository(hub, MOE_TINY) / "snapshots" / "head"
    dropped = glm_checkpoint(directory)
    main(hub, MOE_TINY, "head")

    (entry,) = catalog.scan()

    assert entry.bytes_per_token == active_bytes_per_token(GLM4_MOE.load(directory, None)) + dropped
    assert entry.bytes_per_token is not None
    assert entry.bytes_per_token * 2 < (directory / "model.safetensors").stat().st_size


def test_an_architecture_with_no_tree_leaves_a_stacked_checkpoint_unpriced(
    caches: tuple[Path, Path],
) -> None:
    """A checkpoint the engine cannot load can still be listed, and its stacks are still
    stacks — but how many of their rows a step reads is the tree's to say, and there is no
    tree. The entry reports no number rather than an invented one."""
    hub, _ = caches
    directory = repository(hub, MOE_TINY) / "snapshots" / "head"
    glm_checkpoint(directory)
    (directory / "config.json").write_text(json.dumps({**GLM_CONFIG, "model_type": "glm9_moe"}))
    main(hub, MOE_TINY, "head")

    (entry,) = catalog.scan()

    assert entry.architecture == "glm9_moe"
    assert entry.bytes_on_disk > 0
    assert entry.bytes_per_token is None


def test_the_thirty_billion_moe_prices_at_the_gigabyte_the_house_measured() -> None:
    """1.711 GB/token and a 356 tok/s ceiling — 8 of 128 experts at 4 bits plus the whole
    untied lm_head, with the 0.175 GB embedding table out because a step gathers one row of
    it. Read off the headers of a checkpoint this suite never opens, let alone loads;
    skipped rather than downloading 17 GB."""
    if not repository(catalog.HUB_CACHE, MOE).is_dir():
        pytest.skip(f"{MOE} is not in the hub cache")

    entry = {found.id: found for found in catalog.scan()}[MOE]

    assert entry.bytes_per_token is not None
    assert round(entry.bytes_per_token / 1e9, 3) == 1.711
    assert round(ceiling(entry.bytes_per_token)) == 356


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
    checkpoint(
        repository(hub, QUANTIZED) / "snapshots" / "head",
        CONFIG,
        extra=((name, [2304, 1152], table),),
    )
    main(hub, QUANTIZED, "head")

    (entry,) = catalog.scan()

    assert entry.bytes_per_token == SHARD_BYTES + (table if priced else 0)


def test_the_thirty_five_billion_moe_prices_the_shared_expert_whole() -> None:
    """8 of 256 experts at 4 bits, the shared expert — which the checkpoint ships as plain
    2-D tensors and the tree stacks as slot 256 of the routed pile — read whole, and the
    tower's position table out: 2,557,394,656 bytes, the same integer `test_footprint`
    asserts over this checkpoint once loaded. That equality is what delta 0 means for the
    sixth family, and it is what this test is for; the number itself still carries the
    vision tower and still misses the recurrent state, which `test_footprint` spells out."""
    if not repository(catalog.HUB_CACHE, MOE_SHARED).is_dir():
        pytest.skip(f"{MOE_SHARED} is not in the hub cache")

    entry = {found.id: found for found in catalog.scan()}[MOE_SHARED]

    assert entry.bytes_per_token == 2_557_394_656


def test_a_real_repository_from_the_hub_cache_reads_its_own_numbers() -> None:
    """The scan against the machine's cache, over the checkpoint the API gate already
    uses. Skipped rather than downloading: this suite is not what pulls 350 MB."""
    if not repository(catalog.HUB_CACHE, QUANTIZED).is_dir():
        pytest.skip(f"{QUANTIZED} is not in the hub cache")

    entries = {entry.id: entry for entry in catalog.scan()}
    assert QUANTIZED in entries
    entry = entries[QUANTIZED]
    assert entry.architecture == "qwen3"
    assert entry.quantization == "4-bit"
    assert entry.context == 40960
    assert (entry.directory / "model.safetensors").is_file()
    assert 0.30 < entry.bytes_on_disk / 1e9 < 0.45

"""What `/admin/models` answers over the catalog: how an id is routed, what residency does to
an entry, and the card, the files, the assets, the image price and the blueprint beside it.

The app is the daemon's own, built through `create_app`: a delete forgets what the prefix
tier spilled under the id it removes, and the row it drops lives in the database the lifespan
opens.
"""

import json
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import mlx.core as mx
import pytest

from mlx_omnia.engine.footprint import active_bytes_per_token
from mlx_omnia.engine.models.glm4_moe import CHECKPOINT as GLM4_MOE
from mlx_omnia.server.services import catalog

from .catalog_stand import (
    CONFIG,
    DENSE,
    DENSE_CONFIG,
    MOE_TINY,
    QUANTIZED,
    TINY,
    client_of,
    glm_checkpoint,
    index,
    installed,
    main,
    moe_checkpoint,
    repository,
    use_caches,
)


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    return use_caches(tmp_path, monkeypatch)


@pytest.fixture
def stack() -> Iterator[ExitStack]:
    with ExitStack() as opened:
        yield opened


def test_an_id_with_a_slash_is_routed_raw_and_percent_encoded(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    installed(hub, QUANTIZED)
    client = client_of(stack)

    raw = client.get(f"/admin/models/{QUANTIZED}")
    encoded = client.get(f"/admin/models/{quote(QUANTIZED, safe='')}")
    assert raw.status_code == 200, raw.text
    assert raw.json()["id"] == QUANTIZED
    assert encoded.status_code == 200, encoded.text
    assert encoded.json() == raw.json()


def test_the_catalog_says_which_entry_is_resident_and_can_list_only_those(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    installed(hub, QUANTIZED)
    installed(hub, DENSE, DENSE_CONFIG)
    client = client_of(stack, QUANTIZED)

    everything = client.get("/admin/models").json()
    assert {entry["id"]: entry["resident"] for entry in everything} == {
        QUANTIZED: True,
        DENSE: False,
    }
    only = client.get("/admin/models", params={"resident": "true"}).json()
    assert [entry["id"] for entry in only] == [QUANTIZED]


def test_deleting_a_resident_model_is_refused_and_says_why(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    installed(hub, QUANTIZED)
    response = client_of(stack, QUANTIZED).delete(f"/admin/models/{QUANTIZED}")
    assert response.status_code == 409, response.text
    assert "resident" in response.json()["detail"]
    assert repository(hub, QUANTIZED).is_dir()


def test_deleting_a_model_that_is_not_resident_takes_its_blobs_with_it(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    """Removing the snapshot alone frees nothing: it is symlinks, and the bytes are in
    `blobs/`."""
    hub, _ = caches
    installed(hub, QUANTIZED)
    blob = repository(hub, QUANTIZED) / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\0" * 2048)

    response = client_of(stack).delete(f"/admin/models/{QUANTIZED}")
    assert response.status_code == 204, response.text
    assert not repository(hub, QUANTIZED).exists()
    assert catalog.scan() == []


def test_a_model_that_is_not_on_disk_is_a_404(caches: tuple[Path, Path], stack: ExitStack) -> None:
    client = client_of(stack)
    assert client.get("/admin/models/nope").status_code == 404
    assert client.delete("/admin/models/nope").status_code == 404


def test_the_card_is_the_readme_raw_and_absent_is_a_404(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    snapshot = installed(hub, QUANTIZED)
    installed(hub, DENSE)
    (snapshot / "README.md").write_text("---\nlicense: mit\n---\n# hello\n")
    client = client_of(stack)
    answer = client.get(f"/admin/models/{quote(QUANTIZED, safe='')}/card")
    assert answer.status_code == 200
    assert answer.text == "---\nlicense: mit\n---\n# hello\n"
    assert client.get(f"/admin/models/{quote(DENSE, safe='')}/card").status_code == 404


def test_the_files_listing_prices_each_name_through_the_symlink(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    snapshot = installed(hub, QUANTIZED)
    blob = snapshot.parent.parent / "blobs" / "cafe"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x89PNG" + b"\0" * 96)
    (snapshot / "ladder.png").symlink_to(blob)
    client = client_of(stack)
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
    caches: tuple[Path, Path], stack: ExitStack, tmp_path: Path
) -> None:
    hub, _ = caches
    snapshot = installed(hub, QUANTIZED)
    (snapshot / "ladder.png").write_bytes(b"\x89PNG-bytes")
    outside = tmp_path / "secret.txt"
    outside.write_text("not yours")
    client = client_of(stack)
    base = f"/admin/models/{quote(QUANTIZED, safe='')}/assets"
    assert client.get(f"{base}/ladder.png").content == b"\x89PNG-bytes"
    assert client.get(f"{base}/missing.png").status_code == 404
    assert client.get(f"{base}/{quote('../secret.txt', safe='')}").status_code == 404
    assert client.get(f"{base}/{quote(str(outside), safe='')}").status_code == 404


def test_a_resident_entry_is_priced_by_the_tree_that_is_answering_the_requests(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    """The estimate off the headers is what an entry carries until it is loaded; from then
    on the number is the engine's walk over the real tree, which is the one that knows what
    the loader dropped. A resident model with no tree under it keeps the estimate."""
    hub, _ = caches
    directory = repository(hub, MOE_TINY) / "snapshots" / "head"
    dropped = glm_checkpoint(directory)
    main(hub, MOE_TINY, "head")
    estimate = active_bytes_per_token(GLM4_MOE.load(directory, None)) + dropped

    listed = client_of(stack, MOE_TINY, active=estimate - dropped).get("/admin/models").json()
    assert [entry["bytes_per_token"] for entry in listed] == [estimate - dropped]

    untreed = client_of(stack, MOE_TINY).get(f"/admin/models/{quote(MOE_TINY, safe='')}").json()
    assert untreed["bytes_per_token"] == estimate
    assert catalog.scan()[0].bytes_per_token == estimate


def test_the_model_route_carries_the_cache_facts(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    installed(hub, "house/kv", CONFIG)

    body = client_of(stack).get("/admin/models").json()

    assert body
    for entry in body:
        assert {"kv_bytes_per_token", "attention_window", "vocab_size", "shape"} <= set(entry)


# ── the blueprint ────────────────────────────────────────────────────────


def test_the_blueprint_traces_the_step_without_loading_the_checkpoint(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    """The route builds the tree to answer, and building it reads no weight. What that has to
    hold is both halves: the graph arrives, and MLX's live footprint does not move — a route
    that quietly loaded 17 GB to draw a picture would answer the same and cost the machine
    the model."""
    hub, _ = caches
    directory = moe_checkpoint(repository(hub, MOE_TINY) / "snapshots" / "head", bits=4)
    index(directory)
    main(hub, MOE_TINY, "head")
    client = client_of(stack)

    mx.clear_cache()
    before = mx.get_active_memory()
    answer = client.get(f"/admin/models/{quote(MOE_TINY, safe='')}/blueprint")
    assert answer.status_code == 200, answer.json()
    assert mx.get_active_memory() - before < 4096, "the route read a tensor"

    drawn = answer.json()
    assert [node["role"] for node in drawn["spine"]][:2] == ["embedding", "stack"]
    (block,) = drawn["blocks"]
    assert block["layers"] == list(range(TINY.num_hidden_layers))
    # The wiring, which is the thing no config carries: the block's input is one side of the
    # first sum, and the norm before the mixer reads the same input.
    edges = {(edge["source"], edge["target"]) for edge in block["edges"] if edge["observed"]}
    # One join and not two: this architecture's step hands the residual to the routed mixer
    # and adds it in there, so only the attention sum is a `+` of its own. Which is the sort
    # of thing the drawing exists to show and no config says.
    (join,) = [node["id"] for node in block["nodes"] if node["role"] == "join"]
    assert ("in", "input_layernorm") in edges
    assert ("in", join) in edges
    assert (join, "mlp") in edges
    # And the kernel the routed mixer resolved to, which is what a dense tree would not have
    # picked: the route decides on the config, and it had one to read.
    routed = [node for node in block["nodes"] if node["kernels"]]
    assert routed, "no operation reported which implementation ran"
    assert any("Route" in name for node in routed for name in node["kernels"])


def test_an_architecture_with_no_loader_is_refused_rather_than_drawn(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    directory = moe_checkpoint(repository(hub, MOE_TINY) / "snapshots" / "head", bits=4)
    index(directory)
    (directory / "config.json").write_text(json.dumps({**asdict(TINY), "model_type": "qwen9"}))
    main(hub, MOE_TINY, "head")

    answer = client_of(stack).get(f"/admin/models/{quote(MOE_TINY, safe='')}/blueprint")

    assert answer.status_code == 409
    assert "qwen9" in answer.json()["detail"]

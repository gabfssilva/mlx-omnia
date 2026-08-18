"""Which checkpoints take a picture, and what one costs before it is sent."""

import json
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest

from mlx_omnia.server.services import catalog

from .catalog_stand import QUANTIZED, client_of, installed, use_caches

SEEING = "house/with-eyes"

SEEING_CONFIG: dict[str, object] = {
    "model_type": "qwen3_5",
    "image_token_id": 100,
    "vision_start_token_id": 101,
    "vision_end_token_id": 102,
    "vision_config": {
        "depth": 2,
        "hidden_size": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "num_heads": 2,
        "intermediate_size": 128,
        "out_hidden_size": 64,
        "in_channels": 3,
        "temporal_patch_size": 2,
        "num_position_embeddings": 64,
        "deepstack_visual_indexes": [],
        "hidden_act": "gelu_pytorch_tanh",
    },
    "text_config": {
        "model_type": "qwen3_5",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention", "linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [8, 4, 4],
        },
    },
}
"""A checkpoint of a family that has eyes. Every field is one the family's own config mirror
requires — the scan does not read a single one of them, but `sight` parses the mirror, and a
config it cannot parse is a model this catalog reports as taking no image."""

PROCESSOR = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
"""qwen3.5's real geometry: patches of 16 folded 2x2, and an area window wide enough that
neither bound moves the sizes below."""


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    return use_caches(tmp_path, monkeypatch)


@pytest.fixture
def stack() -> Iterator[ExitStack]:
    with ExitStack() as opened:
        yield opened


def _seeing(hub: Path, *, processor: bool = True) -> None:
    snapshot = installed(hub, SEEING, SEEING_CONFIG)
    if processor:
        (snapshot / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))


def test_a_checkpoint_with_a_tower_says_it_takes_an_image(caches: tuple[Path, Path]) -> None:
    hub, _ = caches
    _seeing(hub)
    installed(hub, QUANTIZED)

    assert {entry.id: entry.sees for entry in catalog.scan()} == {SEEING: True, QUANTIZED: False}


def test_the_same_tower_without_its_processor_takes_none(caches: tuple[Path, Path]) -> None:
    """The truth is per checkpoint and not per architecture: the config declares the tower,
    the file that says how to cut an image for it never landed, and the model the loader
    builds refuses pictures. Reported on the family's name it would be offered one."""
    hub, _ = caches
    _seeing(hub, processor=False)

    assert [entry.sees for entry in catalog.scan()] == [False]


def test_an_image_is_priced_before_it_is_sent(caches: tuple[Path, Path], stack: ExitStack) -> None:
    hub, _ = caches
    _seeing(hub)

    body = client_of(stack).get(
        f"/admin/models/{SEEING}/image", params={"height": 712, "width": 1236}
    )

    assert body.status_code == 200, body.text
    # 1236x712 rounded to the patch block is 1248x704 — 78x44 patches, folded 2x2.
    assert body.json() == {"height": 704, "width": 1248, "tokens": 858}


def test_a_model_that_takes_no_image_says_so_rather_than_pricing_one(
    caches: tuple[Path, Path], stack: ExitStack
) -> None:
    hub, _ = caches
    installed(hub, QUANTIZED)

    refusal = client_of(stack).get(
        f"/admin/models/{QUANTIZED}/image", params={"height": 8, "width": 8}
    )

    assert refusal.status_code == 409, refusal.text
    assert "takes no image" in refusal.json()["detail"]


def test_an_image_with_no_size_is_refused(caches: tuple[Path, Path], stack: ExitStack) -> None:
    hub, _ = caches
    _seeing(hub)
    client = client_of(stack)

    size = {"height": 8, "width": 8}
    flat = client.get(f"/admin/models/{SEEING}/image", params={**size, "height": 0})
    assert flat.status_code == 422
    # No size at all is a body no schema accepts, and the daemon answers those in the
    # dialect of whoever asked — 400 with a named field, not FastAPI's own 422.
    absent = client.get(f"/admin/models/{SEEING}/image")
    assert absent.status_code == 400, absent.text
    assert "height" in absent.text
    assert client.get("/admin/models/house/nobody/image", params=size).status_code == 404

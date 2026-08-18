"""The quantize job: what it writes, under which id, and what the entry then carries.

The fixtures and the two models live in `quantize_stand` and `quantize_models`; what is
here is the entry the job produces — its id, its widths, its provenance and its price.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi.testclient import TestClient

import mlx_omnia
from mlx_omnia.engine import task
from mlx_omnia.engine.quant.quantization import MXFP, NVFP, Affine, infer_quantization
from mlx_omnia.server.services import catalog, quantize
from mlx_omnia.server.services.quantize import packing
from mlx_omnia.server.services.quantize.plan import slug
from tests.server.polling import progress, view, wait_for
from tests.server.quantize_models import CALIBRATED, CALIBRATION, DRAFT_LEAVES, LEAVES
from tests.server.quantize_stand import (
    DEADLINE,
    REPO,
    SOURCE_REPO,
    SOURCE_SHA,
    Packer,
    bf16_source,
    blocked,
    caches,
    client,
    difference,
    drafter,
    hub_source,
    logits,
    source,
    start,
)

__all__ = [
    "bf16_source",
    "blocked",
    "caches",
    "client",
    "drafter",
    "hub_source",
    "source",
]


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

    assert progress(finished) == (len(DRAFT_LEAVES), len(DRAFT_LEAVES))
    entries = catalog.scan()
    assert [(entry.id, entry.quantization) for entry in entries] == [("local/draft-4bit", "4-bit")]

    packed = task.load_drafter(entries[0].directory)
    assert {
        path for path, module in packed.named_modules() if isinstance(module, nn.QuantizedLinear)
    } == set(DRAFT_LEAVES)


@pytest.mark.parametrize("method", CALIBRATED)
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

    assert progress(finished) == (len(LEAVES), len(LEAVES))
    entries = catalog.scan()
    assert [entry.id for entry in entries] == [REPO]
    assert entries[0].quantization == "4-bit"
    assert (entries[0].directory / "tokenizer.json").is_file(), "the entry cannot load alone"

    loaded = mlx_omnia.load(REPO, local_files_only=True)
    memory = mlx_omnia.load(source, quantize=Affine(group_size=64, bits=4), cache=False)

    assert difference(logits(loaded), logits(memory)) == 0.0
    assert list((caches / quantize.STAGING).iterdir()) == [], "the staging outlived the job"


def test_the_job_reports_the_leaf_it_is_packing_and_how_many_are_left(
    client: TestClient, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the screen has to draw while it waits. Leaves and not bytes: the plan is what
    the job already holds, and it is the unit the packing advances by."""
    packer = Packer(at=2)
    monkeypatch.setattr(packing, "quantize_weights", packer)
    job_id = start(client, source=str(source), repo=REPO)
    assert packer.reached.wait(DEADLINE), "the job never reached the second leaf"

    frame = view(client, job_id)["progress"]

    assert isinstance(frame, dict)
    assert (frame["message"], frame["completed"], frame["total"]) == ("embed", 1, len(LEAVES))

    packer.proceed.set()

    assert progress(wait_for(client, job_id, "ok")) == (len(LEAVES), len(LEAVES))


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

    assert {leaf: infer_quantization(tensors, leaf, input_dims=64) for leaf in LEAVES} == {
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
    assert logits(mlx_omnia.load(REPO, local_files_only=True)).shape == (1, 5, 32)


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

    assert {leaf: infer_quantization(tensors, leaf, input_dims=64) for leaf in LEAVES} == {
        "attn": format,
        "embed": None,
        "head": format,
        "mlp": format,
    }
    assert entry.quantization == mode
    assert logits(mlx_omnia.load(REPO, local_files_only=True)).shape == (1, 5, 32)


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
    recorded_block = json.loads((entry.directory / "config.json").read_text())["mlx_omnia"]

    assert recorded_block["source"] == {"repository": SOURCE_REPO, "commit": SOURCE_SHA}
    assert recorded_block["digest"] == entry.directory.name
    assert recorded_block["repo"] == REPO
    assert recorded_block["method"] == "rtn"
    assert recorded_block["format_version"] == task._FORMAT_VERSION


def test_a_source_without_a_trunk_fails_the_calibrated_job_and_writes_nothing(
    client: TestClient, source: Path, caches: Path
) -> None:
    """The one thing the pass still demands of an architecture: a list of blocks to find.
    `Tiny` has four leaves and no trunk, and the failure is the engine's own message."""
    failed = wait_for(
        client,
        start(client, source=str(source), repo=REPO, method="gptq", **CALIBRATION),
        "error",
    )

    error = failed["error"]
    assert isinstance(error, str) and "no list of blocks" in error
    assert catalog.scan() == []
    assert not (caches / slug(REPO)).exists()

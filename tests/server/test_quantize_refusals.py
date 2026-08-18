"""What the quantize routes refuse, and what a `DELETE` mid-job leaves behind."""

import json
from pathlib import Path

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from mlx_omnia.engine import task
from mlx_omnia.engine.quant.quantization import inventory
from mlx_omnia.server.services import catalog, quantize
from mlx_omnia.server.services.quantize import packing
from mlx_omnia.server.services.quantize.plan import slug
from tests.server.polling import wait_for
from tests.server.quantize_models import LEAVES, Tiny
from tests.server.quantize_stand import (
    DEADLINE,
    REPO,
    Packer,
    Writer,
    blocked,
    caches,
    client,
    source,
    start,
)

__all__ = ["blocked", "caches", "client", "source"]


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
    assert writer.reached.wait(DEADLINE), "the write never began"
    (staged,) = writer.staged
    assert staged.name.startswith(".tmp-"), "the entry was written under its final name"
    assert (staged / "model.safetensors").is_file(), "the cancellation came before the write"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    writer.proceed.set()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None
    assert not staged.exists(), "the `.tmp-` outlived the job"
    assert not (caches / slug(REPO)).exists(), "the entry took its final name"
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
    monkeypatch.setattr(packing, "quantize_weights", packer)
    job_id = start(client, source=str(source), repo=REPO)
    assert packer.reached.wait(DEADLINE), "the job never reached the second leaf"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    packer.proceed.set()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None
    assert packer.calls < len(LEAVES), "the job packed every leaf and only then noticed"
    assert catalog.scan() == []
    assert not (caches / slug(REPO)).exists()


def test_a_second_job_for_the_same_repo_id_is_refused_while_the_first_runs(
    client: TestClient, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two jobs on one repo id are two jobs staging into one directory, and the first to
    finish renames it out from under the second. The refusal goes out while the first job is
    demonstrably inside the packing — a guard tested before the job starts is a guard tested
    against nothing."""
    packer = Packer(at=1)
    monkeypatch.setattr(packing, "quantize_weights", packer)
    first = start(client, source=str(source), repo=REPO)
    assert packer.reached.wait(DEADLINE), "the job never began packing"

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
    assert not (caches / slug(REPO)).exists()


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
                for leaf in inventory(Tiny())
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

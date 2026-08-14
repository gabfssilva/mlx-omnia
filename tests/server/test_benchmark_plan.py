"""`POST /admin/benchmarks/plan`: what would run, what would not, and why — with nothing
written. The property that matters most is that it and the job agree, because the moment
they do not there are two answers to "does this fit"."""

from pathlib import Path

from benchmark_stand import HUGE, SMALL, SPEED_BODY, WINDOWED, stand, wait_for
from fastapi.testclient import TestClient

from mlx_omnia.server import benchmarks

MODEL = "house/small"
BIG = "house/huge"
BANDED = "house/banded"


def plan(client: TestClient, body: dict[str, object]) -> dict[str, object]:
    response = client.post("/admin/benchmarks/plan", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def test_the_plan_and_the_job_expand_to_the_same_shapes(tmp_path: Path) -> None:
    """One function, used twice. A second copy of the arithmetic would go stale the first
    time the cache dtype moved."""
    with stand(tmp_path, [(MODEL, SMALL), (BIG, HUGE)]) as house:
        body = {
            **SPEED_BODY,
            "models": [MODEL, BIG],
            "contexts": [512, 2048, 131072],
            "generates": [128, 256],
        }
        drawn = plan(house.client, body)

        response = house.client.post("/admin/benchmarks", json=body)
        assert response.status_code == 202
        wait_for(house.client, response.json()["id"], "ok")
        written = house.client.get("/admin/benchmarks/runs").json()

        planned = {(row["model"], row["key"]) for row in drawn["shapes"] + drawn["skipped"]}
        assert {(row["model"], row["key"]) for row in written} == planned


def test_a_shape_that_does_not_fit_is_skipped_with_both_numbers(tmp_path: Path) -> None:
    with stand(tmp_path, [(BIG, HUGE)]) as house:
        drawn = plan(
            house.client,
            {**SPEED_BODY, "models": [BIG], "contexts": [131072], "generates": [2048]},
        )

        assert drawn["shapes"] == []
        (skipped,) = drawn["skipped"]
        assert skipped["reason"] == "kv_over_budget"
        assert skipped["needed_bytes"] > skipped["budget_bytes"]


def test_the_same_request_runs_on_a_checkpoint_whose_cache_stops_growing(
    tmp_path: Path,
) -> None:
    """Which is the whole reason the window is a field of the catalog."""
    with stand(tmp_path, [(BIG, HUGE), (BANDED, WINDOWED)]) as house:
        drawn = plan(
            house.client,
            {
                **SPEED_BODY,
                "models": [BIG, BANDED],
                "contexts": [131072],
                "generates": [2048],
            },
        )

        assert [row["model"] for row in drawn["shapes"]] == [BANDED]
        assert [row["model"] for row in drawn["skipped"]] == [BIG]


def test_turning_the_skip_off_moves_the_already_measured_back_into_the_shapes(
    tmp_path: Path,
) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        response = house.client.post("/admin/benchmarks", json={**SPEED_BODY, "models": [MODEL]})
        wait_for(house.client, response.json()["id"], "ok")

        skipping = plan(house.client, {**SPEED_BODY, "models": [MODEL]})
        keeping = plan(house.client, {**SPEED_BODY, "models": [MODEL], "skip_if_measured": False})

        assert skipping["shapes"] == []
        assert len(skipping["already"]) == 1
        assert keeping["already"] == []
        assert len(keeping["shapes"]) == 1


def test_an_estimate_with_no_history_and_no_ceiling_is_absent(tmp_path: Path) -> None:
    """`null` rather than a guessed number: a bar drawn from an invented total is worse
    than no bar."""
    with stand(tmp_path, [(MODEL, SMALL), (BIG, {"model_type": "unknown"})]) as house:
        drawn = plan(house.client, {**SPEED_BODY, "models": [MODEL]})

        assert len(drawn["shapes"]) == 1
        # This checkpoint is priced, so there is a ceiling to fall back on.
        assert drawn["estimate_seconds"] is not None and drawn["estimate_seconds"] > 0

        unpriced = plan(house.client, {**SPEED_BODY, "models": [BIG]})
        assert len(unpriced["shapes"]) == 1
        assert unpriced["estimate_seconds"] is None


def test_a_fidelity_pair_of_two_vocabularies_is_skipped_before_anything_loads(
    tmp_path: Path,
) -> None:
    with stand(tmp_path, [(MODEL, SMALL), (BIG, {**HUGE, "vocab_size": 4096})]) as house:
        drawn = plan(
            house.client,
            {"kind": "fidelity", "pairs": [{"model": MODEL, "reference": BIG}]},
        )

        (skipped,) = drawn["skipped"]
        assert skipped["reason"] == "vocabulary_mismatch"
        assert skipped["key"] == benchmarks.fidelity_key("wikitext103", 10000, 42, BIG)
        assert house.engine.resident == []


def test_the_plan_writes_nothing(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        plan(house.client, {**SPEED_BODY, "models": [MODEL]})

        assert house.client.get("/admin/benchmarks/runs").json() == []


def test_the_same_vocabulary_with_another_architecture_warns_and_runs(tmp_path: Path) -> None:
    """Qwen3-14B and Qwen3-30B-A3B share a tokenizer, so KL between them computes and says
    nothing this view means. It warns rather than refuses: a fine-tune prints identically to
    its base, so blocking on a different print would block only the pairs somebody
    deliberately declared."""
    with stand(tmp_path, [(MODEL, SMALL), (BIG, HUGE)]) as house:
        drawn = plan(
            house.client, {"kind": "fidelity", "pairs": [{"model": MODEL, "reference": BIG}]}
        )

        assert [row["reason"] for row in drawn["warnings"]] == ["shape_mismatch"]
        assert drawn["skipped"][0]["reason"] == "not_implemented"
        assert "the architectures do not" in drawn["warnings"][0]["detail"]


def test_two_checkpoints_of_one_architecture_carry_no_warning(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL), ("house/twin", SMALL)]) as house:
        drawn = plan(
            house.client,
            {"kind": "fidelity", "pairs": [{"model": MODEL, "reference": "house/twin"}]},
        )

        assert drawn["warnings"] == []

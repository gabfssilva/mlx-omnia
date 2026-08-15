"""`POST /admin/benchmarks`: what the batch writes, what it refuses, and the one thing that
makes any of the numbers worth keeping — that nothing else runs while it measures."""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from mlx_omnia import GenerationOptions, Text
from mlx_omnia.server import benchmarks
from mlx_omnia.server.engine import Engine
from tests.server.benchmark_stand import (
    HUGE,
    SMALL,
    SPEED_BODY,
    Scripted,
    composite,
    stand,
)
from tests.server.polling import wait_for

MODEL = "house/small"
OTHER = "house/other"
BIG = "house/huge"


def start(client: TestClient, body: dict[str, object]) -> str:
    response = client.post("/admin/benchmarks", json=body)
    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["kind"] == "benchmark"
    job_id = payload["id"]
    assert isinstance(job_id, str)
    return job_id


def test_two_shapes_become_two_rows_under_one_key(tmp_path: Path) -> None:
    """And the key is the same on both models, which is what authorises drawing them on one
    axis."""
    with stand(tmp_path, [(MODEL, SMALL), (OTHER, SMALL)]) as house:
        job_id = start(house.client, {**SPEED_BODY, "models": [MODEL, OTHER]})
        finished = wait_for(house.client, job_id, "ok")

        assert finished["error"] is None
        rows = house.client.get("/admin/benchmarks/runs").json()
        assert {row["model"] for row in rows} == {MODEL, OTHER}
        assert len({row["key"] for row in rows}) == 1
        key = rows[0]["key"]
        under = house.client.get("/admin/benchmarks/runs", params={"key": key}).json()
        assert len(under) == 2
        assert all(row["speed"]["decode_tps"] for row in under)


def test_the_warm_up_runs_once_per_checkpoint_and_not_once_per_shape(tmp_path: Path) -> None:
    """Two shapes of one model cost one warm-up, a load probe each and one round each: five
    generations. A warm-up per shape would be six."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        job_id = start(house.client, {**SPEED_BODY, "models": [MODEL], "contexts": [512, 2048]})
        wait_for(house.client, job_id, "ok")

        assert house.model.runs == 5


def test_every_checkpoint_gets_its_own_warm_up(tmp_path: Path) -> None:
    """Once per batch would leave the second model measured on a cold first round: two models
    at one shape are two warm-ups, two load probes and two rounds."""
    with stand(tmp_path, [(MODEL, SMALL), (OTHER, SMALL)]) as house:
        job_id = start(house.client, {**SPEED_BODY, "models": [MODEL, OTHER]})
        wait_for(house.client, job_id, "ok")

        assert house.model.runs == 6


def test_a_shape_that_does_not_fit_is_recorded_and_not_raised(tmp_path: Path) -> None:
    with stand(tmp_path, [(BIG, HUGE)]) as house:
        job_id = start(
            house.client, {**SPEED_BODY, "models": [BIG], "contexts": [131072], "generates": [2048]}
        )
        finished = wait_for(house.client, job_id, "ok")

        assert finished["error"] is None
        (row,) = house.client.get("/admin/benchmarks/runs").json()
        assert row["state"] == "not_run"
        assert row["reason"] == "kv_over_budget"
        assert row["speed"]["context"] == 131072


def test_more_than_one_stream_is_measured(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        job_id = start(house.client, {**SPEED_BODY, "models": [MODEL], "concurrencies": [1, 4]})
        wait_for(house.client, job_id, "ok")

        rows = house.client.get("/admin/benchmarks/runs").json()
        states = {row["speed"]["concurrency"]: row["reason"] for row in rows}
        assert states == {1: None, 4: None}


def test_a_value_outside_the_fixed_sets_is_refused_by_name(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        response = house.client.post(
            "/admin/benchmarks", json={**SPEED_BODY, "models": [MODEL], "contexts": [3000]}
        )

        assert response.status_code == 400
        assert "3000" in response.json()["detail"]


def test_a_model_the_catalog_does_not_know_is_a_404(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        response = house.client.post(
            "/admin/benchmarks", json={**SPEED_BODY, "models": ["house/ghost"]}
        )

        assert response.status_code == 404


def test_skip_if_measured_is_what_the_second_run_asks(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        wait_for(house.client, start(house.client, {**SPEED_BODY, "models": [MODEL]}), "ok")
        first = house.model.runs
        wait_for(house.client, start(house.client, {**SPEED_BODY, "models": [MODEL]}), "ok")

        assert house.model.runs == first, "the shape was measured again despite the skip"
        wait_for(
            house.client,
            start(house.client, {**SPEED_BODY, "models": [MODEL], "skip_if_measured": False}),
            "ok",
        )
        assert house.model.runs > first
        # Nothing is overwritten: the series under one key is what answers whether an engine
        # version moved the number, so the history holds both and the key holds the newest.
        assert len(house.client.get("/admin/benchmarks/runs").json()) == 2
        key = house.client.get("/admin/benchmarks/runs").json()[0]["key"]
        assert len(house.client.get("/admin/benchmarks/runs", params={"key": key}).json()) == 1


def test_quality_expands_its_rows_and_then_says_it_is_not_implemented(tmp_path: Path) -> None:
    """A job ending `ok` with an invented number is the one outcome this section exists to
    not produce."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        job_id = start(
            house.client,
            {"kind": "quality", "models": [MODEL], "datasets": ["mmlu"], "items": 1400},
        )
        finished = wait_for(house.client, job_id, "error")

        assert finished["error"] is not None
        assert "59.8" in finished["error"]
        (row,) = house.client.get("/admin/benchmarks/runs", params={"kind": "quality"}).json()
        assert row["state"] == "not_run"
        assert row["reason"] == "not_implemented"
        assert row["quality"]["dataset"] == "mmlu"
        assert row["key"] == benchmarks.quality_key("mmlu", 1400, 42, 5)


def test_fidelity_refuses_a_pair_whose_logits_are_not_the_same_vector(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL), (BIG, {**HUGE, "vocab_size": 4096})]) as house:
        job_id = start(
            house.client,
            {"kind": "fidelity", "pairs": [{"model": MODEL, "reference": BIG}]},
        )
        wait_for(house.client, job_id, "error")

        (row,) = house.client.get("/admin/benchmarks/runs", params={"kind": "fidelity"}).json()
        assert row["reason"] == "vocabulary_mismatch"
        assert row["fidelity"]["reference"] == BIG
        assert "ref:" in row["key"]


def test_the_reservation_makes_everybody_else_wait_at_the_door(tmp_path: Path) -> None:
    """The one thing that makes every number above worth keeping. Without it a chat request
    lands between two rounds and moves the median without appearing anywhere.

    Against the engine rather than through HTTP: what the reservation is about is `submit`,
    which is the same door every dialect goes through.
    """
    del tmp_path

    async def run() -> list[str]:
        order: list[str] = []
        engine = Engine(lambda _model_id: composite(Scripted()))
        engine.start()
        token = await engine.acquire_queue()

        async def outsider() -> None:
            job = await engine.submit(MODEL, Text("hello"), GenerationOptions(max_tokens=2))
            order.append("chat")
            while await job.chunks.get() is not None:
                pass

        waiting = asyncio.create_task(outsider())
        # Long enough that a `submit` which did not wait would already have appended.
        await asyncio.sleep(0.2)
        assert order == [], "a request got past a reserved queue"
        holder = await engine.submit(MODEL, Text("mine"), GenerationOptions(max_tokens=2), token)
        while await holder.chunks.get() is not None:
            pass
        order.append("benchmark")
        await engine.release_queue(token)
        await asyncio.wait_for(waiting, 10)
        engine.stop()
        return order

    assert asyncio.run(run()) == ["benchmark", "chat"]


def test_releasing_a_token_nobody_holds_changes_nothing(tmp_path: Path) -> None:
    """The release rides a `finally`, so it has to be safe to call twice."""
    del tmp_path

    async def run() -> None:
        engine = Engine(lambda _model_id: composite(Scripted()))
        token = await engine.acquire_queue()
        await engine.release_queue(token)
        await engine.release_queue(token)
        await engine.release_queue(object())
        # Still usable: a leaked lock would hang here instead of returning.
        await asyncio.wait_for(engine.acquire_queue(), 5)

    asyncio.run(run())


def test_cancelling_the_batch_gives_the_queue_back(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        job_id = start(house.client, {**SPEED_BODY, "models": [MODEL], "rounds": 8})
        house.client.delete(f"/admin/jobs/{job_id}")
        wait_for(house.client, job_id, "cancelled", "ok", "error")

        # The next batch runs, which it could not if the reservation had been leaked.
        again = start(house.client, {**SPEED_BODY, "models": [MODEL], "skip_if_measured": False})
        finished = wait_for(house.client, again, "ok")
        assert finished["error"] is None


def test_the_row_records_what_it_was_measured_with(tmp_path: Path) -> None:
    """Engine, mlx and who else was resident. A field left empty here is a number nobody can
    place afterwards."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        wait_for(house.client, start(house.client, {**SPEED_BODY, "models": [MODEL]}), "ok")

        (row,) = house.client.get("/admin/benchmarks/runs").json()

        assert row["engine_version"]
        assert row["mlx_version"]
        assert row["residents"]
        assert row["speed"]["step_weight_bytes"] is not None
        assert row["speed"]["step_kv_bytes"] is not None


def test_progress_names_the_shape_and_counts_it(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        job_id = start(house.client, {**SPEED_BODY, "models": [MODEL], "contexts": [512, 2048]})
        finished = wait_for(house.client, job_id, "ok")

        assert finished["progress"]["total"] == 2
        assert finished["progress"]["message"].startswith(MODEL)


def test_the_sampler_travels_into_the_key_and_two_of_them_are_two_rows(tmp_path: Path) -> None:
    """A drawn token pays for filters an argmax does not. The same shape under two samplers
    is two questions, so `skip_if_measured` does not confuse them for one another."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        greedy_job = start(house.client, {**SPEED_BODY, "models": [MODEL]})
        wait_for(house.client, greedy_job, "ok")
        drawn_job = start(
            house.client,
            {
                **SPEED_BODY,
                "models": [MODEL],
                "sampling": {"temperature": 0.7, "top_p": 0.95},
            },
        )
        wait_for(house.client, drawn_job, "ok")

        keys = {row["key"] for row in house.client.get("/admin/benchmarks/runs").json()}
        assert len(keys) == 2
        assert any(" · greedy · " in key for key in keys)
        assert any(" · t0.7·p0.95 · " in key for key in keys)


def test_the_engine_says_when_the_queue_is_reserved(tmp_path: Path) -> None:
    """`/admin/state` carries this straight through, and what the app does with it is stop
    polling the routes that walk the disk while a measurement runs."""
    del tmp_path

    async def run() -> None:
        engine = Engine(lambda _model_id: composite(Scripted()))
        assert engine.reserved is False
        token = await engine.acquire_queue()
        assert engine.reserved is True
        await engine.release_queue(token)
        assert engine.reserved is False

    asyncio.run(run())

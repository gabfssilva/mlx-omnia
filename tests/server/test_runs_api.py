"""`GET /admin/benchmarks/runs`: the two directions the history is read in, the rows that
did not produce a number, and the export."""

import csv
import io
from pathlib import Path

import pytest

from mlx_omnia.server.db.models.benchmarks import BenchmarkRun, QualityResult, SpeedResult
from mlx_omnia.server.services import benchmarks
from tests.server.benchmark_stand import SMALL, Stand, stand

MODEL = "house/small"
OTHER = "house/other"
KEY = "4k→256 · 1 streams · r3+1 · greedy · warm · queue"


def record(
    house: Stand,
    run_id: str,
    model: str,
    *,
    key: str = KEY,
    created_at: float = 1.0,
    state: str = "ok",
    reason: str | None = None,
    decode_tps: float | None = 100.0,
) -> None:
    async def write() -> None:
        await benchmarks.insert_run(
            BenchmarkRun(
                id=run_id,
                kind="speed",
                model=model,
                key=key,
                state="ok" if state == "ok" else "not_run",
                reason=reason,
                engine_version="0.9.2",
                mlx_version="0.30.1",
                created_at=created_at,
            ),
            SpeedResult(
                run_id=run_id,
                context=4096,
                generate=256,
                concurrency=1,
                rounds=3,
                stream_source="queue",
                page_cache="warm",
                decode_tps=decode_tps,
            ),
        )

    house.run(write)


def test_a_key_answers_with_one_row_per_model(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house, "old", MODEL, created_at=1.0)
        record(house, "new", MODEL, created_at=2.0)
        record(house, "other", OTHER, created_at=1.5)

        rows = house.client.get("/admin/benchmarks/runs", params={"key": KEY}).json()

        assert [row["id"] for row in rows] == ["other", "new"]


def test_a_model_answers_with_its_whole_series(tmp_path: Path) -> None:
    """Which is what says whether an engine version moved the number."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house, "old", MODEL, created_at=1.0)
        record(house, "new", MODEL, created_at=2.0)

        rows = house.client.get("/admin/benchmarks/runs", params={"model": MODEL}).json()

        assert [row["id"] for row in rows] == ["new", "old"]


def test_a_row_that_did_not_run_stays_in_the_listing_with_its_reason(tmp_path: Path) -> None:
    """It is drawn dashed, not left as a hole: a shape that did not fit is an answer."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house, "ok", MODEL)
        record(
            house,
            "refused",
            OTHER,
            state="not_run",
            reason="kv_over_budget",
            decode_tps=None,
        )

        rows = house.client.get("/admin/benchmarks/runs", params={"key": KEY}).json()

        refused = next(row for row in rows if row["id"] == "refused")
        assert refused["state"] == "not_run"
        assert refused["reason"] == "kv_over_budget"
        assert refused["speed"]["context"] == 4096
        assert refused["speed"]["decode_tps"] is None


def test_one_run_by_id_and_a_delete_that_takes_the_body_with_it(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house, "one", MODEL)

        assert house.client.get("/admin/benchmarks/runs/one").json()["model"] == MODEL
        assert house.client.delete("/admin/benchmarks/runs/one").status_code == 204
        assert house.client.get("/admin/benchmarks/runs/one").status_code == 404
        assert house.client.delete("/admin/benchmarks/runs/one").status_code == 404
        assert house.run(lambda: benchmarks.run("one")) is None


def test_a_body_that_does_not_match_its_header_leaves_no_row(tmp_path: Path) -> None:
    """One door, one refusal: the kind is taken from the result's own type and checked before
    either half is written, so a `speed` header carrying a quality body writes neither."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:

        async def mismatched() -> None:
            await benchmarks.insert_run(
                BenchmarkRun(
                    id="one",
                    kind="speed",
                    model=MODEL,
                    key=KEY,
                    state="ok",
                    engine_version="0.9.2",
                    mlx_version="0.30.1",
                    created_at=1.0,
                ),
                QualityResult(
                    run_id="one",
                    dataset="mmlu",
                    items=1400,
                    seed=42,
                    shots=5,
                    scoring="loglikelihood",
                ),
            )

        with pytest.raises(ValueError):
            house.run(mismatched)

        # The header row itself, and not what a view answers: a header stranded without its
        # body is exactly the run no view can see, so asserting through one would pass on the
        # write this is here to forbid.
        assert house.run(lambda: BenchmarkRun.objects.filter(id="one").exists()) is False
        assert house.run(lambda: benchmarks.run("one")) is None


def test_the_export_is_flat_and_carries_no_detail_blob(tmp_path: Path) -> None:
    """A spreadsheet has no use for the per-round JSON, and a cell holding it makes the
    file unreadable in the reader it was exported for."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house, "one", MODEL)
        record(house, "two", OTHER)

        response = house.client.get("/admin/benchmarks/runs.csv")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert len(rows) == 2
        assert set(rows[0]) >= {"model", "key", "decode_tps", "ceiling_fraction"}
        assert "per_round" not in rows[0]
        assert rows[0]["decode_tps"] == "100.0"


def test_an_unknown_kind_is_refused_rather_than_answered_empty(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        response = house.client.get("/admin/benchmarks/runs", params={"kind": "latency"})

        # 400 and not 422: the daemon answers a body no schema accepts in the dialect of
        # whoever sent it, and `/admin` gets OpenAI's envelope.
        assert response.status_code == 400
        assert "kind" in response.json()["error"]["message"]

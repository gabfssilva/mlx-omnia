"""`GET /admin/benchmarks/runs`: the two directions the history is read in, the rows that
did not produce a number, and the export."""

import csv
import io
from pathlib import Path

from benchmark_stand import SMALL, stand

from mlx_omnia_server.store import Run, SpeedResult, Store

MODEL = "house/small"
OTHER = "house/other"
KEY = "4k→256 · 1 streams · r3+1 · greedy · warm · queue"


def record(
    store: Store,
    run_id: str,
    model: str,
    *,
    key: str = KEY,
    created_at: float = 1.0,
    state: str = "ok",
    reason: str | None = None,
    decode_tps: float | None = 100.0,
) -> None:
    store.insert_run(
        Run(
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
            context=4096,
            generate=256,
            concurrency=1,
            rounds=3,
            stream_source="queue",
            page_cache="warm",
            decode_tps=decode_tps,
        ),
    )


def test_a_key_answers_with_one_row_per_model(tmp_path: Path) -> None:
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house.store, "old", MODEL, created_at=1.0)
        record(house.store, "new", MODEL, created_at=2.0)
        record(house.store, "other", OTHER, created_at=1.5)

        rows = house.client.get("/admin/benchmarks/runs", params={"key": KEY}).json()

        assert [row["id"] for row in rows] == ["other", "new"]


def test_a_model_answers_with_its_whole_series(tmp_path: Path) -> None:
    """Which is what says whether an engine version moved the number."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house.store, "old", MODEL, created_at=1.0)
        record(house.store, "new", MODEL, created_at=2.0)

        rows = house.client.get("/admin/benchmarks/runs", params={"model": MODEL}).json()

        assert [row["id"] for row in rows] == ["new", "old"]


def test_a_row_that_did_not_run_stays_in_the_listing_with_its_reason(tmp_path: Path) -> None:
    """It is drawn dashed, not left as a hole: a shape that did not fit is an answer."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house.store, "ok", MODEL)
        record(
            house.store,
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
        record(house.store, "one", MODEL)

        assert house.client.get("/admin/benchmarks/runs/one").json()["model"] == MODEL
        assert house.client.delete("/admin/benchmarks/runs/one").status_code == 204
        assert house.client.get("/admin/benchmarks/runs/one").status_code == 404
        assert house.client.delete("/admin/benchmarks/runs/one").status_code == 404
        assert house.store.run("one") is None


def test_the_export_is_flat_and_carries_no_detail_blob(tmp_path: Path) -> None:
    """A spreadsheet has no use for the per-round JSON, and a cell holding it makes the
    file unreadable in the reader it was exported for."""
    with stand(tmp_path, [(MODEL, SMALL)]) as house:
        record(house.store, "one", MODEL)
        record(house.store, "two", OTHER)

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

        assert response.status_code == 422

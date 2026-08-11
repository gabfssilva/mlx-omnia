"""Dataset definitions and the preview that checks a mapping before it costs twenty minutes.

The Hub is not reached: `hf_hub_download` and the file listing are replaced by a parquet
this suite writes, because what is under test is the mapping and the refusal, not the
network. What the real repositories look like is in the fixture's shape — a flat column, a
list column and a struct — which is the three layouts the definitions in circulation use.
"""

import json
from collections.abc import Generator
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_omnia_server import datasets
from mlx_omnia_server.store import Store

FILES = [
    "README.md",
    "all/test-00000-of-00001.parquet",
    "all/validation-00000-of-00001.parquet",
    "abstract_algebra/test-00000-of-00001.parquet",
]

MMLU_LIKE = {
    "question": ["What is 2 + 2?", "Which is a prime?"],
    "choices": [["3", "4", "5", "6"], ["4", "6", "7", "9"]],
    "answer": [1, 2],
    "subject": ["arithmetic", "arithmetic"],
}

ARC_LIKE = {
    "question": ["Which is a metal?"],
    "choices": [{"text": ["wood", "iron"], "label": ["A", "B"]}],
    "answerKey": ["B"],
}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    written = tmp_path / "test-00000-of-00001.parquet"
    parquet.write_table(arrow.table(MMLU_LIKE), written)
    nested = tmp_path / "arc.parquet"
    parquet.write_table(arrow.table(ARC_LIKE), nested)

    def download(repo: str, filename: str, **_options: object) -> str:
        del filename
        return str(nested if "arc" in repo else written)

    monkeypatch.setattr(datasets, "hf_hub_download", download)
    monkeypatch.setattr(datasets, "_files", lambda _repo: FILES)
    app = FastAPI()
    app.state.store = Store(tmp_path / "server.db")
    app.include_router(datasets.router)
    with TestClient(app) as opened:
        yield opened


MMLU_BODY: dict[str, object] = {
    "id": "mine",
    "use": "multiple_choice",
    "repo": "cais/mmlu",
    "config": "all",
    "split": "test",
    "template": "{question}\nAnswer:",
    "columns": {"question": "question", "choices": "choices", "answer": "answer"},
}


def test_the_presets_are_rows_and_say_so(client: TestClient) -> None:
    listing = client.get("/admin/benchmarks/datasets").json()

    found = {row["id"]: row for row in listing}
    assert {"mmlu", "arc-challenge", "hellaswag", "gsm8k", "wikitext103"} <= set(found)
    assert found["mmlu"]["builtin"] is True
    assert found["mmlu"]["columns"]["choices"] == "choices"
    assert found["wikitext103"]["use"] == "corpus"


def test_a_definition_is_declared_read_back_and_deleted(client: TestClient) -> None:
    assert client.post("/admin/benchmarks/datasets", json=MMLU_BODY).status_code == 201

    read = client.get("/admin/benchmarks/datasets/mine").json()
    assert read["repo"] == "cais/mmlu"
    assert read["builtin"] is False
    assert client.delete("/admin/benchmarks/datasets/mine").status_code == 204
    assert client.get("/admin/benchmarks/datasets/mine").status_code == 404
    assert client.delete("/admin/benchmarks/datasets/mine").status_code == 404


def test_a_preset_is_not_overwritten_under_its_own_id(client: TestClient) -> None:
    """The id travels inside the key of every run made with it, so silently redefining
    `mmlu` would make two different measurements share one name."""
    response = client.post("/admin/benchmarks/datasets", json={**MMLU_BODY, "id": "mmlu"})

    assert response.status_code == 409


def test_the_preview_renders_the_first_item_and_lists_the_continuations(
    client: TestClient,
) -> None:
    body = client.post("/admin/benchmarks/datasets/preview", json=MMLU_BODY)

    assert body.status_code == 200, body.text
    payload = body.json()
    assert payload["prompt"] == "What is 2 + 2?\nAnswer:"
    assert payload["continuations"] == [" 3", " 4", " 5", " 6"]
    assert payload["answer"] == "1"
    assert payload["rows"] == 2


def test_a_column_that_does_not_exist_is_a_400_naming_the_ones_that_do(
    client: TestClient,
) -> None:
    """Which is the whole point of the preview: the alternative is 25% accuracy at the end
    of a run, and nothing saying why."""
    response = client.post(
        "/admin/benchmarks/datasets/preview",
        json={**MMLU_BODY, "columns": {**MMLU_BODY["columns"], "choices": "options"}},  # type: ignore[dict-item]
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "options" in detail
    assert "choices" in detail and "subject" in detail


def test_a_placeholder_nothing_is_mapped_to_is_refused_rather_than_left_in_the_prompt(
    client: TestClient,
) -> None:
    response = client.post(
        "/admin/benchmarks/datasets/preview",
        json={**MMLU_BODY, "template": "{context}\n{question}\nAnswer:"},
    )

    assert response.status_code == 400
    assert "{context}" in response.json()["detail"]


def test_a_struct_column_is_reached_through_the_dots(client: TestClient) -> None:
    """ARC keeps its options under `choices.text`, which is a mapping and not a special
    case in the code."""
    response = client.post(
        "/admin/benchmarks/datasets/preview",
        json={
            "id": "arc",
            "use": "multiple_choice",
            "repo": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "split": "test",
            "template": "{question}\nAnswer:",
            "columns": {"question": "question", "choices": "choices.text", "answer": "answerKey"},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["continuations"] == [" wood", " iron"]


def test_a_corpus_has_no_continuations(client: TestClient) -> None:
    response = client.post(
        "/admin/benchmarks/datasets/preview",
        json={
            "id": "corpus",
            "use": "corpus",
            "repo": "cais/mmlu",
            "config": "all",
            "split": "test",
            "template": "{question}",
            "columns": {"question": "question"},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["continuations"] == []


def test_the_split_decides_which_parquet_is_read() -> None:
    assert datasets.parquet_path(FILES, "all", "test") == "all/test-00000-of-00001.parquet"
    assert (
        datasets.parquet_path(FILES, "abstract_algebra", "test")
        == "abstract_algebra/test-00000-of-00001.parquet"
    )
    assert datasets.parquet_path(FILES, "all", "validation").endswith(
        "validation-00000-of-00001.parquet"
    )


def test_a_split_the_repository_does_not_ship_is_named(client: TestClient) -> None:
    response = client.post(
        "/admin/benchmarks/datasets/preview", json={**MMLU_BODY, "split": "train"}
    )

    assert response.status_code == 400
    assert "train" in response.json()["detail"]


def test_the_columns_column_is_json_text_in_the_row(client: TestClient, tmp_path: Path) -> None:
    """It is a mapping whose keys are the scoring's vocabulary; a table under it would make
    every new role a migration."""
    client.post("/admin/benchmarks/datasets", json=MMLU_BODY)

    stored = Store(tmp_path / "server.db").dataset("mine")

    assert stored is not None
    assert json.loads(stored.columns)["choices"] == "choices"

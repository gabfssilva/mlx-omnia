"""What the comparison band is drawn from: a run, reduced to the axes of its kind."""

from __future__ import annotations

from dataclasses import dataclass

from mlx_omnia.app.api.benchmarks import (
    Run,
    human,
)
from mlx_omnia.app.api.engine import CatalogEntry
from mlx_omnia.app.ui.band import Axis

TITLES = {"speed": "Speed", "fidelity": "Fidelity", "quality": "Quality"}


def speed_axes(context: int, generate: int, concurrency: int) -> list[Axis]:
    streams = f"{concurrency} stream{'s' if concurrency > 1 else ''}"
    return [
        Axis("load", "Load", "warm cache", " s", 1, down=True),
        Axis("pre", "Prefill", f"{streams} · {human(context)}", "", 0),
        Axis("ttft", "TTFT", f"{streams} · p50", " ms", 0, down=True),
        Axis("dec", "Decode", f"{streams} · {human(context)}→{generate}", "", 1),
    ]


def speed_values(run: Run | None) -> dict[str, float | None]:
    body = None if run is None else run["speed"]
    return {
        "load": None if body is None else body["load_s"],
        "pre": None if body is None else body["prefill_tps"],
        "ttft": None if body is None else body["ttft_p50_ms"],
        "dec": None if body is None else body["decode_tps"],
    }


def fidelity_axes(corpus: str) -> list[Axis]:
    return [
        Axis("kl", "KL", corpus, "", 3, down=True),
        Axis("kl95", "KL p95", "tail", "", 3, down=True),
        Axis("top1", "Top-1", "agreement", "%", 1),
        Axis("ppl", "Perplexity", "vs reference", "%", 1, down=True),
    ]


def fidelity_values(run: Run | None) -> dict[str, float | None]:
    body = None if run is None else run["fidelity"]
    top1 = None if body is None else body["top1"]
    return {
        "kl": None if body is None else body["kl_mean"],
        "kl95": None if body is None else body["kl_p95"],
        "top1": None if top1 is None else top1 * 100,
        "ppl": None if body is None else body["ppl_delta"],
    }


def quality_axes(datasets: list[tuple[str, str]]) -> list[Axis]:
    return [Axis(name, name, key, "%", 1) for name, key in datasets]


def quality_values(runs: list[Run]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for run in runs:
        body = run["quality"]
        if body is not None:
            values[body["dataset"]] = (
                None if body["accuracy"] is None else body["accuracy"] * 100
            )
    return values


@dataclass
class Candidate:
    entry: CatalogEntry
    run: Run | None
    # Same vocabulary, other architecture. It computes and it does not mean what this view
    # means — a warning, never a refusal.
    other_shape: bool


def split(
    catalog: list[CatalogEntry], reference: str, runs: list[Run]
) -> tuple[list[Candidate], list[Candidate], list[CatalogEntry]]:
    referred = next((entry for entry in catalog if entry["id"] == reference), None)
    by_model = {run["model"]: run for run in runs}
    measured: list[Candidate] = []
    comparable: list[Candidate] = []
    incompatible: list[CatalogEntry] = []
    for entry in catalog:
        if entry["id"] == reference:
            continue
        same_vocab = (
            referred is None
            or entry["vocab_size"] is None
            or referred["vocab_size"] is None
            or entry["vocab_size"] == referred["vocab_size"]
        )
        if not same_vocab:
            incompatible.append(entry)
            continue
        found = by_model.get(entry["id"])
        candidate = Candidate(
            entry,
            found,
            referred is not None
            and entry["shape"] is not None
            and referred["shape"] is not None
            and entry["shape"] != referred["shape"],
        )
        if found is not None and found["state"] == "ok":
            measured.append(candidate)
        else:
            comparable.append(candidate)
    measured.sort(
        key=lambda one: (one.run or {}).get("fidelity", {}).get("kl_mean") or 0.0  # type: ignore[union-attr]
    )
    return measured, comparable, incompatible

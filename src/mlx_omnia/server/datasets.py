"""Dataset definitions for the quality view, and the preview that checks one before it runs.

A definition is data, not code. MMLU, ARC, HellaSwag, GSM8K and wikitext are rows of
`benchmark_datasets`, seeded by the v4 migration — not branches in a scoring function — so
declaring a sixth is a POST and not a release. What a row says is where the rows come from
(`repo`, `config`, `split`), what they are for (`use`), which of the repository's columns
fill which role (`columns`), and how a row becomes a prompt (`template`).

`preview` is why the form is worth having. A column mapped to the wrong name is a silent
error: the run completes, twenty minutes later, at chance accuracy, and nothing says why.
The preview reads the parquet header and the first rows, applies the template, and answers
with the prompt as the model would see it and the continuations that would be scored — so a
wrong mapping fails in a second, by name, with the columns the repository actually ships.

Parquet through `pyarrow` rather than the Hub's rows API (decision of task 59.8): the file
is pulled once by `hf_hub_download` into the cache the daemon already uses, and every read
after it is local. The rows API needs no dependency and needs the network on every call,
including for the run itself — and a benchmark that stops because a web service rate-limited
it is not a benchmark.
"""

import json
import time
from collections.abc import Mapping, Sequence
from typing import Literal

import pyarrow.parquet as parquet
from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.deps import StoreDep
from mlx_omnia.server.store import Dataset, DatasetUse

_PREVIEW_ROWS = 8
"""Read off the first row group, which is what makes the preview a second and not a
download of the whole split."""


class DatasetBody(BaseModel):
    """What a definition is, over the wire. `columns` maps the roles the template and the
    scoring name — `question`, `choices`, `answer`, `group`, `text` — onto the columns the
    repository ships, and a dotted name reaches into a struct (`choices.text` is how ARC
    keeps its options)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    use: Literal["multiple_choice", "generation", "corpus"]
    repo: str = Field(min_length=1)
    split: str = Field(min_length=1)
    template: str = Field(min_length=1)
    config: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    size: int | None = None


class PreviewBody(DatasetBody):
    """The same body, before it is saved — the form previews what it is about to write, not
    what is already written."""


class Preview(BaseModel):
    prompt: str
    """The first item rendered, as the model would read it."""
    continuations: list[str]
    """What would be scored against that prompt. Empty for a corpus, which is read and not
    answered."""
    answer: str | None
    columns: list[str]
    """Everything the parquet header declares, so a mapping can be fixed from the response
    that refused it."""
    rows: int


def _files(repo: str) -> list[str]:
    try:
        return HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception as error:
        raise HTTPException(
            status_code=404, detail=f"{repo!r} is not a dataset repository the Hub answers for"
        ) from error


def parquet_path(files: Sequence[str], config: str | None, split: str) -> str:
    """Which file holds this split. The layouts in circulation put the config in a
    directory (`all/test-00000-of-00001.parquet`), in the file name, or nowhere at all when
    the dataset has one config — so the match is on containment and the earliest name wins,
    which is shard zero."""
    candidates = [name for name in files if name.endswith(".parquet") and split in name]
    if config is not None:
        narrowed = [name for name in candidates if config in name]
        candidates = narrowed or candidates
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=f"no parquet file for split {split!r}"
            + (f" of config {config!r}" if config else ""),
        )
    return sorted(candidates)[0]


def pluck(row: Mapping[str, object], path: str, columns: Sequence[str]) -> object:
    """One column, by the name the definition gave it, reaching into a struct through the
    dots. A name the header does not carry is a 400 listing what it does carry — never a
    `KeyError` from inside a run."""
    head, _, rest = path.partition(".")
    if head not in row:
        raise HTTPException(
            status_code=400,
            detail=f"{path!r} is not a column of this split: {sorted(columns)}",
        )
    value = row[head]
    for step in rest.split(".") if rest else []:
        if not isinstance(value, dict) or step not in value:
            raise HTTPException(
                status_code=400, detail=f"{path!r} does not reach into {head!r}: {value!r}"
            )
        value = value[step]
    return value


def render(
    template: str, row: Mapping[str, object], columns: Mapping[str, str], header: Sequence[str]
) -> str:
    """The template's own placeholders, filled from the mapped columns. A placeholder no
    role answers for is refused by name: silently leaving `{context}` in the prompt is a
    prompt the model reads literally."""
    filled = template
    for role, column in columns.items():
        placeholder = "{" + role + "}"
        if placeholder in filled:
            filled = filled.replace(placeholder, str(pluck(row, column, header)))
    if "{" in filled and "}" in filled[filled.index("{") :]:
        unfilled = filled[filled.index("{") : filled.index("}", filled.index("{")) + 1]
        raise HTTPException(
            status_code=400,
            detail=f"{unfilled} has no column mapped to it: mapped roles are {sorted(columns)}",
        )
    return filled


def _continuations(
    use: DatasetUse, row: Mapping[str, object], columns: Mapping[str, str], header: Sequence[str]
) -> list[str]:
    match use:
        case "multiple_choice":
            named = columns.get("choices")
            if named is None:
                raise HTTPException(
                    status_code=400,
                    detail="a multiple-choice dataset needs a 'choices' column mapped",
                )
            options = pluck(row, named, header)
            if not isinstance(options, list):
                raise HTTPException(
                    status_code=400, detail=f"{named!r} is not a list of options: {options!r}"
                )
            return [f" {option}" for option in options]
        case "generation":
            named = columns.get("answer")
            return [] if named is None else [f" {pluck(row, named, header)}"]
        case "corpus":
            return []


router = APIRouter()


def _row(dataset: Dataset) -> DatasetBody:
    return DatasetBody(
        id=dataset.id,
        use=dataset.use,
        repo=dataset.repo,
        split=dataset.split,
        template=dataset.template,
        config=dataset.config,
        columns=json.loads(dataset.columns),
        size=dataset.size,
    )


class DatasetView(DatasetBody):
    builtin: bool
    """Seeded by the migration. It changes nothing about how it is read — the flag is what
    lets the form say which rows came with the daemon."""


@router.get("/admin/benchmarks/datasets")
def listing(store: StoreDep) -> list[DatasetView]:
    return [
        DatasetView(**_row(dataset).model_dump(), builtin=dataset.builtin)
        for dataset in store.datasets()
    ]


@router.get("/admin/benchmarks/datasets/{dataset_id}")
def one(dataset_id: str, store: StoreDep) -> DatasetView:
    dataset = store.dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset {dataset_id!r}")
    return DatasetView(**_row(dataset).model_dump(), builtin=dataset.builtin)


@router.post("/admin/benchmarks/datasets", status_code=201)
def save(body: DatasetBody, store: StoreDep) -> DatasetView:
    """Create or replace. A definition is keyed by the id it was declared under, and that id
    is what the key of every run made with it carries — replacing one under a name that has
    already been measured is what the history is there to make visible."""
    existing = store.dataset(body.id)
    if existing is not None and existing.builtin:
        raise HTTPException(
            status_code=409,
            detail=f"{body.id!r} is one of the definitions the daemon ships: declare yours"
            " under another id",
        )
    dataset = Dataset(
        id=body.id,
        use=body.use,
        repo=body.repo,
        config=body.config,
        split=body.split,
        columns=json.dumps(body.columns),
        template=body.template,
        size=body.size,
        builtin=False,
        created_at=time.time(),
    )
    store.save_dataset(dataset)
    return DatasetView(**_row(dataset).model_dump(), builtin=False)


@router.delete("/admin/benchmarks/datasets/{dataset_id}", status_code=204)
def remove(dataset_id: str, store: StoreDep) -> None:
    """The runs made with it stay: a measurement is history, and the definition it was taken
    under is already inside its key."""
    if not store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"no dataset {dataset_id!r}")


@router.post("/admin/benchmarks/datasets/preview")
def preview(body: PreviewBody) -> Preview:
    """The first item, rendered. Sync like the catalog's own handlers: it is a download and
    a parquet read, which is the threadpool's work and not the loop's."""
    path = hf_hub_download(
        body.repo, parquet_path(_files(body.repo), body.config, body.split), repo_type="dataset"
    )
    table = parquet.ParquetFile(path)
    header = list(table.schema_arrow.names)
    batch = next(table.iter_batches(batch_size=_PREVIEW_ROWS))
    rows = batch.to_pylist()
    if not rows:
        raise HTTPException(status_code=400, detail=f"{body.repo!r} split {body.split!r} is empty")
    first = rows[0]
    answer = body.columns.get("answer")
    return Preview(
        prompt=render(body.template, first, body.columns, header),
        continuations=_continuations(body.use, first, body.columns, header),
        answer=None if answer is None else str(pluck(first, answer, header)),
        columns=header,
        rows=table.metadata.num_rows,
    )

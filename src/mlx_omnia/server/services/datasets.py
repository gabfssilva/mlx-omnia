"""Dataset definitions for the quality view, and the preview that checks one before it runs.

A definition is data, not code: MMLU, ARC, HellaSwag, GSM8K and wikitext are rows of
`benchmark_datasets` seeded by the migration, so declaring a sixth is a POST and not a release.
What a row says is where the rows come from (`repo`, `config`, `split`), what they are for
(`use`), which of the repository's columns fill which role (`columns`), and how a row becomes a
prompt (`template`).

`preview` is why the form is worth having. A column mapped to the wrong name is a silent error:
the run completes, twenty minutes later, at chance accuracy. The preview reads the parquet
header and the first rows, applies the template, and answers with the prompt as the model would
see it — so a wrong mapping fails in a second, by name.

Parquet through `pyarrow` rather than the Hub's rows API: the file is pulled once into the cache
the daemon already uses, and every read after it is local.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pyarrow.parquet as parquet
from huggingface_hub import HfApi, hf_hub_download

from mlx_omnia.server.db.models.datasets import Dataset, DatasetUse

_PREVIEW_ROWS = 8
"""Read off the first row group, which is what makes the preview a second and not a download of
the whole split."""


class Invalid(Exception):
    """The definition cannot be applied to the split it names."""


class Unknown(Exception):
    """No dataset, or no repository the Hub answers for."""


class Conflict(Exception):
    """A definition the daemon ships, under an id somebody else asked for."""


@dataclass(frozen=True)
class Definition:
    """What a definition is, as a domain value. `columns` maps the roles the template and the
    scoring name — `question`, `choices`, `answer`, `group`, `text` — onto the columns the
    repository ships, and a dotted name reaches into a struct (`choices.text` is how ARC keeps
    its options)."""

    id: str
    use: DatasetUse
    repo: str
    split: str
    template: str
    config: str | None = None
    columns: Mapping[str, str] = field(default_factory=dict)
    size: int | None = None


@dataclass(frozen=True)
class Preview:
    prompt: str
    """The first item rendered, as the model would read it."""
    continuations: list[str]
    """What would be scored against that prompt. Empty for a corpus, which is read and not
    answered."""
    answer: str | None
    columns: list[str]
    """Everything the parquet header declares, so a mapping can be fixed from the response that
    refused it."""
    rows: int


def _files(repo: str) -> list[str]:
    try:
        return HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception as error:
        raise Unknown(f"{repo!r} is not a dataset repository the Hub answers for") from error


def parquet_path(files: Sequence[str], config: str | None, split: str) -> str:
    """Which file holds this split. The layouts in circulation put the config in a directory, in
    the file name, or nowhere at all when the dataset has one config — so the match is on
    containment and the earliest name wins, which is shard zero."""
    candidates = [name for name in files if name.endswith(".parquet") and split in name]
    if config is not None:
        narrowed = [name for name in candidates if config in name]
        candidates = narrowed or candidates
    if not candidates:
        raise Invalid(
            f"no parquet file for split {split!r}" + (f" of config {config!r}" if config else "")
        )
    return sorted(candidates)[0]


def pluck(row: Mapping[str, object], path: str, columns: Sequence[str]) -> object:
    """One column, by the name the definition gave it, reaching into a struct through the dots. A
    name the header does not carry is refused listing what it does carry — never a `KeyError`
    from inside a run."""
    head, _, rest = path.partition(".")
    if head not in row:
        raise Invalid(f"{path!r} is not a column of this split: {sorted(columns)}")
    value = row[head]
    for step in rest.split(".") if rest else []:
        if not isinstance(value, dict) or step not in value:
            raise Invalid(f"{path!r} does not reach into {head!r}: {value!r}")
        value = value[step]
    return value


def render(
    template: str, row: Mapping[str, object], columns: Mapping[str, str], header: Sequence[str]
) -> str:
    """The template's own placeholders, filled from the mapped columns. A placeholder no role
    answers for is refused by name: silently leaving `{context}` in the prompt is a prompt the
    model reads literally."""
    filled = template
    for role, column in columns.items():
        placeholder = "{" + role + "}"
        if placeholder in filled:
            filled = filled.replace(placeholder, str(pluck(row, column, header)))
    if "{" in filled and "}" in filled[filled.index("{") :]:
        unfilled = filled[filled.index("{") : filled.index("}", filled.index("{")) + 1]
        raise Invalid(f"{unfilled} has no column mapped to it: mapped roles are {sorted(columns)}")
    return filled


def continuations(
    use: DatasetUse, row: Mapping[str, object], columns: Mapping[str, str], header: Sequence[str]
) -> list[str]:
    match use:
        case "multiple_choice":
            named = columns.get("choices")
            if named is None:
                raise Invalid("a multiple-choice dataset needs a 'choices' column mapped")
            options = pluck(row, named, header)
            if not isinstance(options, list):
                raise Invalid(f"{named!r} is not a list of options: {options!r}")
            return [f" {option}" for option in options]
        case "generation":
            named = columns.get("answer")
            return [] if named is None else [f" {pluck(row, named, header)}"]
        case "corpus":
            return []


def _use(value: str) -> DatasetUse:
    match value:
        case "multiple_choice" | "generation" | "corpus":
            return value
        case _:
            raise ValueError(f"{value!r} is not a dataset use")


def definition_of(dataset: Dataset) -> Definition:
    columns: object = json.loads(dataset.columns)
    return Definition(
        id=dataset.id,
        use=_use(dataset.use),
        repo=dataset.repo,
        split=dataset.split,
        template=dataset.template,
        config=dataset.config,
        columns={
            str(role): str(column)
            for role, column in (columns.items() if isinstance(columns, dict) else ())
        },
        size=dataset.size,
    )


async def listing() -> list[Dataset]:
    return list(await Dataset.objects.order_by("id").all())


async def one(dataset_id: str) -> Dataset | None:
    return await Dataset.objects.get_or_none(id=dataset_id)


async def save(definition: Definition) -> Dataset:
    """Create or replace. A definition is keyed by the id it was declared under, and that id is
    what the key of every run made with it carries — replacing one under a name that has already
    been measured is what the history is there to make visible."""
    existing = await one(definition.id)
    if existing is not None and existing.builtin:
        raise Conflict(
            f"{definition.id!r} is one of the definitions the daemon ships: declare yours"
            " under another id"
        )
    dataset = Dataset(
        id=definition.id,
        use=definition.use,
        repo=definition.repo,
        config=definition.config,
        split=definition.split,
        columns=json.dumps(dict(definition.columns)),
        template=definition.template,
        size=definition.size,
        builtin=False,
        created_at=time.time(),
    )
    if existing is not None:
        await Dataset.objects.delete(id=definition.id)
    await dataset.save()
    return dataset


async def delete(dataset_id: str) -> bool:
    """The runs made with it stay: a measurement is history, and the definition it was taken
    under is already inside its key."""
    return await Dataset.objects.delete(id=dataset_id) == 1


def preview(definition: Definition) -> Preview:
    """The first item, rendered. Blocking: it is a download and a parquet read."""
    path = hf_hub_download(
        definition.repo,
        parquet_path(_files(definition.repo), definition.config, definition.split),
        repo_type="dataset",
    )
    table = parquet.ParquetFile(path)
    header = list(table.schema_arrow.names)
    batch = next(table.iter_batches(batch_size=_PREVIEW_ROWS))
    rows: list[dict[str, object]] = batch.to_pylist()
    if not rows:
        raise Invalid(f"{definition.repo!r} split {definition.split!r} is empty")
    first = rows[0]
    answer = definition.columns.get("answer")
    return Preview(
        prompt=render(definition.template, first, definition.columns, header),
        continuations=continuations(definition.use, first, definition.columns, header),
        answer=None if answer is None else str(pluck(first, answer, header)),
        columns=header,
        rows=table.metadata.num_rows,
    )

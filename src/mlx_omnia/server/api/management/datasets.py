"""The dataset definitions a quality benchmark is measured over."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.services import datasets

router = APIRouter()


class DatasetBody(BaseModel):
    """What a definition is, over the wire. `columns` maps the roles the template and the
    scoring name onto the columns the repository ships, and a dotted name reaches into a
    struct."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    use: Literal["multiple_choice", "generation", "corpus"]
    repo: str = Field(min_length=1)
    split: str = Field(min_length=1)
    template: str = Field(min_length=1)
    config: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    size: int | None = None

    def definition(self) -> datasets.Definition:
        return datasets.Definition(
            id=self.id,
            use=self.use,
            repo=self.repo,
            split=self.split,
            template=self.template,
            config=self.config,
            columns=self.columns,
            size=self.size,
        )


class PreviewBody(DatasetBody):
    """The same body, before it is saved — the form previews what it is about to write."""


class DatasetView(DatasetBody):
    builtin: bool
    """Seeded by the migration. The flag is what lets the form say which rows came with the
    daemon."""


def _dataset_view(dataset: datasets.Dataset) -> DatasetView:
    found = datasets.definition_of(dataset)
    return DatasetView(
        id=found.id,
        use=found.use,
        repo=found.repo,
        split=found.split,
        template=found.template,
        config=found.config,
        columns=dict(found.columns),
        size=found.size,
        builtin=dataset.builtin,
    )


@router.get("/admin/benchmarks/datasets")
async def dataset_listing() -> list[DatasetView]:
    return [_dataset_view(dataset) for dataset in await datasets.listing()]


@router.get("/admin/benchmarks/datasets/{dataset_id}")
async def one_dataset(dataset_id: str) -> DatasetView:
    dataset = await datasets.one(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset {dataset_id!r}")
    return _dataset_view(dataset)


@router.post("/admin/benchmarks/datasets", status_code=201)
async def save_dataset(body: DatasetBody) -> DatasetView:
    """Create or replace. A definition is keyed by the id it was declared under, and that id is
    what the key of every run made with it carries."""
    try:
        return _dataset_view(await datasets.save(body.definition()))
    except datasets.Conflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.delete("/admin/benchmarks/datasets/{dataset_id}", status_code=204)
async def remove_dataset(dataset_id: str) -> None:
    """The runs made with it stay: a measurement is history, and the definition it was taken
    under is already inside its key."""
    if not await datasets.delete(dataset_id):
        raise HTTPException(status_code=404, detail=f"no dataset {dataset_id!r}")


@router.post("/admin/benchmarks/datasets/preview")
def preview_dataset(body: PreviewBody) -> datasets.Preview:
    """The first item, rendered. Sync like the catalog's own handlers: it is a download and a
    parquet read, which is the threadpool's work and not the loop's."""
    try:
        return datasets.preview(body.definition())
    except datasets.Invalid as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except datasets.Unknown as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

"""The `benchmark_datasets` table: a dataset is data, not code.

MMLU and GSM8K are rows seeded by the initial migration, not branches in a scoring function.
`columns` maps the roles a template names onto the columns the repository ships, as JSON text.
"""

from __future__ import annotations

from typing import Literal

import ormar

from mlx_omnia.server.db import base

DatasetUse = Literal["multiple_choice", "generation", "corpus"]

DATASET_USES: tuple[DatasetUse, ...] = ("multiple_choice", "generation", "corpus")


class Dataset(ormar.Model):
    ormar_config = base.ormar_config.copy(tablename="benchmark_datasets")

    id: str = ormar.Text(primary_key=True)
    use: str = ormar.Text(choices=list(DATASET_USES))
    repo: str = ormar.Text()
    config: str | None = ormar.Text(nullable=True)
    split: str = ormar.Text()
    columns: str = ormar.Text(default="{}")
    template: str = ormar.Text()
    size: int | None = ormar.Integer(nullable=True)
    builtin: bool = ormar.Boolean(default=False)
    created_at: float = ormar.Float()

"""The `prefix_cache` table: the index over the prefix spans spilled to disk.

The row is the index and the weight is a file. `key` is the chained digest of everything that
changes what the bytes mean; `model` is denormalised beside it so forgetting one model's
conversations is one delete and one rmtree. `used_at` moves on a hit, which is what makes the
eviction order a use and not an age.

The file keys the table by `(key, kind)`, and ormar carries one primary key. It is sqlite's
own `rowid` rather than `key`: ormar folds rows that share a primary key into one, and a span
and its anchor share a key — reading them back through `key` would answer with one of the two
and price the tier at half of what is on the disk. Writes still filter on `(key, kind)`.
"""

from __future__ import annotations

import ormar

from mlx_omnia.server.db import base


class PrefixCacheFile(ormar.Model):
    """One payload of one span. `kind` is `rows` for the span itself, `anchor` for the state
    that stood on its far boundary."""

    ormar_config = base.ormar_config.copy(tablename="prefix_cache")

    rowid: int | None = ormar.Integer(primary_key=True, autoincrement=True, nullable=True)
    key: str = ormar.Text()
    kind: str = ormar.Text()
    model: str = ormar.Text()
    path: str = ormar.Text()
    bytes: int = ormar.Integer()
    created_at: float = ormar.Float()
    used_at: float = ormar.Float()

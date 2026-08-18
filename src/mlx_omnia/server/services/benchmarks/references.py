"""The fidelity reference cache: the index, its lifecycle, and the reuse of an entry already
on disk."""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

from mlx_omnia.server.db.models.benchmarks import ReferenceCache
from mlx_omnia.server.services.benchmarks.specs import REFERENCE_CACHE, TOPK


def reference_key(reference: str, corpus: str, tokens: int, seed: int) -> str:
    """What decides whether a pass has already been paid for. The model id goes through a
    digest because it carries slashes and is a path here."""
    digest = hashlib.sha256(f"{reference}\0{corpus}\0{tokens}\0{seed}".encode()).hexdigest()
    return digest[:16]


def reference_path(reference: str, corpus: str, tokens: int, seed: int) -> Path:
    return REFERENCE_CACHE / f"{reference_key(reference, corpus, tokens, seed)}.npz"


async def references() -> list[ReferenceCache]:
    return list(await ReferenceCache.objects.order_by(["-created_at", "-id"]).all())


async def reference(entry_id: str) -> ReferenceCache | None:
    return await ReferenceCache.objects.get_or_none(id=entry_id)


async def existing_reference(
    reference_id: str, corpus: str, tokens: int, seed: int
) -> ReferenceCache | None:
    """The entry a run would reuse, or `None` — which is what says the expensive pass has to
    happen. An index row whose file is gone is not an entry: the disk is the truth, and the row
    is dropped so the next pass rebuilds it."""
    key = reference_key(reference_id, corpus, tokens, seed)
    for entry in await references():
        if reference_key(entry.reference, entry.corpus, entry.tokens, entry.seed) != key:
            continue
        if Path(entry.path).is_file():
            return entry
        await delete_reference(entry.id)
        return None
    return None


async def save_reference(
    reference_id: str, corpus: str, tokens: int, seed: int, path: Path
) -> ReferenceCache:
    """The index row for a file that is already written. Separate from the pass that writes it
    so that pass — which does not exist yet — has one thing to call."""
    entry = ReferenceCache(
        id=uuid.uuid4().hex,
        reference=reference_id,
        corpus=corpus,
        tokens=tokens,
        seed=seed,
        topk=TOPK,
        path=str(path),
        bytes=path.stat().st_size,
        created_at=time.time(),
    )
    await entry.save()
    return entry


async def build_reference(reference_id: str, corpus: str, tokens: int, seed: int) -> ReferenceCache:
    """The expensive pass: read the corpus, run the reference teacher-forced over it, and keep
    the top-k logits plus the logsumexp per position.

    Not implemented — it is the other half of the `# TODO` in `work`. Reusing an entry already
    on disk is, because that is the part the routes have to be honest about.
    """
    found = await existing_reference(reference_id, corpus, tokens, seed)
    if found is not None:
        return found
    # TODO(59.9): the teacher-forced pass. Read `tokens` tokens of `corpus` through the dataset
    # definition, run `reference_id` over them with the cache the engine already builds, and
    # write `mx.savez` of the top-`TOPK` ids, their logits and the logsumexp per position to
    # `reference_path(...)`; then `save_reference(...)`.
    raise NotImplementedError(
        "building a reference's logit cache is not implemented: the teacher-forced pass is"
        " task 59.9. What exists is the index, its lifecycle and the reuse of an entry"
        " already on disk."
    )


async def delete_reference(entry_id: str) -> bool:
    return await ReferenceCache.objects.delete(id=entry_id) == 1


async def discard_reference(entry_id: str) -> bool:
    """The row and the file. Dropping one without the other is either a leak on disk or a hit
    that reads a path that is gone."""
    entry = await reference(entry_id)
    if entry is None:
        return False
    Path(entry.path).unlink(missing_ok=True)
    return await delete_reference(entry_id)

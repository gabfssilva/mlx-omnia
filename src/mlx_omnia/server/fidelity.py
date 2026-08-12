"""The reference side of the fidelity view: one expensive pass, kept, so every candidate
after the first is cheap.

Fidelity asks how far a checkpoint drifted from the one it was made out of, and it asks it
of the logits rather than of the answers: KL against the reference, top-1 agreement, top-5
overlap, delta perplexity, teacher-forced over a fixed corpus with nothing generated. A
3-bit can lose 0.3 points of MMLU — inside the noise of 1400 items — and change its first
choice on one token in ten, and only this side of the section sees that.

The reference is read once per `(reference, corpus, tokens, seed)` and kept as **top-64 plus
the logsumexp** per position. That is exact enough for KL and for top-1 agreement, and it
turns "compare five quantizations" into one expensive pass and five cheap ones. The full
`relative_diff` over the whole vector needs both models in one pass and stays where it
already lives: the parity tests.

The weights are on disk and not in sqlite. Five megabytes a reference does not break the
database, but it bloats the one file a user backs up — so the row is an index and the array
is an `.npz` under `~/.cache/mlx_omnia/fidelity/`, and `DELETE` unlinks both. A cache nobody
can see is rubbish accumulating in silence; that is why the listing exists and reports what
it weighs.
"""

import hashlib
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlx_omnia.server.profiles import StoreDep
from mlx_omnia.server.store import ReferenceCache, Store

CACHE = Path.home() / ".cache" / "mlx_omnia" / "fidelity"

TOPK = 64
"""Kept per position. Wide enough that the tail outside it contributes to KL only through
the logsumexp, which is kept beside it."""


def cache_key(reference: str, corpus: str, tokens: int, seed: int) -> str:
    """What decides whether a pass has already been paid for. The model id goes through a
    digest because it carries slashes and is a path here."""
    digest = hashlib.sha256(f"{reference}\0{corpus}\0{tokens}\0{seed}".encode()).hexdigest()
    return digest[:16]


def cache_path(reference: str, corpus: str, tokens: int, seed: int) -> Path:
    return CACHE / f"{cache_key(reference, corpus, tokens, seed)}.npz"


def existing(
    store: Store, reference: str, corpus: str, tokens: int, seed: int
) -> ReferenceCache | None:
    """The entry a run would reuse, or `None` — which is what says the expensive pass has to
    happen. An index row whose file is gone is not an entry: the disk is the truth, and the
    row is dropped so the next pass rebuilds it instead of reading a path that is not there."""
    key = cache_key(reference, corpus, tokens, seed)
    for entry in store.references():
        if cache_key(entry.reference, entry.corpus, entry.tokens, entry.seed) != key:
            continue
        if Path(entry.path).is_file():
            return entry
        store.delete_reference(entry.id)
        return None
    return None


def record(
    store: Store, reference: str, corpus: str, tokens: int, seed: int, path: Path
) -> ReferenceCache:
    """The index row for a file that is already written. Separate from the pass that writes
    it so the pass — which does not exist yet — has one thing to call."""
    entry = ReferenceCache(
        id=uuid.uuid4().hex,
        reference=reference,
        corpus=corpus,
        tokens=tokens,
        seed=seed,
        topk=TOPK,
        path=str(path),
        bytes=path.stat().st_size,
        created_at=time.time(),
    )
    store.save_reference(entry)
    return entry


def build(store: Store, reference: str, corpus: str, tokens: int, seed: int) -> ReferenceCache:
    """The expensive pass: read the corpus, run the reference teacher-forced over it, and
    keep the top-k logits plus the logsumexp per position.

    Not implemented — it is the other half of the `# TODO` in `benchmarks._work`. Reusing an
    entry that is already on disk is implemented, because that is the part the routes below
    have to be honest about.
    """
    found = existing(store, reference, corpus, tokens, seed)
    if found is not None:
        return found
    # TODO(59.9): the teacher-forced pass. Read `tokens` tokens of `corpus` through the
    # dataset definition, run `reference` over them with the cache the engine already
    # builds, and write `mx.savez` of the top-`TOPK` ids, their logits and the logsumexp per
    # position to `cache_path(...)`; then `record(...)`. The comparison that turns two of
    # these into KL, top-1, top-5 and Δppl belongs beside it.
    raise NotImplementedError(
        "building a reference's logit cache is not implemented: the teacher-forced pass is"
        " task 59.9. What exists is the index, its lifecycle and the reuse of an entry"
        " already on disk."
    )


class ReferenceView(BaseModel):
    id: str
    reference: str
    corpus: str
    tokens: int
    seed: int
    topk: int
    bytes: int
    created_at: float
    present: bool
    """Whether the file the row points at is still there. A row without its file is not a
    cache hit, and saying so beats a listing that reports space nothing occupies."""


class References(BaseModel):
    entries: list[ReferenceView]
    total_bytes: int
    """What the cache costs on disk, summed — the number that makes the purge button
    something other than a guess."""


router = APIRouter()


@router.get("/admin/benchmarks/references")
def listing(store: StoreDep) -> References:
    entries = [
        ReferenceView(
            id=entry.id,
            reference=entry.reference,
            corpus=entry.corpus,
            tokens=entry.tokens,
            seed=entry.seed,
            topk=entry.topk,
            bytes=entry.bytes,
            created_at=entry.created_at,
            present=Path(entry.path).is_file(),
        )
        for entry in store.references()
    ]
    return References(
        entries=entries,
        total_bytes=sum(entry.bytes for entry in entries if entry.present),
    )


@router.delete("/admin/benchmarks/references/{entry_id}", status_code=204)
def remove(entry_id: str, store: StoreDep) -> None:
    """The row and the file. Dropping one without the other is either a leak on disk or a
    hit that reads a path that is gone."""
    entry = store.reference(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no reference cache {entry_id!r}")
    Path(entry.path).unlink(missing_ok=True)
    store.delete_reference(entry_id)

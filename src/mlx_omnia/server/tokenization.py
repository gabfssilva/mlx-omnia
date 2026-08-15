"""`POST /admin/models/{id}/tokenize`: what this model's tokenizer makes of that text.

RPC, and APP.md already assumes it as one — nothing is created, and there is nothing to
GET afterwards. The ids come from the loaded model rather than from `tokenizer.json`: which
tokenizer a checkpoint gets is decided by its architecture (Gemma 3 reads that same file
with one of its own), and a second dispatch here would be a table to keep in sync with the
loaders'.

Only a resident model answers, and a name that is not resident is refused rather than
loaded. The caller is a chat window counting its context as the user types; a load is
seconds and tens of gigabytes, and it is asked for on purpose through
`PUT /admin/models/{id}/residency`, never as the side effect of counting.
"""

import asyncio
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from mlx_omnia.engine.language import tokenizer_of
from mlx_omnia.server.deps import EngineDep


class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


@dataclass(frozen=True)
class Tokens:
    ids: list[int]


router = APIRouter()


@router.post("/admin/models/{model_id:path}/tokenize")
async def tokenize(model_id: str, body: TokenizeRequest, engine: EngineDep) -> Tokens:
    """`resolve` is reached only for a model the check above found resident, and for one of
    those it returns before it can suspend: the check and the lookup are the same step of
    the loop, so this route cannot become the load it just refused.

    The encode goes to a thread. The BPE is Python, a full context is hundreds of thousands
    of characters, and the loop it would otherwise sit on is the one carrying a token
    stream.
    """
    if model_id not in engine.resident:
        raise HTTPException(
            status_code=409,
            detail=f"{model_id!r} is not resident: load it before tokenizing with it",
        )
    tokenizer = tokenizer_of(await engine.resolve(model_id))
    if tokenizer is None:
        raise HTTPException(status_code=500, detail=f"{model_id!r} exposes no tokenizer")
    # Read out in the thread and not after it: the tokenizer hands the ids over as it
    # makes them, so `to_thread` on `encode` alone would move a generator and leave the
    # work on the loop.
    def encode() -> list[int]:
        return list(tokenizer.encode(body.text))

    return Tokens(ids=await asyncio.to_thread(encode))

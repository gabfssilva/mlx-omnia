"""The mlx-lm arm: the external baseline, over the same checkpoint as everything else.

`generate_step` yields `(id, logprobs)` and the id is handed on unread — the timed loop counts
and never converts, so this arm pays exactly the synchronizations mlx-lm's own loop pays and
no more.
"""

from collections.abc import Iterator, Sequence

import mlx.core as mx

from mlx_omnia.bench.arm import Arm, TokenId, arm
from mlx_omnia.bench.forcing import forced

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def build(
    repo: str, *, tokens: int = 128, dtype: str | None = None, name: str = "mlx-lm"
) -> Arm:
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    # `load` answers a 2-tuple or a 3-tuple depending on what it was asked for; the model is
    # the first either way, and unpacking would have to know which.
    model = load(repo)[0]
    if dtype is not None:
        model.set_dtype(DTYPES[dtype])
        mx.eval(model.parameters())

    def generate(
        prompt: Sequence[int], script: Sequence[int] | None, limit: int
    ) -> Iterator[int | TokenId]:
        sampler = None if script is None else forced(script)
        for drawn, _ in generate_step(mx.array(list(prompt)), model, sampler=sampler):
            yield drawn

    return arm(name, generate, tokens=tokens)

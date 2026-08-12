"""The universal embed strategy: the three copies as the three gathers they already are.

`build` accepts every embedding and every atlas pair, so it registers last and makes the
delegator total. The token's row comes out of the embedding table by index, and each angle
row out of its atlas flattened to `[positions, width]` — the same three pure copies the
fused kernel performs, in three dispatches instead of one. No arithmetic on either side,
so the outputs are bit-identical; what the specialization buys is dispatch latency at the
head of a decode step.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx


@dataclass(frozen=True)
class DefaultEmbed:
    embedding: mx.array
    full_atlas: mx.array
    sliding_atlas: mx.array

    @classmethod
    def build(
        cls,
        embedding: mx.array,
        *,
        full_atlas: mx.array,
        sliding_atlas: mx.array,
    ) -> Self:
        return cls(embedding, full_atlas, sliding_atlas)

    def __call__(
        self, tokens: mx.array, position: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        assert tokens.size == 1
        hidden = self.embedding[tokens.reshape(-1)].reshape(-1)
        full = self.full_atlas.reshape(-1, self.full_atlas.shape[-1])[position]
        sliding = self.sliding_atlas.reshape(-1, self.sliding_atlas.shape[-1])[position]
        return hidden, full, sliding

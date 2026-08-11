"""The decode-head embed primitive's contract: what a strategy is and what a model declares.

The primitive: the single decoded token's embedding row plus one row out of each of two
position-indexed RoPE angle atlases — (tokens [1 element], position) -> (hidden [hidden],
full_angles [full_width], sliding_angles [sliding_width]). Three pure copies, no
arithmetic: every strategy is bit-identical to the stock gathers, and what a
specialization removes is dispatches, not work.

Two atlases because a hybrid trunk runs two rope bases (a full-attention one and a
sliding-window one) whose angle rows have different widths; a caller with a single atlas
passes the same array twice or ignores the second output. The atlases themselves —
`[..., positions, width]` fp32, precomputed once — are the caller's, and the position must
be inside them.

One module per specialization implements it; the `Embed` delegator in `__init__.py`
resolves which one serves a given embedding and atlas pair, once, at construction.
"""

from typing import Protocol

import mlx.core as mx


class EmbedStrategy(Protocol):
    """(tokens [1 element], position) -> (hidden [hidden], full_angles [full_width],
    sliding_angles [sliding_width])."""

    def __call__(
        self, tokens: mx.array, position: int
    ) -> tuple[mx.array, mx.array, mx.array]: ...

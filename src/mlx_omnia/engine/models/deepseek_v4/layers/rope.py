from dataclasses import dataclass

import mlx.core as mx

from mlx_omnia.engine.core.rope import YarnJson, yarn


@dataclass(frozen=True)
class Rotary:
    """A rope table already padded to the head's width.

    The rotation covers only the last `qk_rope_head_dim` of a 512-wide head, and
    `mx.fast.rope` rotates the *first* `dims`. Padding the period table with `inf` on the
    leading pairs is what reconciles the two: an infinite period is a zero angle, so those
    dimensions come out untouched and no row permutation is needed. `inverse` is the same
    table with the periods negated — the output de-rotation costs nothing extra.
    """

    forward: mx.array
    inverse: mx.array

    def __call__(
        self, x: mx.array, offset: int | mx.array, *, inverse: bool = False
    ) -> mx.array:
        """`offset` is an array wherever the caller's position lives in a graph. `mx.fast.rope`
        takes either and the two are bit-identical, which is what lets one traced step rotate
        by a position no host read ever settles."""
        return mx.fast.rope(
            x,
            x.shape[-1],
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self.inverse if inverse else self.forward,
        )


def rotary(
    rope_dims: int,
    head_dim: int,
    base: float,
    scaling: YarnJson | None,
    *,
    freq_scale: int = 1,
) -> Rotary:
    """The period table for one head width. YaRN gives the blended table (the house's
    `yarn` computes the same array the reference does); V4 declares no `mscale`, and the
    default one — 1.277 at factor 16 — would scale every q and k it touched."""
    table = yarn(rope_dims, base, scaling).freqs
    if table is None:
        table = base ** (mx.arange(0, rope_dims, 2, dtype=mx.float32) / rope_dims)
    if freq_scale != 1:
        table = table / freq_scale
    pad = mx.full(((head_dim - rope_dims) // 2,), mx.inf)
    return Rotary(mx.concatenate([pad, table]), mx.concatenate([pad, -table]))

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.hyper_connection import FusedHyperConnection
from mlx_omnia.engine.core.kernels.hyper_connection import HyperConnection as HyperJunction
from mlx_omnia.engine.models.deepseek_v4.config import DeepseekV4Config

# The junction the free `hc_expand` delegates to, one entry per `(hc_mult, hidden)`:
# the block re-expands outside the layer that produced `post`/`comb`.
_JUNCTIONS: dict[tuple[int, int], HyperJunction] = {}


def hc_expand(
    x: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
    fn: mx.array | None = None,
) -> tuple[mx.array, mx.array | None]:
    """The sublayer output broadcast back over the copies, plus the copies mixed — and,
    given the next junction's `fn`, that junction's gemv partials on the way out."""
    hc, hidden = residual.shape[-2:]
    return _JUNCTIONS[(hc, hidden)].expand(x, residual, post, comb, fn)


class HyperConnection(nn.Module):
    """One mHC junction: collapse the `hc_mult` residual copies into the sublayer's input,
    and produce the two tensors that re-expand its output.

    `fn` reads the RMS-normed concatenation of the copies and emits three groups: a
    per-copy input gate, a per-copy output gate, and an `hc_mult x hc_mult` mixing matrix
    made doubly stochastic by alternating row and column normalizations (20 of them). All
    of it in fp32; the copies are only cast back at the end.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)
        junction = HyperJunction(
            hc_mult=self.hc_mult,
            hidden=config.hidden_size,
            iters=config.hc_sinkhorn_iters,
            eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )
        _JUNCTIONS.setdefault((self.hc_mult, config.hidden_size), junction)
        self._junction = junction
        self.fused = isinstance(junction.strategy, FusedHyperConnection)

    def __call__(
        self, x: mx.array, norm: nn.RMSNorm, partials: mx.array | None = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """The sublayer's rms-normed input — the collapse only ever feeds `norm` — plus
        the two re-expansion tensors. `partials` are the mixes gemv's partial sums when
        the preceding expansion already computed them."""
        if partials is None:
            partials = (x.flatten(-2) @ self.fn.T)[..., None, :]
        return self._junction(x, partials, self.scale, self.base, norm.weight)


class HyperHead(nn.Module):
    """The last collapse: the trunk's `hc_mult` copies into one hidden state."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.fn = mx.zeros((config.hc_mult, config.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((config.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        mixes = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps) @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)

from collections.abc import Sequence
from typing import NamedTuple, assert_never

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import Draftable, Tracing
from mlx_omnia.engine.core.cache import (
    DeltaCache,
    KVCache,
    LayerCache,
)
from mlx_omnia.engine.core.kernels.add_norm import RowsAddRmsNorm
from mlx_omnia.engine.models.nemotron_h.config import MAMBA, NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.layers.block import NemotronHBlock, NemotronHLayer
from mlx_omnia.engine.models.nemotron_h.layers.mamba import NemotronHMamba


def _joins(
    blocks: Sequence[nn.Module], norm_f: nn.RMSNorm
) -> list[RowsAddRmsNorm] | None:
    """One residual join per block: block `i`'s add fused with the norm that reads the
    sum — block `i + 1`'s for every block but the last, whose sum `norm_f` reads.

    Always the rows kernel and never the single-token fusion: the decode and the
    verification forward must round the join identically, and `RowsAddRmsNorm` is the
    one strategy that serves both shapes with one arithmetic (`add_norm/rows.py` —
    also `mx.fast.rms_norm`'s own, which is what keeps the compiled graphs agreeing
    with prefill). All or nothing: one norm the kernel cannot serve returns None and
    the caller keeps the plain chain."""
    norms: list[nn.RMSNorm] = []
    for block in blocks[1:]:
        assert isinstance(block, NemotronHBlock)
        norms.append(block.norm)
    norms.append(norm_f)
    joins = [RowsAddRmsNorm.build(leaf, tokens=None) for leaf in norms]
    if any(join is None for join in joins):
        return None
    return [join for join in joins if join is not None]


class NemotronHActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class NemotronHTrunk(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NemotronHBlock(config, kind) for kind in config.pattern]
        self.norm_f = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


class NemotronH(nn.Module, Draftable[LayerCache], Tracing[LayerCache]):

    _joins_cache: list[RowsAddRmsNorm] | None
    _joins_resolved: bool

    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        # Off the module's own attributes: `nn.Module.__setattr__` files a list into the
        # parameter tree, and these hold the norms' weights by reference rather than being
        # weights. Resolved on first use and not here, because the leaves this reads are
        # the checkpoint's and the checkpoint has not been loaded yet.
        object.__setattr__(self, "_joins_cache", None)
        object.__setattr__(self, "_joins_resolved", False)
        self.config = config
        self.backbone = NemotronHTrunk(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. The residual joins, resolved outside the graph that bakes
        them: `RowsAddRmsNorm.build` reads a leaf's size to decide applicability, and a
        trace runs its Python once — a join left unresolved here would be chosen against
        whatever the first token happened to see.

        Nothing is captured. The joins hold the norms' weights as frozen fields, and an
        array that is both a declared input and a strategy's field is the uncaptured-input
        error; left out they bake in as constants, which is what a decode wants.

        Both halves of the claim hold without this trunk doing anything: every position it
        reads is `LayerCache.offset` passed straight into `mx.fast.rope` with `rope_dims=0`
        — never a Python int — and the columns a promoted buffer has not written are cut by
        `FixedKVCache.readable` inside `core.attend`.
        """
        del cache
        self.joins()
        return ()

    def make_cache(self) -> list[LayerCache]:
        caches: list[LayerCache] = []
        for kind in self.config.pattern:
            match kind:
                case "M":
                    caches.append(DeltaCache())
                case "*":
                    caches.append(KVCache())
                case "E" | "-":
                    caches.append(LayerCache())
                case _:
                    assert_never(kind)
        return caches

    def raw_embed(self, ids: mx.array) -> mx.array:
        """`core.api.Draftable`. The trunk puts nothing over the lookup here, so the raw
        pair and the cooked one are the same tensors — which is why the protocol names them
        rather than trusting a family to have only one."""
        return self.backbone.embeddings(ids)

    def raw_logits(self, hidden: mx.array) -> mx.array:
        """`core.api.Draftable`. `norm_f` is *not* applied: the rows an MTP step returns
        already went through the step's own `final_layernorm`, and normalizing twice is a
        different model. The trunk's own path applies `norm_f` before calling this."""
        return self.lm_head(hidden)

    def block_outputs(
        self, ids: mx.array, cache: Sequence[LayerCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """`core.api.Draftable`: the same forward, plus the output of blocks `at`
        concatenated on the last dim. What reads it is an MTP step, which asks for one — the
        last block, whose rows are what the step's `hnorm` was trained on."""
        activations = self.activations(ids, cache)
        return activations.logits, mx.concatenate([activations.blocks[index] for index in at], -1)

    def checkpoints(self, cache: Sequence[LayerCache], rows: int) -> bool:
        """`core.api.Draftable`. Yes, when every mamba mixer holds the fused step: that is
        the kernel `NemotronHMamba.verify_rows` writes the per-row states out of, and the
        one thing about a rejected round that is not the cache's own answer.

        Worth having on this trunk in particular. 23 of its 52 layers are recurrent and the
        23 sparse ones hold nothing but an offset — replaying them reads the MoE weights a
        second time, and on this checkpoint that is 57% of the bytes and the reason a
        rejected round would cost more than the token it bought.
        """
        del cache, rows
        from mlx_omnia.engine.core.kernels.mamba_step.fused import FusedMambaStep

        for block, kind in zip(self.backbone.layers, self.config.pattern, strict=True):
            if kind != MAMBA:
                continue
            assert isinstance(block, NemotronHBlock)
            mixer = block.mixer
            assert isinstance(mixer, NemotronHMamba)
            if not isinstance(mixer.mamba_step().strategy, FusedMambaStep):
                return False
        return True

    def joins(self) -> list[RowsAddRmsNorm] | None:
        """The residual joins, resolved once — after load, when the norms' leaves are
        final. `None` is a trunk whose hidden size the rows kernel declines, and the plain
        add-then-norm chain is what it gets."""
        if not self._joins_resolved:
            object.__setattr__(
                self, "_joins_cache", _joins(self.backbone.layers, self.backbone.norm_f)
            )
            object.__setattr__(self, "_joins_resolved", True)
        return self._joins_cache

    def activations(
        self, ids: mx.array, cache: Sequence[NemotronHLayer] | None = None
    ) -> NemotronHActivations:
        """One forward for every shape this trunk runs in — prefill, one-token decode, and
        the `width + 1` rows of a verification — over growing caches or promoted ones.

        The joins are the reason it is worth saying so. Block `i`'s residual add and the
        norm block `i + 1` reads it through are one kernel, which crosses a block boundary
        and so cannot live inside a block; running it here instead of only under a compiled
        decode is what keeps the two from being two arithmetics that have to agree.
        `RowsAddRmsNorm` is bit-identical to `nn.RMSNorm` over the same sum, in fp32 and
        bf16 alike, which is what makes the fused chain and the plain one the same numbers
        and not merely close ones.
        """
        layers: Sequence[NemotronHLayer] = self.make_cache() if cache is None else cache
        x = self.backbone.embeddings(ids)
        embeddings = x
        blocks: list[mx.array] = []
        joins = self.joins()
        if joins is None:
            for block, layer_cache in zip(self.backbone.layers, layers, strict=True):
                x = block(x, layer_cache)
                blocks.append(x)
            normed = self.backbone.norm_f(x)
        else:
            first = self.backbone.layers[0]
            assert isinstance(first, NemotronHBlock)
            normed = first.norm(x)
            for block, layer_cache, join in zip(
                self.backbone.layers, layers, joins, strict=True
            ):
                assert isinstance(block, NemotronHBlock)
                x, normed = join(x, block.mix(normed, layer_cache))
                blocks.append(x)
        return NemotronHActivations(embeddings, blocks, normed, self.lm_head(normed))

    def __call__(
        self, ids: mx.array, cache: Sequence[NemotronHLayer] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

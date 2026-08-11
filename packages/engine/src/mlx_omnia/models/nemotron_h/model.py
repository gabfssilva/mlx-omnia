from collections.abc import Callable, Sequence
from typing import NamedTuple, assert_never

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.models.nemotron_h.config import MAMBA, NemotronHConfig
from mlx_omnia.models.nemotron_h.layers.block import NemotronHBlock


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


class NemotronH(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = NemotronHTrunk(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        """`speculative.Speculable`. The trunk puts nothing over the lookup here, so the raw
        pair and the cooked one are the same tensors — which is why the protocol names them
        rather than trusting a family to have only one."""
        return self.backbone.embeddings(ids)

    def raw_logits(self, hidden: mx.array) -> mx.array:
        """`speculative.Speculable`. `norm_f` is *not* applied: the rows an MTP step returns
        already went through the step's own `final_layernorm`, and normalizing twice is a
        different model. The trunk's own path applies `norm_f` before calling this."""
        return self.lm_head(hidden)

    def block_outputs(
        self, ids: mx.array, cache: list[LayerCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """`generate.BlockOutputs`: the same forward, plus the output of blocks `at`
        concatenated on the last dim. What reads it is an MTP step, which asks for one — the
        last block, whose rows are what the step's `hnorm` was trained on."""
        activations = self.activations(ids, cache)
        return activations.logits, mx.concatenate([activations.blocks[index] for index in at], -1)

    def verify(
        self, ids: mx.array, cache: list[LayerCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array, Callable[[int], None]]:
        """`speculative.Verifiable`: `block_outputs`, plus a cheap way back.

        Only 23 of these 52 layers keep state a rewind cannot subtract. The 6 attention layers
        hold keys, which are dropped by moving the offset; the 23 sparse ones hold nothing but
        the offset itself. Replaying all of them — which is what the round does for a trunk
        that cannot say otherwise — reads the MoE weights a second time, and on this trunk
        that is 57% of the bytes and the reason a rejected round costs more than the token it
        bought.

        The inputs the replay needs are already here: block `i` was fed block `i-1`'s output,
        and `activations` collected every one. Holding a reference to the 23 that matter costs
        nothing the forward had not already allocated.
        """
        # Before the forward, which is the only moment a restore point means anything: by the
        # time `rewind` is called the recurrent layers have already taken every row, including
        # the ones about to be thrown away.
        restores = [
            layer.checkpoint()
            for layer, kind in zip(cache, self.config.pattern, strict=True)
            if kind == MAMBA
        ]
        activations = self.activations(ids, cache)
        inputs = [activations.embeddings, *activations.blocks[:-1]]

        def rewind(kept: int) -> None:
            for restore in restores:
                restore()
            for index, kind in enumerate(self.config.pattern):
                if kind == MAMBA:
                    self.backbone.layers[index](inputs[index][:, :kept], cache[index])
                else:
                    # The keys of a rejected row stay in the buffer and are overwritten by the
                    # next update, exactly as `KVCache.trim` leaves them; a layer holding only
                    # a count needs only the count.
                    cache[index].offset += kept - ids.shape[1]

        features = mx.concatenate([activations.blocks[index] for index in at], -1)
        return activations.logits, features, rewind

    def activations(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> NemotronHActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.backbone.embeddings(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.backbone.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.backbone.norm_f(x)
        return NemotronHActivations(embeddings, blocks, normed, self.lm_head(normed))

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits

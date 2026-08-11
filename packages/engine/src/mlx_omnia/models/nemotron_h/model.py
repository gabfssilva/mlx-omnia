from collections.abc import Callable, Sequence
from typing import NamedTuple, assert_never

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import DeltaCache, FixedDeltaCache, FixedKVCache, KVCache, LayerCache
from mlx_omnia.core.kernels.add_norm import RowsAddRmsNorm
from mlx_omnia.models.nemotron_h.config import ATTENTION, MAMBA, NemotronHConfig
from mlx_omnia.models.nemotron_h.layers.block import NemotronHBlock
from mlx_omnia.models.nemotron_h.layers.mamba import NemotronHMamba

_HEADROOM = 768


def _fit(offset: int) -> int:
    """The smallest 256-multiple holding `offset` plus generation headroom."""
    return (offset + _HEADROOM + 255) // 256 * 256


def _regrow(layer: FixedKVCache, capacity: int) -> tuple[mx.array, mx.array, int]:
    """A full fixed buffer copied into a larger one, rows and position preserved."""
    keys, values = layer.fetch()
    rows = layer.rows
    shape = list(keys.shape)
    shape[2] = capacity
    grown_keys = mx.zeros(shape, dtype=keys.dtype)
    grown_values = mx.zeros(shape, dtype=values.dtype)
    grown_keys[..., :rows, :] = keys[..., :rows, :]
    grown_values[..., :rows, :] = values[..., :rows, :]
    mx.eval(grown_keys, grown_values)
    return grown_keys, grown_values, rows


def _joins(
    blocks: list[nn.Module], norm_f: nn.RMSNorm
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

    def compile_decode(
        self, cache: list[LayerCache], capacity: int | None = None
    ) -> Callable[[mx.array], mx.array]:
        """Promote a completed prefill cache and compile one-token forwards.

        The 6 attention layers get a fixed buffer with a graph-visible position; the 23
        recurrent ones get their window and state moved into a graph-visible container
        (`FixedDeltaCache`); the sparse and dense ones hold only an offset, which never
        enters the graph. The attention mask is the fixed buffer's own fill: every column
        up to and including the row this step writes.

        The default capacity is sized to the prompt rather than a constant: the fixed
        buffer is read whole by every step's attention, so an oversized one is bytes on
        the decode's critical path. When a generation outgrows it, the wrapper promotes
        again into a larger buffer and recompiles — a copy and a retrace, paid once per
        doubling and never per token.
        """
        blocks = self.backbone.layers

        def build(fitting: int) -> tuple[Callable[[mx.array], mx.array], list[LayerCache], int]:
            promoted: list[LayerCache] = []
            for layer, kind in zip(cache, self.config.pattern, strict=True):
                if kind == ATTENTION:
                    assert isinstance(layer, KVCache | FixedKVCache)
                    grown = FixedKVCache(*_regrow(layer, fitting)) if isinstance(
                        layer, FixedKVCache
                    ) else FixedKVCache.promote(layer, fitting)
                    promoted.append(grown)
                elif kind == MAMBA:
                    assert isinstance(layer, DeltaCache)
                    promoted.append(
                        layer
                        if isinstance(layer, FixedDeltaCache)
                        else FixedDeltaCache.promote(layer)
                    )
                else:
                    promoted.append(layer)
            cache[:] = promoted
            state = [
                layer.state if isinstance(layer, FixedKVCache) else layer.graph
                for layer in promoted
                if isinstance(layer, FixedKVCache | FixedDeltaCache)
            ]
            anchor = next(layer for layer in promoted if isinstance(layer, FixedKVCache))
            columns = mx.arange(fitting)

            joins = _joins(blocks, self.backbone.norm_f)

            def forward(ids: mx.array) -> mx.array:
                # Read before any layer advances it: the row this step writes lands at
                # the pre-update position, so `<=` keeps it attendable and `<` would
                # drop it.
                mask = (columns <= anchor.position).reshape(1, 1, 1, fitting)
                x = self.backbone.embeddings(ids[None])
                if joins is None:
                    for block, layer_cache in zip(blocks, promoted, strict=True):
                        assert isinstance(block, NemotronHBlock)
                        x = block(x, layer_cache, mask)
                    return self.lm_head(self.backbone.norm_f(x))[:, -1, :]
                first = blocks[0]
                assert isinstance(first, NemotronHBlock)
                normed = first.norm(x)
                for block, layer_cache, join in zip(blocks, promoted, joins, strict=True):
                    assert isinstance(block, NemotronHBlock)
                    x, normed = join(x, block.mix(normed, layer_cache, mask))
                return self.lm_head(normed)[:, -1, :]

            return mx.compile(forward, inputs=state, outputs=state), promoted, fitting

        offset = cache[0].offset
        fit = capacity if capacity is not None else _fit(offset)
        if offset >= fit:
            fit = _fit(offset)
        compiled, promoted, fit = build(fit)
        base = offset
        steps = 0

        def decode(ids: mx.array) -> mx.array:
            # The python-side offsets are assigned, not incremented: the trace's own pass
            # through the layers already bumped them once, and only once.
            nonlocal steps, compiled, promoted, fit
            if base + steps + 1 >= fit:
                compiled, promoted, fit = build(_fit(base + steps))
            logits = compiled(ids)
            steps += 1
            for layer in promoted:
                layer.offset = base + steps
            return logits

        return decode

    def compile_verify(
        self, cache: list[LayerCache], *, width: int, at: Sequence[int]
    ) -> (
        tuple[
            Callable[[mx.array], tuple[mx.array, mx.array]],
            Callable[[int], None],
        ]
        | None
    ):
        """The speculative round's fixed-shape half, compiled: one forward over the
        `width + 1` rows a round always verifies, and a rewind that costs no forward.

        The rewind is vllm's shape: the recurrent layers' verify kernel wrote the state
        after every row to its own slot, and the conv windows are raw projection rows,
        so keeping `kept` rows is picking a slot and a slice. The attention layers'
        fixed buffers rewind by moving the graph-visible position back — the rows past
        it are stale and overwritten by the next round.
        """
        rows = width + 1
        pattern = self.config.pattern
        # The per-token-checkpoint kernel is what makes the compiled round worth
        # taking; a configuration it declines falls back to replay by returning None.
        from mlx_omnia.core.kernels.mamba_step.fused import FusedMambaStep

        for block, kind in zip(self.backbone.layers, pattern, strict=True):
            if kind != MAMBA:
                continue
            assert isinstance(block, NemotronHBlock)
            mixer = block.mixer
            assert isinstance(mixer, NemotronHMamba)
            if not isinstance(mixer._mamba_step().strategy, FusedMambaStep):
                return None
        if ATTENTION not in pattern:
            return None
        offset = cache[0].offset
        promoted: list[LayerCache] = []
        for layer, kind in zip(cache, pattern, strict=True):
            if kind == ATTENTION:
                assert isinstance(layer, KVCache)
                promoted.append(FixedKVCache.promote(layer, _fit(offset)))
            elif kind == MAMBA:
                assert isinstance(layer, DeltaCache)
                fixed = FixedDeltaCache.promote(layer)
                window, state = fixed.graph
                fixed.graph.extend(
                    [
                        mx.zeros((rows, *state.shape[1:]), dtype=state.dtype),
                        mx.zeros((rows, window.shape[-1]), dtype=window.dtype),
                        mx.zeros_like(window),
                    ]
                )
                promoted.append(fixed)
            else:
                promoted.append(layer)
        cache[:] = promoted
        blocks = self.backbone.layers
        steps = mx.arange(rows)

        joins = _joins(blocks, self.backbone.norm_f)

        def build(fitting: int) -> tuple[Callable[[mx.array], tuple[mx.array, mx.array]], int]:
            # A generation that outgrows the fixed attention buffers promotes again into
            # larger ones and recompiles — the mamba containers are size-free and stay.
            for index, kind in enumerate(pattern):
                if kind != ATTENTION:
                    continue
                layer = promoted[index]
                assert isinstance(layer, FixedKVCache)
                if layer.state[0].shape[2] < fitting:
                    promoted[index] = FixedKVCache(*_regrow(layer, fitting))
            cache[:] = promoted
            state_containers = [
                layer.state if isinstance(layer, FixedKVCache) else layer.graph
                for layer in promoted
                if isinstance(layer, FixedKVCache | FixedDeltaCache)
            ]
            anchor = next(layer for layer in promoted if isinstance(layer, FixedKVCache))
            columns = mx.arange(fitting)

            def forward(ids: mx.array) -> tuple[mx.array, mx.array]:
                mask = (columns[None, :] <= anchor.position + steps[:, None]).reshape(
                    1, 1, rows, fitting
                )
                x = self.backbone.embeddings(ids[None])
                outputs: list[mx.array] = []
                first = blocks[0]
                assert isinstance(first, NemotronHBlock)
                normed = first.norm(x) if joins is not None else x
                for index, (block, layer_cache, kind) in enumerate(
                    zip(blocks, promoted, pattern, strict=True)
                ):
                    assert isinstance(block, NemotronHBlock)
                    if joins is None:
                        normed = block.norm(x)
                    if kind == MAMBA:
                        mixer = block.mixer
                        assert isinstance(mixer, NemotronHMamba)
                        mixed = mixer.verify_rows(normed, layer_cache)
                    else:
                        mixed = block.mix(normed, layer_cache, mask)
                    if joins is None:
                        x = x + mixed
                    else:
                        x, normed = joins[index](x, mixed)
                    outputs.append(x)
                final = self.backbone.norm_f(x) if joins is None else normed
                logits = self.lm_head(final)
                features = mx.concatenate([outputs[index] for index in at], -1)
                return logits[0], features

            return (
                mx.compile(forward, inputs=state_containers, outputs=state_containers),
                fitting,
            )

        compiled, fit = build(_fit(offset))
        current = offset
        before = offset

        def verify(ids: mx.array) -> tuple[mx.array, mx.array]:
            nonlocal current, before, compiled, fit
            if current + rows > fit:
                compiled, fit = build(_fit(current))
            before = current
            logits, features = compiled(ids)
            current = before + rows
            for layer in promoted:
                layer.offset = current
            return logits, features

        def rewind(kept: int) -> None:
            nonlocal current
            current = before + kept
            for layer in promoted:
                if isinstance(layer, FixedKVCache):
                    layer.state[2] = mx.array([current], dtype=mx.int32)
                elif isinstance(layer, FixedDeltaCache):
                    window_before = layer.graph[4][0]
                    conv_rows = layer.graph[3][:kept]
                    tail = self.config.conv_kernel - 1
                    layer.graph[0] = mx.concatenate([window_before, conv_rows], axis=0)[
                        None, -tail:
                    ]
                    layer.graph[1] = layer.graph[2][kept - 1][None]
                layer.offset = current

        return verify, rewind

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

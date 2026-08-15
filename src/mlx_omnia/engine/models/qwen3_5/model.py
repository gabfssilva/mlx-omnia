from collections.abc import Callable, Collection, Iterator, Sequence
from typing import NamedTuple, TypeIs, assert_never

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_omnia.engine.core.cache import (
    DeltaCache,
    FixedDeltaCache,
    FixedKVCache,
    KVCache,
    LayerCache,
    fit,
    regrow,
)
from mlx_omnia.engine.core.decode import DecodePlan, compiled_decode
from mlx_omnia.engine.core.prefill import prefill
from mlx_omnia.engine.core.prompt_cache import PromptCache
from mlx_omnia.engine.generate import (
    BlockOutputs,
    Meter,
    Penalty,
    Sampler,
    greedy,
    stream_ids,
    stream_text,
)
from mlx_omnia.engine.language import (
    TEXT,
    GenerationOptions,
    LanguagePrompt,
    Text,
    Tokenizer,
    prefix_cache,
)
from mlx_omnia.engine.model import ModelInput, ModelSignature
from mlx_omnia.engine.models.qwen3_5.config import Qwen35Config
from mlx_omnia.engine.models.qwen3_5.layers.block import Qwen35Block, Qwen35Trunk
from mlx_omnia.engine.models.qwen3_5.vision import (
    Grid,
    ProcessorConfig,
    Qwen35Vision,
    multimodal_positions,
    process_image,
)
from mlx_omnia.engine.parsers import Parser, Segment
from mlx_omnia.engine.speculative import Persistent, Speculable, Step
from mlx_omnia.engine.vision import RGB_IMAGE, Image


class Qwen35Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Qwen35(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen35Trunk(config.text_config)
        if config.vision_config is not None:
            self.visual = Qwen35Vision(config.vision_config)
        if not config.tied:
            text = config.text_config
            self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        return [
            KVCache() if kind == "full_attention" else DeltaCache()
            for kind in self.config.text_config.layer_types
        ]

    def compile_decode(
        self,
        cache: list[LayerCache],
        capacity: int | None = None,
        *,
        rope_delta: int | None = None,
    ) -> Callable[[mx.array], mx.array]:
        """Promote a completed prefill cache and compile one-token forwards.

        Per-block traces already collapse the elementwise work around each mixer, but the
        trunk between them stays interpreted: forty round trips through Python per token,
        forty offsets read as constants, and a growing KV buffer whose shape changes under
        the tracer. Here the whole trunk is one trace. The 10 full-attention layers get a
        fixed buffer with a graph-visible position, the 30 DeltaNet ones get their window
        and state moved into a graph-visible container, and the attention mask is the
        buffer's own fill — every column up to and including the row this step writes.

        Nothing about the arithmetic moves. `mx.fast.rope` takes the position as an array
        and returns the same bits it returns for the equivalent int, which is what makes
        the projections traceable at all; the mixers call the same closures the per-block
        traces compile.

        `rope_delta` is the multimodal clock: after an image the position resumes at
        `pos + max(grid)/merge` rather than at `pos + image_tokens`, so a decode that
        followed a picture rotates by `row + delta` on all three MRoPE sections. Left
        `None` the trunk runs the text rotation, which is the same partial rope.
        """
        text = self.config.text_config
        blocks = self.model.layers
        kinds = text.layer_types
        attends = [kind == "full_attention" for kind in kinds]
        if not any(attends):
            raise ValueError("decode compilation needs at least one attention layer to anchor")

        def promote(layers: list[LayerCache], fitting: int) -> list[LayerCache]:
            promoted: list[LayerCache] = []
            for layer, full in zip(layers, attends, strict=True):
                if full:
                    assert isinstance(layer, KVCache | FixedKVCache)
                    if isinstance(layer, FixedKVCache):
                        # A rebuild that is not a growth (the delegators moved under the
                        # graph, not the buffer) keeps the buffer it already has.
                        promoted.append(
                            layer if layer.state[0].shape[2] >= fitting else regrow(layer, fitting)
                        )
                    else:
                        promoted.append(FixedKVCache.promote(layer, fitting))
                else:
                    assert isinstance(layer, DeltaCache)
                    promoted.append(
                        layer
                        if isinstance(layer, FixedDeltaCache)
                        else FixedDeltaCache.promote(layer)
                    )
            return promoted

        def prepare(_: Sequence[LayerCache]) -> None:
            for block in blocks:
                assert isinstance(block, Qwen35Block)
                block.prepare_decode()

        def epochs() -> tuple[int, ...]:
            """The delegators this graph baked. A per-block step run in between re-resolves
            them: streaming with a prefix cache takes that path and streaming without takes
            this one, so the same resident model alternates."""
            return tuple(block.epoch for block in blocks if isinstance(block, Qwen35Block))

        def step(ids: mx.array, slots: Sequence[LayerCache], mask: mx.array | None) -> mx.array:
            anchor = next(layer for layer in slots if isinstance(layer, FixedKVCache))
            positions = (
                None
                if rope_delta is None
                else mx.broadcast_to(anchor.position + rope_delta, (3, 1))
            )
            x = self.model.embed_tokens(ids[None])
            for block, layer_cache in zip(blocks, slots, strict=True):
                assert isinstance(block, Qwen35Block)
                assert isinstance(layer_cache, FixedKVCache | FixedDeltaCache)
                x = block.graph_step(x, layer_cache, positions, mask)
            normed = self.model.norm(x)
            logits = (
                self.model.embed_tokens.as_linear(normed)
                if self.config.tied
                else self.lm_head(normed)
            )
            return logits[:, -1, :]

        plan = DecodePlan(step=step, promote=promote, prepare=prepare, epochs=epochs)
        return compiled_decode(plan, cache, capacity)

    def raw_embed(self, ids: mx.array) -> mx.array:
        """`speculative.Speculable`. The trunk puts nothing over the lookup, so the raw
        embedding and the cooked one are the same tensor."""
        return self.model.embed_tokens(ids)

    def raw_logits(self, hidden: mx.array) -> mx.array:
        """`speculative.Speculable`. `model.norm` is *not* applied: the rows an MTP step
        returns already went through the step's own final norm, and normalizing twice is a
        different model. The trunk's own path applies `model.norm` before calling this."""
        if self.config.tied:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def _tapped(self, activations: "Qwen35Activations", at: Sequence[int]) -> mx.array:
        """The rows a proposer conditions on. The last tap is the **normed** stream, not the
        last block's raw output: vllm's runner hands the MTP step what `Qwen3_5Model.forward`
        returns, which is `self.norm(hidden)` (`gpu_model_runner.py:5328`) — the step's
        `pre_fc_norm_hidden` was trained on rows that already went through `model.norm`."""
        last = len(activations.blocks) - 1
        return mx.concatenate(
            [
                activations.norm if index in (-1, last) else activations.blocks[index]
                for index in at
            ],
            -1,
        )

    def block_outputs(
        self, ids: mx.array, cache: list[LayerCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """`generate.BlockOutputs`: the same forward, plus the rows `at` names concatenated
        on the last dim. What reads it is an MTP step, which asks for one — the last."""
        activations = self.activations(ids, cache)
        return activations.logits, self._tapped(activations, at)

    def verify(
        self, ids: mx.array, cache: list[LayerCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array, Callable[[int], None]]:
        """`speculative.Verifiable`: `block_outputs`, plus a cheap way back.

        Three of every four layers are DeltaNet and keep state a rewind cannot subtract.
        The attention layers hold keys, which are dropped by moving the offset; the
        recurrent ones restore the state the round checkpointed and replay only the rows
        that were kept — the inputs the replay needs are the previous block's outputs,
        which `activations` already collected.
        """
        kinds = self.config.text_config.layer_types
        # Before the forward, which is the only moment a restore point means anything: by
        # the time `rewind` is called the recurrent layers have already taken every row,
        # including the ones about to be thrown away.
        restores = [
            layer.checkpoint()
            for layer, kind in zip(cache, kinds, strict=True)
            if kind == "linear_attention"
        ]
        activations = self.activations(ids, cache)
        inputs = [activations.embeddings, *activations.blocks[:-1]]

        def rewind(kept: int) -> None:
            for restore in restores:
                restore()
            for index, kind in enumerate(kinds):
                if kind == "linear_attention":
                    self.model.layers[index](inputs[index][:, :kept], cache[index])
                else:
                    # The keys of a rejected row stay in the buffer and are overwritten by
                    # the next update, exactly as `KVCache.trim` leaves them; a layer
                    # holding only a count needs only the count.
                    cache[index].offset += kept - ids.shape[1]

        return activations.logits, self._tapped(activations, at), rewind

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

        The DeltaNet layers write the state after every row to a slot
        (`Qwen35DeltaNet.verify_rows`), and the conv windows are raw projection rows, so
        keeping `kept` rows is picking a slot and a slice. The attention layers' fixed
        buffers rewind by moving the graph-visible position back — the rows past it are
        stale and overwritten by the next round.

        Refusals fall back to the replay round, which is correct everywhere: a sparse
        trunk (the fused MoE step is a T=1 shape), a delta rule the fused kernel does not
        tile (`verify_rows`' checkpoints are the kernel's), and a tap that is not the
        normed stream (`_tapped` says why the last one is).
        """
        rows = width + 1
        text = self.config.text_config
        kinds = text.layer_types
        if text.num_experts:
            return None
        if "full_attention" not in kinds:
            return None
        if tuple(at) not in ((-1,), (len(kinds) - 1,)):
            return None
        from mlx_omnia.engine.core.kernels.gated_delta import FusedGatedDelta

        blocks = self.model.layers
        for block, kind in zip(blocks, kinds, strict=True):
            if kind == "full_attention":
                continue
            assert isinstance(block, Qwen35Block)
            if not isinstance(block.linear_attn.rule().strategy, FusedGatedDelta):
                return None

        offset = cache[0].offset
        promoted: list[LayerCache] = []
        for layer, kind in zip(cache, kinds, strict=True):
            if kind == "full_attention":
                assert isinstance(layer, KVCache)
                promoted.append(FixedKVCache.promote(layer, fit(offset)))
            else:
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
        cache[:] = promoted
        steps = mx.arange(rows)

        def build(fitting: int) -> tuple[
            Callable[[mx.array], tuple[mx.array, mx.array]], int, list[int]
        ]:
            # A generation that outgrows the fixed attention buffers promotes again into
            # larger ones and recompiles — the delta containers are size-free and stay.
            for index, kind in enumerate(kinds):
                if kind != "full_attention":
                    continue
                layer = promoted[index]
                assert isinstance(layer, FixedKVCache)
                if layer.state[0].shape[2] < fitting:
                    promoted[index] = regrow(layer, fitting)
            cache[:] = promoted
            state_containers = [
                layer.state if isinstance(layer, FixedKVCache) else layer.graph
                for layer in promoted
                if isinstance(layer, FixedKVCache | FixedDeltaCache)
            ]
            anchor = next(layer for layer in promoted if isinstance(layer, FixedKVCache))
            columns = mx.arange(fitting)
            for block in blocks:
                assert isinstance(block, Qwen35Block)
                block.prepare_decode()

            def forward(ids: mx.array) -> tuple[mx.array, mx.array]:
                # Fill and causal in one mask: row j attends every column up to the
                # position it writes, `position + j`.
                mask = (columns[None, :] <= anchor.position + steps[:, None]).reshape(
                    1, 1, rows, fitting
                )
                x = self.model.embed_tokens(ids[None])
                for block, layer_cache, kind in zip(blocks, promoted, kinds, strict=True):
                    assert isinstance(block, Qwen35Block)
                    if kind == "full_attention":
                        assert isinstance(layer_cache, FixedKVCache)
                        projected = block.self_attn(
                            block.input_layernorm(x), layer_cache, None, mask
                        )
                    else:
                        assert isinstance(layer_cache, FixedDeltaCache)
                        projected = block.linear_attn.verify_rows(
                            block.input_layernorm(x), layer_cache
                        )
                    attended = x + projected
                    x = attended + block.mlp(block.post_attention_layernorm(attended))
                normed = self.model.norm(x)
                logits = (
                    self.model.embed_tokens.as_linear(normed)
                    if self.config.tied
                    else self.lm_head(normed)
                )
                assert isinstance(normed, mx.array) and isinstance(logits, mx.array)
                return logits[0], normed

            epochs = [block.epoch for block in blocks if isinstance(block, Qwen35Block)]
            return (
                mx.compile(forward, inputs=state_containers, outputs=state_containers),
                fitting,
                epochs,
            )

        compiled, room, epochs = build(fit(offset))
        current = offset
        before = offset

        def stale() -> bool:
            currently = [block.epoch for block in blocks if isinstance(block, Qwen35Block)]
            return currently != epochs

        def verify(ids: mx.array) -> tuple[mx.array, mx.array]:
            nonlocal current, before, compiled, room, epochs
            if current + rows > room:
                compiled, room, epochs = build(fit(current))
            elif stale():
                compiled, room, epochs = build(room)
            before = current
            logits, features = compiled(ids)
            current = before + rows
            for layer in promoted:
                layer.offset = current
            return logits, features

        def rewind(kept: int) -> None:
            nonlocal current
            current = before + kept
            tail = text.linear_conv_kernel_dim - 1
            for layer in promoted:
                if isinstance(layer, FixedKVCache):
                    layer.state[2] = mx.array([current], dtype=mx.int32)
                elif isinstance(layer, FixedDeltaCache):
                    window_before = layer.graph[4][0]
                    conv_rows = layer.graph[3][:kept]
                    layer.graph[0] = mx.concatenate([window_before, conv_rows], axis=0)[
                        None, -tail:
                    ]
                    layer.graph[1] = layer.graph[2][kept - 1][None]
                layer.offset = current

        return verify, rewind

    def activations(
        self,
        ids: mx.array,
        cache: list[LayerCache] | None = None,
        *,
        positions: mx.array | None = None,
        embeddings: mx.array | None = None,
    ) -> Qwen35Activations:
        """`embeddings` replaces the token lookup (an image's rows are already spliced
        in) and `positions` is the `[3, L]` MRoPE clock; both default to the text path."""
        cache = cache if cache is not None else self.make_cache()
        x = embeddings if embeddings is not None else self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache, positions)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tied:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Qwen35Activations(embedded, blocks, normed, logits)

    def __call__(
        self,
        ids: mx.array,
        cache: list[LayerCache] | None = None,
        *,
        positions: mx.array | None = None,
        embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.activations(ids, cache, positions=positions, embeddings=embeddings).logits


class MultimodalPrompt(NamedTuple):
    """A prompt whose image rows are already spliced in, with the clock they run on.

    `delta` is the trap. After an image the position resumes at
    `pos + max(grid_h, grid_w)/merge`, not at `pos + image_tokens`: a 22x28 image spends
    154 cache rows and 14 positions, so everything after it runs 140 behind the row count
    forever. Removing it does not break greedy — 18 of the 24 layers are DeltaNet and
    carry position through the recurrence, and the rope rotates 64 of 256 dims — the
    model just writes fluent, on-topic, wrong text. What catches it is teacher forcing.
    """

    ids: list[int]
    embeddings: mx.array
    positions: mx.array
    delta: int


def multimodal_prompt(
    model: Qwen35, ids: Sequence[int], images: Sequence[tuple[mx.array, Grid]]
) -> MultimodalPrompt:
    """Runs the tower over each image, scatters its rows onto the placeholder run, and
    builds the 3-D clock the trunk reads."""
    config = model.config
    vision = config.vision_config
    if vision is None:
        raise ValueError("this checkpoint carries no vision tower")
    tower = model.visual
    assert isinstance(tower, Qwen35Vision)

    tokens = model.model.embed_tokens(mx.array(list(ids)))[None]
    merge = vision.spatial_merge_size
    rows = mx.concatenate(
        [tower(patches.astype(tokens.dtype), grid) for patches, grid in images], axis=0
    ) if images else None
    if rows is not None:
        placeholder = mx.array(list(ids)) == config.image_token_id
        indices = mx.array(np.flatnonzero(np.array(placeholder)))
        if indices.shape[0] != rows.shape[0]:
            raise ValueError(
                f"{rows.shape[0]} image rows for {indices.shape[0]} placeholder tokens"
            )
        tokens[0, indices] = rows.astype(tokens.dtype)

    positions, delta = multimodal_positions(
        list(ids), [grid for _, grid in images], config.image_token_id, merge
    )
    return MultimodalPrompt(list(ids), tokens, mx.array(positions), delta)


def decode_clock(prompt: MultimodalPrompt, row: int) -> mx.array:
    """The 3-D rotation for the token sitting at cache row `row`: transformers'
    `arange(past) + rope_delta`, one row at a time."""
    return mx.full((3, 1), row + prompt.delta, dtype=mx.int32)


def stream_multimodal_ids(
    model: Qwen35,
    prompt: MultimodalPrompt,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    meter: Meter | None = None,
) -> Iterator[int]:
    """The chassis' lazy loop with the 3-D clock threaded through: prefill reads the
    prompt's own positions, and every step after it rotates by `row + delta`, which is
    what transformers rebuilds as `arange(past) + rope_delta`."""
    cache = model.make_cache()
    row = len(prompt.ids)
    history = mx.array(prompt.ids)
    if meter is not None:
        # The placeholder run is prompt: those rows are read, and the tower's cost lands
        # in the prefill mark like any other.
        meter.prefill(len(prompt.ids))

    def step(ids: mx.array, positions: mx.array, embeddings: mx.array | None) -> mx.array:
        logits = model(ids, cache, positions=positions, embeddings=embeddings)[:, -1, :]
        return sampler(logits if penalty is None else penalty(logits, history))[0]

    def advance(y: mx.array) -> mx.array:
        nonlocal row
        positions = decode_clock(prompt, row)
        out = step(y[None, None], positions, None)
        row += 1
        return out

    ids = mx.array(prompt.ids)

    def feed(part: slice) -> mx.array:
        return model(
            ids[part][None],
            cache,
            positions=prompt.positions[:, part],
            embeddings=prompt.embeddings[:, part],
        )

    window = prefill(feed, ids.size, cache)
    y = step(ids[window][None], prompt.positions[:, window], prompt.embeddings[:, window])
    mx.async_eval(y)
    for _ in range(max_tokens):
        if penalty is not None:
            history = mx.concatenate([history, y[None]])
        next_y = advance(y)
        mx.async_eval(next_y)
        token = y.item()
        assert isinstance(token, int)
        if token in stop:
            return
        if meter is not None:
            meter.token()
        yield token
        y = next_y


type Qwen35Input = Text | Image | LanguagePrompt


def _parser(input: Qwen35Input) -> Parser | None:
    """Which dialect this prompt's checkpoint speaks, off the prompt itself — the
    capability put it there when it rendered. Reading it from a field of the facade instead is
    how this model came to suppress nothing: the loader never set one, and the only writer was
    a test."""
    match input:
        case Text():
            return input.parser
        case LanguagePrompt(parts=parts):
            return next((part.parser for part in parts if isinstance(part, Text)), None)
        case Image():
            return None
        case _:
            assert_never(input)


def sees(config: Qwen35Config, processor: ProcessorConfig | None) -> bool:
    """Whether this checkpoint takes images: a tower in the config, the processor file
    beside it, and the three ids the marker is spelled with.

    A function and not an expression inside the facade because the catalog asks the same
    question of a directory it has not loaded — and an answer that disagreed with the
    facade's would offer a picture to a model that refuses it, or refuse one the model
    would have read."""
    return (
        processor is not None
        and config.vision_config is not None
        and config.image_token_id >= 0
        and config.vision_start_token_id >= 0
        and config.vision_end_token_id >= 0
    )


class Qwen35LanguageModel:
    def __init__(
        self,
        model: Qwen35,
        tokenizer: Tokenizer,
        processor: ProcessorConfig | None,
        *,
        stop: Collection[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.stop = stop
        self.prefix: PromptCache[KVCache | DeltaCache] | None = None
        """What this model kept of the prompts before it. A trunk with a recurrent layer has
        nothing to cut at a common prefix and `stream_ids` refuses it there — this holds the
        trie for the all-attention configurations of the same architecture."""
        self.drafter: nn.Module | None = None
        """`speculative.Drafting`. Out of the trunk's tree so whoever accounts for memory
        can weigh it: with an MTP head the two are one checkpoint on disk and still two
        trees here, and only one of them is under `self.model`."""
        self._block: int | None = None
        self._vision = sees(model.config, processor)

    def speculate_with(self, drafter: nn.Module, *, block_size: int | None = None) -> None:
        """Take an MTP step for this trunk — `speculative.Drafting`, the same door
        `language.TextLanguageModel` opens. A facade of its own only because of the vision
        path; the checks are the protocol's, not the family's."""
        if not isinstance(drafter, Step):
            raise TypeError(f"{type(drafter).__name__} is not an MTP step for this engine")
        if not isinstance(self.model, Speculable):
            raise TypeError(
                f"{type(self.model).__name__} does not lend its embedding table and its head "
                "(`speculative.Speculable`), which is the whole of an MTP step's vocabulary"
            )
        if not isinstance(self.model, BlockOutputs):
            raise TypeError(
                f"{type(self.model).__name__} has no `block_outputs`, so there is nothing "
                "for the step to condition on"
            )
        block = drafter.block if block_size is None else block_size
        if block < 2:
            raise ValueError(f"a block of {block} proposes nothing")
        self.drafter = drafter
        self._block = block

    def _proposer(self, options: GenerationOptions) -> Persistent[LayerCache] | None:
        """The proposer for this request, or `None` for every request that cannot be
        verified against the target's own argmax — those decode as if no head were loaded.

        `Persistent` and not `Chained`, measured on the nvfp4 27B entry (2026-08-14): the
        full-history drafter lifts acceptance from 0.73 to 0.85 at depth one and from 0.45
        to 0.63 at depth two, which buys more than the compiled chain's two milliseconds —
        54.2 against 48.7 tok/s at block 3 on the prose prompt."""
        drafter = self.drafter
        if drafter is None or self._block is None or not options.speculate:
            return None
        if options.sampler is not greedy or options.penalty is not None:
            return None
        if options.constraint is not None or options.reasoning_budget is not None:
            return None
        assert isinstance(drafter, Step)
        assert isinstance(self.model, Speculable)
        return Persistent(self.model, drafter, block=self._block, tap=-1)

    @property
    def native_signature(self) -> ModelSignature:
        inputs = frozenset({TEXT, RGB_IMAGE}) if self._vision else frozenset({TEXT})
        return ModelSignature(inputs, frozenset({TEXT}))

    @property
    def image_marker(self) -> str | None:
        """What the chat template emits per image, as text: the vision wrapper around one
        placeholder. Cutting a rendered conversation there gives back the parts this model
        already knows how to prompt with."""
        if not self._vision:
            return None
        config = self.model.config
        wrapper = [config.vision_start_token_id, config.image_token_id, config.vision_end_token_id]
        return self.tokenizer.decode_bytes(wrapper).decode("utf-8")

    def accepts(self, input: ModelInput) -> TypeIs[Qwen35Input]:
        match input:
            case Text():
                return True
            case Image():
                return self._vision
            case LanguagePrompt():
                return self._vision and input.content_types <= self.native_signature.inputs
            case _:
                return False

    def prepare(self, input: Image | LanguagePrompt) -> MultimodalPrompt:
        config = self.model.config
        processor = self.processor
        vision = config.vision_config
        if not self._vision or processor is None or vision is None:
            raise TypeError("this language model does not accept images")

        parts = input.parts if isinstance(input, LanguagePrompt) else (input,)
        ids: list[int] = []
        images: list[tuple[mx.array, Grid]] = []
        for part in parts:
            match part:
                case Text():
                    ids.extend(self.tokenizer.encode(part.read()))
                case Image(pixels=pixels):
                    processed = process_image(pixels, processor)
                    _, grid = processed
                    placeholders = grid.t * grid.h * grid.w // vision.spatial_merge_size**2
                    ids.append(config.vision_start_token_id)
                    ids.extend([config.image_token_id] * placeholders)
                    ids.append(config.vision_end_token_id)
                    images.append(processed)
                case _:
                    raise TypeError(f"unsupported prompt part {type(part).__name__}")
        return multimodal_prompt(self.model, ids, images)

    def stream(self, input: Qwen35Input, options: GenerationOptions) -> Iterator[Segment]:
        stop = self.stop if options.stop is None else options.stop
        parser = _parser(input)
        match input:
            case Text():
                rendered = input.read()
                draft = self._proposer(options)
                self.prefix = prefix_cache(self.prefix, options.prefix_budget)
                ids = stream_ids(
                    self.model,
                    self.tokenizer.encode(rendered),
                    max_tokens=options.max_tokens,
                    sampler=options.sampler,
                    stop=stop,
                    penalty=options.penalty,
                    meter=options.meter,
                    # The two do not compose (`stream_ids` says why), and a round is worth
                    # more than a prefix: the trie saves the prefill of a turn, the head
                    # saves a read of the weights per token of every turn.
                    prefix=None if draft is not None else self.prefix,
                    constraint=options.constraint,
                    draft=draft,
                )
            case Image() | LanguagePrompt():
                parts = input.parts if isinstance(input, LanguagePrompt) else ()
                rendered = "".join(part.read() for part in parts if isinstance(part, Text))
                prompt = self.prepare(input)
                # No prefix here, and it is not an omission: an image is one id repeated per
                # patch, so two different pictures on the same grid produce the same ids — a
                # trie keyed on ids would hand one image's attention to another.
                ids = stream_multimodal_ids(
                    self.model,
                    prompt,
                    max_tokens=options.max_tokens,
                    sampler=options.sampler,
                    stop=stop,
                    penalty=options.penalty,
                    meter=options.meter,
                )
            case _:
                assert_never(input)

        yield from stream_text(ids, self.tokenizer, parser=parser, prompt=rendered)

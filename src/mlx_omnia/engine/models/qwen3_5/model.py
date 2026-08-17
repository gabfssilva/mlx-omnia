from collections.abc import Collection, Iterator, Sequence
from typing import NamedTuple, TypeIs, assert_never

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_omnia.engine.core.api import Draftable, Step, Tracing
from mlx_omnia.engine.core.cache import (
    DeltaCache,
    KVCache,
    LayerCache,
)
from mlx_omnia.engine.core.prefill import prefill
from mlx_omnia.engine.generate import (
    Meter,
    Penalty,
    Sampler,
    greedy,
    stream_text,
)
from mlx_omnia.engine.language import (
    TEXT,
    GenerationOptions,
    LanguagePrompt,
    Text,
    TextLanguageModel,
    Tokenizer,
)
from mlx_omnia.engine.model import ModelInput, ModelSignature
from mlx_omnia.engine.models.qwen3_5.config import Qwen35Config
from mlx_omnia.engine.models.qwen3_5.layers.block import Qwen35Block, Qwen35Layer, Qwen35Trunk
from mlx_omnia.engine.models.qwen3_5.vision import (
    Grid,
    ProcessorConfig,
    Qwen35Vision,
    multimodal_positions,
    process_image,
)
from mlx_omnia.engine.parsers import Parser, Segment
from mlx_omnia.engine.speculative import Persistent
from mlx_omnia.engine.vision import RGB_IMAGE, Image


class Qwen35Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Qwen35(nn.Module, Draftable[LayerCache], Tracing[LayerCache]):

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

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. Every delegator the trunk's graph will bake, resolved outside
        it — a strategy inspects the checkpoint's tensors to decide applicability, and an
        array read inside `mx.compile` raises. The blocks also drop whatever their *own*
        traces resolved: those hold that trace's tracers as their weight fields, and a graph
        reading them a second time is the uncaptured-input error.

        Nothing is captured. The delegators hold the leaves they resolved against, so an
        array that is both a declared input and a strategy's field is exactly what the
        compile rejects; left out they bake in as constants.

        Both halves of the claim hold. Every position this trunk rotates by is
        `FixedKVCache.position` — a graph tensor, read in `Qwen35Attention.__call__` before
        the update moves it — and the columns a promoted buffer has not written are cut by
        `FixedKVCache.readable` inside `core.attend`.
        """
        del cache
        for block in self.model.layers:
            assert isinstance(block, Qwen35Block)
            block.prepare_decode()
        return ()

    @property
    def trace_epoch(self) -> tuple[int, ...]:
        """`core.api.Tracing`. The delegators a graph baked. A per-block step run in between
        re-resolves them: streaming with a prefix cache takes that path and streaming
        without takes this one, so the same resident model alternates."""
        return tuple(block.epoch for block in self.model.layers)

    def raw_embed(self, ids: mx.array) -> mx.array:
        """`core.api.Draftable`. The trunk puts nothing over the lookup, so the raw
        embedding and the cooked one are the same tensor."""
        return self.model.embed_tokens(ids)

    def raw_logits(self, hidden: mx.array) -> mx.array:
        """`core.api.Draftable`. `model.norm` is *not* applied: the rows an MTP step
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
        self, ids: mx.array, cache: Sequence[LayerCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """`core.api.Draftable`: the same forward, plus the rows `at` names concatenated
        on the last dim. What reads it is an MTP step, which asks for one — the last."""
        activations = self.activations(ids, cache)
        return activations.logits, self._tapped(activations, at)

    def checkpoints(self, cache: Sequence[LayerCache], rows: int) -> bool:
        """`core.api.Draftable`. Yes, when every DeltaNet layer's rule is the fused one:
        that is the kernel `Qwen35DeltaNet.verify_rows` writes the per-row states out of,
        and the shapes it declines are shapes it cannot tile. Both of its routes are held
        to it — the unrolled walk stores the same slots, and the A/B flag between them can
        move under a running generation, so the answer cannot depend on which is live.

        Three of every four layers here are DeltaNet and none of them can be trimmed, so
        without this a rejected round is a second forward of the whole trunk.
        """
        del cache, rows
        from mlx_omnia.engine.core.kernels.gated_delta import FusedGatedDelta, GatedDelta

        for block, kind in zip(
            self.model.layers, self.config.text_config.layer_types, strict=True
        ):
            if kind == "full_attention":
                continue
            assert isinstance(block, Qwen35Block)
            rule = block.linear_attn.rule()
            if not isinstance(rule, GatedDelta) or not isinstance(rule.strategy, FusedGatedDelta):
                return False
        return True

    def activations(
        self,
        ids: mx.array,
        cache: Sequence[Qwen35Layer] | None = None,
        *,
        positions: mx.array | None = None,
        embeddings: mx.array | None = None,
    ) -> Qwen35Activations:
        """`embeddings` replaces the token lookup (an image's rows are already spliced
        in) and `positions` is the `[3, L]` MRoPE clock; both default to the text path."""
        layers: Sequence[Qwen35Layer] = self.make_cache() if cache is None else cache
        x = embeddings if embeddings is not None else self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, layers, strict=True):
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
        cache: Sequence[Qwen35Layer] | None = None,
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


class Qwen35LanguageModel(TextLanguageModel[Qwen35Input]):
    """The chassis with the vision path over it: everything a text request needs — the trie,
    the MTP door, continuous batching — is `language.TextLanguageModel`'s, and what is the
    family's is the picture."""

    model: Qwen35

    def __init__(
        self,
        model: Qwen35,
        tokenizer: Tokenizer,
        processor: ProcessorConfig | None,
        *,
        stop: Collection[int] = (),
    ) -> None:
        super().__init__(model, tokenizer, stop=stop)
        self.processor = processor
        self._vision = sees(model.config, processor)

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
        assert isinstance(self.model, Draftable)
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
        """A text prompt is the chassis' generation — batched when the trunk batches; what is
        answered here is the one thing the chassis has no path for, a prompt with a picture
        in it."""
        if isinstance(input, Text):
            yield from super().stream(input, options)
            return
        parts = input.parts if isinstance(input, LanguagePrompt) else ()
        rendered = "".join(part.read() for part in parts if isinstance(part, Text))
        prompt = self.prepare(input)
        # No prefix here, and it is not an omission: an image is one id repeated per patch,
        # so two different pictures on the same grid produce the same ids — a trie keyed on
        # ids would hand one image's attention to another.
        ids = stream_multimodal_ids(
            self.model,
            prompt,
            max_tokens=options.max_tokens,
            sampler=options.sampler,
            stop=self.stop if options.stop is None else options.stop,
            penalty=options.penalty,
            meter=options.meter,
        )
        yield from stream_text(ids, self.tokenizer, parser=_parser(input), prompt=rendered)

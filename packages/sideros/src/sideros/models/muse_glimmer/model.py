from collections.abc import Collection, Iterator, Sequence
from typing import NamedTuple, TypeIs, assert_never

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sideros.core.cache import KVCache
from sideros.core.prefill import prefill
from sideros.core.prompt_cache import PromptCache
from sideros.generate import Meter, Penalty, Sampler, greedy, stream_ids, stream_text
from sideros.language import (
    TEXT,
    GenerationOptions,
    LanguagePrompt,
    Text,
    Tokenizer,
    prefix_cache,
)
from sideros.model import ModelInput, ModelSignature
from sideros.models.muse_glimmer.config import MuseGlimmerConfig
from sideros.models.muse_glimmer.dflash import MuseGlimmerAssistant, MuseGlimmerDFlash
from sideros.models.muse_glimmer.layers.block import MuseGlimmerTrunk
from sideros.models.muse_glimmer.vision import (
    Grid,
    MuseGlimmerVision,
    ProcessorConfig,
    VisionAdapter,
    process_image,
)
from sideros.parsers import Parser, Segment
from sideros.vision import RGB_IMAGE, Image


class MuseGlimmerActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class MuseGlimmer(nn.Module):
    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.config = config.text_config
        self.full_config = config
        self.model = MuseGlimmerTrunk(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if config.vision_config is not None:
            self.visual = MuseGlimmerVision(config.vision_config)
            self.vision_adapter = VisionAdapter(
                config.out_hidden_size, config.projector_hidden_size
            )
            self.vision_projection = nn.Linear(
                config.projector_hidden_size, self.config.hidden_size, bias=False
            )

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def embed(self, ids: mx.array) -> mx.array:
        return mx.fast.rms_norm(self.model.embed_tokens(ids), None, self.config.rms_norm_eps)

    def image_features(self, patches: mx.array, grid: Grid) -> mx.array:
        """One row per trunk token: tower, adapter, projection, then the same weightless
        RMS the text embeddings get (`perception_emb_norm`)."""
        tower = self.visual
        assert isinstance(tower, MuseGlimmerVision)
        adapted = self.vision_adapter(tower(patches, grid))
        projected = self.vision_projection(adapted)
        return mx.fast.rms_norm(projected, None, self.config.rms_norm_eps)

    def head(self, normed: mx.array) -> mx.array:
        cap = self.config.final_logit_softcapping
        logits = self.lm_head(normed) * self.config.output_multiplier
        return mx.tanh(logits / cap) * cap

    def activations(
        self,
        ids: mx.array,
        cache: list[KVCache] | None = None,
        *,
        embeddings: mx.array | None = None,
    ) -> MuseGlimmerActivations:
        """`embeddings` replaces the token lookup — an image's rows are already spliced
        in, already normed."""
        cache = cache if cache is not None else self.make_cache()
        x = embeddings if embeddings is not None else self.embed(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return MuseGlimmerActivations(embedded, blocks, normed, self.head(normed))

    def block_outputs(
        self, ids: mx.array, cache: list[KVCache], *, at: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """The forward, plus the output of blocks `at` concatenated on the last dim —
        `generate.BlockOutputs`. What reads it is the DFlash drafter, whose whole input is
        the trunk's own reading of the context; the five it asks for out of 52 are the
        selection this signature exists for."""
        x = self.embed(ids)
        wanted = set(at)
        kept: dict[int, mx.array] = {}
        for index, (block, layer_cache) in enumerate(zip(self.model.layers, cache, strict=True)):
            x = block(x, layer_cache)
            if index in wanted:
                kept[index] = x
        features = mx.concatenate([kept[index] for index in at], axis=-1)
        return self.head(self.model.norm(x)), features

    def __call__(
        self,
        ids: mx.array,
        cache: list[KVCache] | None = None,
        *,
        embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.activations(ids, cache, embeddings=embeddings).logits


class MultimodalPrompt(NamedTuple):
    """A prompt whose image rows are already spliced in. The trunk runs a plain 1-D
    clock — no positional algebra beyond the cache offset."""

    ids: list[int]
    embeddings: mx.array


def multimodal_prompt(
    model: MuseGlimmer, ids: Sequence[int], images: Sequence[tuple[mx.array, Grid]]
) -> MultimodalPrompt:
    """Runs the tower over each image and scatters its rows onto the placeholder run."""
    config = model.full_config
    if config.vision_config is None:
        raise ValueError("this checkpoint carries no vision tower")

    tokens = model.embed(mx.array(list(ids)))[None]
    rows = (
        mx.concatenate(
            [
                model.image_features(patches.astype(tokens.dtype), grid)
                for patches, grid in images
            ],
            axis=0,
        )
        if images
        else None
    )
    if rows is not None:
        placeholder = mx.array(list(ids)) == config.image_token_id
        indices = mx.array(np.flatnonzero(np.array(placeholder)))
        if indices.shape[0] != rows.shape[0]:
            raise ValueError(
                f"{rows.shape[0]} image rows for {indices.shape[0]} placeholder tokens"
            )
        tokens[0, indices] = rows.astype(tokens.dtype)
    return MultimodalPrompt(list(ids), tokens)


def stream_multimodal_ids(
    model: MuseGlimmer,
    prompt: MultimodalPrompt,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    meter: Meter | None = None,
) -> Iterator[int]:
    """The chassis' lazy loop with the prompt's embeddings fed at prefill; decode is the
    plain text path."""
    cache = model.make_cache()
    history = mx.array(prompt.ids)
    if meter is not None:
        meter.prefill(len(prompt.ids))

    def step(ids: mx.array, embeddings: mx.array | None) -> mx.array:
        logits = model(ids, cache, embeddings=embeddings)[:, -1, :]
        return sampler(logits if penalty is None else penalty(logits, history))[0]

    ids = mx.array(prompt.ids)

    def feed(part: slice) -> mx.array:
        return model(ids[part][None], cache, embeddings=prompt.embeddings[:, part])

    window = prefill(feed, ids.size, cache)
    y = step(ids[window][None], prompt.embeddings[:, window])
    mx.async_eval(y)
    for _ in range(max_tokens):
        if penalty is not None:
            history = mx.concatenate([history, y[None]])
        next_y = step(y[None, None], None)
        mx.async_eval(next_y)
        token = y.item()
        assert isinstance(token, int)
        if token in stop:
            return
        if meter is not None:
            meter.token()
        yield token
        y = next_y


type MuseGlimmerInput = Text | Image | LanguagePrompt


def _parser(input: MuseGlimmerInput) -> Parser | None:
    match input:
        case Text():
            return input.parser
        case LanguagePrompt(parts=parts):
            return next((part.parser for part in parts if isinstance(part, Text)), None)
        case Image():
            return None
        case _:
            assert_never(input)


class MuseGlimmerLanguageModel:
    def __init__(
        self,
        model: MuseGlimmer,
        tokenizer: Tokenizer,
        processor: ProcessorConfig | None,
        *,
        stop: Collection[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.stop = stop
        self.prefix: PromptCache[KVCache] | None = None
        self.drafter: nn.Module | None = None
        """The DFlash checkpoint this model speculates with, when it was given one
        (`speculative.Drafting`). It is not part of `model` — two checkpoints, two trees —
        and it is `None` for every model nobody paired one with."""
        self._draft_block: int | None = None
        config = model.full_config
        self._vision = (
            processor is not None
            and config.vision_config is not None
            and config.image_token_id >= 0
        )

    @property
    def native_signature(self) -> ModelSignature:
        inputs = frozenset({TEXT, RGB_IMAGE}) if self._vision else frozenset({TEXT})
        return ModelSignature(inputs, frozenset({TEXT}))

    def speculate_with(self, drafter: nn.Module, *, block_size: int | None = None) -> None:
        """Take a DFlash drafter for this target — `speculative.Drafting`.

        The three checks are what a checkpoint id cannot promise: the tree is the right
        architecture, it folds the width this trunk produces, and every block it reads
        exists in this trunk. A drafter trained against another model of the same
        architecture passes all three and drafts badly, which costs speed and never
        correctness — the target's own argmax is what accepts."""
        if not isinstance(drafter, MuseGlimmerAssistant):
            raise TypeError(f"{type(drafter).__name__} is not a Muse-Glimmer DFlash drafter")
        config = drafter.config
        if config.hidden_size != self.model.config.hidden_size:
            raise ValueError(
                f"the drafter folds {config.hidden_size}-wide blocks and this trunk is "
                f"{self.model.config.hidden_size}-wide"
            )
        if max(config.target_layer_ids) >= self.model.config.num_hidden_layers:
            raise ValueError(
                f"the drafter reads block {max(config.target_layer_ids)} and this trunk has "
                f"{self.model.config.num_hidden_layers}"
            )
        self.drafter = drafter
        self._draft_block = block_size

    @property
    def image_marker(self) -> str | None:
        """The chat template emits one bare `<|patch|>` per image — no start/end
        wrapper; the processor expands it to the grid's token count."""
        if not self._vision:
            return None
        marker = [self.model.full_config.image_token_id]
        return self.tokenizer.decode_bytes(marker).decode("utf-8")

    def accepts(self, input: ModelInput) -> TypeIs[MuseGlimmerInput]:
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
        config = self.model.full_config
        processor = self.processor
        vision = config.vision_config
        if not self._vision or processor is None or vision is None:
            raise TypeError("this language model does not accept images")

        parts = input.parts if isinstance(input, LanguagePrompt) else (input,)
        ids: list[int] = []
        images: list[tuple[mx.array, Grid]] = []
        for part in parts:
            match part:
                case Text(value=value):
                    ids.extend(self.tokenizer.encode(value))
                case Image(pixels=pixels):
                    processed = process_image(pixels, processor)
                    _, grid = processed
                    ids.extend([config.image_token_id] * grid.tokens(vision.merge_size))
                    images.append(processed)
                case _:
                    raise TypeError(f"unsupported prompt part {type(part).__name__}")
        return multimodal_prompt(self.model, ids, images)

    def _proposer(self, options: GenerationOptions) -> MuseGlimmerDFlash | None:
        """The drafter for this request, or `None` — which is every request that cannot be
        verified against the target's own argmax.

        Not a refusal: speculation is greedy-only here (`speculative`'s docstring), and a
        request that samples, penalizes or decodes under a grammar is answered the way it
        would have been with no drafter loaded at all. What the switch buys is speed, so
        the case where it buys nothing is a case where it does nothing.

        One proposer per generation: the cache under it is this request's accepted prefix.
        """
        if self.drafter is None or not options.speculate:
            return None
        if options.sampler is not greedy or options.penalty is not None:
            return None
        if options.constraint is not None:
            return None
        assert isinstance(self.drafter, MuseGlimmerAssistant)
        return MuseGlimmerDFlash(self.model, self.drafter, block_size=self._draft_block)

    def stream(self, input: MuseGlimmerInput, options: GenerationOptions) -> Iterator[Segment]:
        stop = self.stop if options.stop is None else options.stop
        parser = _parser(input)
        match input:
            case Text(value=rendered):
                self.prefix = prefix_cache(self.prefix, options.prefix_budget)
                draft = self._proposer(options)
                ids = stream_ids(
                    self.model,
                    self.tokenizer.encode(rendered),
                    max_tokens=options.max_tokens,
                    sampler=options.sampler,
                    stop=stop,
                    penalty=options.penalty,
                    meter=options.meter,
                    # The two do not compose (`stream_ids` says why), and a round is worth
                    # more than a prefix: the trie saves the prefill of a turn, the drafter
                    # saves a read of the weights per token of every turn.
                    prefix=None if draft is not None else self.prefix,
                    constraint=options.constraint,
                    draft=draft,
                )
            case Image() | LanguagePrompt():
                parts = input.parts if isinstance(input, LanguagePrompt) else ()
                rendered = "".join(part.value for part in parts if isinstance(part, Text))
                prompt = self.prepare(input)
                # No prefix trie: an image is one id repeated per patch, so two different
                # pictures on the same grid produce the same ids.
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

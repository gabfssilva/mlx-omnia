from collections.abc import Collection, Iterator, Sequence
from typing import NamedTuple, TypeIs, assert_never

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sideros.core.cache import DeltaCache, KVCache
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
from sideros.models.qwen3_5.config import Qwen35Config
from sideros.models.qwen3_5.layers.block import Qwen35Trunk
from sideros.models.qwen3_5.vision import (
    Grid,
    ProcessorConfig,
    Qwen35Vision,
    multimodal_positions,
    process_image,
)
from sideros.parsers import Parser, Segment
from sideros.vision import RGB_IMAGE, Image


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

    def make_cache(self) -> list[KVCache | DeltaCache]:
        return [
            KVCache() if kind == "full_attention" else DeltaCache()
            for kind in self.config.text_config.layer_types
        ]

    def activations(
        self,
        ids: mx.array,
        cache: list[KVCache | DeltaCache] | None = None,
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
        cache: list[KVCache | DeltaCache] | None = None,
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
        config = model.config
        self._vision = (
            processor is not None
            and config.vision_config is not None
            and config.image_token_id >= 0
            and config.vision_start_token_id >= 0
            and config.vision_end_token_id >= 0
        )

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
                self.prefix = prefix_cache(self.prefix, options.prefix_budget)
                ids = stream_ids(
                    self.model,
                    self.tokenizer.encode(rendered),
                    max_tokens=options.max_tokens,
                    sampler=options.sampler,
                    stop=stop,
                    penalty=options.penalty,
                    meter=options.meter,
                    prefix=self.prefix,
                    constraint=options.constraint,
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

from collections.abc import Collection, Iterator, Sequence
from typing import NamedTuple, TypeIs, assert_never

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.attention import ragged_mask
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.prefill import prefill
from mlx_omnia.engine.generate import Meter, Penalty, Sampler, greedy, stream_ids, stream_text
from mlx_omnia.engine.language import (
    TEXT,
    GenerationOptions,
    LanguagePrompt,
    Text,
    Tokenizer,
    prefix_cache,
)
from mlx_omnia.engine.model import ModelInput, ModelSignature
from mlx_omnia.engine.models.step3p7.config import FULL, SLIDING, Step3p7Config
from mlx_omnia.engine.models.step3p7.layers.block import Step3p7Trunk
from mlx_omnia.engine.models.step3p7.vision import Step3p7Vision
from mlx_omnia.engine.parsers import Parser, Segment
from mlx_omnia.engine.processors.step3p7 import ImageFeatures, Step3p7Processor, process_images
from mlx_omnia.engine.vision import RGB_IMAGE, Image


class Step3p7Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Step3p7(nn.Module):
    """The VLM: step3p5 trunk + perception_encoder vision tower + linear projector.
    Image features are scattered into ``input_ids == image_token_id`` via masked_scatter."""

    continuous_batching = True

    def __init__(self, config: Step3p7Config) -> None:
        super().__init__()
        self.config = config
        text = config.text_config
        self.model = Step3p7Trunk(text)
        vision = config.vision_config
        if vision is not None:
            self.vision_model = Step3p7Vision(vision)
            self.vit_large_projector = nn.Linear(
                vision.width * 4, text.hidden_size, bias=config.projector_bias
            )
        if not config.tied:
            self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def activations(
        self,
        ids: mx.array,
        cache: Sequence[KVStore] | None = None,
        *,
        embeddings: mx.array | None = None,
    ) -> Step3p7Activations:
        text = self.config.text_config
        cache = cache if cache is not None else self.make_cache()
        x = embeddings if embeddings is not None else self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        types = text.types
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if SLIDING in types:
            if isinstance(offset, mx.array):
                sliding = ragged_mask(length, offset, text.sliding_window)
            else:
                sliding = self._sliding_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(self.model.layers, types, cache, strict=True):
            mask = full if kind == FULL else sliding
            x = block(x, mask, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tied:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Step3p7Activations(blocks, logits)

    def __call__(
        self,
        ids: mx.array,
        cache: Sequence[KVStore] | None = None,
        *,
        embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.activations(ids, cache, embeddings=embeddings).logits

    def _sliding_mask(self, length: int, offset: int) -> mx.array | str | None:
        window = self.config.text_config.sliding_window
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)


class Step3p7Prompt(NamedTuple):
    ids: list[int]
    embeddings: mx.array | None


type Step3p7Input = Text | Image | LanguagePrompt


def stream_step3p7_ids(
    model: Step3p7,
    prompt: Step3p7Prompt,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    meter: Meter | None = None,
) -> Iterator[int]:
    """The chassis' lazy loop with the scattered embeddings threaded through: prefill
    reads the prompt's own embeddings, and every step after it uses plain token
    lookup (no image tokens in the decode)."""
    cache = model.make_cache()
    history = mx.array(prompt.ids)
    if meter is not None:
        meter.prefill(len(prompt.ids))

    def step(ids: mx.array, emb: mx.array | None) -> mx.array:
        logits = model(ids, cache, embeddings=emb)[:, -1, :]
        return sampler(logits if penalty is None else penalty(logits, history))[0]

    ids = mx.array(prompt.ids)
    embeddings = prompt.embeddings

    def feed(block: slice) -> mx.array:
        return model(
            ids[block][None],
            cache,
            embeddings=None if embeddings is None else embeddings[:, block],
        )

    window = prefill(feed, ids.size, cache)
    y = step(ids[window][None], None if embeddings is None else embeddings[:, window])
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


def sees(config: Step3p7Config, processor: Step3p7Processor | None) -> bool:
    """Whether this checkpoint takes images: a tower in the config, the special ids the
    processor reads out of `tokenizer_config.json`, and the id the marker is spelled with.
    The catalog asks the same question of a directory it has not loaded, and the two
    answers have to be one function."""
    return (
        processor is not None
        and config.vision_config is not None
        and config.image_token_id >= 0
    )


class Step3p7LanguageModel:
    def __init__(
        self,
        model: Step3p7,
        tokenizer: Tokenizer,
        processor: Step3p7Processor | None,
        *,
        stop: Collection[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.stop = stop
        self.prefix: object | None = None
        self._vision = sees(model.config, processor)

    @property
    def native_signature(self) -> ModelSignature:
        inputs = frozenset({TEXT, RGB_IMAGE}) if self._vision else frozenset({TEXT})
        return ModelSignature(inputs, frozenset({TEXT}))

    @property
    def image_marker(self) -> str | None:
        if not self._vision:
            return None
        config = self.model.config
        return self.tokenizer.decode_bytes([config.image_token_id]).decode("utf-8")

    @property
    def config(self) -> Step3p7Config:
        return self.model.config

    def accepts(self, input: ModelInput) -> TypeIs[Step3p7Input]:
        match input:
            case Text():
                return True
            case Image():
                return self._vision
            case LanguagePrompt():
                return self._vision and input.content_types <= self.native_signature.inputs
            case _:
                return False

    def prepare(self, input: Image | LanguagePrompt) -> Step3p7Prompt:
        config = self.model.config
        processor = self.processor
        if not self._vision or processor is None:
            raise TypeError("this language model does not accept images")

        parts = input.parts if isinstance(input, LanguagePrompt) else (input,)
        left_ids: list[int] = []
        right_ids: list[int] = []
        images: list[Image] = []
        seen_image = False
        for part in parts:
            match part:
                case Text(value=value):
                    if seen_image:
                        right_ids.extend(self.tokenizer.encode(value))
                    else:
                        left_ids.extend(self.tokenizer.encode(value))
                case Image():
                    images.append(part)
                    seen_image = True
                case _:
                    raise TypeError(f"unsupported prompt part {type(part).__name__}")

        image_features = process_images(images, processor, config)
        ids = image_features.assemble_ids(left_ids, right_ids)
        embeddings = self._vision_embeddings(mx.array(ids)[None], image_features)
        return Step3p7Prompt(ids, embeddings)

    def stream(self, input: Step3p7Input, options: GenerationOptions) -> Iterator[Segment]:
        stop = self.stop if options.stop is None else options.stop
        parser = _parser(input)
        match input:
            case Text(value=rendered):
                self.prefix = prefix_cache(self.prefix, options.prefix_budget)  # type: ignore[arg-type]
                ids = stream_ids(
                    self.model,
                    self.tokenizer.encode(rendered),
                    max_tokens=options.max_tokens,
                    sampler=options.sampler,
                    stop=stop,
                    penalty=options.penalty,
                    meter=options.meter,
                    prefix=self.prefix,  # type: ignore[arg-type]
                    constraint=options.constraint,
                )
            case Image() | LanguagePrompt():
                parts = input.parts if isinstance(input, LanguagePrompt) else ()
                rendered = "".join(part.value for part in parts if isinstance(part, Text))
                prompt = self.prepare(input)
                ids = stream_step3p7_ids(
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

    def _vision_embeddings(
        self, ids: mx.array, image_features: ImageFeatures
    ) -> mx.array:
        tokens = self.model.model.embed_tokens(ids)[None]
        if image_features.num_images == 0:
            return tokens
        tower = self.model.vision_model
        assert isinstance(tower, Step3p7Vision)
        all_features = tower.process(image_features)
        projected = self.model.vit_large_projector(all_features)
        projected = projected.astype(tokens.dtype)
        image_mask = ids == self.config.image_token_id
        indices = mx.array(np.flatnonzero(np.array(image_mask)))
        if indices.shape[0] != projected.shape[0]:
            raise ValueError(
                f"{projected.shape[0]} image rows for {indices.shape[0]} placeholder tokens"
            )
        tokens[0, indices] = projected
        return tokens


def _parser(input: Step3p7Input) -> Parser | None:
    match input:
        case Text():
            return input.parser
        case LanguagePrompt(parts=parts):
            return next((part.parser for part in parts if isinstance(part, Text)), None)
        case Image():
            return None
        case _:
            assert_never(input)

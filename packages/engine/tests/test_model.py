from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeIs

import pytest

from mlx_omnia.language import Text
from mlx_omnia.model import (
    AtomicInput,
    Capability,
    ContentType,
    Modality,
    Model,
    ModelInput,
    ModelSignature,
)

TEXT = ContentType(Modality.TEXT, "text/plain")
TOKENS = ContentType(Modality.TEXT, "application/x.tokens")
PDF = ContentType(Modality.DOCUMENT, "application/pdf")
IMAGE = ContentType(Modality.IMAGE, "image/rgb")


@dataclass(frozen=True)
class TextInput:
    value: str

    @property
    def content_type(self) -> ContentType:
        return TEXT


@dataclass(frozen=True)
class TokenInput:
    values: tuple[int, ...]

    @property
    def content_type(self) -> ContentType:
        return TOKENS


@dataclass(frozen=True)
class PDFDocument:
    data: bytes

    @property
    def content_type(self) -> ContentType:
        return PDF


@dataclass(frozen=True)
class ImageInput:
    pixels: tuple[int, ...]

    @property
    def content_type(self) -> ContentType:
        return IMAGE


@dataclass(frozen=True)
class Options:
    repeats: int = 1


class TextModel:
    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[TextInput]:
        return isinstance(input, TextInput)

    def stream(self, input: TextInput, options: Options) -> Iterator[str]:
        for _ in range(options.repeats):
            yield input.value


def use_text_model(model: Model[TextInput, str, Options], input: ModelInput) -> str:
    if model.accepts(input):
        return "".join(model.stream(input, Options()))
    return ""


def test_model_is_structural_and_accepts_narrows_input() -> None:
    model = TextModel()
    assert isinstance(model, Model)
    assert use_text_model(model, TextInput("hello")) == "hello"
    assert use_text_model(model, TokenInput((1, 2))) == ""


def test_signature_derives_modalities() -> None:
    signature = ModelSignature(
        frozenset({TEXT, TOKENS}),
        frozenset({ContentType(Modality.VECTOR, "application/x.embedding")}),
    )
    assert signature.modalities.inputs == frozenset({Modality.TEXT})
    assert signature.modalities.outputs == frozenset({Modality.VECTOR})


def test_atomic_input_has_one_content_type() -> None:
    input: AtomicInput = TextInput("hello")
    assert input.content_type == TEXT


def test_model_can_emit_one_item() -> None:
    assert list(TextModel().stream(TextInput("one"), Options())) == ["one"]


def test_model_can_emit_multiple_items() -> None:
    assert list(TextModel().stream(TextInput("many"), Options(repeats=3))) == [
        "many",
        "many",
        "many",
    ]


class PDFToText:
    input_type = PDF
    target_types = frozenset({TEXT})

    def accepts(self, input: ModelInput) -> bool:
        return isinstance(input, PDFDocument)

    def prepare(self, input: ModelInput) -> Text:
        if not isinstance(input, PDFDocument):
            raise TypeError(f"expected PDFDocument, got {type(input).__name__}")
        return Text(input.data.decode())


@dataclass(frozen=True)
class CaptionOptions:
    prefix: str


class CaptionModel:
    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({IMAGE}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[ImageInput]:
        return isinstance(input, ImageInput)

    def stream(self, input: ImageInput, options: CaptionOptions) -> Iterator[str]:
        yield options.prefix
        yield str(sum(input.pixels))


class ImageToText:
    input_type = IMAGE
    target_types = frozenset({TEXT})

    def __init__(
        self,
        model: Model[ImageInput, str, CaptionOptions],
        options: CaptionOptions,
    ) -> None:
        self.model = model
        self.options = options

    def accepts(self, input: ModelInput) -> bool:
        return isinstance(input, ImageInput)

    def prepare(self, input: ModelInput) -> Text:
        if not isinstance(input, ImageInput):
            raise TypeError(f"expected ImageInput, got {type(input).__name__}")
        return Text("".join(self.model.stream(input, self.options)))


def test_python_capability_prepares_the_target_input() -> None:
    capability = PDFToText()
    assert isinstance(capability, Capability)
    assert capability.prepare(PDFDocument(b"paper")) == Text("paper")


def test_model_backed_capability_prepares_the_target_input() -> None:
    capability = ImageToText(CaptionModel(), CaptionOptions("pixels="))
    assert capability.prepare(ImageInput((1, 2, 3))) == Text("pixels=6")


def test_capability_rejects_the_wrong_concrete_input() -> None:
    with pytest.raises(TypeError, match="expected PDFDocument, got ImageInput"):
        PDFToText().prepare(ImageInput((1,)))


class FailingCaptionModel(CaptionModel):
    def stream(self, input: ImageInput, options: CaptionOptions) -> Iterator[str]:
        raise RuntimeError("caption failed")
        yield


def test_model_backed_capability_propagates_internal_errors() -> None:
    capability = ImageToText(FailingCaptionModel(), CaptionOptions(""))
    with pytest.raises(RuntimeError, match="caption failed"):
        capability.prepare(ImageInput((1,)))

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeIs

import pytest

from mlx_omnia.language import TEXT, LanguagePrompt, Text
from mlx_omnia.model import (
    Capability,
    CompositeModel,
    ContentType,
    DuplicateCapability,
    IncompatibleCapabilityTarget,
    InvalidCapabilityOutput,
    Modality,
    ModelInput,
    ModelSignature,
    NativeInputOverride,
    UnsupportedInput,
)

PDF = ContentType(Modality.DOCUMENT, "application/pdf")
IMAGE = ContentType(Modality.IMAGE, "image/rgb")
AUDIO = ContentType(Modality.AUDIO, "audio/pcm")
EMBEDDING = ContentType(Modality.VECTOR, "application/x.embedding")


@dataclass(frozen=True)
class PDFDocument:
    value: str

    @property
    def content_type(self) -> ContentType:
        return PDF


@dataclass(frozen=True)
class Image:
    value: str

    @property
    def content_type(self) -> ContentType:
        return IMAGE


@dataclass(frozen=True)
class Options:
    suffix: str


class RecordingModel:
    def __init__(
        self,
        input_type: ContentType = TEXT,
        output_type: ContentType = TEXT,
    ) -> None:
        self.input_type = input_type
        self.output_type = output_type
        self.calls: list[tuple[ModelInput, Options]] = []

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({self.input_type}), frozenset({self.output_type}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text) and input.content_type == self.input_type

    def stream(self, input: Text, options: Options) -> Iterator[str]:
        self.calls.append((input, options))
        yield input.value + options.suffix


class PDFToText:
    input_type = PDF
    target_types = frozenset({TEXT})

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls = 0
        self.events = events

    def accepts(self, input: ModelInput) -> bool:
        return isinstance(input, PDFDocument)

    def prepare(self, input: ModelInput) -> Text:
        if not isinstance(input, PDFDocument):
            raise TypeError
        self.calls += 1
        if self.events is not None:
            self.events.append("prepare")
        return Text(input.value)


def test_native_route_preserves_output_and_options() -> None:
    base = RecordingModel()
    options = Options("!")
    composite = CompositeModel(base, [])
    assert list(composite.stream(Text("native"), options)) == ["native!"]
    assert base.calls == [(Text("native"), options)]


def test_pluggable_route_prepares_once_before_base_stream() -> None:
    events: list[str] = []
    capability = PDFToText(events)

    class OrderedModel(RecordingModel):
        def stream(self, input: Text, options: Options) -> Iterator[str]:
            events.append("stream")
            yield from super().stream(input, options)

    composite = CompositeModel(OrderedModel(), [capability])
    assert list(composite.stream(PDFDocument("paper"), Options("?"))) == ["paper?"]
    assert capability.calls == 1
    assert events == ["prepare", "stream"]


def test_unsupported_input_is_rejected() -> None:
    composite = CompositeModel(RecordingModel(), [])
    with pytest.raises(UnsupportedInput) as error:
        list(composite.stream(PDFDocument("paper"), Options("")))
    assert error.value.content_type == PDF


def test_unsupported_aggregate_input_reports_its_part_types() -> None:
    composite = CompositeModel(RecordingModel(), [])
    prompt = LanguagePrompt((Image("image"),))
    with pytest.raises(UnsupportedInput) as error:
        list(composite.stream(prompt, Options("")))
    assert error.value.content_types == frozenset({IMAGE})


def test_duplicate_capability_is_rejected() -> None:
    with pytest.raises(DuplicateCapability):
        CompositeModel(RecordingModel(), [PDFToText(), PDFToText()])


class TextToText(PDFToText):
    input_type = TEXT


def test_native_input_cannot_be_replaced() -> None:
    with pytest.raises(NativeInputOverride):
        CompositeModel(RecordingModel(), [TextToText()])


class PDFToImage(PDFToText):
    target_types = frozenset({IMAGE})


def test_capability_target_must_be_native() -> None:
    with pytest.raises(IncompatibleCapabilityTarget):
        CompositeModel(RecordingModel(), [PDFToImage()])


class BrokenPDFToText(PDFToText):
    def prepare(self, input: ModelInput) -> Text:
        return Text("wrong")


class RejectingModel(RecordingModel):
    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return False


def test_prepared_output_is_checked_at_runtime() -> None:
    composite = CompositeModel(RejectingModel(), [BrokenPDFToText()])
    with pytest.raises(InvalidCapabilityOutput) as error:
        list(composite.stream(PDFDocument("paper"), Options("")))
    assert error.value.input_type == PDF
    assert error.value.output_type == TEXT


def test_signatures_modalities_and_sources_remain_distinguishable() -> None:
    base = RecordingModel(output_type=EMBEDDING)
    capability = PDFToText()
    capabilities: list[Capability[Text]] = [capability]
    composite = CompositeModel(base, capabilities)
    capabilities.clear()

    assert composite.native_signature == ModelSignature(
        frozenset({TEXT}), frozenset({EMBEDDING})
    )
    assert composite.signature == ModelSignature(
        frozenset({TEXT, PDF}), frozenset({EMBEDDING})
    )
    assert composite.modalities.inputs == frozenset({Modality.TEXT, Modality.DOCUMENT})
    assert composite.modalities.outputs == frozenset({Modality.VECTOR})
    assert composite.input_sources[TEXT] is base
    assert composite.input_sources[PDF] is capability
    assert composite.accepts(Text("native"))
    assert composite.accepts(PDFDocument("plugged"))
    assert not composite.accepts(Image("unsupported"))


@pytest.mark.parametrize(
    ("output_type", "expected"),
    [
        (TEXT, "language"),
        (IMAGE, "vision"),
        (AUDIO, "speech"),
        (EMBEDDING, "embedding"),
    ],
)
def test_same_composite_preserves_different_task_outputs(
    output_type: ContentType,
    expected: str,
) -> None:
    model = RecordingModel(output_type=output_type)
    composite = CompositeModel(model, [PDFToText()])
    assert list(composite.stream(PDFDocument(expected), Options(""))) == [expected]

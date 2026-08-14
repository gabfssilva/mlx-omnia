from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, TypeIs, runtime_checkable


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    VECTOR = "vector"


@dataclass(frozen=True)
class ContentType:
    modality: Modality
    media_type: str


@runtime_checkable
class AtomicInput(Protocol):
    @property
    def content_type(self) -> ContentType: ...


@runtime_checkable
class AggregateInput(Protocol):
    @property
    def content_types(self) -> frozenset[ContentType]: ...


type ModelInput = AtomicInput | AggregateInput


@dataclass(frozen=True)
class Modalities:
    inputs: frozenset[Modality]
    outputs: frozenset[Modality]


@dataclass(frozen=True)
class ModelSignature:
    inputs: frozenset[ContentType]
    outputs: frozenset[ContentType]

    @property
    def modalities(self) -> Modalities:
        return Modalities(
            frozenset(content.modality for content in self.inputs),
            frozenset(content.modality for content in self.outputs),
        )


@runtime_checkable
class Model[I: ModelInput, O, Options](Protocol):
    @property
    def native_signature(self) -> ModelSignature: ...

    def accepts(self, input: ModelInput) -> TypeIs[I]: ...

    def stream(self, input: I, options: Options) -> Iterator[O]: ...


@runtime_checkable
class Capability[N: ModelInput](Protocol):
    @property
    def input_type(self) -> ContentType: ...

    @property
    def target_types(self) -> frozenset[ContentType]: ...

    def accepts(self, input: ModelInput) -> bool: ...

    def prepare(self, input: ModelInput) -> N: ...


@runtime_checkable
class Wrapping(Protocol):
    """A model that delegates to the one beneath it. `model` is the name every facade in the
    package gives what it wraps — `CompositeModel` over the task-level model, that one over
    the `nn.Module` — and following it is what reaches, through a protocol that only declares
    `stream`, whatever the facade is hiding."""

    model: object


class DuplicateCapability(ValueError):
    def __init__(self, content_type: ContentType) -> None:
        self.content_type = content_type
        super().__init__(f"duplicate capability for {content_type.media_type}")


class NativeInputOverride(ValueError):
    def __init__(self, content_type: ContentType) -> None:
        self.content_type = content_type
        super().__init__(f"capability cannot replace native input {content_type.media_type}")


class IncompatibleCapabilityTarget(ValueError):
    def __init__(
        self,
        input_type: ContentType,
        target_types: frozenset[ContentType],
    ) -> None:
        self.input_type = input_type
        self.target_types = target_types
        super().__init__(f"capability for {input_type.media_type} has no compatible target")


class UnsupportedInput(TypeError):
    def __init__(self, input: ModelInput) -> None:
        if isinstance(input, AtomicInput):
            self.content_type: ContentType | None = input.content_type
            self.content_types = frozenset({input.content_type})
        else:
            self.content_type = None
            self.content_types = input.content_types
        formats = ", ".join(sorted(content.media_type for content in self.content_types))
        super().__init__(f"unsupported input types: {formats}")


class InvalidCapabilityOutput(TypeError):
    def __init__(self, input_type: ContentType, output_type: ContentType) -> None:
        self.input_type = input_type
        self.output_type = output_type
        super().__init__(
            f"capability for {input_type.media_type} produced {output_type.media_type}"
        )


class CompositeModel[N: ModelInput, O, Options]:
    def __init__(
        self,
        model: Model[N, O, Options],
        capabilities: Sequence[Capability[N]],
    ) -> None:
        native_inputs = model.native_signature.inputs
        indexed: dict[ContentType, Capability[N]] = {}
        for capability in capabilities:
            if capability.input_type in native_inputs:
                raise NativeInputOverride(capability.input_type)
            if capability.input_type in indexed:
                raise DuplicateCapability(capability.input_type)
            if not capability.target_types <= native_inputs:
                raise IncompatibleCapabilityTarget(
                    capability.input_type,
                    capability.target_types,
                )
            indexed[capability.input_type] = capability
        self.model = model
        self._capabilities: Mapping[ContentType, Capability[N]] = MappingProxyType(indexed)

    @property
    def native_signature(self) -> ModelSignature:
        return self.model.native_signature

    @property
    def signature(self) -> ModelSignature:
        return ModelSignature(
            self.native_signature.inputs | self._capabilities.keys(),
            self.native_signature.outputs,
        )

    @property
    def modalities(self) -> Modalities:
        return self.signature.modalities

    @property
    def input_sources(self) -> Mapping[ContentType, object]:
        sources: dict[ContentType, object] = {
            content_type: self.model for content_type in self.native_signature.inputs
        }
        sources.update(self._capabilities)
        return MappingProxyType(sources)

    def accepts(self, input: ModelInput) -> TypeIs[ModelInput]:
        if self.model.accepts(input):
            return True
        if not isinstance(input, AtomicInput):
            return False
        capability = self._capabilities.get(input.content_type)
        return capability is not None and capability.accepts(input)

    def prepare(self, input: ModelInput) -> N:
        """Resolve native input directly or through one registered capability."""
        if self.model.accepts(input):
            return input
        if not isinstance(input, AtomicInput):
            raise UnsupportedInput(input)
        capability = self._capabilities.get(input.content_type)
        if capability is None or not capability.accepts(input):
            raise UnsupportedInput(input)
        prepared = capability.prepare(input)
        if not self.model.accepts(prepared):
            raise InvalidCapabilityOutput(capability.input_type, prepared.content_type)
        return prepared

    def stream(self, input: ModelInput, options: Options) -> Iterator[O]:
        return self.model.stream(self.prepare(input), options)

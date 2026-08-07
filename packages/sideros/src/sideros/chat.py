"""The checkpoint's own chat template: the only thing that knows how this model spells a
conversation. Rendering produces the literal text — `<|im_start|>` and all — which the
tokenizer encodes back into the single ids of those added tokens.

The environment reproduces the one transformers renders with
(`chat_template_utils.py:489`): immutable sandbox, `trim_blocks`/`lstrip_blocks` on
(without them the template's own indentation lands in the prompt), `tojson` without HTML
escaping, `raise_exception` and `strftime_now`. What enters the scope besides the
arguments is the special-token map from `tokenizer_config.json`
(`template_kwargs = {**self.special_tokens_map, **kwargs}`,
`tokenization_utils_base.py:3120`).

One filter is ours and not transformers': `from_json`, for a template that reads a tool
call's `arguments` key by key and gets them as text — the shape the OpenAI dialect
delivers and the server forwards unchanged. Without it that render raises instead of
producing the prompt.

A base model has no template and gets no guessed one: an invented template produces
fluent, wrong text.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, runtime_checkable

import jinja2
import jinja2.ext
import jinja2.nodes
import jinja2.parser
import jinja2.runtime
from jinja2.sandbox import ImmutableSandboxedEnvironment

from sideros.language import TEXT, LanguagePrompt, Text
from sideros.model import AtomicInput, ContentType, Modality, ModelInput, Wrapping
from sideros.tools import ToolFamily, tool_family
from sideros.vision import RGB_IMAGE, Image

__all__ = [
    "CHAT",
    "Chat",
    "ChatCapability",
    "ChatMessage",
    "ChatTemplate",
    "ImageMarkerMismatch",
    "ImagePart",
    "MultimodalChatCapability",
    "TextPart",
    "chat_capabilities",
    "chat_template",
    "tool_family_of",
]

CHAT = ContentType(Modality.TEXT, "application/vnd.sideros.chat")


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    image: Image


type ContentPart = TextPart | ImagePart


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | Sequence[ContentPart]


@dataclass(frozen=True)
class Chat:
    """A conversation as model input. `tools` are JSON schemas in the shape the template
    expects; `enable_thinking` only reaches the template when someone sets it, which is how
    transformers treats its kwargs."""

    messages: tuple[ChatMessage, ...]
    tools: tuple[Mapping[str, object], ...] = ()
    enable_thinking: bool | None = None

    @property
    def content_type(self) -> ContentType:
        return CHAT


class ImageMarkerMismatch(ValueError):
    def __init__(self, images: int, markers: int) -> None:
        self.images = images
        self.markers = markers
        super().__init__(f"{images} images in the conversation, {markers} in the prompt")


def _images(chat: Chat) -> tuple[Image, ...]:
    """In document order, which is the order the markers come out of the render."""
    found: list[Image] = []
    for message in chat.messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for part in content:
            if part["type"] == "image":
                found.append(part["image"])
    return tuple(found)


class _TokenJson(TypedDict):
    content: str


type _Token = str | _TokenJson


class _ConfigJson(TypedDict):
    chat_template: NotRequired[str]
    bos_token: NotRequired[_Token | None]
    eos_token: NotRequired[_Token | None]
    unk_token: NotRequired[_Token | None]
    sep_token: NotRequired[_Token | None]
    pad_token: NotRequired[_Token | None]
    cls_token: NotRequired[_Token | None]
    mask_token: NotRequired[_Token | None]
    model_specific_special_tokens: NotRequired[dict[str, _Token]]


def _token(value: _Token | None) -> str | None:
    """Either the literal string, or the `{"content": ...}` a slow tokenizer writes."""
    if isinstance(value, str):
        return value
    return None if value is None else value["content"]


def _special_tokens(config: _ConfigJson) -> dict[str, str]:
    """The seven transformers names (`SPECIAL_TOKENS_ATTRIBUTES`) plus the model's own
    (`image_token` and friends), which arrive under `model_specific_special_tokens`."""
    named: dict[str, _Token | None] = {
        "bos_token": config.get("bos_token"),
        "eos_token": config.get("eos_token"),
        "unk_token": config.get("unk_token"),
        "sep_token": config.get("sep_token"),
        "pad_token": config.get("pad_token"),
        "cls_token": config.get("cls_token"),
        "mask_token": config.get("mask_token"),
    }
    specific = config.get("model_specific_special_tokens", {})
    tokens = {
        name: content
        for name, value in (named | specific).items()
        if (content := _token(value)) is not None
    }
    return tokens


class _Generation(jinja2.ext.Extension):
    """`{% generation %}…{% endgeneration %}` marks the assistant-generated span for
    training masks; for rendering it is transparent. Without the extension the tag is a
    syntax error, and LFM2.5 uses it (`chat_template.jinja:77`)."""

    tags = {"generation"}  # noqa: RUF012

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_emit"), [], [], body).set_lineno(lineno)

    def _emit(self, caller: jinja2.runtime.Macro) -> str:
        return caller()


def _compile(source: str, now: Callable[[], datetime]) -> jinja2.Template:
    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(
        value: object,
        ensure_ascii: bool = False,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format: str) -> str:
        return now().strftime(format)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[_Generation, jinja2.ext.loopcontrols]
    )
    environment.filters["tojson"] = tojson
    environment.filters["from_json"] = json.loads
    # Through `globals=` rather than `environment.globals[...]`: the environment's map is
    # inferred from jinja2's default namespace and rejects any other signature.
    return environment.from_string(
        source, globals={"raise_exception": raise_exception, "strftime_now": strftime_now}
    )


@dataclass(frozen=True)
class ChatTemplate:
    template: jinja2.Template
    special_tokens: Mapping[str, str]
    source: str
    """The jinja2 text this was compiled from. A compiled template cannot be read back into
    one, and the source is where the tool family is a fact of the checkpoint: the template
    writes the assistant's own calls when it replays a history, so the envelope it spells is
    the envelope the model emits. Guessed from the generated text instead, Qwen3.6's marker
    reads as Qwen's and its XML call goes to a parser that cannot read it."""

    @classmethod
    def from_source(
        cls,
        source: str,
        special_tokens: Mapping[str, str] | None = None,
        *,
        now: Callable[[], datetime] = datetime.now,
    ) -> "ChatTemplate":
        return cls(
            _compile(source, now), special_tokens if special_tokens is not None else {}, source
        )

    @property
    def tool_family(self) -> ToolFamily | None:
        return tool_family(self.source)

    def render(self, chat: Chat, *, add_generation_prompt: bool = True) -> str:
        thinking = {} if chat.enable_thinking is None else {"enable_thinking": chat.enable_thinking}
        return self.template.render(
            messages=list(chat.messages),
            tools=list(chat.tools) or None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **self.special_tokens,
            **thinking,
        )


def chat_template(
    directory: Path, *, now: Callable[[], datetime] = datetime.now
) -> ChatTemplate | None:
    """The checkpoint's chat template, or None when it ships none.

    Two shapes in circulation: newer exports move the template into a file of its own,
    older ones keep it inside `tokenizer_config.json`. A base model has neither, and gets
    no guessed one — an invented template produces fluent, wrong text.
    """
    path = directory / "tokenizer_config.json"
    config: _ConfigJson = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    file = directory / "chat_template.jinja"
    source = file.read_text(encoding="utf-8") if file.exists() else config.get("chat_template")
    if source is None:
        return None
    return ChatTemplate.from_source(source, _special_tokens(config), now=now)


def chat_capabilities(directory: Path) -> list["ChatCapability"]:
    """What a text model's `_task` hands to `CompositeModel`: the conversation as an input
    when the checkpoint carries a template, nothing when it does not."""
    template = chat_template(directory)
    return [] if template is None else [ChatCapability(template)]


@dataclass(frozen=True)
class ChatCapability:
    """A conversation as the input of a text model. A `Chat` carrying an image is refused
    here — what knows where an image goes in the prompt is `MultimodalChatCapability`."""

    template: ChatTemplate

    @property
    def input_type(self) -> ContentType:
        return CHAT

    @property
    def target_types(self) -> frozenset[ContentType]:
        return frozenset({TEXT})

    def accepts(self, input: ModelInput) -> bool:
        return isinstance(input, Chat) and not _images(input)

    def prepare(self, input: ModelInput) -> Text:
        assert isinstance(input, Chat)
        return Text(self.template.render(input), self.template.tool_family)


@dataclass(frozen=True)
class MultimodalChatCapability:
    """The template renders one marker per image (`<|vision_start|><|image_pad|>
    <|vision_end|>` on Qwen3.5); cutting there gives the parts back in document order.
    Expanding each image into the number of placeholders its grid asks for stays with the
    model, which is what processes the pixels."""

    template: ChatTemplate
    image_marker: str

    @property
    def input_type(self) -> ContentType:
        return CHAT

    @property
    def target_types(self) -> frozenset[ContentType]:
        return frozenset({TEXT, RGB_IMAGE})

    def accepts(self, input: ModelInput) -> bool:
        return isinstance(input, Chat)

    def prepare(self, input: ModelInput) -> Text | LanguagePrompt:
        """The family rides on the text the same way it does without images: what the
        streamer is handed is all it gets, and a checkpoint that spells `<tool_call>` spells
        it whether or not the turn carried a picture. The pieces after the first carry it
        too — which of them the generation continues from is not this side's to know."""
        assert isinstance(input, Chat)
        rendered = self.template.render(input)
        family = self.template.tool_family
        images = _images(input)
        if not images:
            return Text(rendered, family)
        chunks = rendered.split(self.image_marker)
        if len(chunks) != len(images) + 1:
            raise ImageMarkerMismatch(len(images), len(chunks) - 1)
        parts: list[AtomicInput] = []
        for chunk, image in zip(chunks, images, strict=False):
            if chunk:
                parts.append(Text(chunk, family))
            parts.append(image)
        if chunks[-1]:
            parts.append(Text(chunks[-1], family))
        return LanguagePrompt(tuple(parts))


@runtime_checkable
class _Composed(Protocol):
    """A model that answers for inputs it does not take natively — `CompositeModel`, whose
    map is where the capability that renders conversations for it lives."""

    @property
    def input_sources(self) -> Mapping[ContentType, object]: ...


def tool_family_of(model: object) -> ToolFamily | None:
    """Which envelope the checkpoint behind a loaded model spells a call in, or `None` when
    nothing here can say.

    The answer is a fact of the chat template, and a caller holding only the model — the
    server hands over a `Chat` and reads back text — reaches it through the capability that
    renders the conversation. Walking down the facades is what `tokenizer_of` does for the
    tokenizer, and for the same reason: what `load` returns is a stack of them.
    """
    while True:
        if isinstance(model, _Composed):
            capability = model.input_sources.get(CHAT)
            if isinstance(capability, ChatCapability | MultimodalChatCapability):
                return capability.template.tool_family
        if not isinstance(model, Wrapping):
            return None
        model = model.model

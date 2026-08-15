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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, runtime_checkable

import jinja2
import jinja2.ext
import jinja2.nodes
import jinja2.parser
import jinja2.runtime
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mlx_omnia.engine.language import TEXT, GenerationOptions, LanguageModel, LanguagePrompt, Text
from mlx_omnia.engine.model import (
    AtomicInput,
    CompositeModel,
    ContentType,
    Modality,
    ModelInput,
    Wrapping,
)
from mlx_omnia.engine.parsers import Parser, Segment, ToolCall, parser_for
from mlx_omnia.engine.vision import RGB_IMAGE, Image

__all__ = [
    "CHAT",
    "DESCRIBE",
    "Chat",
    "ChatCapability",
    "ChatMessage",
    "ChatTemplate",
    "Effort",
    "ImageMarkerMismatch",
    "ImagePart",
    "MultimodalChatCapability",
    "SeeingChat",
    "TextPart",
    "ToolCallFunction",
    "ToolCallRequest",
    "chat_capabilities",
    "chat_template",
    "composite",
    "parser_of",
    "template_of",
]

CHAT = ContentType(Modality.TEXT, "application/vnd.mlx_omnia.engine.chat")

type Effort = Literal["auto", "off", "on", "low", "medium", "high", "xhigh", "max"]
"""How hard the checkpoint is asked to think, as the templates in circulation can be told it.

Three of the values say something no level does. `auto` says nothing at all — no kwarg
reaches the template, so what happens is the template's own default, which is thinking on
for Qwen3 and off for Gemma 4. `off` and `on` are the yes/no a dialect that has only a
switch can express: Anthropic's `thinking.type` and Gemini's `thinkingBudget` name a state
and never a rung, and collapsing either onto `medium` would write a level into the prompt
that no client asked for.

The five rungs are `reasoning_effort`'s own vocabulary, and they are passed through as
written: `xhigh` is Anthropic's, `max` is DeepSeek V4's, and a template that reads the
kwarg spells whatever it is handed. A template that does not read it ignores it, which is
what makes sending both kwargs safe.
"""


def _thinking(effort: Effort) -> dict[str, object]:
    """The chat-template kwargs one effort asks for.

    `enable_thinking` travels with every level because it is the kwarg the templates in
    circulation actually branch on — `reasoning_effort` alone reaches a Qwen template that
    never reads it and turns thinking off by omission.

    `reasoning_strength` is the same rung under the name the atem templates read: they write
    it into their system block as prose and default it to `high`. Sending all three is safe
    for the reason the type's docstring gives — a template ignores a kwarg it does not read —
    and sending only the other two meant the level a client asked for reached that whole
    family as nothing at all.
    """
    match effort:
        case "auto":
            return {}
        case "off":
            return {"enable_thinking": False}
        case "on":
            return {"enable_thinking": True}
        case _:
            return {
                "enable_thinking": True,
                "reasoning_effort": effort,
                "reasoning_strength": effort,
            }


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    image: Image


type ContentPart = TextPart | ImagePart


class ToolCallFunction(TypedDict):
    name: str
    arguments: Mapping[str, object]
    """The arguments as a value, never as the JSON text of one.

    Measured over the fifteen `model_type` with a template that declares `tools`: seven read
    either shape, **eight read only the mapping** (`qwen3_5`, `qwen3_5_moe`, `nemotron_h`,
    `muse_glimmer`, `bailing_hybrid`, `laguna`, `lfm2_moe` — they do `arguments|items`, and
    atem raises a message written by hand saying it wants a dict), and none reads only the
    text. A dialect that receives the text on the wire decodes it at its own boundary, which
    is what a dialect is for; handing it through raises inside the render, which is a 500 on
    the second turn of every tool loop.
    """


class ToolCallRequest(TypedDict):
    """One call as the templates read it back. The nesting under `function` is the shape they
    are written against — they unwrap it themselves (`if tool_call.function is defined`), and
    the id is carried because some of them render it."""

    id: str
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | Sequence[ContentPart]
    tool_calls: NotRequired[Sequence[ToolCallRequest]]
    """The calls this assistant turn made, when it made any. Declared here and not bolted on
    by whoever renders a conversation: the template iterates the messages and reads what is in
    each dict, so a turn that called something is a shape of message and not a dialect's
    private extension."""
    tool_call_id: NotRequired[str]
    """Which call a `tool` turn answers."""


@dataclass(frozen=True)
class Chat:
    """A conversation as model input. `tools` are JSON schemas in the shape the template
    expects; `reasoning_effort` only reaches the template when it is not `auto`, which is how
    transformers treats its kwargs.

    How *long* the checkpoint may think is not here: it is spent in ids and enforced against
    the ones the loop draws, so it travels with the generation
    (`GenerationOptions.reasoning_budget`) and not with the prompt."""

    messages: tuple[ChatMessage, ...]
    tools: tuple[Mapping[str, object], ...] = ()
    reasoning_effort: Effort = "auto"

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


_PROBE_USER = "omnia:asked"
_PROBE_SYSTEM = "omnia:told"
_PROBE_ASSISTANT = "omnia:answered"


def _plain(content: str | Sequence[ContentPart]) -> str | None:
    """The text of a turn, or `None` when it holds a part that is not text."""
    if isinstance(content, str):
        return content
    said: list[str] = []
    for part in content:
        if part["type"] != "text":
            return None
        said.append(part["text"])
    return "".join(said)


def _merged_system(messages: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...] | None:
    """Every system turn as one, at the front. `None` when one of them carries something
    other than text and cannot be moved."""
    said: list[str] = []
    rest: list[ChatMessage] = []
    for message in messages:
        if message["role"] != "system":
            rest.append(message)
            continue
        text = _plain(message["content"])
        if text is None:
            return None
        if text:
            said.append(text)
    if not said:
        return tuple(rest)
    return ({"role": "system", "content": "\n\n".join(said)}, *rest)


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

    _keeps_system: dict[tuple[bool, bool, Effort], bool] = field(
        default_factory=dict, compare=False, repr=False
    )
    """What the probe below has already answered for this template. Rendering markers costs
    one render, and the answer is a property of the template rather than of a conversation."""

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
    def parser(self) -> Parser | None:
        return parser_for(self.source)

    @property
    def replays_calls(self) -> bool:
        """Whether this template renders back a turn that called something.

        Read off the source and not by rendering one and looking: a template that never
        mentions the key cannot read it, and that is decidable without guessing what an
        empty render meant. SmolLM3's and Falcon-H1's declare tools in the prompt — they
        instruct the model to write `<tool_call>` — and have no branch for `tool_calls` at
        all, so the assistant's own call is dropped from every history they render.
        """
        return "tool_calls" in self.source

    def keeps_system(
        self, *, trailing: bool, tools: Sequence[Mapping[str, object]], effort: Effort
    ) -> bool:
        """Whether this template renders a system turn where the conversation puts it.

        Answered by rendering markers rather than by reading the source: a template can
        refuse the turn (`raise_exception('System message must be at the beginning.')`,
        which the whole Qwen3 family does), drop it, or move it to the front, and only the
        first of those is visible in the text. What the probe checks is that all three
        markers came out in the order they went in.

        `trailing` and the presence of tools are separate questions because templates branch
        on both — Qwen3.8 renders the opening system turn inside its tools block — and the
        effort is in the key for the same reason: the kwargs it turns into are read by the
        same branches. What the tools *are* is not in the key; where a system turn lands
        does not depend on which functions were declared.
        """
        key = (trailing, bool(tools), effort)
        known = self._keeps_system.get(key)
        if known is not None:
            return known
        probe: list[ChatMessage] = [
            {"role": "user", "content": _PROBE_USER},
            {"role": "system", "content": _PROBE_SYSTEM},
        ]
        marks = [_PROBE_USER, _PROBE_SYSTEM]
        if not trailing:
            probe.append({"role": "assistant", "content": _PROBE_ASSISTANT})
            marks.append(_PROBE_ASSISTANT)
        try:
            rendered = self.template.render(
                messages=probe,
                tools=list(tools) or None,
                documents=None,
                add_generation_prompt=True,
                **self.special_tokens,
                **_thinking(effort),
            )
        except Exception:
            # Any failure to render the probe is the same answer: this template cannot be
            # handed the turn where it sits. Which exception it was is the template's own
            # business — they raise `TemplateError` by hand, and reach for the undefined and
            # the wrong type by accident.
            kept = False
        else:
            at = [rendered.find(mark) for mark in marks]
            kept = all(found >= 0 for found in at) and at == sorted(at)
        self._keeps_system[key] = kept
        return kept

    def render(self, chat: Chat, *, add_generation_prompt: bool = True) -> str:
        return self.template.render(
            messages=list(self._messages(chat)),
            tools=list(chat.tools) or None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **self.special_tokens,
            **_thinking(chat.reasoning_effort),
        )

    def _placed(self, chat: Chat) -> tuple[ChatMessage, ...]:
        """The conversation with its system turns where this template can read them.

        A client may put a system turn after the conversation has started — Claude Code
        sends one on every request, carrying the agents it offers — and upstream keeps it
        there. Templates disagree about whether that is renderable at all, so the probe is
        asked and the answer decides: a template that keeps the turn gets the conversation
        untouched, and the prefix already in the trie stays valid.

        A template that does not gets every system turn merged into one at the front. That
        moves the instruction ahead of the words it was written after, which is a different
        prompt — and the alternative is a 500 on every request from that client.

        A system turn carrying anything but text is left alone: merging it would move an
        image out of the order the markers are counted in, and every template that refuses
        the placement refuses images in a system turn anyway.
        """
        messages = chat.messages
        loose = [
            index
            for index, message in enumerate(messages)
            if index > 0 and message["role"] == "system"
        ]
        if not loose:
            return messages
        last = len(messages) - 1
        if all(
            self.keeps_system(
                trailing=index == last, tools=chat.tools, effort=chat.reasoning_effort
            )
            for index in loose
        ):
            return messages
        merged = _merged_system(messages)
        return messages if merged is None else merged

    def _messages(self, chat: Chat) -> Iterator[ChatMessage]:
        """The conversation as this template can read it.

        For every template that replays a call this is the conversation untouched. For the
        ones that cannot, the call is written into the content it would otherwise be dropped
        from, in the family's own spelling — which is what the model itself wrote, and what a
        history is. `tool_calls` is left on the message either way: a template that ignores
        the key ignores it, and taking it off would be this deciding what the next template
        gets to see.
        """
        family = None if self.replays_calls or (parser := self.parser) is None else parser.tools
        for message in self._placed(chat):
            calls = message.get("tool_calls")
            if family is None or not calls:
                yield message
                continue
            content = message["content"]
            if not isinstance(content, str):
                # A call beside pictures is not a shape any of these templates renders, and
                # splicing an envelope between the parts would move the image markers.
                yield message
                continue
            written = "".join(
                family.write(ToolCall(call["function"]["name"], call["function"]["arguments"]))
                for call in calls
            )
            yield {**message, "content": f"{content}{written}"}


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
        return Text(self.template.render(input), self.template.parser)


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
        """The dialect rides on the text the same way it does without images: what the
        streamer is handed is all it gets, and a checkpoint that spells `<tool_call>` spells
        it whether or not the turn carried a picture. The pieces after the first carry it
        too — which of them the generation continues from is not this side's to know."""
        assert isinstance(input, Chat)
        rendered = self.template.render(input)
        parser = self.template.parser
        images = _images(input)
        if not images:
            return Text(rendered, parser)
        chunks = rendered.split(self.image_marker)
        if len(chunks) != len(images) + 1:
            raise ImageMarkerMismatch(len(images), len(chunks) - 1)
        parts: list[AtomicInput] = []
        for chunk, image in zip(chunks, images, strict=False):
            if chunk:
                parts.append(Text(chunk, parser))
            parts.append(image)
        if chunks[-1]:
            parts.append(Text(chunks[-1], parser))
        return LanguagePrompt(tuple(parts))


DESCRIBE = (
    "Describe what is visible in this image in one or two sentences."
    " Describe only, do not comment."
)
"""What a vision model is asked for when it stands in for eyes the text model has not got.

It forbids commentary because the caption is read by a decoder that cannot check it: asked
to describe an image "so that questions can be answered about it", the models in
circulation write out the questions too, and a small decoder answers the ones it was handed
instead of the one that was asked."""


_MARKER = "\x00mlx_omnia:image\x00"
"""Where a picture stands in the conversation handed to the template, and therefore where its
caption is cut back into the rendered prompt. A control character because the cut has to be
unambiguous: anything spellable is something a user can type, and a marker found in a message
would splice a caption into the middle of someone's sentence. The count is checked after the
render either way."""


@dataclass(frozen=True)
class SeeingChat:
    """A conversation carrying images as the input of a text model that takes none. Every
    image is replaced by what `vision` says about it, and the conversation is then rendered
    by the text model's own template.

    What the two other capabilities do with the checkpoint's own tower, this does with a
    second model: the images never reach the decoder, only prose about them. Nothing here
    is that decoder's semantics — it is prompted with text a user could have typed."""

    vision: LanguageModel[ModelInput]
    main_template: ChatTemplate
    """The chat template of the model that answers, which is what the conversation is
    rendered by once the captions are in it."""
    vision_template: str = DESCRIBE
    """What the model that looks is asked for, which is a prompt and not a template of the
    kind above: a checkpoint being asked to describe a picture is having a conversation, and
    its own template renders that one on the far side of `stream`."""
    vision_max_tokens: int = 128
    """The caption's own budget, which is not the request's: what the decoder is told about
    a picture is this capability's business, and a request asking for one word of answer
    still needs the whole description."""

    @property
    def input_type(self) -> ContentType:
        return CHAT

    @property
    def target_types(self) -> frozenset[ContentType]:
        return frozenset({TEXT})

    def accepts(self, input: ModelInput) -> bool:
        return isinstance(input, Chat)

    def prepare(self, input: ModelInput) -> Text:
        """The prompt as something still being written, not as a string.

        The render happens first and once, over a conversation whose pictures are a marker:
        the template has to see the whole thing to produce a prompt, and only text can pass
        through jinja. Cutting the result on the marker gives back what is already written on
        either side of each caption, so the pieces before the first picture reach the
        decoder's prefill while the model that looks is still writing.
        """
        assert isinstance(input, Chat)
        marked = Chat(
            tuple(self._marked(message) for message in input.messages),
            input.tools,
            input.reasoning_effort,
        )
        rendered = self.main_template.render(marked)
        images = _images(input)
        chunks = rendered.split(_MARKER)
        if len(chunks) != len(images) + 1:
            raise ImageMarkerMismatch(len(images), len(chunks) - 1)
        if not images:
            return Text(rendered, self.main_template.parser)
        return Text(self._arriving(chunks, images), self.main_template.parser)

    def _marked(self, message: ChatMessage) -> ChatMessage:
        """Flat text and not parts: a text-only checkpoint's template reads `content` as a
        string — Qwen3's calls `startswith` on it — and raises on the list a multimodal one
        expects. Each picture goes in as `_MARKER`, which is where its caption will be cut
        back in."""
        content = message["content"]
        if isinstance(content, str):
            return message
        return {
            "role": message["role"],
            "content": "\n".join(
                part["text"] if part["type"] == "text" else _MARKER for part in content
            ),
        }

    def _arriving(self, chunks: list[str], images: tuple[Image, ...]) -> Iterator[str]:
        for chunk, image in zip(chunks, images, strict=False):
            if chunk:
                yield chunk
            yield "[image: "
            yield from self._caption(image)
            yield "]"
        if chunks[-1]:
            yield chunks[-1]

    def _caption(self, image: Image) -> Iterator[str]:
        """The description as the model writes it, piece by piece — which is the whole of what
        the decoder downstream is waiting on, and the reason it does not wait for all of it.

        Content alone: what the vision model thought on the way to the description is not the
        description, and read into the prompt it is a decoder answering someone's
        deliberation about the picture. Thinking is off for the reason the budget is small —
        a caption is a lookup, and a model reasoning its way to one spends the budget before
        reaching the text.
        """
        question: ChatMessage = {
            "role": "user",
            "content": (
                {"type": "image", "image": image},
                {"type": "text", "text": self.vision_template},
            ),
        }
        ask = Chat((question,), reasoning_effort="off")
        started = False
        for piece in self.vision.stream(ask, GenerationOptions(max_tokens=self.vision_max_tokens)):
            if piece.channel != "content":
                continue
            # Only the leading run is stripped, and it is stripped by holding the pieces that
            # are all space: the trailing end cannot be found without keeping the last piece
            # back, which is one caption's worth of latency for a space before a `]`.
            text = piece.text if started else piece.text.lstrip()
            if not text:
                continue
            started = True
            yield text


@runtime_checkable
class _Composed(Protocol):
    """A model that answers for inputs it does not take natively — `CompositeModel`, whose
    map is where the capability that renders conversations for it lives."""

    @property
    def input_sources(self) -> Mapping[ContentType, object]: ...


def template_of(model: object) -> ChatTemplate | None:
    """The chat template behind a loaded model, or `None` when nothing here holds one.

    A caller that hands the engine a `Chat` and reads back text never sees the prompt: what
    renders one is the capability inside the composite, and the only way to it from outside
    is down the facades. Walking them is what `tokenizer_of` does for the tokenizer, and for
    the same reason — what `load` returns is a stack of them.
    """
    while True:
        if isinstance(model, _Composed):
            capability = model.input_sources.get(CHAT)
            if isinstance(capability, ChatCapability | MultimodalChatCapability):
                return capability.template
        if not isinstance(model, Wrapping):
            return None
        model = model.model


def composite(
    model: LanguageModel[ModelInput],
    vision: LanguageModel[ModelInput],
    vision_template: str | None = None,
) -> CompositeModel[ModelInput, Segment, GenerationOptions]:
    """`model` with `vision` behind it: a text-only decoder answering conversations that
    carry images, by being told what is in them.

    Both are what `load` hands back. A conversation without images never reaches the vision
    model — the composite offers the input to `model` first, and a text checkpoint already
    takes those through its own `ChatCapability`; only what that one refuses falls through
    to here.

    Parameters
    ----------
    model : LanguageModel[ModelInput]
        The model that answers. Its chat template is what the conversation is rendered by.
    vision : LanguageModel[ModelInput]
        The model that looks. Any model taking images and writing text.
    vision_template : str | None, optional
        What `vision` is asked for. `DESCRIBE` when omitted.

    Returns
    -------
    CompositeModel[ModelInput, Segment, GenerationOptions]
        `model`, with a conversation carrying images among its inputs.

    Raises
    ------
    ValueError
        If `model` ships no chat template, which leaves nothing to render the conversation
        the captions were written into.
    """
    main_template = template_of(model)
    if main_template is None:
        raise ValueError("the model ships no chat template")
    asked = DESCRIBE if vision_template is None else vision_template
    return CompositeModel(model, [SeeingChat(vision, main_template, asked)])


def parser_of(model: object) -> Parser | None:
    """Which dialect the checkpoint behind a loaded model speaks, or `None` when nothing
    here can say. It is a fact of the chat template, so it is read off the one `template_of`
    walks down to rather than looked for a second time."""
    template = template_of(model)
    return None if template is None else template.parser

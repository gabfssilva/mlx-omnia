"""The render against transformers' own `apply_chat_template`, byte for byte.

The fixture carries the template text itself, so the render test needs no local
checkpoint; what does need one (reading the template off the directory, encoding the ids)
is gated per repo. Exact equality is the right bar here: a prompt that differs by one
space is no longer the prompt the model was trained on.
"""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, TypeIs, cast

import numpy as np
import pytest
from conftest import local_snapshot

from sideros.bpe import ByteLevelBPE
from sideros.chat import (
    CHAT,
    DESCRIBE,
    Chat,
    ChatCapability,
    ChatMessage,
    ChatTemplate,
    Effort,
    ImageMarkerMismatch,
    MultimodalChatCapability,
    SeeingChat,
    chat_capabilities,
    chat_template,
    composite,
    parser_of,
)
from sideros.language import TEXT, GenerationOptions, LanguagePrompt, Text
from sideros.model import CompositeModel, ModelInput, ModelSignature, UnsupportedInput
from sideros.parsers import Parser, Segment
from sideros.parsers.qwen import PARSER as QWEN
from sideros.parsers.qwen_xml import PARSER as QWEN_XML
from sideros.vision import RGB_IMAGE, Image

FIXTURE = Path(__file__).parent / "fixtures" / "chat_template.json"
GOLDEN: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
CLOCK = datetime.fromisoformat(GOLDEN["clock"])

CASES = [
    pytest.param(case, id=f"{case['repo'].split('/')[-1]}-{case['name']}")
    for case in GOLDEN["cases"]
]
REPOS = list(GOLDEN["repos"])


def has_image(case: dict[str, Any]) -> bool:
    return any(
        not isinstance(message["content"], str)
        and any(part["type"] == "image" for part in message["content"])
        for message in case["messages"]
    )


IMAGE_CASES = [pytest.param(case, id=case["name"]) for case in GOLDEN["cases"] if has_image(case)]
FIRST_IMAGE_CASE = next(case for case in GOLDEN["cases"] if has_image(case))


def template_of(case: dict[str, Any]) -> ChatTemplate:
    """The checkpoint's template, or the case's own when it carries one (the synthetic one
    that exercises the environment's whitespace control)."""
    meta = GOLDEN["repos"][case["repo"]]
    return ChatTemplate.from_source(
        case["template"] or meta["template"], meta["special_tokens"], now=lambda: CLOCK
    )


def chat_of(case: dict[str, Any]) -> Chat:
    """The fixture is written in the kwargs transformers was called with, which is the
    ground truth this compares against. `enable_thinking` is the bool the effort resolves
    to, so the case is read back through the same mapping the render applies."""
    kwargs = case["kwargs"]
    thinking = kwargs.get("enable_thinking")
    effort: Effort = "auto" if thinking is None else "on" if thinking else "off"
    return Chat(
        tuple(cast(list[ChatMessage], case["messages"])),
        tuple(kwargs.get("tools", ())),
        effort,
    )


def checkpoint_of(repo: str) -> Path:
    directory = local_snapshot(repo)
    if directory is None:
        pytest.skip(f"{repo} not in the local HF cache")
    return directory


@pytest.mark.parametrize("case", CASES)
def test_render_matches_transformers(case: dict[str, Any]) -> None:
    assert template_of(case).render(chat_of(case)) == case["rendered"]


@pytest.mark.parametrize("case", CASES)
def test_ids_match_transformers(case: dict[str, Any]) -> None:
    """The rendered text encoded back with the checkpoint's tokenizer: this is where the
    added tokens (`<|im_start|>`, `<|image_pad|>`) have to come out as one id, not bytes."""
    tokenizer = ByteLevelBPE.from_file(checkpoint_of(case["repo"]) / "tokenizer.json")
    assert list(tokenizer.encode(template_of(case).render(chat_of(case)))) == case["ids"]


@pytest.mark.parametrize("repo", REPOS)
def test_template_and_tokens_come_from_the_checkpoint(repo: str) -> None:
    """Both file shapes and the special-token map, read off the real directory."""
    template = chat_template(checkpoint_of(repo), now=lambda: CLOCK)
    assert template is not None
    assert template.special_tokens == GOLDEN["repos"][repo]["special_tokens"]
    case = next(c for c in GOLDEN["cases"] if c["repo"] == repo and not c["template"])
    assert template.render(chat_of(case)) == case["rendered"]


def test_directory_without_template_has_no_capability(tmp_path: Path) -> None:
    """A base model gets no guessed template, so it gets no chat capability either: what
    refuses the conversation is the model's signature, not a template we made up."""
    assert chat_template(tmp_path) is None
    assert chat_capabilities(tmp_path) == []
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|end|>"}))
    assert chat_template(tmp_path) is None


def marker_of(repo: str) -> str:
    """What the template emits per image, assembled from the tokens the checkpoint itself
    declares."""
    tokens = GOLDEN["repos"][repo]["special_tokens"]
    return "".join(tokens[key] for key in ("vision_bos_token", "image_token", "vision_eos_token"))


def with_images(case: dict[str, Any]) -> tuple[Chat, list[Image]]:
    """The same fixture case with real `Image`s where the pixel placeholder was."""
    images: list[Image] = []
    messages: list[ChatMessage] = []
    for message in case["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append(cast(ChatMessage, message))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if part["type"] != "image":
                parts.append(part)
                continue
            images.append(Image(np.full((4, 4, 3), len(images), dtype=np.uint8)))
            parts.append({"type": "image", "image": images[-1]})
        messages.append(cast(ChatMessage, {**message, "content": parts}))
    return Chat(tuple(messages)), images


@pytest.mark.parametrize("case", IMAGE_CASES)
def test_multimodal_capability_splits_on_the_marker(case: dict[str, Any]) -> None:
    """Putting the marker back where the images are reproduces the render exactly: that is
    what says nothing was swallowed or duplicated by the cut."""
    marker = marker_of(case["repo"])
    chat, images = with_images(case)
    prompt = MultimodalChatCapability(template_of(case), marker).prepare(chat)
    assert isinstance(prompt, LanguagePrompt)
    found = [part for part in prompt.parts if isinstance(part, Image)]
    assert len(found) == len(images)
    assert all(ours is theirs for ours, theirs in zip(found, images, strict=True))
    rebuilt = "".join(part.value if isinstance(part, Text) else marker for part in prompt.parts)
    assert rebuilt == case["rendered"]


def test_multimodal_capability_without_images_prepares_text() -> None:
    """Carrying the family the template declares, like the text-only capability: what the
    streamer is handed is all it gets, and a checkpoint spells its tool envelope whether or
    not the turn had a picture in it."""
    case = next(c for c in GOLDEN["cases"] if c["name"] == "user")
    template = template_of(case)
    capability = MultimodalChatCapability(template, marker_of(FIRST_IMAGE_CASE["repo"]))
    assert capability.prepare(chat_of(case)) == Text(case["rendered"], template.parser)


def test_multimodal_capability_rejects_a_template_without_the_marker() -> None:
    """One image in the conversation and no marker in the prompt misaligns every other
    one — staying silent here comes out as fluent, wrong text."""
    chat, _ = with_images(FIRST_IMAGE_CASE)
    template = ChatTemplate.from_source("{{ messages[0]['content'][1]['text'] }}")
    with pytest.raises(ImageMarkerMismatch):
        MultimodalChatCapability(template, marker_of(FIRST_IMAGE_CASE["repo"])).prepare(chat)


def test_chat_capability_prepares_text() -> None:
    case = next(c for c in GOLDEN["cases"] if c["name"] == "user")
    capability = ChatCapability(template_of(case))
    assert capability.accepts(chat_of(case))
    assert capability.prepare(chat_of(case)) == Text(case["rendered"], QWEN)


def test_chat_capability_refuses_a_conversation_with_images() -> None:
    chat, _ = with_images(FIRST_IMAGE_CASE)
    assert not ChatCapability(template_of(FIRST_IMAGE_CASE)).accepts(chat)


# --- the source, and the family that is read off it -----------------------------------

QWEN3 = "mlx-community/Qwen3-0.6B-4bit"
QWEN36 = "mlx-community/Qwen3.6-35B-A3B-6bit"

FAMILIES = [
    pytest.param(QWEN3, QWEN, id="qwen3"),
    # The same `<tool_call>` marker, filled with `<function=...>` XML instead of JSON. Two
    # families, and a prompt carrying the wrong one would have the envelope read by a machine
    # that cannot spell it — which is why the recognizers look past the marker.
    pytest.param(QWEN36, QWEN_XML, id="qwen3.6"),
]


@pytest.mark.parametrize("repo", REPOS)
def test_the_template_keeps_the_source_it_compiled(repo: str) -> None:
    """A compiled jinja2 template cannot be read back into its text, and the text is the one
    place where the family a checkpoint spells a call in is a fact rather than a guess."""
    source = GOLDEN["repos"][repo]["template"]
    assert ChatTemplate.from_source(source).source == source


@pytest.mark.parametrize(("repo", "family"), FAMILIES)
def test_the_prepared_prompt_carries_the_family_of_the_template_that_rendered_it(
    repo: str, family: Parser | None
) -> None:
    """What the streamer is handed is the rendered text and nothing else, so the family
    travels with it: the template is on this side of `prepare`, the suppression machine on
    the other."""
    meta = GOLDEN["repos"][repo]
    template = ChatTemplate.from_source(meta["template"], meta["special_tokens"])
    prepared = ChatCapability(template).prepare(Chat(({"role": "user", "content": "Hi"},)))
    assert prepared.parser is family


class _Echo:
    """A text model that echoes the prompt: what is measured here is `CompositeModel`'s
    routing, not generation."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.read())


def test_composite_model_routes_a_chat_through_the_capability() -> None:
    case = next(c for c in GOLDEN["cases"] if c["name"] == "user")
    model = CompositeModel(_Echo(), [ChatCapability(template_of(case))])
    assert model.signature.inputs == frozenset({TEXT, CHAT})
    assert list(model.stream(chat_of(case), GenerationOptions(max_tokens=1))) == [
        Segment("content", case["rendered"])
    ]
    chat, _ = with_images(FIRST_IMAGE_CASE)
    with pytest.raises(UnsupportedInput):
        list(model.stream(chat, GenerationOptions(max_tokens=1)))


class _Eyes:
    """A vision model that answers one caption, and records what it was asked."""

    def __init__(self, *pieces: Segment) -> None:
        self.pieces = pieces or (Segment("content", "a cat"),)
        self.asked: list[tuple[Chat, GenerationOptions]] = []

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT, RGB_IMAGE}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Chat]:
        return isinstance(input, Chat)

    def stream(self, input: Chat, options: GenerationOptions) -> Iterator[Segment]:
        self.asked.append((input, options))
        yield from self.pieces


def test_seeing_chat_puts_the_caption_where_the_image_was() -> None:
    chat, images = with_images(FIRST_IMAGE_CASE)
    eyes = _Eyes()
    prepared = SeeingChat(eyes, template_of(FIRST_IMAGE_CASE)).prepare(chat)

    assert isinstance(prepared, Text)
    assert prepared.read().count("[image: a cat]") == len(images)
    # The conversation the eyes were handed is the picture and the instruction, and nothing
    # of the conversation it came from: what the decoder is asking is not what is being
    # described.
    asked, options = eyes.asked[0]
    assert asked.messages == (
        {
            "role": "user",
            "content": ({"type": "image", "image": images[0]}, {"type": "text", "text": DESCRIBE}),
        },
    )
    assert asked.reasoning_effort == "off"
    assert options.max_tokens == 128
    assert len(eyes.asked) == len(images)


def test_seeing_chat_asks_nothing_until_the_prompt_is_read() -> None:
    """The point of the whole arrangement. `prepare` hands back a prompt that is still being
    written, so the model that looks runs while whoever holds it is already prefilling —
    a caption produced eagerly here would be the wait it exists to remove."""
    chat, _ = with_images(FIRST_IMAGE_CASE)
    eyes = _Eyes()
    prepared = SeeingChat(eyes, template_of(FIRST_IMAGE_CASE)).prepare(chat)
    assert eyes.asked == []

    assert not isinstance(prepared.value, str)
    pieces = prepared.value
    # Everything the template wrote before the picture is there to be prefilled first, and
    # the eyes have still not been asked for anything.
    assert next(pieces)
    assert eyes.asked == []


def test_seeing_chat_keeps_only_what_the_eyes_said() -> None:
    """The reasoning the vision model spends on its way to a description is not the
    description — read into the prompt as one, it is a decoder answering someone's
    deliberation about the picture."""
    chat, _ = with_images(FIRST_IMAGE_CASE)
    eyes = _Eyes(
        Segment("reasoning", "the user wants a colour"),
        Segment("content", " a cat "),
        Segment("tool", '{"name": "zoom"}'),
    )
    rendered = SeeingChat(eyes, template_of(FIRST_IMAGE_CASE)).prepare(chat).read()
    assert "[image: a cat " in rendered
    assert "the user wants a colour" not in rendered
    assert "zoom" not in rendered


def test_seeing_chat_asks_for_the_description_it_was_given() -> None:
    chat, _ = with_images(FIRST_IMAGE_CASE)
    eyes = _Eyes()
    SeeingChat(eyes, template_of(FIRST_IMAGE_CASE), "Name the colours.").prepare(chat).read()
    asked, _ = eyes.asked[0]
    assert asked.messages[0]["content"][1] == {"type": "text", "text": "Name the colours."}


def test_composite_sends_only_a_conversation_with_images_to_the_eyes() -> None:
    """The text model is offered the input first, so a conversation it already takes never
    pays for a second model."""
    case = next(c for c in GOLDEN["cases"] if c["name"] == "user")
    eyes = _Eyes()
    model = composite(CompositeModel(_Echo(), [ChatCapability(template_of(case))]), eyes)

    assert list(model.stream(chat_of(case), GenerationOptions(max_tokens=1))) == [
        Segment("content", case["rendered"])
    ]
    assert eyes.asked == []

    chat, images = with_images(FIRST_IMAGE_CASE)
    generated = list(model.stream(chat, GenerationOptions(max_tokens=1)))
    assert len(eyes.asked) == len(images)
    assert "[image: a cat]" in generated[0].text


def test_composite_refuses_a_model_that_ships_no_template() -> None:
    """Nothing renders the conversation the captions were written into, and a guessed
    template produces fluent, wrong text."""
    with pytest.raises(ValueError, match="chat template"):
        composite(CompositeModel(_Echo(), []), _Eyes())


@pytest.mark.parametrize(("repo", "family"), FAMILIES)
def test_the_family_is_reachable_from_the_model_alone(repo: str, family: Parser | None) -> None:
    """The other reader is the server, which holds the model and nothing else — it hands
    over a `Chat` and reads back text, so the route to the checkpoint's own answer is down
    the facades to the capability that renders the conversation."""
    meta = GOLDEN["repos"][repo]
    template = ChatTemplate.from_source(meta["template"], meta["special_tokens"])
    assert parser_of(CompositeModel(_Echo(), [ChatCapability(template)])) is family
    # A model that takes no conversation answers nothing rather than a family by default.
    assert parser_of(CompositeModel(_Echo(), [])) is None
    assert parser_of(_Echo()) is None


def test_a_call_that_arrived_as_text_reaches_the_template_as_data() -> None:
    """`from_json` is the one filter here that transformers does not ship. The OpenAI dialect
    delivers `arguments` as text and the server forwards it unchanged, so a template that
    reads the arguments key by key has to parse them first; without the filter the render
    raises instead of producing the prompt."""
    template = ChatTemplate.from_source(
        "{% for call in messages[0]['tool_calls'] %}"
        "{{ (call['function']['arguments'] | from_json)['city'] }}"
        "{% endfor %}"
    )
    message = cast(
        ChatMessage,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "weather", "arguments": '{"city": "Recife"}'}}],
        },
    )
    assert template.render(Chat((message,))) == "Recife"


def test_generation_prompt_is_optional() -> None:
    """Without it the render closes the last turn and opens no assistant one — that is what
    separates building a prompt from reconstructing a history."""
    case = next(c for c in GOLDEN["cases"] if c["name"] == "user")
    template = template_of(case)
    assert template.render(chat_of(case), add_generation_prompt=False) != case["rendered"]
    assert case["rendered"].startswith(template.render(chat_of(case), add_generation_prompt=False))


ECHO = ChatTemplate.from_source(
    "{% if enable_thinking is defined %}on={{ enable_thinking }};{% endif %}"
    "{% if reasoning_effort is defined %}effort={{ reasoning_effort }};{% endif %}"
    "{% if reasoning_strength is defined %}strength={{ reasoning_strength }};{% endif %}"
)
"""A template that reports which thinking kwargs reached it. `is defined` is the branch the
templates in circulation take, which is why an unset kwarg has to be absent and not false."""


def rendered(effort: Effort) -> str:
    return ECHO.render(Chat(({"role": "user", "content": "hi"},), reasoning_effort=effort))


def test_auto_sends_no_thinking_kwarg_at_all() -> None:
    """`auto` is the template's own default, and the only way to ask for it is to say
    nothing: `enable_thinking=false` is off, not unset."""
    assert rendered("auto") == ""


def test_the_switch_travels_without_a_level() -> None:
    """What a dialect with only a switch can say. A level here would be one this server
    invented for a client that named none."""
    assert rendered("off") == "on=False;"
    assert rendered("on") == "on=True;"


def test_a_level_travels_with_the_switch() -> None:
    """All three kwargs, because each family spells the same rung differently and a template
    that reads none of the others would otherwise have thinking turned off by omission, or —
    the atem case — take its own default while the client's level reached nothing. The rung
    is passed through as written."""
    assert rendered("high") == "on=True;effort=high;strength=high;"
    assert rendered("xhigh") == "on=True;effort=xhigh;strength=xhigh;"
    assert rendered("max") == "on=True;effort=max;strength=max;"

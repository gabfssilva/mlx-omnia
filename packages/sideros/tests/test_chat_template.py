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
    Chat,
    ChatCapability,
    ChatMessage,
    ChatTemplate,
    Effort,
    ImageMarkerMismatch,
    MultimodalChatCapability,
    chat_capabilities,
    chat_template,
    tool_family_of,
)
from sideros.language import TEXT, GenerationOptions, LanguagePrompt, Text
from sideros.model import CompositeModel, ModelInput, ModelSignature, UnsupportedInput
from sideros.suppress import Segment
from sideros.tools import ToolFamily
from sideros.tools.families.qwen import FAMILY as QWEN
from sideros.tools.families.qwen_xml import FAMILY as QWEN_XML
from sideros.vision import Image

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
    assert tokenizer.encode(template_of(case).render(chat_of(case))) == case["ids"]


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
    assert capability.prepare(chat_of(case)) == Text(case["rendered"], template.tool_family)


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
    repo: str, family: ToolFamily | None
) -> None:
    """What the streamer is handed is the rendered text and nothing else, so the family
    travels with it: the template is on this side of `prepare`, the suppression machine on
    the other."""
    meta = GOLDEN["repos"][repo]
    template = ChatTemplate.from_source(meta["template"], meta["special_tokens"])
    prepared = ChatCapability(template).prepare(Chat(({"role": "user", "content": "Hi"},)))
    assert prepared.tool_family is family


class _Echo:
    """A text model that echoes the prompt: what is measured here is `CompositeModel`'s
    routing, not generation."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


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


@pytest.mark.parametrize(("repo", "family"), FAMILIES)
def test_the_family_is_reachable_from_the_model_alone(repo: str, family: ToolFamily | None) -> None:
    """The other reader is the server, which holds the model and nothing else — it hands
    over a `Chat` and reads back text, so the route to the checkpoint's own answer is down
    the facades to the capability that renders the conversation."""
    meta = GOLDEN["repos"][repo]
    template = ChatTemplate.from_source(meta["template"], meta["special_tokens"])
    assert tool_family_of(CompositeModel(_Echo(), [ChatCapability(template)])) is family
    # A model that takes no conversation answers nothing rather than a family by default.
    assert tool_family_of(CompositeModel(_Echo(), [])) is None
    assert tool_family_of(_Echo()) is None


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
    """Both kwargs, because a template that reads only `enable_thinking` would otherwise
    have thinking turned off by omission — and the rung is passed through as written."""
    assert rendered("high") == "on=True;effort=high;"
    assert rendered("xhigh") == "on=True;effort=xhigh;"
    assert rendered("max") == "on=True;effort=max;"

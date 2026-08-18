"""39: an image through the four dialects, over the app `create_app` builds.

Each SDK attaches an image its own way — a `data:` URL nested under `image_url`, the same URL
flat on the part, a base64 `source` block, an `inlineData` blob whose bytes the client
base64s with the *url-safe* alphabet — and all four have to arrive as the same pixels in the
same place among the words. So the stand mounts every dialect at once and the four requests
are compared against one another as well as against what the model was handed.

The stand lives in `images_stand.py` and the png reader's own tests in
`test_images_reader.py`.
"""

import anthropic
import httpx
import pytest
from google import genai
from google.genai import errors
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mlx_omnia import Chat
from tests.server.images_stand import (
    ANSWER,
    ASKED,
    BUDGET,
    MODEL,
    TEXT_ONLY,
    Stand,
    claude,
    fresh_state,
    gemini,
    openai,
    stand,
    through_anthropic,
    through_chat,
    through_gemini,
    through_responses,
)
from tests.server.png_fixtures import BASE64, PIXELS, spelling

__all__ = ["claude", "fresh_state", "gemini", "openai", "stand"]


def test_the_same_image_reaches_the_model_through_the_four_dialects(
    openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """The box this stage exists for. Each SDK attaches the image the way its own API spells
    it, and what the model was handed is the same render cut in the same place with the same
    pixels between the halves — the digest is over the bytes, so a channel dropped, an alpha
    kept or a row misfiltered lands somewhere else.

    The Gemini leg is the one that also pins the alphabet: its SDK base64s bytes with `-_`
    rather than `+/`, and a reader using the permissive `b64decode` default drops those two
    characters instead of failing, which turns the image into noise without an error.
    """
    answers = {
        "chat/completions": through_chat(openai),
        "responses": through_responses(openai),
        "anthropic": through_anthropic(claude),
        "gemini": through_gemini(gemini),
    }

    assert answers == dict.fromkeys(answers, ANSWER)


def test_a_message_of_text_parts_reaches_the_model_as_characters(
    stand: Stand, openai: OpenAI
) -> None:
    """The trap `content_of` is written for: a template concatenates `content`, so a one-part
    list handed over as a list renders as its own repr — `[{'type': 'text', ...}]` inside the
    prompt. Only a message with an image in it becomes parts."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]}
    ]
    answer = openai.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=BUDGET, temperature=0
    )

    assert answer.choices[0].message.content == "<user>Hi!</user>"
    conversation = stand.engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert conversation.messages[0]["content"] == "Hi!"


def test_an_image_for_a_text_only_model_is_named_and_not_the_template_refusal(
    openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """A checkpoint with no vision tower and one with no chat template are the same
    `UnsupportedInput` on the way out of the engine, and they are not the same thing to say:
    the client that just attached an image has to hear about the image."""
    with pytest.raises(BadRequestError) as by_chat:
        through_chat(openai, model=TEXT_ONLY)
    said = str(by_chat.value)
    assert "image" in said and "chat template" not in said

    with pytest.raises(BadRequestError) as by_responses:
        through_responses(openai, model=TEXT_ONLY)
    assert "image" in str(by_responses.value)

    with pytest.raises(anthropic.BadRequestError) as by_claude:
        through_anthropic(claude, model=TEXT_ONLY)
    assert "image" in str(by_claude.value)

    with pytest.raises(errors.ClientError) as by_gemini:
        through_gemini(gemini, model=TEXT_ONLY)
    failure = by_gemini.value
    assert failure.code == 400 and failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "image" in failure.message


def test_a_remote_url_is_refused_rather_than_fetched(openai: OpenAI) -> None:
    """Fetching it would have the daemon making requests of its own, at a client's word, from
    inside the network it was told to serve. Named, so the client knows to inline the bytes."""
    with pytest.raises(BadRequestError) as raised:
        through_chat(openai, url="https://example.invalid/cat.png")

    assert "data URL" in str(raised.value)


def test_bytes_that_are_not_a_png_are_refused_in_each_dialect(
    stand: Stand, openai: OpenAI, gemini: genai.Client
) -> None:
    """A jpeg reaches here as bytes like any other, and reading it as a png would give the
    tower noise to describe. The refusal names the format rather than the byte it stopped on."""
    with pytest.raises(BadRequestError) as by_chat:
        through_chat(openai, url="data:image/png;base64,bm90IGEgcG5n")
    assert "png" in str(by_chat.value)

    with pytest.raises(errors.ClientError) as by_gemini:
        through_gemini(gemini, mime_type="image/jpeg")
    failure = by_gemini.value
    assert failure.code == 400
    assert failure.message is not None and "mimeType" in failure.message

    refused = httpx.post(
        f"{stand.base_url}/api/anthropic/v1/messages",
        json={
            "model": MODEL,
            "max_tokens": BUDGET,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": BASE64,
                            },
                        }
                    ],
                }
            ],
        },
        timeout=30,
    )
    assert refused.status_code == 400, refused.text
    body = refused.json()
    assert body["type"] == "error"
    assert "media_type" in body["error"]["message"]


def test_an_image_after_a_tool_result_keeps_its_place_in_the_round(
    stand: Stand, claude: anthropic.Anthropic
) -> None:
    """The one dialect where a message spells more than one kind of block: a `tool_result` is a
    turn of its own and comes out before the message's own words, and the image stays where the
    client put it among them."""
    reply = claude.messages.create(
        model=MODEL,
        max_tokens=BUDGET,
        messages=[
            {"role": "user", "content": ASKED},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_photo",
                        "input": {"of": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "here"},
                    {"type": "text", "text": "and this:"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": BASE64},
                    },
                ],
            },
        ],
    )

    block = reply.content[0]
    assert block.type == "text"
    assert block.text == (
        f"<user>{ASKED}</user><assistant></assistant><tool>here</tool>"
        f"<user>and this:{spelling(PIXELS)}</user>"
    )
    conversation = stand.engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert [message["role"] for message in conversation.messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]


def test_the_stand_touches_no_hub_cache(stand: Stand) -> None:
    """`create_app` mounts the catalog, and the catalog reads the machine's own cache: what
    this module patched is what the listing answers from."""
    listed = httpx.get(f"{stand.base_url}/admin/models", timeout=30)

    assert listed.status_code == 200, listed.text
    assert listed.json() == []


def test_counting_the_tokens_of_a_conversation_with_an_image_is_refused(
    claude: anthropic.Anthropic,
) -> None:
    """How many tokens a picture becomes is decided by the checkpoint's own processor, and
    `count_tokens` renders text: a count that skipped the image would be a number the request
    it is about never pays. Refused by name so the client knows which turn to drop, and not
    silently short.

    The generation first, because only a resident model answers this route at all: the
    refusal under test is the one about the image, not the one about the load."""
    through_anthropic(claude)

    with pytest.raises(anthropic.BadRequestError) as raised:
        claude.messages.count_tokens(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ASKED},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": BASE64,
                            },
                        },
                    ],
                }
            ],
        )

    body = raised.value.body
    assert isinstance(body, dict)
    error = body["error"]
    assert isinstance(error, dict)
    assert "image" in str(error["message"])

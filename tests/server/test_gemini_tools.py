"""Gate 37.3, the tool channel: a `functionCall` part out, a `functionResponse` part back in.

Split off `test_gemini.py` for size; the stand is the shared one in `gemini_stand.py`.
"""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from mlx_omnia import Chat
from tests.server.gemini_stand import (
    ANSWERED,
    ARGUMENTS,
    ASKED,
    CALLER,
    DESCRIPTION,
    ENTRIES,
    ENVELOPE,
    MODEL,
    MUTE,
    PREAMBLE,
    RESULT,
    SCHEMA,
    STRANGER,
    TEMPLATE,
    TOOL_BODY,
    Stand,
    Turn,
    candidate,
    client,
    fresh_state,
    offered,
    stand,
    submitted,
    url,
)

__all__ = ["client", "fresh_state", "stand"]


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(
    stand: Stand, client: genai.Client
) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function and
    answers with a `functionCall` part beside its text, the result goes back as the
    `functionResponse` part of the next content, and the second answer is one only a model that
    was handed the result can give.

    The conversation is compared against the one `chat/completions` builds out of the same
    round. Same characters into the model through two dialects — which is what makes one
    checkpoint answer both the same way, and what a `functionResponse` keyed by name instead of
    by id has to survive.
    """
    config = offered()
    first = client.models.generate_content(model=CALLER, contents=ASKED, config=config)

    assert first.text == PREAMBLE
    calls = first.function_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].args == {"city": "Paris"}
    assert candidate(first).finish_reason == types.FinishReason.STOP

    second = client.models.generate_content(
        model=CALLER,
        contents=[
            types.UserContent(ASKED),
            types.Content(
                role="model",
                parts=[
                    types.Part(text=PREAMBLE),
                    types.Part(
                        function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"})
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="get_weather", response={"output": RESULT}
                    )
                ],
            ),
        ],
        config=config,
    )

    assert second.text == ANSWERED
    expected: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": PREAMBLE,
            "tool_calls": [
                {
                    "id": "get_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.loads(ARGUMENTS)},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "get_weather"},
    )
    built = Chat(expected, tools=ENTRIES)
    assert submitted(stand) == built
    # And the same characters, which the comparison above cannot see: a template renders a
    # tool entry with `tojson`, and two dicts that differ only in the order their keys went
    # in are equal and do not write the same prompt.
    assert TEMPLATE.render(submitted(stand)) == TEMPLATE.render(built)


def test_the_call_rides_the_frame_that_closes_the_stream(client: genai.Client) -> None:
    """The call whole in one terminal frame — A11's decision — and that frame is the one the
    finish reason and the counts already ride: a client reading a completed turn reads it
    there. The text before it arrives piece by piece, envelope suppressed, which is what says
    the reader ran over the stream and not over the finished answer."""
    chunks = list(
        client.models.generate_content_stream(model=CALLER, contents=ASKED, config=offered())
    )

    assert "".join(chunk.text or "" for chunk in chunks) == PREAMBLE
    assert all(chunk.function_calls is None for chunk in chunks[:-1])
    last = chunks[-1]
    calls = last.function_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].args == {"city": "Paris"}
    assert candidate(last).finish_reason == types.FinishReason.STOP
    assert last.usage_metadata is not None


def test_a_content_that_only_called_something_carries_no_text_part(
    stand: Stand, client: genai.Client
) -> None:
    """The empty text part this dialect does not write: a part carrying `""` beside a call is a
    model that answered with nothing before it called, and the SDK's `.text` reads it as an
    answer. The stream says the same thing — one frame, the call's."""
    answer = client.models.generate_content(model=MUTE, contents=ASKED, config=offered())

    content = candidate(answer).content
    assert content is not None and content.parts is not None
    assert [part.text for part in content.parts] == [None]
    assert answer.function_calls is not None and len(answer.function_calls) == 1

    streamed = httpx.post(
        url(stand, f"{MUTE}:streamGenerateContent"),
        json={"contents": [{"parts": [{"text": ASKED}]}], "tools": TOOL_BODY},
        timeout=30,
    )
    frames = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert len(frames) == 1, frames
    assert "functionCall" in frames[0]
    assert '"text"' not in frames[0]


def test_mode_none_neither_offers_the_declarations_nor_reads_a_call_back(
    stand: Stand, client: genai.Client
) -> None:
    """`NONE` is honoured where it can be honoured: the declarations never reach the prompt, so
    the model has nothing to call rather than an instruction not to. And a turn nobody was
    offered a function for is text — answering with a call would be answering with the one
    thing the client asked us not to do."""
    config = offered(types.FunctionCallingConfigMode.NONE)
    answer = client.models.generate_content(model=MUTE, contents=ASKED, config=config)

    assert answer.text == ENVELOPE
    assert answer.function_calls is None
    assert submitted(stand).tools == ()


def test_a_checkpoint_whose_envelope_nothing_here_parses_answers_with_the_text(
    client: genai.Client,
) -> None:
    """Which family a checkpoint speaks is a fact of its chat template and not of the text it
    writes: read off the output instead, Qwen3.6's `<tool_call><function=…>` is taken for Qwen's
    JSON and the envelope is held for a parser that cannot read it. A template that spells none
    leaves the channel shut, and what the model wrote reaches the client whole."""
    answer = client.models.generate_content(model=STRANGER, contents=ASKED, config=offered())

    assert answer.text == PREAMBLE + ENVELOPE
    assert answer.function_calls is None


def test_forcing_a_call_is_refused_by_name(stand: Stand) -> None:
    """`ANY` and `VALIDATED` constrain decoding to a call and there is no such constraint here:
    answering `AUTO` to a client that asked for one is a call the model may never have made.
    The envelope a refused body comes back in is 37.1's — what this asserts is that it is
    refused at all, and that the field is named where the client can read it."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "tools": TOOL_BODY,
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        },
        timeout=30,
    )

    assert 400 <= response.status_code < 500, response.text
    assert "mode" in response.text


@pytest.mark.parametrize("spelling", ["parametersJsonSchema", "parameters_json_schema"])
def test_a_json_schema_reaches_the_prompt_under_either_of_its_two_names(
    stand: Stand, spelling: str
) -> None:
    """Proto's JSON mapping accepts a field's own name as well as its camelCase form: the REST
    reference documents `parametersJsonSchema` and the SDK sends `parameters_json_schema`, so a
    dialect that took one of them refuses half the clients that speak it. Both arrive at the
    template as `parameters`, which is the only field a template has."""
    declaration: dict[str, object] = {
        "name": "get_weather",
        "description": DESCRIPTION,
        spelling: SCHEMA,
    }
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "tools": [{"functionDeclarations": [declaration]}],
        },
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert submitted(stand).tools == ENTRIES


def test_the_sdks_own_schema_travels_untouched(stand: Stand) -> None:
    """`parameters` is the other spelling, and what rides it is the OpenAPI subset the SDK
    builds out of its `Schema`: types spelled `OBJECT` and `STRING`. It reaches the model as
    written — normalizing a client's schema is rewriting what it declared — and a declaration
    with no description carries none rather than a null."""
    schema: dict[str, object] = {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}}
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": schema}]}],
        },
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert submitted(stand).tools == (
        {"type": "function", "function": {"name": "get_weather", "parameters": schema}},
    )

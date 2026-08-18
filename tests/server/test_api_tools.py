"""Tool calls through the dialect: the round trip, the frames, and what the conversion owes
the template."""

import json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from mlx_omnia import Chat
from tests.server import openai_stand
from tests.server.openai_script import ANSWER, CALL, CALLER, PAIR, PREAMBLE, RESULT, TOOLS
from tests.server.openai_stand import (
    Recording,
    offer,
    sse,
)

fresh_state = openai_stand.fresh_state
"""Overrides `conftest`'s per-test wipe, which would delete the database under the
module server while it is still answering."""

pytest_plugins = ("tests.server.openai_stand",)
"""The per-file server. `fresh_state` is imported to override `conftest`'s per-test wipe,
which would delete the database under a server that is still answering."""


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(client: OpenAI) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function
    and answers with a call instead of text, the result of that call goes back in as a turn
    of its own, and the second answer is one only a model that was handed the result can
    give — `Script` reads it out of the `<tool_response>` the checkpoint's own template
    renders, so nothing but the conversion having worked puts it there."""
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Weather in Paris?"}]
    first = client.chat.completions.create(model=CALLER, messages=messages, tools=TOOLS)
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == PREAMBLE
    made = choice.message.tool_calls
    assert made is not None and len(made) == 1
    call = made[0]
    assert call.type == "function"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": call.id, "content": RESULT})
    second = client.chat.completions.create(model=CALLER, messages=messages, tools=TOOLS)
    assert second.choices[0].message.content == ANSWER
    assert second.choices[0].message.tool_calls is None
    assert second.choices[0].finish_reason == "stop"


def test_the_sdk_accumulator_builds_both_calls_out_of_the_stream(client: OpenAI) -> None:
    """What judges the frames is the SDK's own accumulator, not our reading of them. Two
    calls is what makes `index` load-bearing: the second frame's entry is merged into a list
    that already holds one, and an entry with no `index` raises there instead of folding
    in."""
    asked: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Weather and time in Paris?"}
    ]
    with client.chat.completions.stream(model=PAIR, messages=asked, tools=TOOLS) as stream:
        final = stream.get_final_completion()

    choice = final.choices[0]
    assert choice.finish_reason == "tool_calls"
    made = choice.message.tool_calls
    assert made is not None and len(made) == 2
    weather, clock = made
    assert weather.type == "function" and clock.type == "function"
    assert weather.function.name == "get_weather"
    assert json.loads(weather.function.arguments) == {"city": "Paris"}
    assert clock.function.name == "get_time"
    assert json.loads(clock.function.arguments) == {"zone": "Europe/Paris"}
    assert weather.id != clock.id


def test_tool_choice_none_neither_offers_the_tools_nor_reads_a_call_back(
    base_url: str, engine: Recording
) -> None:
    """`none` is honoured where it can be honoured: the tools never reach the prompt, so
    there is nothing to call rather than an instruction not to. And a turn nobody was
    offered a tool for is text — answering with a call would be answering with the one thing
    the client asked us not to do."""
    payload = offer(base_url, CALLER, tools=TOOLS, tool_choice="none").json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": PREAMBLE + CALL}

    job = engine.jobs[-1]
    assert isinstance(job.input, Chat)
    assert job.input.tools == ()


def test_a_request_without_tools_is_answered_with_the_envelope_it_wrote(base_url: str) -> None:
    """The suppression cannot charge a client that declared no tool: what the model wrote
    comes back whole, envelope and all, and no `tool_calls` are invented beside it.

    The frames are the model's segments and not the tokenizer's pieces, and that is the
    checkpoint's own doing rather than this route's: a model whose template declares a tool
    family segments on the way out, so an envelope leaves as one piece however many ids it
    took. What this route decides is only whether to *read* that channel — and with no tool
    offered it does not, so the text goes out as it came."""
    payload = offer(base_url, CALLER).json()
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": PREAMBLE + CALL}
    assert payload["choices"][0]["finish_reason"] == "stop"

    frames = [json.loads(frame) for frame in sse(base_url, CALLER)]
    deltas = [frame["choices"][0]["delta"] for frame in frames]
    assert deltas[0] == {"role": "assistant", "content": ""} and deltas[-1] == {}
    assert "".join(delta.get("content", "") for delta in deltas) == PREAMBLE + CALL
    assert not any("tool_calls" in delta for delta in deltas), "nobody offered a tool"
    finishes = [frame["choices"][0]["finish_reason"] for frame in frames]
    assert finishes[-1] == "stop" and set(finishes[:-1]) == {None}


def test_the_message_the_dialect_answers_with_is_a_message_it_accepts(base_url: str) -> None:
    """A client replays the answer it was given: the assistant turn goes back verbatim, the
    `index` inside the call and the `content: null` included. A field the request model
    refuses would 400 the second turn of every round trip that goes through an SDK."""
    answered = offer(base_url, CALLER, tools=TOOLS).json()["choices"][0]["message"]
    assert answered["content"] == PREAMBLE
    assert answered["tool_calls"][0]["index"] == 0

    replay = offer(
        base_url,
        CALLER,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            answered,
            {"role": "tool", "tool_call_id": answered["tool_calls"][0]["id"], "content": RESULT},
        ],
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["choices"][0]["message"]["content"] == ANSWER


def test_the_tools_and_the_call_a_result_answers_reach_the_conversation(
    base_url: str, engine: Recording
) -> None:
    """The frontier this stage owns: what the dialect's fields become in the `Chat` the
    engine is handed. The template renders from that and from nothing else, so a key the
    conversion drops is a key no checkpoint can put back — `tool_call_id` in particular,
    which the Qwen template never renders and only this can see.

    `arguments` crosses that frontier as a **mapping**, not as the JSON text this dialect
    spells it in. Eight of the fifteen template families in circulation do
    `arguments|items`, so the text raised inside the render — a 500 on the second turn of
    every tool loop against Qwen3.6, Nemotron, Muse-Glimmer, Ling, Laguna and LFM2.5.
    """
    history = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
    ]
    assert offer(base_url, CALLER, tools=TOOLS, messages=history).status_code == 200

    job = engine.jobs[-1]
    assert isinstance(job.input, Chat)
    assert job.input.tools == tuple(TOOLS)
    assert job.input.messages == (
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "call_1"},
    )


def test_a_replayed_call_whose_arguments_are_not_json_is_refused_by_name(
    base_url: str,
) -> None:
    """400 and not 500. The client sent something that is not a call; letting it reach the
    render turns the client's fault into the server's, and what the client would read is
    `generation_failed`."""
    history = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
    ]
    refusal = offer(base_url, CALLER, tools=TOOLS, messages=history)
    assert refusal.status_code == 400
    body = refusal.json()["error"]
    assert body["code"] == "invalid_tool_arguments"
    assert "get_weather" in body["message"]


def test_arguments_that_are_json_but_not_an_object_are_refused(base_url: str) -> None:
    """`"[1, 2]"` parses and is still not a call: the templates iterate the arguments by key,
    and a list has none."""
    history = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "[1, 2]"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
    ]
    refusal = offer(base_url, CALLER, tools=TOOLS, messages=history)
    assert refusal.status_code == 400
    assert refusal.json()["error"]["code"] == "invalid_tool_arguments"

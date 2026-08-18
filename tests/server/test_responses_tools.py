"""Tool calling over `/api/openai/v1/responses`: a call is an item of its own here and a key
of the assistant's message everywhere else, and both directions of that translation are what
these tests pin.
"""

import json

import pytest
from openai import BadRequestError, OpenAI
from openai.types.responses import FunctionToolParam, ResponseInputParam

from mlx_omnia import ChatMessage as Turn
from tests.server.responses_script import (
    ANSWERED,
    ARGUMENTS,
    ASKED,
    CALLER,
    CUT,
    DESCRIPTION,
    ENTRIES,
    ENVELOPE,
    MUTE,
    PREAMBLE,
    RESULT,
    SCHEMA,
    SCRIPTED,
    STRANGER,
    TOOLS,
    last,
)
from tests.server.responses_stand import client, code, only_call, rendered, stand

__all__ = ["client", "stand"]


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(client: OpenAI) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function and
    answers with a call beside its text, the result of that call goes back in as items of the
    input, and the second answer is one only a model that was handed the result can give —
    `Script` reads it out of the turn the template rendered.

    Both prompts are compared against the render of the conversation `chat/completions` builds
    out of the same round: the tools nested under `function`, the call on the assistant's turn,
    the result as a turn of its own. Same characters, same ids into the model — which is what
    keeps one checkpoint answering three dialects the same way.
    """
    first = client.responses.create(model=CALLER, input=ASKED, tools=TOOLS)

    assert first.output_text == PREAMBLE
    called = only_call([item for item in first.output if item.type == "function_call"])
    call_id, arguments = called.call_id, called.arguments
    assert json.loads(arguments) == {"city": "Paris"}
    assert last().prompt == rendered(({"role": "user", "content": ASKED},), ENTRIES)

    items: ResponseInputParam = [
        {"role": "user", "content": ASKED},
        {
            "type": "function_call",
            "call_id": call_id,
            "name": "get_weather",
            "arguments": arguments,
        },
        {"type": "function_call_output", "call_id": call_id, "output": RESULT},
    ]
    second = client.responses.create(model=CALLER, input=items, tools=TOOLS)

    assert second.output_text == ANSWERED
    assert [item.type for item in second.output] == ["message"]
    replayed: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": call_id},
    )
    assert last().prompt == rendered(replayed, ENTRIES)


def test_two_calls_replayed_fold_into_the_one_turn_that_made_them(client: OpenAI) -> None:
    """A call is an item of its own here and a key of the assistant's message everywhere else,
    so two of them in a row are one turn and not two — two would tell the model it answered
    twice, and a template that numbers the turns would render a conversation that never
    happened."""
    items: ResponseInputParam = [
        {"role": "user", "content": ASKED},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call", "call_id": "call_2", "name": "get_time", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": RESULT},
        {"type": "function_call_output", "call_id": "call_2", "output": "noon"},
    ]
    client.responses.create(model=SCRIPTED, input=items)

    folded: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": {}},
                },
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "call_1"},
        {"role": "tool", "content": "noon", "tool_call_id": "call_2"},
    )
    assert last().prompt == rendered(folded)


def test_the_sdk_accumulates_the_call_out_of_the_named_frames(client: OpenAI) -> None:
    """What judges the frames is the SDK's own accumulator: an arguments delta for an item it
    was never told about raises there instead of folding in, and `response.completed` is what
    it turns into a final response.

    One delta and the arguments whole inside it — A11's decision — which is the assertion the
    accumulator cannot make: it would concatenate any number of fragments into the same
    string.
    """
    with client.responses.stream(model=CALLER, input=ASKED, tools=TOOLS) as stream:
        seen = list(stream)
        final = stream.get_final_response()

    kinds = [event.type for event in seen]
    assert kinds[-5:] == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    fragments = [
        event.delta for event in seen if event.type == "response.function_call_arguments.delta"
    ]
    assert fragments == [ARGUMENTS], "the arguments arrived in fragments"
    assert [event.sequence_number for event in seen] == list(range(len(seen)))
    assert final.output_text == PREAMBLE
    assert [item.type for item in final.output] == ["message", "function_call"]
    called = only_call([item for item in final.output if item.type == "function_call"])
    assert called.arguments == ARGUMENTS
    assert final.status == "completed"


def test_a_turn_that_only_called_something_carries_no_message_item(client: OpenAI) -> None:
    """The empty message this dialect does not write: an item announced with no text in it is
    an assistant that answered `""` before it called, and a client rendering the transcript
    shows a blank turn. The stream says the same thing — no text frames at all — which is what
    says the item is opened on the first text there is and not before."""
    answer = client.responses.create(model=MUTE, input=ASKED, tools=TOOLS)

    assert [item.type for item in answer.output] == ["function_call"]
    assert answer.output_text == ""

    with client.responses.stream(model=MUTE, input=ASKED, tools=TOOLS) as stream:
        kinds = [event.type for event in stream]
    assert not [kind for kind in kinds if kind.startswith("response.output_text")]
    assert not [kind for kind in kinds if kind.startswith("response.content_part")]
    assert kinds[1] == "response.output_item.added"


def test_tool_choice_none_neither_offers_the_tools_nor_reads_a_call_back(
    client: OpenAI,
) -> None:
    """`none` is honoured where it can be honoured: the tools never reach the prompt, so there
    is nothing to call rather than an instruction not to. And a turn nobody was offered a tool
    for is text — answering with a call would be answering with the one thing the client asked
    us not to do."""
    answer = client.responses.create(model=MUTE, input=ASKED, tools=TOOLS, tool_choice="none")

    assert answer.output_text == ENVELOPE
    assert [item.type for item in answer.output] == ["message"]
    assert last().prompt == rendered(({"role": "user", "content": ASKED},))


def test_strict_is_refused_by_name(client: OpenAI) -> None:
    """`strict` on a *tool* constrains the arguments of a call, and what a grammar constrains
    here is the whole answer (`text.format`): an argument that violated the schema would come
    back as one that was checked against it. The field is declared so that saying so is a named
    error and not the generic refusal an undeclared one would get — and it has to be declared,
    because the SDK's own tool type requires the key on every tool. `false` answers, or the
    refusal would be unreachable."""
    strict: list[FunctionToolParam] = [
        {
            "type": "function",
            "name": "get_weather",
            "description": DESCRIPTION,
            "parameters": SCHEMA,
            "strict": True,
        }
    ]
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=SCRIPTED, input=ASKED, tools=strict)

    assert code(raised.value.body) == "strict_unsupported"
    answer = client.responses.create(model=MUTE, input=ASKED, tools=TOOLS)
    assert [item.type for item in answer.output] == ["function_call"]


def test_a_checkpoint_whose_envelope_nothing_here_parses_answers_with_the_text(
    client: OpenAI,
) -> None:
    """Which family a checkpoint speaks is a fact of its chat template and not of the text it
    writes: read off the output instead, Qwen3.6's `<tool_call><function=…>` is taken for Qwen's
    JSON and the envelope is held for a parser that cannot read it. A template that spells none
    leaves the channel shut, and what the model wrote reaches the client whole."""
    answer = client.responses.create(model=STRANGER, input=ASKED, tools=TOOLS)

    assert answer.output_text == PREAMBLE + ENVELOPE
    assert [item.type for item in answer.output] == ["message"]


def test_an_envelope_the_budget_cut_in_half_comes_back_as_the_text_it_is(client: OpenAI) -> None:
    """Held as a possible call and it is not one: what the model wrote goes out as text. The
    silent failure this rules out is the opposite — an envelope suppressed and no call produced
    reaches the client as a model that chose to call nothing, which is exactly the shape of a
    correct refusal."""
    answer = client.responses.create(model=CUT, input=ASKED, tools=TOOLS)

    assert answer.output_text == '<tool_call>\n{"name": "get_weat'
    assert [item.type for item in answer.output] == ["message"]

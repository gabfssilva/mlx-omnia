"""The tool path of `/api/anthropic/v1/messages`: a call is a `tool_use` block of the
assistant's message and a result is a `tool_result` block of the *user's*, while the
conversation the engine takes has one turn per result."""

import json

import anthropic
import pytest
from anthropic.types import Message as Reply
from anthropic.types import MessageParam

from mlx_omnia import Chat
from mlx_omnia import ChatMessage as Turn
from tests.server.anthropic_script import (
    ANSWERED,
    ARGUMENTS,
    ASKED,
    BUDGET,
    CALLER,
    ECHO,
    ENTRIES,
    ENVELOPE,
    MUTE,
    PREAMBLE,
    RESULT,
    STRANGER,
    TEMPLATE,
    TOOLS,
)
from tests.server.anthropic_stand import (
    Stand,
    client,
    entry,
    envelope,
    frames,
    fresh_state,
    only_text,
    stand,
    text,
)

__all__ = ["client", "fresh_state", "stand"]
"""The fixtures live in the stand module and are imported for pytest to find them here."""


def conversation(turns: tuple[Turn, ...]) -> str:
    """What the template writes for the conversation `chat/completions` would have built out of
    the same round: the tools nested under `function`, the call on the assistant's turn, the
    result as a turn of its own. Rendered rather than spelled out, because what is under test
    is the conversation and not the toy template — the two sides are the same instrument, and
    what has to agree is what reached it."""
    return TEMPLATE.render(Chat(turns, tools=ENTRIES))


def only_use(reply: Reply) -> tuple[str, str]:
    """The one `tool_use` block of an answer: its id and the name it called. The id is the
    dialect's own (`toolu_`), because it is what a `tool_result` comes back addressed to."""
    uses = [block for block in reply.content if block.type == "tool_use"]
    assert len(uses) == 1, f"expected one call, got {reply.content!r}"
    use = uses[0]
    assert use.id.startswith("toolu_")
    assert use.input == {"city": "Paris"}
    return use.id, use.name


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(
    client: anthropic.Anthropic,
) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function and
    answers with a `tool_use` block beside its text, the result goes back as the `tool_result`
    block of the next user message, and the second answer is one only a model that was handed
    the result can give.

    `stop_reason` is this dialect's own vocabulary for it — `tool_use`, where OpenAI says
    `tool_calls` — and a client that reads anything else there never executes the call.
    """
    first = client.messages.create(
        model=CALLER, messages=[{"role": "user", "content": ASKED}], max_tokens=BUDGET, tools=TOOLS
    )

    assert [block.type for block in first.content] == ["text", "tool_use"]
    assert first.stop_reason == "tool_use"
    use_id, name = only_use(first)
    assert name == "get_weather"

    replay: list[MessageParam] = [
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": PREAMBLE},
                {
                    "type": "tool_use",
                    "id": use_id,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": use_id, "content": RESULT}],
        },
    ]
    second = client.messages.create(model=CALLER, messages=replay, max_tokens=BUDGET, tools=TOOLS)

    assert only_text(second) == ANSWERED
    assert second.stop_reason == "end_turn"


def test_the_tools_and_the_blocks_of_a_round_become_the_conversation_openai_would_build(
    client: anthropic.Anthropic,
) -> None:
    """The frontier this stage owns, read where it is visible: the echo answers with the
    rendered prompt, so what the blocks became is in the reply. A `tool_result` is a turn of
    its own and comes out before the text of the message that carried it; a `tool_use` is a key
    of the turn that made it; `input` becomes JSON text, because that is what the templates in
    circulation render.

    The comparison is against the conversation `chat/completions` builds out of the same round.
    Same characters into the model through two dialects, which is what makes one checkpoint
    answer both the same way.
    """
    reply = client.messages.create(
        model=ECHO,
        max_tokens=BUDGET,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": ASKED},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": PREAMBLE},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": RESULT},
                    {"type": "text", "text": "And Lyon?"},
                ],
            },
        ],
    )

    expected: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": PREAMBLE,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.loads(ARGUMENTS)},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "toolu_1"},
        {"role": "user", "content": "And Lyon?"},
    )
    assert only_text(reply) == conversation(expected)


def test_the_sdk_accumulates_the_call_out_of_the_named_events(
    client: anthropic.Anthropic, stand: Stand
) -> None:
    """What judges the frames is the SDK's own accumulator: it indexes every delta into the
    block `content_block_start` announced, and reparses the tool input from the `partial_json`
    it has accumulated so far. A block index off by one lands the arguments in the text block.

    One delta with the arguments whole inside it is A11's decision, and the assertion the
    accumulator cannot make: it would concatenate any number of fragments into the same
    string.
    """
    with client.messages.stream(
        model=CALLER, messages=[{"role": "user", "content": ASKED}], max_tokens=BUDGET, tools=TOOLS
    ) as stream:
        deltas = list(stream.text_stream)
        final = stream.get_final_message()

    assert "".join(deltas) == PREAMBLE, "the envelope reached the client as text"
    assert [block.type for block in final.content] == ["text", "tool_use"]
    assert final.stop_reason == "tool_use"
    assert only_use(final)[1] == "get_weather"

    captured = frames(stand, model=CALLER, tools=TOOLS)
    fragments = [
        text(entry(payload["delta"])["partial_json"])
        for name, payload in captured
        if name == "content_block_delta" and "partial_json" in entry(payload["delta"])
    ]
    assert fragments == [ARGUMENTS], "the arguments arrived in fragments"
    opened = [payload for name, payload in captured if name == "content_block_start"]
    assert [entry(payload["content_block"])["type"] for payload in opened] == ["text", "tool_use"]
    assert [payload["index"] for payload in opened] == [0, 1]


def test_an_answer_that_only_called_something_has_no_text_block(
    client: anthropic.Anthropic, stand: Stand
) -> None:
    """The empty text block this dialect does not write: a block announced with nothing in it
    is an assistant that answered `""` before it called, and a client rendering the transcript
    shows a blank turn. The stream says the same thing — the call's block is index 0, which is
    what says the text block is opened on the first text there is and not before."""
    reply = client.messages.create(
        model=MUTE, messages=[{"role": "user", "content": ASKED}], max_tokens=BUDGET, tools=TOOLS
    )

    assert [block.type for block in reply.content] == ["tool_use"]
    assert reply.stop_reason == "tool_use"

    opened = [
        payload
        for name, payload in frames(stand, model=MUTE, tools=TOOLS)
        if name == "content_block_start"
    ]
    assert len(opened) == 1
    assert entry(opened[0]["content_block"])["type"] == "tool_use"
    assert opened[0]["index"] == 0


def test_tool_choice_none_neither_offers_the_tools_nor_reads_a_call_back(
    client: anthropic.Anthropic,
) -> None:
    """`none` is honoured where it can be honoured: the tools never reach the prompt, so there
    is nothing to call rather than an instruction not to. And a turn nobody was offered a tool
    for is text — answering with a call would be answering with the one thing the client asked
    us not to do."""
    reply = client.messages.create(
        model=MUTE,
        messages=[{"role": "user", "content": ASKED}],
        max_tokens=BUDGET,
        tools=TOOLS,
        tool_choice={"type": "none"},
    )

    assert only_text(reply) == ENVELOPE
    assert reply.stop_reason == "end_turn"


def test_a_checkpoint_whose_envelope_nothing_here_parses_answers_with_the_text(
    client: anthropic.Anthropic,
) -> None:
    """Which family a checkpoint speaks is a fact of its chat template and not of the text it
    writes: read off the output instead, Qwen3.6's `<tool_call><function=…>` is taken for
    Qwen's JSON and the envelope is held for a parser that cannot read it. A template that
    spells none leaves the channel shut, and what the model wrote reaches the client whole —
    text, and no call it never made."""
    reply = client.messages.create(
        model=STRANGER,
        messages=[{"role": "user", "content": ASKED}],
        max_tokens=BUDGET,
        tools=TOOLS,
    )

    assert only_text(reply) == PREAMBLE + ENVELOPE
    assert reply.stop_reason == "end_turn"


def test_forcing_a_call_is_refused_by_name(client: anthropic.Anthropic) -> None:
    """`any` and `tool` are a constraint on decoding and there is none here: answering `auto`
    to a client that asked for one is a call the model may never have made. Named in the
    message, in this dialect's envelope."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=ECHO,
            messages=[{"role": "user", "content": ASKED}],
            max_tokens=BUDGET,
            tools=TOOLS,
            tool_choice={"type": "any"},
        )

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "tool_choice" in message

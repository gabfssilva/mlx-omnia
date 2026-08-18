"""`/api/openai/v1/responses` judged by the official SDK against a real server process.

The judge matters more here than in the chat dialect. A Responses stream is a sequence of
*named* frames whose only consumer is an accumulator: it refuses anything before
`response.created`, indexes each delta into an item and a content part it was told about
first, and rebuilds the whole `Response` at the end. Asserting on the JSON this route writes
would be asserting that it writes what this file thinks it should; `client.responses.stream`
is what says whether the frames add up to an answer.

The non-streaming half is here; the frames are in `test_responses_streams.py`, the tool calls
in `test_responses_tools.py` and the formats in `test_responses_structured.py`. The stand
they share is `responses_stand.py`.
"""

import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from openai.types.responses import ResponseInputParam

from mlx_omnia import ChatMessage
from mlx_omnia import ChatMessage as Turn
from tests.server.responses_script import (
    ANSWER,
    ASKED,
    BARE,
    CACHED,
    PIECES,
    RESULT,
    REUSED,
    SCRIPTED,
    last,
)
from tests.server.responses_stand import Stand, client, code, rendered, save_profile, stand

__all__ = ["client", "stand"]


def test_the_sdk_reads_the_answer_and_the_model_s_own_numbers(client: OpenAI) -> None:
    """The whole non-streaming shape at once, because the SDK parses it as one: an `output`
    that is not a list of typed items has no `output_text` at all.

    The usage is asserted against the prompt that actually reached the model, and the budget
    against the options the engine was handed — the two fields the dialect renames on the way
    down, and the two a route can get wrong without the answer looking any different."""
    answer = client.responses.create(
        model=SCRIPTED, input="Where is Paris?", max_output_tokens=7, temperature=0
    )

    assert answer.output_text == ANSWER
    assert answer.status == "completed"
    assert answer.id.startswith("resp_")
    assert answer.model == SCRIPTED
    item = answer.output[0]
    assert item.type == "message"
    assert item.role == "assistant"
    assert item.status == "completed"
    call = last()
    assert call.options.max_tokens == 7
    usage = answer.usage
    assert usage is not None
    assert usage.output_tokens == len(PIECES)
    assert usage.input_tokens == len(call.prompt)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.input_tokens_details.cached_tokens == 0, (
        "absent would read as a server without the field, and zero is a miss"
    )


def test_a_reused_prefix_is_a_subset_of_the_input_count(client: OpenAI) -> None:
    """This dialect counts the way the chat one does and unlike the Anthropic one:
    `input_tokens` stays the whole prompt and `cached_tokens` says how much of it the trie
    handed over, so `total_tokens` is unmoved by a hit. `cache_write_tokens` is zero because
    it is — the trie fills from the forward the turn was running anyway."""
    answer = client.responses.create(model=CACHED, input="Where is Paris?", temperature=0)

    usage = answer.usage
    assert usage is not None
    assert usage.input_tokens_details.cached_tokens == REUSED
    assert usage.input_tokens == len(last().prompt)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


def test_instructions_are_the_system_turn_the_chat_dialect_would_have_sent(
    client: OpenAI,
) -> None:
    """`instructions` is a field and not a message, and the checkpoint's template has one
    place to put it. The prompt is compared against the render of the conversation the chat
    dialect builds out of `[system, user]` — the same ids, which is what makes one model
    answer two dialects the same way.

    Written out rather than fetched through `chat/completions`: that handler is another
    agent's file this wave, and what the two dialects have to agree on is the conversation,
    which is this side of it."""
    client.responses.create(
        model=SCRIPTED, input="Where is Paris?", instructions="Answer in one word."
    )

    turns: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Where is Paris?"},
    )
    assert last().prompt == rendered(turns)


def test_a_list_of_items_is_the_conversation_it_spells(client: OpenAI) -> None:
    """The other half of `input`: typed items, including the ones this route wrote itself. An
    assistant turn comes back as `output_text` parts, and a client replaying it carries the
    `id`, `status` and `annotations` the dialect gave it — dropped here, since none of them
    is a turn.

    `developer` is the dialect's newer name for a system turn and the template knows only the
    older one, so a route that passed it through would render a role no checkpoint has."""
    items: ResponseInputParam = [
        {"role": "developer", "content": "Answer in one word."},
        {"role": "user", "content": [{"type": "input_text", "text": "Where is Paris?"}]},
        {
            "id": "msg_replayed",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
        },
        {"role": "user", "content": "And Lyon?"},
    ]
    answer = client.responses.create(model=SCRIPTED, input=items)

    turns: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Where is Paris?"},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": "And Lyon?"},
    )
    assert last().prompt == rendered(turns)
    assert answer.output_text == ANSWER


def test_a_profile_fills_what_the_request_left_out_and_nothing_else(
    stand: Stand, client: OpenAI
) -> None:
    """A profile is selected by the one field every dialect has, the model name. Its system
    prompt gives way to `instructions` for the same reason the chat dialect's gives way to a
    system message: two system turns is a conversation the template has to pick between.

    Its knobs give way the same way, and the request's own default is not what decides it —
    `repetition_penalty` is 1.0 unset, which is exactly the value that turns the penalty off,
    so a route reading the value instead of `model_fields_set` would drop the profile's."""
    save_profile(
        stand,
        SCRIPTED,
        "brief",
        {"sampling": {"repetition_penalty": 1.2}, "system_prompt": "You are terse."},
    )

    client.responses.create(model=f"{SCRIPTED}:brief", input="Where is Paris?")
    from_profile: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Where is Paris?"},
    )
    assert last().prompt == rendered(from_profile)
    assert last().options.penalty is not None

    client.responses.create(
        model=f"{SCRIPTED}:brief",
        input="Where is Paris?",
        instructions="Answer in one word.",
        extra_body={"repetition_penalty": 1.0},
    )
    from_request: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Where is Paris?"},
    )
    assert last().prompt == rendered(from_request)
    assert last().options.penalty is None


def test_store_is_refused_by_name(client: OpenAI) -> None:
    """Persistent responses are not this server's: an id it handed back would name nothing.
    Refusing by name is the difference between a client that knows and one that keeps a
    conversation id and finds out later."""
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=SCRIPTED, input="Where is Paris?", store=True)

    assert code(raised.value.body) == "store_unsupported"
    # The field is declared so the refusal can be named, which only holds if `false` answers.
    assert client.responses.create(model=SCRIPTED, input="Hi", store=False).output_text == ANSWER


def test_an_empty_input_is_refused_and_the_next_request_is_answered(client: OpenAI) -> None:
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=SCRIPTED, input="")

    assert code(raised.value.body) == "empty_input"
    assert client.responses.create(model=SCRIPTED, input="Hi").output_text == ANSWER


def test_a_model_that_takes_no_conversation_is_refused_by_name(client: OpenAI) -> None:
    """A base model ships no chat template, so nothing turns a conversation into a prompt. It
    is a client error and not a missing model: the checkpoint is there."""
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=BARE, input="Where is Paris?")

    assert code(raised.value.body) == "unsupported_input"


def test_an_unknown_model_is_the_sdk_s_not_found(client: OpenAI) -> None:
    with pytest.raises(NotFoundError) as raised:
        client.responses.create(model="nope", input="Where is Paris?")

    assert code(raised.value.body) == "model_not_found"


def test_a_generation_the_budget_cut_is_incomplete_and_says_why(client: OpenAI) -> None:
    """`completed` over a sentence `max_output_tokens` cut is a truncation reported as the
    final answer. The dialect has a status for it and a field that names the reason, and this
    is the only place a client can read either."""
    answer = client.responses.create(model=SCRIPTED, input="Where is Paris?", max_output_tokens=1)

    assert answer.status == "incomplete"
    assert answer.incomplete_details is not None
    assert answer.incomplete_details.reason == "max_output_tokens"


def test_a_generation_that_ended_on_its_own_is_completed(client: OpenAI) -> None:
    """The other half: the same script under a budget it does not reach."""
    answer = client.responses.create(model=SCRIPTED, input="Where is Paris?", max_output_tokens=64)

    assert answer.status == "completed"
    assert answer.incomplete_details is None


def test_the_text_and_the_call_of_one_generation_replay_as_one_turn(client: OpenAI) -> None:
    """The canonical loop of this dialect: the client sends back `input + response.output`,
    and a generation that wrote text *and* called something is two items — a message and a
    function_call. They are one turn of the model, and two assistant turns in the prompt tell
    it that it answered twice."""
    items: ResponseInputParam = [
        {"role": "user", "content": ASKED},
        # The item as the dialect wrote it, `id` and `status` included: what a client
        # replays is `input + response.output`, and those two are part of what it got back.
        {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Let me check.", "annotations": []}],
        },
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": RESULT},
    ]
    client.responses.create(model=SCRIPTED, input=items)

    folded: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "call_1"},
    )
    assert last().prompt == rendered(folded)

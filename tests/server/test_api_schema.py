"""`response_format`, whole: what enters the prompt, what is checked afterwards, what a
grammar refuses, and what a second attempt costs."""

import json

import pytest
from openai import APIError, OpenAI

from mlx_omnia import Chat
from mlx_omnia.engine.schema import json_instruction
from tests.server import openai_stand
from tests.server.openai_script import (
    BREAKER,
    CALLER,
    FAULTY,
    JSON_SCHEMA,
    LEARNER,
    MODEL,
    MUTE,
    PREAMBLE,
    TOOLS,
    WEATHER,
    WHOLE,
    WRITER,
    Weather,
)
from tests.server.openai_stand import (
    Recording,
    offer,
)

fresh_state = openai_stand.fresh_state
"""Overrides `conftest`'s per-test wipe, which would delete the database under the
module server while it is still answering."""

pytest_plugins = ("tests.server.openai_stand",)
"""The per-file server. `fresh_state` is imported to override `conftest`'s per-test wipe,
which would delete the database under a server that is still answering."""


def test_json_object_answers_with_the_document_and_not_with_the_text_around_it(
    base_url: str, engine: Recording
) -> None:
    """Level 1 end to end: the schema — here only the demand for JSON — enters the prompt as a
    turn of its own, and what comes back is the value that was checked.

    The prose and the fence stay out of the answer. A client that asked for `json_object` and
    got `Sure! ```json…``` ` back was handed something `json.loads` refuses, and the document
    is the one thing this server can promise: it is what it validated.

    The instruction is the engine's own `json_instruction` and not a second copy: what is asked
    for and what `validate` enforces cannot be two readings of the same vocabulary.
    """
    response = offer(base_url, WRITER, response_format={"type": "json_object"})
    payload = response.json()
    assert response.status_code == 200, response.text
    content = payload["choices"][0]["message"]["content"]
    assert json.loads(content) == {"city": "Paris", "celsius": 22}
    assert "Sure!" not in content
    assert payload["schema_attempts"] == 1, "the default is one generation, and it is visible"

    conversation = engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert conversation.messages[-1] == {"role": "system", "content": json_instruction(None)}


def test_a_json_schema_answer_validates_into_the_model_that_asked_for_it(client: OpenAI) -> None:
    """The typed round trip through the official SDK at level 1: the schema goes into the
    prompt and the answer is measured against it afterwards, which is a check that can fail —
    what the client gets when it does not is a document its own model accepts. The guarantee
    is `strict`, and it is the test below."""
    completion = client.chat.completions.create(
        model=WRITER,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=64,
        response_format=JSON_SCHEMA,
    )
    content = completion.choices[0].message.content
    assert content is not None
    assert Weather.model_validate_json(content) == Weather(city="Paris", celsius=22)


def test_the_sdk_s_parse_helper_comes_back_typed_off_a_decode_that_could_not_violate(
    client: OpenAI,
) -> None:
    """Level 3 through the door the SDK actually uses, against a real checkpoint.

    `parse()` puts `strict: true` on every param it builds out of a pydantic model, so this is
    the helper's own path and not a hand-written body. What makes it typed is the mask: the
    schema is compiled against this checkpoint's own token table and the ids that would break
    it are at -inf before the draw. The same request without `strict` was measured on this
    checkpoint and comes back 422 `malformed_json` — a 0.6B asked nicely writes prose — so what
    is being read here is the guarantee and not the model's good manners.

    `finish_reason` is checked because the SDK raises on `length` instead of parsing: a run the
    budget cut is not an answer this level promised anything about. And the content is checked
    against the schema's own keys, because the grammar is what keeps a reasoning block, a fence
    and a field nobody declared out of a document that would otherwise carry all three.
    """
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=64,
        temperature=0,
        response_format=Weather,
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop"
    assert isinstance(choice.message.parsed, Weather)
    assert choice.message.content is not None
    assert sorted(json.loads(choice.message.content)) == ["celsius", "city"]


def test_a_schema_the_grammar_will_not_take_is_a_400_in_the_compiler_s_own_words(
    base_url: str,
) -> None:
    """A guarantee this server cannot give is refused and never quietly demoted: dropping to
    level 1 here would answer a client that asked for a decode that cannot violate the schema
    with a check against a schema keyword nobody enforces.

    The message is the compiler's — `Unimplemented keys: ["uniqueItems"]` is a reason where
    "grammar error" is not, and it is what tells the client which keyword to drop. Sending the
    same schema without `strict` is the way through, and it answers.

    Against the real checkpoint and not a scripted model: a refusal is only the compiler's if
    there was a token table to refuse against, and a double has none (the test below).
    """
    unique = {
        "type": "json_schema",
        "json_schema": {
            "name": "cities",
            "schema": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "strict": True,
        },
    }
    refused = offer(base_url, MODEL, response_format=unique)
    assert refused.status_code == 400, refused.text
    error = refused.json()["error"]
    assert error["code"] == "grammar_refused"
    assert "uniqueItems" in error["message"]


def test_strict_without_a_schema_is_refused_rather_than_read_as_json_object(
    base_url: str,
) -> None:
    """The one shape of `strict` that means nothing: there is no schema for decoding to be
    constrained by, and honouring it as "answer in JSON" would call a demand a guarantee."""
    empty = {"type": "json_schema", "json_schema": {"name": "weather", "strict": True}}
    refused = offer(base_url, WRITER, response_format=empty)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "strict_without_schema"


def test_strict_and_tools_are_refused_together_because_a_grammar_leaves_no_room_for_a_call(
    base_url: str,
) -> None:
    """The one combination `strict` makes impossible rather than expensive: the mask allows
    the schema's ids from the first token, so an offered function can never be called. A 200
    with the tools silently uncallable is the answer this refusal exists instead of.

    `tool_choice: "none"` is not this case — the tools never enter the prompt, so there is
    nothing being dropped — and the test above it proves the non-strict pair still works.
    """
    strict = {
        "type": "json_schema",
        "json_schema": {"name": "weather", "schema": WEATHER, "strict": True},
    }
    refused = offer(base_url, MODEL, tools=TOOLS, response_format=strict)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "strict_with_tools"

    ignored = offer(
        base_url, MODEL, tools=TOOLS, tool_choice="none", response_format=strict, temperature=0
    )
    assert ignored.status_code == 200, ignored.text


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    base_url: str,
) -> None:
    """A scripted model is not in any catalog and holds no tokenizer, so there is no token
    table to compile against. What the client must not get is the answer anyway: `strict` asks
    for a guarantee, and a 200 carrying a free decode is that guarantee broken silently."""
    strict = {
        "type": "json_schema",
        "json_schema": {"name": "weather", "schema": WEATHER, "strict": True},
    }
    refused = offer(base_url, WRITER, response_format=strict)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "not_constrainable"


def test_a_document_that_breaks_the_schema_is_a_named_error_and_never_a_success(
    base_url: str,
) -> None:
    """The failure this level exists to make impossible: JSON that parses, a document that
    does not conform, and a 200 with it inside. The path is in the message because a client
    that is told "invalid" and not *where* is left diffing document against schema by hand.

    422 and not a 5xx: the SDKs retry a 5xx by themselves, and a whole generation bought behind
    the client's back is exactly the cost this level is supposed to show.
    """
    response = offer(base_url, BREAKER, response_format=JSON_SCHEMA)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "schema_violation"
    assert "$.celsius is required and missing" in error["message"]
    assert "after 1 generation" in error["message"], "what it cost is part of the answer"


def test_an_answer_with_no_json_in_it_is_the_other_failure_and_says_which(base_url: str) -> None:
    """A model that answered in prose is not a model that broke the schema, and the two are
    not one error: what a client does about them differs, and only one of them has a path."""
    response = offer(base_url, MUTE, response_format={"type": "json_object"})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "malformed_json"


def test_the_second_attempt_is_the_client_s_to_buy_carries_the_failure_and_is_counted(
    base_url: str, engine: Recording
) -> None:
    """The retry policy, whole: nothing is retried by default, the client raises the ceiling,
    and every generation it cost is on the answer — in `schema_attempts` and in `usage`.

    What goes into the second prompt is what the first one wrote and what was wrong with it.
    A retry over the same conversation would be the same generation, and `Corrected` answers
    off the prompt precisely so that a server that resubmitted it blind would fail here.
    """
    alone = offer(base_url, LEARNER, response_format=JSON_SCHEMA)
    assert alone.status_code == 422, "one generation is the default, and it does not retry"

    response = offer(base_url, LEARNER, response_format=JSON_SCHEMA, max_schema_attempts=2)
    payload = response.json()
    assert response.status_code == 200, response.text
    assert json.loads(payload["choices"][0]["message"]["content"]) == json.loads(WHOLE)
    assert payload["schema_attempts"] == 2
    assert payload["usage"]["completion_tokens"] == 2, "both generations are on the bill"

    conversation = engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert conversation.messages[-2] == {"role": "assistant", "content": FAULTY}
    told = conversation.messages[-1]["content"]
    assert isinstance(told, str) and "$.celsius is required and missing" in told


def test_a_stream_checks_what_it_already_sent_and_fails_with_an_error_frame(
    client: OpenAI,
) -> None:
    """A stream has one pass: the frames are gone by the time the document can be checked, so
    the violation travels the way a generation that died travels — a frame carrying `error`,
    which is the only way to fail a request that already answered 200. Closing with
    `finish_reason: "stop"` instead would hand the client a document it believes was checked."""
    stream = client.chat.completions.create(
        model=BREAKER,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=64,
        response_format=JSON_SCHEMA,
        stream=True,
    )
    drawn: list[str] = []
    with pytest.raises(APIError) as failure:
        for chunk in stream:
            if piece := chunk.choices[0].delta.content:
                drawn.append(piece)

    assert "$.celsius" in str(failure.value)
    assert drawn == [FAULTY], "what did arrive is still the client's"


def test_a_stream_cannot_buy_a_second_attempt_and_says_so(base_url: str) -> None:
    """The one combination that cannot be honoured at all: a second attempt is a second
    generation, and the first one's frames have already left."""
    refused = offer(
        base_url, WRITER, response_format=JSON_SCHEMA, max_schema_attempts=2, stream=True
    )
    assert refused.status_code == 400, refused.text
    message = refused.json()["error"]["message"]
    assert "max_schema_attempts" in message and "stream" in message


def test_a_turn_that_called_something_is_not_an_answer_to_check(base_url: str) -> None:
    """`response_format` is about the answer, and a turn that called a function has not
    answered yet: checking the preamble against the schema would turn every tool call under a
    `response_format` into a 422, which is refusing the one thing the tools were offered for."""
    payload = offer(base_url, CALLER, tools=TOOLS, response_format=JSON_SCHEMA).json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == PREAMBLE
    assert payload["schema_attempts"] == 1

"""`text.format` over `/api/openai/v1/responses`: the three levels of structured output and
what each of them promises.

Level 1 asks in the prompt and checks the answer, level 3 constrains the decode, and the
third member of the union asks for nothing at all.
"""

import json
from importlib import import_module

import pytest
from openai import BadRequestError, OpenAI, UnprocessableEntityError
from openai.types.responses import ResponseTextConfigParam

from mlx_omnia import ChatMessage
from mlx_omnia.engine.schema import json_instruction
from tests.server.responses_script import (
    ASKED,
    BREAKER,
    CALLER,
    CHECKED,
    DOCUMENT,
    GUARANTEED,
    GUIDED,
    ONLY_JSON,
    PREAMBLE,
    SCHEMA,
    SCRIPTED,
    TOOLS,
    WRITER,
    WRITTEN,
    last,
)
from tests.server.responses_stand import Stand, client, code, entry, rendered, stand

SCANS = import_module("mlx_omnia.server.services.catalog.scan")
"""The module `entry_of` calls `scan` from — the package re-exports the function under the
submodule's own name, so the package attribute is not the one a patch has to replace."""

__all__ = ["client", "stand"]


def test_a_checked_answer_is_the_document_and_not_the_text_around_it(client: OpenAI) -> None:
    """Level 1 end to end in this dialect: the schema enters the prompt as its last turn and
    what comes back is the value that was checked.

    The prose and the fence stay out of the answer. A client that asked for a schema and got
    `Sure! ```json…``` ` back was handed something `json.loads` refuses, and the document is
    the one thing this route can promise: it is what it validated. The instruction is the
    engine's own `json_instruction` and not a second copy — what is asked for and what
    `validate` enforces cannot be two readings of the same vocabulary.
    """
    answer = client.responses.create(model=WRITER, input=ASKED, text=CHECKED)

    assert json.loads(answer.output_text) == DOCUMENT
    assert "Sure!" not in answer.output_text
    turns: tuple[ChatMessage, ...] = (
        {"role": "user", "content": ASKED},
        {"role": "system", "content": json_instruction(SCHEMA)},
    )
    assert last().prompt == rendered(turns)


def test_a_format_of_text_is_the_default_and_asks_for_nothing(client: OpenAI) -> None:
    """The third member of the union, and what keeps the test above from being an assertion
    that every request gets an instruction: `{"type": "text"}` is what a client that wants
    prose sends, and a route that read it as "some format was named" would put a JSON
    instruction in the prompt and refuse the answer that came back."""
    plain: ResponseTextConfigParam = {"format": {"type": "text"}}
    answer = client.responses.create(model=WRITER, input=ASKED, text=plain)

    assert answer.output_text == "".join(WRITTEN)
    assert last().prompt == rendered(({"role": "user", "content": ASKED},))


def test_a_document_that_breaks_the_schema_is_a_named_error_and_never_a_success(
    client: OpenAI,
) -> None:
    """The failure this level exists to make visible: JSON that parses, a document that does
    not conform, and a 200 with it inside. The path is in the message because a client told
    "invalid" and not *where* is left diffing document against schema by hand.

    422 and not a 5xx: the SDKs retry a 5xx by themselves, and a whole generation bought
    behind the client's back is exactly the cost this level is supposed to show."""
    with pytest.raises(UnprocessableEntityError) as raised:
        client.responses.create(model=BREAKER, input=ASKED, text=CHECKED)

    body = entry(raised.value.body)
    assert code(body) == "schema_violation"
    message = body["message"]
    assert isinstance(message, str)
    assert "$.city is required and missing" in message
    assert "after 1 generation" in message, "what it cost is part of the answer"


def test_an_answer_with_no_json_in_it_is_the_other_failure_and_says_which(
    client: OpenAI,
) -> None:
    """A model that answered in prose is not a model that broke the schema, and the two are
    not one error: what a client does about them differs, and only one of them has a path.

    `json_object` is the format with no schema under it — all it asks is that the answer be
    JSON, which is the half of level 1 that has nothing to validate against."""
    with pytest.raises(UnprocessableEntityError) as raised:
        client.responses.create(model=SCRIPTED, input=ASKED, text=ONLY_JSON)

    assert code(raised.value.body) == "malformed_json"
    turns: tuple[ChatMessage, ...] = (
        {"role": "user", "content": ASKED},
        {"role": "system", "content": json_instruction(None)},
    )
    assert last().prompt == rendered(turns)


def test_a_strict_format_is_decoded_under_the_schema_and_never_asked_for_it(
    stand: Stand, client: OpenAI
) -> None:
    """Level 3's wiring, which is all a stand of scripted models can show: the schema is
    compiled against the model the request named and the walk that comes back is what the
    generation runs under. A route that compiled it and dropped the walk would answer 200 with
    a free decode, which is the guarantee broken silently.

    Nothing goes into the prompt: the mask is the whole of it, and a schema in the prompt as
    well would be paying for the answer twice. And nothing is checked afterwards either — the
    answer is not measured against a schema decoding could not violate.
    """
    answer = client.responses.create(model=GUIDED, input=ASKED, text=GUARANTEED)

    assert json.loads(answer.output_text) == DOCUMENT
    assert stand.engine.compiled[-1] == SCHEMA
    assert last().options.constraint is not None
    assert last().prompt == rendered(({"role": "user", "content": ASKED},))


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scripted model is in no catalog and holds no tokenizer, so there is no token table to
    compile against. What the client must not get is the answer anyway: `strict` asks for a
    guarantee, and a 200 carrying a free decode is that guarantee broken silently.

    The catalog is emptied for the read the engine makes on the way to that refusal: it walks
    the machine's own Hub cache, and what this asserts has nothing to do with what someone
    happened to have downloaded."""
    monkeypatch.setattr(SCANS, "scan", list)

    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=WRITER, input=ASKED, text=GUARANTEED)

    assert code(raised.value.body) == "not_constrainable"


def test_a_strict_format_and_tools_are_refused_together(client: OpenAI) -> None:
    """The one combination `strict` makes impossible rather than expensive: the mask allows the
    schema's ids from the first token, so an offered function can never be called. A 200 with
    the tools silently uncallable is the answer this refusal exists instead of.

    `tool_choice: "none"` is not this case — the tools never enter the prompt, so there is
    nothing being dropped."""
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=GUIDED, input=ASKED, tools=TOOLS, text=GUARANTEED)

    assert code(raised.value.body) == "strict_with_tools"
    answered = client.responses.create(
        model=GUIDED, input=ASKED, tools=TOOLS, tool_choice="none", text=GUARANTEED
    )
    assert json.loads(answered.output_text) == DOCUMENT


def test_a_turn_that_called_something_is_not_an_answer_to_check(client: OpenAI) -> None:
    """A format is about the answer, and a turn that called a function has not answered yet:
    checking the preamble against the schema would turn every tool call under a format into a
    422, which is refusing the one thing the tools were offered for."""
    answer = client.responses.create(model=CALLER, input=ASKED, tools=TOOLS, text=CHECKED)

    assert [item.type for item in answer.output] == ["message", "function_call"]
    assert answer.output_text == PREAMBLE

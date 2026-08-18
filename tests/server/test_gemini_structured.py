"""Gate 37.3, the structured answers: `responseMimeType`, `responseJsonSchema`, and the two
shapes a schema cannot be honoured in.

Split off `test_gemini.py` for size; the stand is the shared one in `gemini_stand.py`.
"""

import json

import httpx
import pytest
from google import genai
from google.genai import errors, types

from mlx_omnia import ChatMessage
from mlx_omnia.engine.schema import json_instruction
from mlx_omnia.server.services import catalog
from tests.server.gemini_stand import (
    ASKED,
    DOCUMENT,
    GUIDED,
    MODEL,
    PROSE,
    SCHEMA,
    TOOL_BODY,
    WRITER,
    Stand,
    client,
    fresh_state,
    stand,
    submitted,
    url,
)

__all__ = ["client", "fresh_state", "stand"]


def test_a_json_mime_type_is_checked_and_comes_back_as_the_document(
    stand: Stand, client: genai.Client
) -> None:
    """Level 1 in the half of it this dialect can ask for: `application/json` with no schema
    beside it is the demand that the answer be JSON, and it is checked once the generation is
    spent — upstream the same field says the model "needs to be prompted", which is exactly
    what the instruction turn is.

    The prose and the fence stay out of the answer: a client that asked for JSON and got
    `Sure! ```json…``` ` back was handed something `json.loads` refuses, and the document is
    the one thing this route can promise — it is what it validated."""
    answer = client.models.generate_content(
        model=WRITER,
        contents=ASKED,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    assert answer.text is not None
    assert json.loads(answer.text) == DOCUMENT
    assert "Sure!" not in answer.text
    turns: tuple[ChatMessage, ...] = (
        {"role": "user", "content": ASKED},
        {"role": "system", "content": json_instruction(None)},
    )
    assert submitted(stand).messages == turns


def test_an_answer_with_no_json_in_it_is_the_daemon_s_failure_and_says_so(
    client: genai.Client,
) -> None:
    """The one way level 1 fails here. INTERNAL and not INVALID_ARGUMENT: what did not
    validate is what this server generated, and of the four statuses this dialect has, that is
    the one that does not blame the client for it. The SDK retries nothing by default, so the
    generation is not bought again behind the client's back."""
    with pytest.raises(errors.ServerError) as raised:
        client.models.generate_content(
            model=PROSE,
            contents=ASKED,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

    failure = raised.value
    assert failure.code == 500
    assert failure.message is not None and "no JSON value" in failure.message
    assert "None None" not in str(failure)


def test_a_stream_checks_what_it_already_sent_and_says_error_at_the_client(
    stand: Stand,
) -> None:
    """A stream has one pass: the frames are gone by the time the document can be checked, so
    the violation travels the way a generation that died travels — a chunk opening with
    `error`. The closing frame this dialect writes carries `finishReason: STOP` and a
    `usageMetadata`, and sending it here would be a failure wearing the shape of an answer.

    Read off the frames rather than through the SDK: `google-genai` 1.46 decodes an in-stream
    `error` chunk into an empty `GenerateContentResponse` instead of raising on it, so what
    the client is given is what this asserts.
    """
    streamed = httpx.post(
        url(stand, f"{PROSE}:streamGenerateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=30,
    )
    frames = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
    failure = json.loads(frames[-1].removeprefix("data: "))["error"]
    assert "no JSON value" in failure["message"]
    assert "finishReason" not in frames[-1] and "usageMetadata" not in frames[-1]


def test_a_schema_is_a_guarantee_here_and_reaches_the_generation_as_a_walk(
    stand: Stand, client: genai.Client
) -> None:
    """This dialect has no `strict` to turn a schema into a check, because upstream a schema is
    constrained decoding: what a client sends `responseJsonSchema` for is an answer that cannot
    violate it. So the schema is compiled against the model the request named and the walk that
    comes back is what the generation runs under — a route that compiled it and dropped the
    walk would answer 200 with a free decode, which is the guarantee broken silently.

    Nothing goes into the prompt: the mask is the whole of it, and a schema in the prompt as
    well would be paying for the answer twice."""
    answer = client.models.generate_content(
        model=GUIDED,
        contents=ASKED,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=SCHEMA
        ),
    )

    assert answer.parsed == DOCUMENT, "the SDK reads its own field back off the answer"
    assert stand.engine.compiled[-1] == SCHEMA
    assert stand.engine.jobs[-1].options.constraint is not None
    alone: tuple[ChatMessage, ...] = ({"role": "user", "content": ASKED},)
    assert submitted(stand).messages == alone


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    client: genai.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model of this stand is in no catalog and holds no tokenizer, so there is no token
    table to compile against. What the client must not get is the answer anyway: a schema here
    asks for a guarantee, and a 200 carrying a free decode is that guarantee broken silently.

    The catalog is emptied for the read the engine makes on the way to that refusal: it walks
    the machine's own Hub cache, and what this asserts has nothing to do with what someone
    happened to have downloaded."""
    monkeypatch.setattr(catalog, "scan", list)

    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(
            model=MODEL,
            contents=ASKED,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=SCHEMA
            ),
        )

    failure = raised.value
    assert failure.code == 400
    assert failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "token table" in failure.message


def test_the_sdks_own_response_schema_is_refused_by_name(client: genai.Client) -> None:
    """`response_schema` is the other spelling, and what rides it is the OpenAPI subset the SDK
    builds out of its `Schema`: types spelled `OBJECT` and `STRING`, keys `property_ordering`.
    What is compiled into a grammar here is a JSON Schema, and translating one spelling into
    the other would be a second reading of a vocabulary this server does not own — a grammar
    subtly unlike the one the client asked for is worse than a refusal that names the field
    that carries it.

    A fresh dict rather than `SCHEMA`: `t_schema` processes what it is handed in place."""
    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(
            model=MODEL,
            contents=ASKED,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        )

    failure = raised.value
    assert failure.code == 400
    assert failure.message is not None and "responseJsonSchema" in failure.message


@pytest.mark.parametrize(
    ("body", "named"),
    [
        ({"generationConfig": {"responseJsonSchema": SCHEMA}}, "responseMimeType"),
        (
            {
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": SCHEMA,
                },
                "tools": TOOL_BODY,
            },
            "tools",
        ),
    ],
    ids=["without the mime type", "with tools"],
)
def test_the_two_shapes_a_schema_cannot_be_honoured_in(
    stand: Stand, body: dict[str, object], named: str
) -> None:
    """A schema with no mime type beside it is the API's own rule — it asks for a document and
    for prose around it at the same time. A schema with tools is the combination the mask makes
    impossible: it allows the schema's ids from the first token, so an offered function can
    never be called, and a 200 with the tools silently uncallable is the answer this refusal
    exists instead of.

    Posted raw, because the SDK is not what is under test here: both bodies are ones a client
    can write, and what is asserted is that each is refused with the field that was wrong in
    the message."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={"contents": [{"parts": [{"text": ASKED}]}], **body},
        timeout=30,
    )

    assert response.status_code == 400, response.text
    assert named in response.json()["error"]["message"]

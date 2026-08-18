"""Gate 37.3: the official `google-genai` SDK against a real server process.

What one generation is, in this dialect's vocabulary: the conversation it built, the counts it
writes, the frames it streams, and the errors its SDK parses. The tool channel is in
`test_gemini_tools.py` and the structured answers in `test_gemini_structured.py`; all three
share the stand in `gemini_stand.py`.
"""

import json
from pathlib import Path

import httpx
import pytest
from google import genai
from google.genai import errors, types

from mlx_omnia import greedy
from mlx_omnia.server.services import catalog
from tests.server.gemini_stand import (
    BASE,
    CACHED,
    FLAKY,
    MODEL,
    PROFILE,
    REUSED,
    SYSTEM,
    Stand,
    candidate,
    client,
    fresh_state,
    stand,
    url,
)

__all__ = ["client", "fresh_state", "stand"]


def test_the_sdk_round_trips_one_generation(client: genai.Client) -> None:
    """The answer is the render, so what the dialect turned the request into is what comes
    back. The counts are of that render — a `promptTokenCount` taken from the message the
    client sent would be a smaller number, and neither the template nor the count exists
    outside the engine."""
    answer = client.models.generate_content(model=MODEL, contents="Hi")

    rendered = "<user>Hi\n"
    assert answer.text == rendered
    assert candidate(answer).finish_reason == types.FinishReason.STOP
    assert answer.model_version == MODEL
    usage = answer.usage_metadata
    assert usage is not None
    assert usage.prompt_token_count == len(rendered)
    assert usage.candidates_token_count == 1
    assert usage.total_token_count == len(rendered) + 1
    assert usage.cached_content_token_count == 0, "absent would read as a server without it"


def test_a_reused_prefix_is_a_subset_of_the_prompt_count(client: genai.Client) -> None:
    """Here — and in the OpenAI dialect, unlike the Anthropic one — the cached count is part
    of the prompt count and not beside it: `promptTokenCount` stays the whole render, and
    `cachedContentTokenCount` says how much of it the trie handed over."""
    answer = client.models.generate_content(model=CACHED, contents="Hi")

    usage = answer.usage_metadata
    assert usage is not None
    assert usage.cached_content_token_count == REUSED
    assert usage.prompt_token_count == len("<user>Hi\n")
    assert usage.total_token_count == len("<user>Hi\n") + 1


def test_a_model_turn_becomes_an_assistant_turn(client: genai.Client) -> None:
    """`model` is this dialect's spelling and nobody else's: a role passed through untouched
    would render as `<model>`, and a template that has never seen the word renders the turn
    as if it were the user's."""
    answer = client.models.generate_content(
        model=MODEL,
        contents=[
            types.UserContent("one"),
            types.ModelContent("two"),
            types.UserContent("three"),
        ],
    )

    assert answer.text == "<user>one\n<assistant>two\n<user>three\n"


def test_the_system_instruction_is_the_first_turn(client: genai.Client) -> None:
    """The field is the request's own, next to `contents` rather than inside them, so nothing
    puts it in the conversation but this conversion."""
    answer = client.models.generate_content(
        model=MODEL,
        contents="Hi",
        config=types.GenerateContentConfig(system_instruction="Be brief."),
    )

    assert answer.text == "<system>Be brief.\n<user>Hi\n"


def test_the_stream_says_in_pieces_what_the_one_answer_says_whole(client: genai.Client) -> None:
    """Judged by the SDK's own reader: it decodes every frame it is given, so a frame in the
    wrong shape raises there instead of reading as an empty chunk. The finish reason and the
    counts ride the last frame only — a client reading usage off any chunk would be taking a
    partial count for the total."""
    chunks = list(
        client.models.generate_content_stream(
            model=MODEL,
            contents=[types.UserContent("one"), types.ModelContent("two")],
        )
    )

    assert "".join(chunk.text or "" for chunk in chunks) == "<user>one\n<assistant>two\n"
    assert len(chunks) == 3, "one frame per piece, and the frame that closes them"
    last = chunks[-1]
    assert candidate(last).finish_reason == types.FinishReason.STOP
    assert last.usage_metadata is not None
    assert last.usage_metadata.candidates_token_count == 2
    assert all(candidate(chunk).finish_reason is None for chunk in chunks[:-1])
    assert all(chunk.usage_metadata is None for chunk in chunks[:-1])


def test_max_output_tokens_reaches_the_engine(client: genai.Client) -> None:
    """The one sampling knob a scripted model can answer for: cut the budget and the answer is
    shorter. A `generationConfig` that never left the request model would give three lines."""
    answer = client.models.generate_content(
        model=MODEL,
        contents=[
            types.UserContent("one"),
            types.ModelContent("two"),
            types.UserContent("three"),
        ],
        config=types.GenerateContentConfig(max_output_tokens=2),
    )

    assert answer.text == "<user>one\n<assistant>two\n"


def test_a_name_with_a_slash_and_a_colon_routes_to_the_checkpoint_and_the_profile(
    client: genai.Client,
) -> None:
    """Both separators in one URL: `models/stand/echo:terse:generateContent`. The method is the
    last colon's, the profile the one before it, and the slash is part of the name — the stand's
    loader answers to `stand/echo` and to nothing else, so a split made anywhere else is a
    request that never reaches a model."""
    answer = client.models.generate_content(model=f"{MODEL}:{PROFILE}", contents="Hi")

    assert answer.text == f"<system>{SYSTEM}\n<user>Hi\n"
    assert answer.model_version == f"{MODEL}:{PROFILE}"


def test_the_profile_fills_the_knob_the_request_left_out(
    stand: Stand, client: genai.Client
) -> None:
    """No dialect has a field for a profile, so the name selects it and the request overrides
    it. `temperature: 0` is the profile's, and the deterministic end of the dial is the one
    sampler that can be named from out here."""
    client.models.generate_content(model=f"{MODEL}:{PROFILE}", contents="Hi")
    assert stand.engine.jobs[-1].options.sampler is greedy

    client.models.generate_content(
        model=f"{MODEL}:{PROFILE}",
        contents="Hi",
        config=types.GenerateContentConfig(temperature=1.0),
    )
    assert stand.engine.jobs[-1].options.sampler is not greedy


def test_an_unknown_method_after_the_colon_is_this_dialect_s_error(client: genai.Client) -> None:
    """`count_tokens` is a real method of the SDK and not one this server serves, so it walks
    the same path with a tail the router would otherwise answer 404 to with a body of its own
    shape. What the assertion is about is the message: `APIError` parses `status`, `message`
    and `code` out of the envelope, and prints `None None.` when they are not there."""
    with pytest.raises(errors.ClientError) as raised:
        client.models.count_tokens(model=MODEL, contents="Hi")

    failure = raised.value
    assert failure.code == 404
    assert failure.status == "NOT_FOUND"
    assert failure.message is not None and "countTokens" in failure.message
    assert "None None" not in str(failure)


def test_an_unknown_model_is_a_client_error_the_sdk_can_read(client: genai.Client) -> None:
    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(model="nope", contents="Hi")

    failure = raised.value
    assert failure.code == 404
    assert failure.status == "NOT_FOUND"
    assert failure.message is not None and "nope" in failure.message
    assert "None None" not in str(failure)


def test_a_model_without_a_chat_template_is_refused(client: genai.Client) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go: the
    client's error, named, and not a 500 out of the worker."""
    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(model=BASE, contents="Hi")

    failure = raised.value
    assert failure.code == 400
    assert failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "chat template" in failure.message


def test_a_knob_this_dialect_does_not_honour_is_refused_by_name(stand: Stand) -> None:
    """Accepting `stopSequences` and never cutting on one tells the client it was applied. The
    envelope a refused body comes back in is the app's handler, and `test_dialect_errors.py`
    is where that is read — what this asserts is that it is refused at all, and that the field
    is named where the client can read it."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": "Hi"}]}],
            "generationConfig": {"stopSequences": ["x"]},
        },
        timeout=30,
    )

    assert 400 <= response.status_code < 500, response.text
    assert "stopSequences" in response.text


def test_a_content_with_no_role_is_a_user_turn(stand: Stand) -> None:
    """What the API documents for an absent role. The SDK always writes one, so nothing above
    reaches the default — and a turn that quietly became the assistant's would put words in the
    model's mouth."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={"contents": [{"parts": [{"text": "Hi"}]}]},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "<user>Hi\n"


def test_the_models_list_carries_the_catalog_and_every_name_it_answers_to(
    stand: Stand, client: genai.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog, not the residents, under this dialect's name for a model — `models/{id}`
    is what the SDK puts back into the path. A profile is listed as a model of its own because
    the name is the only field a client has to select one with.

    The scan is replaced rather than read: it walks the machine's own Hub cache, and what this
    asserts is the shape and the profile, not what someone happened to have downloaded.
    """
    entry = catalog.CatalogEntry(
        id=MODEL,
        directory=Path("/nowhere"),
        store=Path("/nowhere"),
        architecture="echo",
        quantization=None,
        dtype=None,
        context=None,
        bytes_on_disk=0,
    )
    monkeypatch.setattr(catalog, "scan", lambda: [entry])

    listed = list(client.models.list())

    assert [model.name for model in listed] == [f"models/{MODEL}", f"models/{MODEL}:{PROFILE}"]
    assert listed[0].supported_actions == ["generateContent", "streamGenerateContent"]


def test_a_generation_that_dies_mid_stream_says_error_instead_of_closing_with_stop(
    stand: Stand, client: genai.Client
) -> None:
    """The failure a stream can hide. The response is already 200 and the frames already
    going out when the decode thread gives out; the closing frame this dialect writes carries
    `finishReason: STOP` and a `usageMetadata` — a failure wearing the shape of an answer.
    What goes out instead is a chunk opening with `error`, carrying the daemon's own words.

    Read off the frames as well as through the SDK: `google-genai` 1.46 decodes an in-stream
    `error` chunk into an empty `GenerateContentResponse` rather than raising on it, so the
    SDK alone cannot tell the failure from a stream that simply stopped — what it can say is
    that no chunk closed the generation as a finished answer.
    """
    for chunk in client.models.generate_content_stream(model=FLAKY, contents="one"):
        assert candidate(chunk).finish_reason is None if chunk.candidates else True
        assert chunk.usage_metadata is None

    streamed = httpx.post(
        url(stand, f"{FLAKY}:streamGenerateContent"),
        json={"contents": [{"parts": [{"text": "one"}]}]},
        timeout=30,
    )
    frames = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
    failure = json.loads(frames[-1].removeprefix("data: "))["error"]
    assert failure["code"] == 500
    assert "RuntimeError" in failure["message"], "the reason is the daemon's own words"
    assert "finishReason" not in frames[-1] and "usageMetadata" not in frames[-1]


def test_a_generation_the_budget_cut_says_max_tokens_and_not_stop(client: genai.Client) -> None:
    """`MAX_TOKENS` is the branch an agent loop takes to continue. `STOP` over a sentence
    `maxOutputTokens` cut is a truncation reported as a finished answer, and nothing else in
    the body says otherwise. Both shapes, because the reason rides the closing frame here."""
    whole = client.models.generate_content(
        model=MODEL,
        contents=[types.UserContent("one"), types.ModelContent("two")],
        config=types.GenerateContentConfig(max_output_tokens=1),
    )
    assert candidate(whole).finish_reason == types.FinishReason.MAX_TOKENS

    chunks = list(
        client.models.generate_content_stream(
            model=MODEL,
            contents=[types.UserContent("one"), types.ModelContent("two")],
            config=types.GenerateContentConfig(max_output_tokens=1),
        )
    )
    assert candidate(chunks[-1]).finish_reason == types.FinishReason.MAX_TOKENS


def test_a_generation_that_ended_on_its_own_still_says_stop(client: genai.Client) -> None:
    """The other half, and what keeps the one above from asserting that every answer is
    truncated: the same model under a budget it does not reach."""
    answer = client.models.generate_content(
        model=MODEL,
        contents=[types.UserContent("one")],
        config=types.GenerateContentConfig(max_output_tokens=64),
    )
    assert candidate(answer).finish_reason == types.FinishReason.STOP

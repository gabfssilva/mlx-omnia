"""An envelope that is not a call, and a call that arrives while it is being written."""

import json

from openai import OpenAI

from tests.server import openai_stand
from tests.server.openai_script import (
    BIG,
    BIG_TOOLS,
    MODEL,
    PATCH,
    SCRIPTS,
    TOOLS,
    TRUNCATED,
    XML_ANSWER,
    XML_CALLER,
    XML_SCRIPT,
)
from tests.server.openai_stand import (
    offer,
    sse,
)

fresh_state = openai_stand.fresh_state
"""Overrides `conftest`'s per-test wipe, which would delete the database under the
module server while it is still answering."""

pytest_plugins = ("tests.server.openai_stand",)
"""The per-file server. `fresh_state` is imported to override `conftest`'s per-test wipe,
which would delete the database under a server that is still answering."""


def test_an_envelope_the_budget_cut_in_half_comes_back_as_the_text_it_is(base_url: str) -> None:
    """Held as a possible call and it is not one: what the model wrote goes out as text. The
    silent failure this rules out is the opposite — an envelope suppressed and no call
    produced reaches the client as a model that chose to call nothing, which is exactly the
    shape of a correct refusal."""
    payload = offer(base_url, TRUNCATED, tools=TOOLS).json()
    choice = payload["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": SCRIPTS[TRUNCATED][0]}
    assert choice["finish_reason"] == "stop"


def test_a_checkpoint_whose_envelope_nothing_can_parse_answers_with_the_text_it_wrote(
    base_url: str,
) -> None:
    """Which family a checkpoint speaks is read off its chat template, and Qwen3.6's answer
    there is "none": it keeps Qwen's marker and fills it with XML. Sniffed from the generated
    text instead, that marker says Qwen — the envelope is held, no call comes out of it, and
    the client is handed the answer with the call moved to the end of it.

    Nothing is suppressed and nothing is an error: the turn is text, and text is what comes
    back, whole and in the pieces it was written in.

    On the reasoning channel, and that is the template's doing rather than this test's:
    Qwen3.6 writes `<think>` into the generation prompt, so the script starts inside the block
    and never closes it. Which channel the text came out on is not what is under test here —
    that it came out whole, in the pieces it was written in, is.
    """
    payload = offer(base_url, XML_CALLER, tools=TOOLS).json()
    choice = payload["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": XML_ANSWER,
    }
    assert choice["finish_reason"] == "stop"

    frames = [json.loads(frame) for frame in sse(base_url, XML_CALLER, tools=TOOLS)]
    assert [frame["choices"][0]["delta"] for frame in frames] == [
        {"role": "assistant", "content": ""},
        *({"reasoning_content": piece} for piece in XML_SCRIPT),
        {},
    ]


def test_the_arguments_of_one_call_arrive_in_more_than_one_frame(base_url: str) -> None:
    """The reversal of A11, measured where it matters: a four-kilobyte argument reaches the
    client while it is being written, not in one frame after the generation ended.

    Counted rather than asserted shape-wise, because one frame carrying everything is also a
    valid stream — it is the *latency* that was wrong, and only the count says so.
    """
    frames = [json.loads(frame) for frame in sse(base_url, BIG, tools=BIG_TOOLS)]
    carrying = [
        frame
        for frame in frames
        for choice in frame.get("choices", [])
        if choice["delta"].get("tool_calls")
    ]
    assert len(carrying) > 1, "the whole call arrived in one frame; nothing is being streamed"

    entries = [
        entry
        for frame in carrying
        for choice in frame["choices"]
        for entry in choice["delta"]["tool_calls"]
    ]

    # The name is on the first entry and on no other: the SDK's accumulator *concatenates*
    # `function.name` across frames the same way it concatenates the arguments, so a name
    # repeated four times reaches the client as `apply_patchapply_patchapply_patch…` and the
    # tool it names does not exist.
    named = [entry for entry in entries if entry.get("function", {}).get("name") is not None]
    assert len(named) == 1
    assert named[0]["function"]["name"] == "apply_patch"

    # And so is the id, for the same reason and one worse: it is what the client sends back
    # as `tool_call_id`, so two frames disagreeing about it is a result that answers nothing.
    identified = [entry for entry in entries if entry.get("id") is not None]
    assert len(identified) == 1
    assert identified[0]["id"].startswith("call_")
    assert identified[0] is named[0], "the id and the name belong to the same first frame"

    # Every entry carries `index`, which is what the accumulator matches two frames by.
    assert all(
        "index" in entry
        for frame in carrying
        for choice in frame["choices"]
        for entry in choice["delta"]["tool_calls"]
    )

    joined = "".join(
        entry["function"].get("arguments", "")
        for frame in carrying
        for choice in frame["choices"]
        for entry in choice["delta"]["tool_calls"]
    )
    assert json.loads(joined) == {"path": "a.py", "body": PATCH}


def test_the_first_frame_of_a_call_arrives_before_the_generation_ends(base_url: str) -> None:
    """Ordering and not a clock: the frame that names the call comes before the last frame
    the generation produces. A stream that announced the call only at the end would put it
    after everything, which is what it did before."""
    frames = [json.loads(frame) for frame in sse(base_url, BIG, tools=BIG_TOOLS)]
    naming = next(
        at
        for at, frame in enumerate(frames)
        for choice in frame.get("choices", [])
        if any(
            entry.get("function", {}).get("name") for entry in choice["delta"].get("tool_calls", [])
        )
    )
    assert naming < len(frames) - 1


def test_the_official_sdk_accumulates_the_streamed_call(client: OpenAI) -> None:
    """Judged by the SDK's own accumulator rather than by our reading of the frames: what a
    harness sees is what that code builds, and it raises on an entry with no `index`."""
    with client.chat.completions.stream(
        model=BIG, messages=[{"role": "user", "content": "patch it"}], tools=BIG_TOOLS
    ) as stream:
        final = stream.get_final_completion()
    calls = final.choices[0].message.tool_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].function.name == "apply_patch"
    assert json.loads(calls[0].function.arguments) == {"path": "a.py", "body": PATCH}


def test_a_forced_tool_choice_produces_a_call_the_model_did_not_have_to_make(
    client: OpenAI,
) -> None:
    """`required`, honoured rather than refused: the decode is constrained to the checkpoint's
    own call envelope over the tools offered, so what comes back cannot be prose.

    Asked something no tool answers, on purpose. A model left to choose would reply in text —
    that is the whole point of the field, and a test that asked about the weather would pass
    with the constraint switched off.
    """
    answer = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello. Do not call anything."}],
        tools=TOOLS,
        tool_choice="required",
        max_tokens=64,
        temperature=0.0,
    )
    choice = answer.choices[0]
    assert choice.finish_reason == "tool_calls"
    calls = choice.message.tool_calls
    assert calls is not None and len(calls) >= 1

    # The name is one of the offered functions and the arguments obey that function's own
    # schema — which is what the grammar's `anyOf` branch ties together. A grammar that let
    # any offered name stand beside any offered arguments would pass a weaker test.
    made = calls[0]
    assert made.type == "function"
    assert made.function.name in {"get_weather", "get_time"}
    arguments = json.loads(made.function.arguments)
    expected = {"get_weather": "city", "get_time": "zone"}[made.function.name]
    assert expected in arguments


def test_forcing_a_call_is_refused_for_a_checkpoint_whose_envelope_has_no_grammar(
    base_url: str,
) -> None:
    """The rule the kernel layer keeps, applied here: a strategy builds only when it
    implements the declared thing exactly.

    Qwen3.6 spells its arguments as XML elements, which the compiler does not know, so no
    grammar can promise the decode produces one. Answering `auto` instead would hand the
    client a turn that may contain no call at all, under a field whose whole purpose is that
    it does — so the refusal names the checkpoint's limit rather than hiding it.
    """
    refusal = offer(base_url, XML_CALLER, tools=TOOLS, tool_choice="required")
    assert refusal.status_code == 400, refusal.text
    body = refusal.json()["error"]
    assert body["code"] == "tool_choice_unsupported"
    assert "envelope" in body["message"]

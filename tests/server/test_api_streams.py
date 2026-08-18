"""What a streamed turn carries: the usage frame, a failure mid-stream, the finish reason,
and the thinking channel."""

import json
import time

import httpx
import pytest
from openai import APIError, OpenAI

from tests.server import openai_stand
from tests.server.openai_script import CALLER, FLAKY, MODEL, MUTE, THINKER
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


def test_include_usage_ends_the_stream_with_a_usage_frame(base_url: str) -> None:
    """The dialect's shape: one extra frame after the finish one and before `[DONE]`,
    carrying the whole request's usage and no choices. No frame before it carries usage —
    a client reading it off any chunk would be taking a partial count for the total."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        frames = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert frames[-1] == "[DONE]"
    last = json.loads(frames[-2])
    usage = last["usage"]
    assert last["choices"] == []
    assert usage["completion_tokens"] > 0
    # Not derivable from the other two: `total` is computed from them, so a streaming path
    # that lost the prompt count would satisfy the sum with a zero in it.
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert all("usage" not in json.loads(frame) for frame in frames[:-2])


def test_the_usage_frame_carries_what_the_turn_cost_in_seconds(base_url: str) -> None:
    """The dialect has no field for a rate, and a chat window that draws one would otherwise
    have to time the stream from outside — where the load, the queue and the prefill are one
    number. The extension rides on the frame that is already the request's total.

    `load_seconds` is not asserted: whether this request found the model resident depends on
    what the suite ran before it, and the key being present with either answer is the
    contract. What is asserted is that the key is there at all.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        frames = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    timings = json.loads(frames[-2])["x_mlx_omnia"]
    assert "load_seconds" in timings
    assert timings["ttft_seconds"] > 0
    assert timings["prefill_tokens_per_second"] > 0
    assert timings["tokens_per_second"] > 0
    assert timings["bytes_per_token"] > 0
    assert 0 < timings["ceiling_fraction"] < 1
    # Present and null, not absent: a turn that did not speculate says so, and a reader that
    # only ever sees this model would otherwise not know the key exists.
    assert timings["speculation"] is None


def test_the_official_sdk_accumulates_the_usage_frame(client: OpenAI) -> None:
    """The SDK's own accumulator over the stream: a frame it cannot fold in is a frame
    that reads as a malformed completion rather than as usage."""
    # A generation that ends on its own: the SDK's accumulator raises `LengthFinishReasonError`
    # on a truncated one, which is its own contract and not what this test is about.
    with client.chat.completions.stream(
        model=CALLER,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        temperature=0,
        stream_options={"include_usage": True},
    ) as stream:
        final = stream.get_final_completion()
    assert final.usage is not None
    assert final.usage.completion_tokens > 0
    assert final.choices[0].message.content


def test_a_client_that_closes_the_connection_leaves_a_cancelled_record(
    base_url: str, engine: Recording
) -> None:
    """Abandoning the stream used to be silent. The job that was cut mid-generation is
    recorded as cancelled, with the tokens it did emit."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a very long story."}],
        "max_tokens": 512,
        "stream": True,
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        for index, _line in enumerate(response.iter_lines()):
            if index >= 3:
                break

    job = engine.jobs[-1]
    deadline = time.time() + 10
    while job.state in ("queued", "running"):
        assert time.time() < deadline, "the worker never reached a terminal state"
        time.sleep(0.02)
    assert job.state == "cancelled"
    assert 0 < job.meter.completion_tokens < 512


def test_a_generation_that_dies_mid_stream_is_an_error_and_not_a_finished_answer(
    client: OpenAI,
) -> None:
    """The failure a stream can hide: the response is already 200 and the frames already
    going out when the decode thread gives out. Closing with `finish_reason: "stop"` and
    `[DONE]` would be a completed answer, and the client would keep whatever text arrived as
    the whole of it — the SDK cannot tell that apart afterwards. A frame carrying `error` is
    what it raises on, which is why the failure travels as one."""
    stream = client.chat.completions.create(
        model=FLAKY,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=8,
        stream=True,
    )

    drawn: list[str] = []
    with pytest.raises(APIError) as failure:
        for chunk in stream:
            if piece := chunk.choices[0].delta.content:
                drawn.append(piece)

    assert "RuntimeError" in str(failure.value), "the reason is the daemon's own words"
    assert drawn == ["Half an "], "what did arrive is still the client's"


def test_a_generation_the_budget_cut_says_length_and_not_stop(client: OpenAI) -> None:
    """`length` is what an agent loop branches on to continue. Answering `stop` for a
    generation `max_tokens` cut hands it half a sentence as the final answer, and nothing else
    in the body says otherwise — the text is there either way.

    The real checkpoint and not a script: what decides is the count the loop wrote, and a
    double that yields text without counting ids has nothing to decide with. Both shapes,
    because the streaming one carries the reason in a frame of its own.
    """
    whole = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=1, temperature=0
    )
    assert whole.choices[0].finish_reason == "length"
    assert whole.usage is not None and whole.usage.completion_tokens == 1

    frames = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
        temperature=0,
        stream=True,
    )
    reasons = [chunk.choices[0].finish_reason for chunk in frames if chunk.choices]
    assert reasons[-1] == "length"


def test_a_generation_that_ended_on_its_own_still_says_stop(client: OpenAI) -> None:
    """The other half of the pair above, and what keeps it from being an assertion that
    every answer is truncated: the same checkpoint with a budget it does not reach."""
    answered = client.chat.completions.create(
        model=CALLER, messages=[{"role": "user", "content": "Hi"}], max_tokens=64
    )
    assert answered.choices[0].finish_reason == "stop"


def test_the_thinking_block_is_a_field_of_its_own_and_never_the_answer(base_url: str) -> None:
    """A11's channel, in the field this dialect has for it — the one DeepSeek, vLLM and Ollama
    answer with over this same route.

    Without it the block goes out as the answer: a client draws `<think>` and everything under
    it as what the model said, and the answer itself starts arriving hundreds of tokens into
    the turn. The markers stay out of both fields, and out of the field that names the channel
    they would be the model quoting itself.
    """
    message = offer(base_url, THINKER).json()["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": "\nParis.",
        "reasoning_content": "\nWeighing it.\n",
    }

    frames = [json.loads(frame) for frame in sse(base_url, THINKER)]
    # The two reasoning frames are the two pieces the block was written in — the seam of
    # `</think>` is what the second one closes, and it leaves as text with no marker in it.
    assert [frame["choices"][0]["delta"] for frame in frames] == [
        {"role": "assistant", "content": ""},
        {"reasoning_content": "\nWeigh"},
        {"reasoning_content": "ing it.\n"},
        {"content": "\nParis."},
        {},
    ]


def test_a_turn_that_thinks_nothing_carries_no_reasoning_at_all(base_url: str) -> None:
    """The field is the block's, not every turn's: a checkpoint that writes no block answers
    exactly what it answered before there was a channel to name."""
    message = offer(base_url, MUTE).json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "I would rather not."}

    frames = [json.loads(frame) for frame in sse(base_url, MUTE)]
    assert [frame["choices"][0]["delta"] for frame in frames] == [
        {"role": "assistant", "content": ""},
        {"content": "I would rather not."},
        {},
    ]

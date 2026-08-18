"""The sampler's doors, and the fields this dialect refuses rather than ignores."""

import json

import httpx
import pytest
from openai import OpenAI

from tests.server import openai_stand
from tests.server.openai_script import BASE_MODEL, CALLER, MODEL, TOOLS
from tests.server.openai_stand import (
    Recording,
    answer,
    ask,
    offer,
    post,
    sse,
)

fresh_state = openai_stand.fresh_state
"""Overrides `conftest`'s per-test wipe, which would delete the database under the
module server while it is still answering."""

pytest_plugins = ("tests.server.openai_stand",)
"""The per-file server. `fresh_state` is imported to override `conftest`'s per-test wipe,
which would delete the database under a server that is still answering."""


def test_a_single_candidate_is_the_greedy_answer(base_url: str) -> None:
    """The two deterministic doors into the sampler — the zero of the dial and a cut that
    leaves one candidate — reach the same tokens the default greedy path reaches."""
    greedy_answer = answer(post(base_url, temperature=0))
    assert answer(post(base_url, temperature=1.4, top_k=1, seed=5)) == greedy_answer


def test_a_seed_replays_the_answer(base_url: str) -> None:
    seeded = answer(post(base_url, temperature=1.0, seed=1234, max_tokens=24))
    assert answer(post(base_url, temperature=1.0, seed=1234, max_tokens=24)) == seeded
    others = {answer(post(base_url, temperature=1.0, seed=s, max_tokens=24)) for s in (7, 99, 5150)}
    assert seeded not in others, "three other seeds all landed on the same answer"


def test_an_unseeded_request_is_drawn_not_argmaxed(base_url: str) -> None:
    """The dialect's default temperature is 1.0. If the server quietly used greedy instead,
    repeated unseeded requests would come back identical."""
    answers = {answer(post(base_url, max_tokens=24)) for _ in range(4)}
    assert len(answers) > 1


def test_sampling_reaches_the_engine(base_url: str) -> None:
    """A penalty strong enough to forbid every id already in the prompt or the answer:
    whatever comes back cannot repeat itself, which no default path would guarantee."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say the word banana ten times."}],
        "max_tokens": 24,
        "temperature": 0,
        "repetition_penalty": 1000.0,
    }
    response = httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body, timeout=60)
    penalized = answer(response)
    assert penalized != answer(post(base_url, temperature=0, max_tokens=24))


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"logit_bias": {"1": 2}}, "logit_bias"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"n": 2}, "n"),
        ({"temperature": -1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"min_p": 1.0}, "min_p"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"stream_options": {"include_logprobs": True}}, "stream_options"),
        # `required` with nothing to call is a contradiction, and the one shape of it this
        # route still refuses: the constraint has no set of functions to pin the call to.
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, "tool_choice"),
        # Retries of a check nobody asked for: the field would be read by nobody, which is
        # the one case a request-level schema cannot catch on its own.
        ({"max_schema_attempts": 2}, "max_schema_attempts"),
        (
            {"max_schema_attempts": 9, "response_format": {"type": "json_object"}},
            "max_schema_attempts",
        ),
        ({"response_format": {"type": "yaml"}}, "response_format"),
    ],
)
def test_a_field_we_do_not_honour_is_refused_by_name(
    base_url: str, client: OpenAI, fields: dict[str, object], named: str
) -> None:
    """Accepting `logit_bias` and ignoring it tells the client it was applied. The name has
    to be in the message, or the client is left guessing which of its fields we mean."""
    response = post(base_url, **fields)
    assert response.status_code == 400, response.text
    assert named in response.json()["error"]["message"]
    assert len(ask(client, "Hi", max_tokens=4)) > 0


def test_a_model_without_a_chat_template_is_refused(base_url: str) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go: a
    client error, not a 500 from the worker."""
    body = {"model": BASE_MODEL, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4}
    response = httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body, timeout=120)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_input"


def test_the_fields_a_client_sends_by_default_reach_the_dialect(base_url: str) -> None:
    """The refusal this dialect owes a client is about what it cannot honour, and not about
    what it would do anyway. Every field here is one an ordinary OpenAI client puts on a
    request without being asked to, and each is accepted at the value that asks for what
    already happens."""
    accepted = offer(
        base_url,
        CALLER,
        tools=TOOLS,
        parallel_tool_calls=True,
        n=1,
        user="someone",
        logprobs=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    assert accepted.status_code == 200


def test_ignoring_a_field_does_not_change_the_prompt(base_url: str, engine: Recording) -> None:
    """An accepted-and-ignored field has to be exactly that. The proof is the conversation
    the engine was handed: the same characters with the field and without it."""
    offer(base_url, CALLER, tools=TOOLS)
    plain = engine.jobs[-1].input
    offer(base_url, CALLER, tools=TOOLS, parallel_tool_calls=True, user="someone", n=1)
    assert engine.jobs[-1].input == plain


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parallel_tool_calls", False),
        ("n", 2),
        ("logprobs", True),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
    ],
)
def test_a_field_that_would_change_the_answer_is_refused_by_name(
    base_url: str, field: str, value: object
) -> None:
    """Each of these asks for something this server does not do, and the difference from the
    ones above is that honouring them by accident is invisible: `n: 2` answered with one
    choice, `parallel_tool_calls: false` answered with two calls. The name of the field has to
    be in the message, because that is the only thing that tells a client which one to drop."""
    refusal = offer(base_url, CALLER, tools=TOOLS, **{field: value})
    # 400 through the dialect's own validation handler, which is where every other refused
    # field in this route lands — the point is the name, not a status of its own.
    assert refusal.status_code == 400, refusal.text
    assert field in refusal.json()["error"]["message"]


def test_a_stop_sequence_cuts_the_answer_before_it(base_url: str) -> None:
    """The engine's `stop` is a set of token ids and cannot express a string, so the sequence
    is honoured over the text: the answer is cut before it, and it never reaches the client."""
    payload = offer(base_url, CALLER, stop=" check").json()
    choice = payload["choices"][0]
    assert choice["message"]["content"] == "Let me"
    assert choice["finish_reason"] == "stop"

    streamed = [json.loads(frame) for frame in sse(base_url, CALLER, stop=" check")]
    text = "".join(
        choice["delta"].get("content", "")
        for frame in streamed
        for choice in frame.get("choices", [])
    )
    assert text == "Let me"
    assert " check" not in text

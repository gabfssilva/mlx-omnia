"""`/api/anthropic/v1/*` against a real server process, judged by the official SDK.

The model under the engine answers with the prompt it was handed, in fixed-size pieces. That
is the only window a test has on what this dialect exists to do: `system` arrives as a field
of the request and has to leave as a turn of the conversation, and what turns one into the
other happens inside the engine, behind a chat template. A double that answered with
something of its own would leave the translation untested — the answer here *is* the rendered
conversation.

Streaming is judged by `client.messages.stream` and not by our reading of the frames: the
SDK's decoder drops a frame with no `event:` line, and its accumulator raises on an event
that arrives before `message_start`. One raw-frame test sits beside it to pin the sequence
itself, which the accumulator tolerates more of than the dialect allows.

The stand, the doubles and the script they answer with are the three modules beside this one,
shared with the suites that carry the rest of this dialect.
"""

import math

import anthropic
import httpx
import pytest

from tests.server.anthropic_script import (
    BASE,
    BUDGET,
    CACHED,
    CATALOGUED,
    CHUNK,
    ECHO,
    FLAKY,
    PRESET,
    REUSED,
    SLOW,
    rendered,
)
from tests.server.anthropic_stand import (
    Stand,
    ask,
    client,
    entry,
    envelope,
    frames,
    fresh_state,
    only_text,
    stand,
    text,
    turns,
)

__all__ = ["client", "fresh_state", "stand"]
"""The fixtures live in the stand module and are imported for pytest to find them here."""


def test_the_system_field_becomes_the_first_turn_and_nothing_else_does(
    client: anthropic.Anthropic,
) -> None:
    """The translation this dialect exists for. The answer is the rendered conversation, so a
    `system` that was dropped, given the wrong role, or appended after the user's own turn is
    visible in it — and the request that sends none is what says the template is not writing
    a system turn by itself."""
    plain = ask(client, "Hello")
    assert only_text(plain) == rendered(("user", "Hello"))

    named = ask(client, "Hello", system="Answer in one word.")
    assert only_text(named) == rendered(("system", "Answer in one word."), ("user", "Hello"))

    blocks = ask(client, "Hello", system=[{"type": "text", "text": "Answer in one word."}])
    assert only_text(blocks) == only_text(named)


def test_the_billing_header_never_reaches_the_prompt(client: anthropic.Anthropic) -> None:
    """Claude Code opens `system` with a block addressed to upstream's billing. It instructs
    this checkpoint in nothing and sits at the front of the prefix every request reuses, so
    it is dropped — and the blocks after it are joined exactly as they were."""
    reply = ask(
        client,
        "Hello",
        system=[
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.233; cc_entrypoint=sdk-cli;",
            },
            {"type": "text", "text": "Answer in one word."},
        ],
    )
    assert only_text(reply) == rendered(("system", "Answer in one word."), ("user", "Hello"))


def test_a_system_field_that_is_only_the_billing_header_opens_no_turn(
    client: anthropic.Anthropic,
) -> None:
    """What is left is nothing, and a system turn holding nothing is a turn the template
    should never have been handed."""
    reply = ask(
        client,
        "Hello",
        system=[{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.233;"}],
    )
    assert only_text(reply) == rendered(("user", "Hello"))


def test_a_profile_fills_the_system_the_request_left_out(client: anthropic.Anthropic) -> None:
    """`model:profile` is how a dialect with no field for a preset selects one. Its prompt
    becomes the conversation's first turn, and it loses to a request that sent one of its
    own — the same precedence the sampling knobs follow."""
    preset = ask(client, "Hello", model=f"{ECHO}:terse")
    assert only_text(preset) == rendered(("system", PRESET), ("user", "Hello"))

    own = ask(client, "Hello", model=f"{ECHO}:terse", system="Speak plainly.")
    assert only_text(own) == rendered(("system", "Speak plainly."), ("user", "Hello"))


def test_usage_counts_the_rendered_prompt_and_the_pieces_emitted(
    client: anthropic.Anthropic,
) -> None:
    """`input_tokens` is not the message the client sent: what reaches the model is the
    conversation the template rendered, which exists nowhere but inside the engine. The echo
    counts one prompt token per character of it, and one output token per piece."""
    prompt = rendered(("user", "Hello"))
    reply = ask(client, "Hello")

    assert reply.usage.input_tokens == len(prompt)
    assert reply.usage.output_tokens == math.ceil(len(prompt) / CHUNK)
    assert reply.role == "assistant"
    assert reply.type == "message"
    assert reply.model == ECHO


def test_a_reused_prefix_is_read_tokens_and_comes_out_of_the_input_count(
    client: anthropic.Anthropic,
) -> None:
    """This dialect's three input fields are disjoint — a client adds them to get the prompt
    — so a reuse counted in both `input_tokens` and `cache_read_input_tokens` would double
    the turn. Nothing is written to the cache: the trie fills from the forward the turn was
    running anyway, so there is no creation to charge."""
    prompt = rendered(("user", "Hello"))
    reply = client.messages.create(model=CACHED, messages=turns("Hello"), max_tokens=BUDGET)

    assert reply.usage.cache_read_input_tokens == REUSED
    assert reply.usage.input_tokens == len(prompt) - REUSED
    assert reply.usage.cache_creation_input_tokens == 0


def test_a_turn_with_no_reuse_says_zero_instead_of_saying_nothing(
    client: anthropic.Anthropic,
) -> None:
    """Absent reads as a server that does not carry the field, and zero as a miss. A client
    tuning a conversation to hit the cache needs to tell those apart, and the echo above
    never reuses anything."""
    reply = ask(client, "Hello")

    assert reply.usage.cache_read_input_tokens == 0
    assert reply.usage.input_tokens == len(rendered(("user", "Hello")))


def test_the_budget_and_the_end_of_the_turn_are_two_stop_reasons(
    client: anthropic.Anthropic,
) -> None:
    """This dialect's vocabulary and not OpenAI's: `end_turn` where the other says `stop`.
    The two are told apart by the count, so the cut answer has to be short *and* say so."""
    whole = ask(client, "Hello")
    assert whole.stop_reason == "end_turn"
    assert whole.stop_sequence is None

    cut = ask(client, "Hello", max_tokens=2)
    assert cut.stop_reason == "max_tokens"
    assert cut.usage.output_tokens == 2
    assert only_text(cut) == rendered(("user", "Hello"))[: 2 * CHUNK]


def test_the_official_sdk_accumulates_the_named_events(client: anthropic.Anthropic) -> None:
    """What judges the frames is the SDK's own accumulator: an event it cannot name raises
    "unexpected event order" instead of folding in, and `input_tokens` reaches it through
    `message_start` alone — a stream that opened with a placeholder count would answer zero
    here while the non-streaming path stayed green."""
    prompt = rendered(("user", "Hello"))
    with client.messages.stream(model=ECHO, messages=turns("Hello"), max_tokens=BUDGET) as stream:
        deltas = list(stream.text_stream)
        final = stream.get_final_message()

    assert len(deltas) > 1, "the whole answer came in one frame; nothing was accumulated"
    assert "".join(deltas) == prompt
    assert only_text(final) == prompt
    assert final.role == "assistant"
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == len(prompt)
    assert final.usage.output_tokens == math.ceil(len(prompt) / CHUNK)


def test_the_stream_is_the_dialects_own_sequence_of_named_events(stand: Stand) -> None:
    """The shape the accumulator is more tolerant of than the dialect is: it would take a
    block that never closed, or a `message_stop` that never came. Every frame is named, and
    the name is also the `type` inside it — a mismatch is what makes an SDK construct the
    wrong member of the union."""
    captured = [(name, payload) for name, payload in frames(stand) if name != "ping"]
    names = [name for name, _ in captured]

    assert names[0] == "message_start"
    assert names[1] == "content_block_start"
    assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert set(names[2:-3]) == {"content_block_delta"}
    assert all(payload["type"] == name for name, payload in captured)

    opened = entry(captured[0][1]["message"])
    assert opened["content"] == [] and opened["stop_reason"] is None
    assert entry(opened["usage"])["input_tokens"] == len(rendered(("user", "Hello")))

    pieces = [text(entry(payload["delta"])["text"]) for _, payload in captured[2:-3]]
    assert "".join(pieces) == rendered(("user", "Hello"))
    assert entry(captured[-2][1]["delta"])["stop_reason"] == "end_turn"


def test_a_long_prefill_is_held_open_by_the_dialects_ping(stand: Stand) -> None:
    """`message_start` waits for the first piece, because that is when the prompt has been
    counted. Without the ping the connection would then go silent for the whole prefill,
    which is how a client loses it on a large prompt."""
    names = [name for name, _ in frames(stand, model=SLOW)]

    assert names[0] == "ping", "the stream said nothing until the model did"
    assert "message_start" in names


def test_an_unknown_model_is_the_dialects_own_not_found(client: anthropic.Anthropic) -> None:
    """The SDK picks the class by status; the body is what a client's error mapping reads,
    and it is this dialect's envelope rather than OpenAI's."""
    with pytest.raises(anthropic.NotFoundError) as raised:
        ask(client, "Hello", model="nope")

    kind, message = envelope(raised.value.body)
    assert kind == "not_found_error"
    assert "nope" in message


def test_a_model_without_a_chat_template_is_refused(client: anthropic.Anthropic) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go: a client
    error, not a 500 out of the worker."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        ask(client, "Hello", model=BASE)

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "chat template" in message


def test_a_field_the_dialect_cannot_honour_is_refused_by_name(
    stand: Stand, client: anthropic.Anthropic
) -> None:
    """`container` names a sandbox that runs somewhere else, and a client told it was honoured
    was told the answer came from a machine it did not. `max_tokens` is the other half —
    required here, so a body without one is refused instead of given a default this dialect
    never had. Both come back named, in the envelope."""
    response = httpx.post(
        stand.url,
        json={"model": ECHO, "messages": turns("Hello"), "max_tokens": 8, "container": "cont_1"},
        timeout=30,
    )
    assert response.status_code == 400
    kind, message = envelope(response.json())
    assert kind == "invalid_request_error"
    assert "container" in message

    response = httpx.post(stand.url, json={"model": ECHO, "messages": turns("Hello")}, timeout=30)
    assert response.status_code == 400
    kind, message = envelope(response.json())
    assert kind == "invalid_request_error"
    assert "max_tokens" in message


def test_a_generation_that_fails_is_told_in_both_shapes(
    stand: Stand, client: anthropic.Anthropic
) -> None:
    """Before the first frame the status is still ours; after it, it is 200 for ever and the
    only place left to say so is the dialect's own `error` event. A stream that ended in
    `message_stop` anyway would hand the client a truncated answer as a complete one."""
    with pytest.raises(anthropic.InternalServerError) as raised:
        ask(client, "Hello", model=FLAKY)
    kind, message = envelope(raised.value.body)
    assert kind == "api_error"
    assert "gave out" in message

    captured = frames(stand, model=FLAKY)
    names = [name for name, _ in captured]
    assert names[-1] == "error"
    assert "message_stop" not in names
    assert entry(captured[-1][1]["error"])["type"] == "api_error"

    with (
        pytest.raises(anthropic.APIStatusError) as broke,
        client.messages.stream(model=FLAKY, messages=turns("Hello"), max_tokens=BUDGET) as stream,
    ):
        stream.get_final_message()
    assert "gave out" in str(broke.value)


def test_the_models_route_lists_the_catalog_and_its_profiles(client: anthropic.Anthropic) -> None:
    """The catalog, not the residents — no dialect's schema carries the notion, and a list of
    loaded models is empty exactly at boot, which is when it is read. A model with a profile
    answers to two names and both are listed: a name a client cannot see is a preset it
    cannot select."""
    listed = list(client.models.list())

    assert [model.id for model in listed] == [CATALOGUED, f"{CATALOGUED}:code"]
    assert {model.type for model in listed} == {"model"}

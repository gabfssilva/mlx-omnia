"""The channels and the sequences: a stop sequence honoured over the characters the model
wrote, a system turn rendered where it sits, and reasoning leaving as a `thinking` block."""

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from mlx_omnia.server.api.anthropic import codec
from mlx_omnia.server.api.anthropic.models import MessagesRequest
from mlx_omnia.server.services.profiles import Sampling
from tests.server.anthropic_script import BUDGET, ECHO, rendered
from tests.server.anthropic_stand import (
    Stand,
    ask,
    body,
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


def post(stand: Stand, **body: object) -> httpx.Response:
    """A body the SDK has no parameter for, sent as it is. Two fields here are beta upstream
    and typed nowhere in the client — `context_management` is the one Claude Code sends on
    every request — and a test that reached them through `extra_body` would be asserting
    about the SDK's escape hatch instead of about the route."""
    asked: dict[str, object] = {
        "model": ECHO,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": BUDGET,
    }
    return httpx.post(stand.url, json=asked | body, timeout=30)


def test_a_stop_sequence_cuts_the_answer_and_comes_back_named(
    client: anthropic.Anthropic,
) -> None:
    """The engine's `stop` is a set of token ids and cannot express a string, so the sequence
    is honoured over the text: what the client reads ends where its sequence began, and the
    reason the turn ended says which one it was. `end_turn` beside a cut answer is a client
    told the model finished on its own."""
    reply = client.messages.create(
        model=ECHO, messages=turns("Hello"), max_tokens=BUDGET, stop_sequences=["llo"]
    )

    assert only_text(reply) == "<user>He"
    assert reply.stop_reason == "stop_sequence"
    assert reply.stop_sequence == "llo"

    ran_out = client.messages.create(
        model=ECHO, messages=turns("Hello"), max_tokens=BUDGET, stop_sequences=["nowhere"]
    )
    assert only_text(ran_out) == rendered(("user", "Hello"))
    assert ran_out.stop_reason == "end_turn"
    assert ran_out.stop_sequence is None


def test_a_sequence_split_across_two_pieces_is_still_one(stand: Stand) -> None:
    """`Hello` straddles the boundary between the first eight characters and the second, so
    a route matching piece by piece would never see it. What the client must also never see
    is the half that arrived first: the frames stop at `<user>`, and the `He` that could
    still have become the sequence is held until the piece that decides it does."""
    captured = frames(stand, stop_sequences=["Hello"])

    written = "".join(
        text(entry(payload["delta"])["text"])
        for named, payload in captured
        if named == "content_block_delta"
    )
    assert written == "<user>"
    ended = next(payload for named, payload in captured if named == "message_delta")
    assert entry(ended["delta"]) == {"stop_reason": "stop_sequence", "stop_sequence": "Hello"}


def test_a_system_turn_between_two_messages_is_rendered_where_it_sits(
    client: anthropic.Anthropic,
) -> None:
    """The other spelling of `system` in this dialect: the field opens the conversation and
    the role appends an instruction mid-way, which is what a client sends to steer a session
    without editing the prompt it has already cached. Both reach the template, and the order
    is the client's — an instruction hoisted to the front is a different conversation."""
    reply = client.messages.create(
        model=ECHO,
        max_tokens=BUDGET,
        system="You are terse.",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "system", "content": "Answer in French now."},
            {"role": "user", "content": "Again"},
        ],
    )

    assert only_text(reply) == rendered(
        ("system", "You are terse."),
        ("user", "Hello"),
        ("assistant", "Hi."),
        ("system", "Answer in French now."),
        ("user", "Again"),
    )


THINKS = "a<think>why</think>b"
"""What the echo is asked, and therefore what it answers: the reasoning marker is in the
prompt, so the segmenter inside the model cuts the answer into the two channels this dialect
has to tell apart. Nothing here scripts a channel by hand."""

THOUGHT = "<user>a"
"""The answer's text, first half — everything before the block. The second is `b</user>`."""


def test_reasoning_leaves_as_a_thinking_block_and_display_decides_its_text(
    client: anthropic.Anthropic,
) -> None:
    """Reasoning is a channel by the time it reaches this module, and this dialect has a block
    for it: text is what the model answered, and `<think>` drawn as the answer is the whole
    turn read wrong. `display` is about the client and not about the model — `omitted` leaves
    the block with its text empty, which is what a client that did not want to read the
    reasoning asked for, and not a turn that never reasoned.

    Three blocks and not two: the echo writes text, reasons, and writes again, and that is
    the order it comes back in. The stream can only hand blocks over as they arrive, so a
    message that gathered the reasoning into one block at the front would be this same
    generation read two different ways."""
    reply = ask(client, THINKS)

    assert [block.type for block in reply.content] == ["text", "thinking", "text"]
    opening, thinking, answer = reply.content
    assert opening.type == "text" and opening.text == THOUGHT
    assert thinking.type == "thinking" and thinking.thinking == "why"
    assert answer.type == "text" and answer.text == "b</user>"

    hidden = client.messages.create(
        model=ECHO,
        messages=turns(THINKS),
        max_tokens=BUDGET,
        thinking={"type": "adaptive", "display": "omitted"},
    )
    blocks = hidden.content
    assert [block.type for block in blocks] == ["text", "thinking", "text"]
    assert blocks[1].type == "thinking" and blocks[1].thinking == ""
    assert blocks[2].type == "text" and blocks[2].text == "b</user>"


def test_the_sdk_accumulates_the_thinking_block_out_of_the_named_events(
    client: anthropic.Anthropic,
) -> None:
    """The stream's side of the block above, judged by the SDK's own accumulator: a
    `thinking_delta` landing in a block that was opened as text is what it raises on, so the
    block turning with the channel is the whole of this."""
    with client.messages.stream(model=ECHO, messages=turns(THINKS), max_tokens=BUDGET) as stream:
        reply = stream.get_final_message()

    assert [block.type for block in reply.content] == ["text", "thinking", "text"]
    assert reply.content[0].type == "text" and reply.content[0].text == THOUGHT
    assert reply.content[1].type == "thinking" and reply.content[1].thinking == "why"
    assert reply.content[2].type == "text" and reply.content[2].text == "b</user>"


def test_a_thinking_block_replayed_by_the_client_is_read_and_left_out_of_the_prompt(
    client: anthropic.Anthropic,
) -> None:
    """The dialect asks a client to send the assistant's blocks back unchanged, so a
    conversation that used thinking is unreplayable if the block is refused — and no template
    in circulation renders a previous turn's reasoning, so it is dropped rather than written
    into the prompt as prose about a question already answered. The echo is what says which
    of the two happened."""
    reply = client.messages.create(
        model=ECHO,
        max_tokens=BUDGET,
        messages=[
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "deliberating", "signature": ""},
                    {"type": "text", "text": "Hi."},
                ],
            },
            {"role": "user", "content": "Again"},
        ],
    )

    assert only_text(reply) == rendered(("user", "Hello"), ("assistant", "Hi."), ("user", "Again"))


def test_thinking_says_what_the_template_is_asked_for() -> None:
    """This dialect names states and not levels, so the three modes reach the template as the
    switch and never as a rung. A request that named no `thinking` leaves it at `auto` rather
    than off: a checkpoint that reasons by design is not asked to stop because a client left
    the field out.

    Off HTTP because the only observable difference is the `Chat` the engine is handed — the
    template on this stand reads no kwarg, and one that did would be testing jinja."""

    def effort(request: MessagesRequest) -> str:
        return codec.to_conversation(request, None).reasoning_effort

    assert effort(body()) == "auto"
    assert effort(body(thinking={"type": "adaptive"})) == "on"
    assert effort(body(thinking={"type": "enabled"})) == "on"
    assert effort(body(thinking={"type": "disabled"})) == "off"


def test_a_profile_names_the_rung_this_dialect_cannot() -> None:
    """The effort lives whole on the profile, and a request that names no `thinking` falls to
    it. One that does throws the switch itself — a state the client named beats a level it
    never asked about."""
    assert codec.to_conversation(body(), None, "high").reasoning_effort == "high"
    named = codec.to_conversation(body(thinking={"type": "disabled"}), None, "high")
    assert named.reasoning_effort == "off"


def test_the_thinking_budget_is_the_ids_the_block_may_spend() -> None:
    """`budget_tokens` reaches the engine as the reasoning budget, and the profile fills the
    silence. With `disabled` it is refused rather than dropped: there is no block to bound."""
    preset = Sampling(reasoning_budget=64)
    asked = body(thinking={"type": "enabled", "budget_tokens": 512})
    assert codec.generation_options(asked, preset, None).reasoning_budget == 512
    assert codec.generation_options(body(), preset, None).reasoning_budget == 64
    assert codec.generation_options(body(), Sampling(), None).reasoning_budget is None
    with pytest.raises(ValidationError):
        body(thinking={"type": "disabled", "budget_tokens": 512})


def test_context_management_is_honoured_where_it_is_already_true_and_refused_where_it_is_not(
    stand: Stand,
) -> None:
    """`clear_thinking_20251015` asks for something this dialect does by construction — no
    thinking block ever enters a prompt here — so it is accepted and the conversation renders
    the same either way. `clear_tool_uses_20250919` is decided by token thresholds against a
    prompt that only exists inside the engine, so it is refused by name: dropping the edit
    would hand the model the conversation the client asked to have shortened."""
    cleared = post(stand, context_management={"edits": [{"type": "clear_thinking_20251015"}]})
    assert cleared.status_code == 200
    blocks = entry(cleared.json())["content"]
    assert isinstance(blocks, list), blocks
    assert text(entry(blocks[0])["text"]) == rendered(("user", "Hello"))

    refused = post(stand, context_management={"edits": [{"type": "clear_tool_uses_20250919"}]})
    assert refused.status_code == 400
    kind, message = envelope(refused.json())
    assert kind == "invalid_request_error"
    assert "context_management" in message

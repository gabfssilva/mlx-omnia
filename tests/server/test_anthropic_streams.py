"""One call, streamed: the input arrives in `input_json_delta` frames as it is written
rather than in one frame after the generation is over."""

import json

import anthropic
import httpx

from tests.server.anthropic_script import BIG, BUDGET, CALLER, ECHO, PATCH, TOOLS
from tests.server.anthropic_stand import (
    Stand,
    client,
    entry,
    frames,
    fresh_state,
    stand,
    text,
)

__all__ = ["client", "fresh_state", "stand"]
"""The fixtures live in the stand module and are imported for pytest to find them here."""


def test_the_input_of_one_call_arrives_in_more_than_one_delta(stand: Stand) -> None:
    """The reversal of A11 in the dialect Claude Code speaks: a four-kilobyte argument is
    handed over as it is written, in `input_json_delta` frames, instead of one frame after
    the generation is over.

    The SDK's accumulator concatenates `partial_json` and parses what it has at every step,
    so several fragments and one fragment both build the same block — what differs is when
    the client can start drawing it, which is what the count measures.
    """
    captured = frames(stand, model=BIG, tools=TOOLS)
    names = [name for name, _ in captured]
    assert "content_block_start" in names

    started = [
        payload
        for name, payload in captured
        if name == "content_block_start" and entry(payload["content_block"])["type"] == "tool_use"
    ]
    assert len(started) == 1, "one call, one block"
    block = entry(started[0]["content_block"])
    assert block["name"] == "apply_patch"
    assert block["input"] == {}, "the block opens empty and the deltas fill it"
    assert text(block["id"]).startswith("toolu_")

    fragments = [
        text(entry(payload["delta"])["partial_json"])
        for name, payload in captured
        if name == "content_block_delta" and entry(payload["delta"])["type"] == "input_json_delta"
    ]
    assert len(fragments) > 1, "the whole input arrived at once; nothing is being streamed"
    assert json.loads("".join(fragments)) == {"path": "a.py", "body": PATCH}

    # Opened, filled and closed, in that order and around the deltas.
    order = [name for name, _ in captured if name.startswith("content_block")]
    opened_at = order.index("content_block_start")
    assert "content_block_stop" in order[opened_at:]


def test_the_official_sdk_accumulates_the_streamed_call(client: anthropic.Anthropic) -> None:
    """Judged by the SDK that Claude Code uses, not by our reading of the frames."""
    with client.messages.stream(
        model=BIG,
        messages=[{"role": "user", "content": "patch it"}],
        max_tokens=BUDGET,
        tools=TOOLS,
    ) as stream:
        final = stream.get_final_message()
    used = [block for block in final.content if block.type == "tool_use"]
    assert len(used) == 1
    assert used[0].name == "apply_patch"
    assert used[0].input == {"path": "a.py", "body": PATCH}
    assert final.stop_reason == "tool_use"


def test_the_tool_choice_claude_code_sends_is_accepted(client: anthropic.Anthropic) -> None:
    """The field that made this dialect unreachable for the client it exists for.

    Claude Code puts `disable_parallel_tool_use` beside `{"type": "auto"}` on its first
    request, and `extra="forbid"` refused the whole body over it — so every session ended
    before the first token. `false` asks for what already happens (nothing here caps how many
    calls a generation writes), so it is accepted and ignored.
    """
    reply = client.messages.create(
        model=CALLER,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=BUDGET,
        tools=TOOLS,
        tool_choice={"type": "auto", "disable_parallel_tool_use": False},
    )
    assert [block.type for block in reply.content] == ["text", "tool_use"]
    assert reply.stop_reason == "tool_use"


def test_disabling_parallel_tool_use_is_refused_by_name(stand: Stand) -> None:
    """`true` is the other half and is not the same request: it asks for at most one call,
    and a generation that wrote two would be answered with something the client said it could
    not take. Refused with the field named, which is what tells a client to drop it."""
    refusal = httpx.post(
        stand.url,
        json={
            "model": CALLER,
            "messages": [{"role": "user", "content": "Weather?"}],
            "max_tokens": BUDGET,
            "tools": TOOLS,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        },
        timeout=30,
    )
    assert refusal.status_code == 400
    assert "disable_parallel_tool_use" in refusal.json()["error"]["message"]


def test_the_prompt_is_untouched_by_the_field_that_is_ignored(stand: Stand) -> None:
    """Accepted-and-ignored has to mean the prompt is the same characters.

    Read off the echo, which answers with the prompt it was handed — so the comparison is
    between two renders and not against a string written here, and it stays about the field
    rather than about the template."""
    body: dict[str, object] = {
        "model": ECHO,
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "max_tokens": BUDGET,
        "tools": TOOLS,
    }

    def echoed(sent: dict[str, object]) -> str:
        answer = httpx.post(stand.url, json=sent, timeout=30)
        assert answer.status_code == 200, answer.text
        blocks = entry(answer.json())["content"]
        assert isinstance(blocks, list)
        return "".join(text(entry(block)["text"]) for block in blocks)

    plain = echoed(body)
    with_field = body | {"tool_choice": {"type": "auto", "disable_parallel_tool_use": False}}
    assert echoed(with_field) == plain

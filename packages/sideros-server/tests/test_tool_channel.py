"""What the four dialects share of tool calling: `Calls`, which is now a reader of the
channel and no longer a second suppression machine.

The class used to hold a `Segmenter` of its own and match the family's markers again over
text that had already been through one. Two machines over one generation do not agree — the
lower one knows the channel the rendered prompt left open and whether the checkpoint's
template spells a family at all, and one built up here knows neither — so the marker-matching
is gone and what is left is `Segment.channel`.

The wire is unchanged, and that is the other half of what is pinned here: reasoning still
leaves as content, and a request whose generation writes no envelope gets back exactly the
text it always did.
"""

from sideros.suppress import Segment
from sideros.tools import ToolCall
from sideros.tools.families.qwen import FAMILY as QWEN
from sideros_server.responses import Calls

CALL = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'


def test_the_channel_decides_and_the_text_is_never_matched_again() -> None:
    """The whole of the change, as a property: an envelope the stream labelled content is
    content, character for character, however much it looks like a call.

    That is not a hypothetical. The streamer opens the tool channel only when the prompt's own
    template spells a family it can parse, and never inside a reasoning block; a machine here
    would answer `<tool_call>` on both, and hand the client a call the model wrote while
    thinking out loud — or one from a checkpoint whose spelling nothing can parse, which is
    text suppressed for a call that never arrives.
    """
    calls = Calls(QWEN)
    assert calls.push(Segment("content", CALL)) == CALL
    assert calls.finish() == ("", ())


def test_an_envelope_is_withheld_from_the_content_and_comes_back_as_a_call() -> None:
    """The tool channel's two halves: nothing of it reaches the client as text, and what it
    spells arrives whole at the end of the generation."""
    calls = Calls(QWEN)
    assert calls.push(Segment("content", "Sure. ")) == "Sure. "
    assert calls.push(Segment("tool", CALL)) == ""
    assert calls.push(Segment("content", "!")) == "!"
    assert calls.finish() == ("", (ToolCall("f", {"x": 1}),))


def test_an_envelope_that_spells_no_call_goes_back_into_the_content() -> None:
    """Held as a possible call and it is not one. Dropped, it reaches the client as a model
    that chose to call nothing — which is the shape of a correct refusal, and the client
    cannot tell the two apart afterwards."""
    broken = '<tool_call>{"arguments": {}}</tool_call>'
    calls = Calls(QWEN)
    assert calls.push(Segment("tool", broken)) == ""
    assert calls.finish() == (broken, ())


def test_all_the_calls_or_none_of_them() -> None:
    """A turn whose calls cannot be read whole is not a turn to half-execute: the readable
    one goes back into the content with the broken one instead of being executed alone."""
    broken = "<tool_call>not json</tool_call>"
    calls = Calls(QWEN)
    assert calls.push(Segment("tool", CALL)) == ""
    assert calls.push(Segment("tool", broken)) == ""
    assert calls.finish() == (CALL + broken, ())


def test_reasoning_leaves_as_content_where_it_always_did() -> None:
    """The channel arrives labelled and nothing here answers with it yet: telling reasoning
    from the answer is a field of the dialect — OpenAI's `reasoning_content` — and inventing
    one would change what every client already reads. Suppressed instead, it would be a turn
    that silently wrote less than it did."""
    calls = Calls(QWEN)
    assert calls.push(Segment("reasoning", "<think>weighing it.</think>")) == (
        "<think>weighing it.</think>"
    )
    assert calls.push(Segment("content", "Answer")) == "Answer"
    assert calls.finish() == ("", ())


def test_content_leaves_on_the_push_that_carried_it() -> None:
    """Nothing is held here any more, and that is latency the client gets back: what could
    still become a marker was already held one level down, and holding it a second time is
    the piece boundary of a response that never calls anything moving for no reason."""
    calls = Calls(QWEN)
    assert [calls.push(Segment("content", piece)) for piece in ("a <tool", "ing shop")] == [
        "a <tool",
        "ing shop",
    ]
    assert calls.finish() == ("", ())

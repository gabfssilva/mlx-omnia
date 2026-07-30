"""The envelope parsers, over text that is spelled the way the checkpoint's own template
writes a call — the fixture carries the real template sources, so the sniff is measured
against the four in circulation and not against an imitation of them.

The two failure modes pinned here are the silent ones: a broken payload inside a closed
envelope has to come out as an error (dropping it reads to the caller exactly like a
model that chose not to call anything), and an argument the tool's schema never declared
has to go through untouched.
"""

import json
from pathlib import Path

import pytest

from sideros.tools import HARMONY, QWEN, MalformedToolCall, ToolCall, ToolFamily, tool_family

FIXTURE = Path(__file__).parent / "fixtures" / "chat_template.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))


def template_source(repo: str) -> str:
    source = GOLDEN["repos"][repo]["template"]
    assert isinstance(source, str)
    return source


def qwen_envelope(name: str, arguments: str) -> str:
    """The shape the Qwen template itself writes when it replays a call."""
    return f'<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>'


HARMONY_CALL = (
    "<|channel|>analysis<|message|>The user wants the weather.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'
)


def test_qwen_reads_one_call_out_of_a_turn_that_also_has_prose() -> None:
    output = "Let me look that up.\n" + qwen_envelope("get_weather", '{"city": "Paris"}')
    assert QWEN.parse_tool_call(output) == (ToolCall("get_weather", {"city": "Paris"}),)


def test_qwen_reads_two_calls_in_the_order_they_were_written() -> None:
    output = "\n".join(
        [
            qwen_envelope("get_weather", '{"city": "Paris"}'),
            qwen_envelope("get_time", '{"zone": "Europe/Paris"}'),
        ]
    )
    assert QWEN.parse_tool_call(output) == (
        ToolCall("get_weather", {"city": "Paris"}),
        ToolCall("get_time", {"zone": "Europe/Paris"}),
    )


def test_a_turn_with_no_envelope_has_no_call() -> None:
    assert QWEN.parse_tool_call("It is sunny in Paris.") == ()
    assert HARMONY.parse_tool_call("<|channel|>final<|message|>It is sunny.<|return|>") == ()


def test_broken_json_inside_a_closed_envelope_is_reported() -> None:
    """The envelope is complete, so the model did mean to call something; what it wrote is
    not a call. Skipping it here would leave the caller reading a refusal."""
    with pytest.raises(MalformedToolCall, match="did not parse as JSON"):
        QWEN.parse_tool_call('<tool_call>\n{"name": "get_weather", </tool_call>')


def test_an_envelope_that_never_closes_is_reported() -> None:
    """Generation cut at max_tokens, mid-call: the same silent refusal if it were dropped."""
    with pytest.raises(MalformedToolCall, match="never closes"):
        QWEN.parse_tool_call('<tool_call>\n{"name": "get_weather", "arguments": {}}')


def test_an_envelope_that_carries_no_name_is_reported() -> None:
    with pytest.raises(MalformedToolCall, match="no name"):
        QWEN.parse_tool_call('<tool_call>\n{"arguments": {"city": "Paris"}}\n</tool_call>')


def test_a_payload_that_is_not_an_object_is_reported() -> None:
    with pytest.raises(MalformedToolCall, match="did not parse as a JSON object"):
        QWEN.parse_tool_call('<tool_call>\n["get_weather"]\n</tool_call>')


def test_an_argument_the_schema_never_declared_goes_through() -> None:
    """Validating the call against the tool's schema belongs to whoever defined the
    function: pruning here makes their bug look like ours, and a model calling with an
    argument nobody declared is a fact the caller has to see."""
    output = qwen_envelope("get_weather", '{"city": "Paris", "in_fahrenheit": true}')
    assert QWEN.parse_tool_call(output) == (
        ToolCall("get_weather", {"city": "Paris", "in_fahrenheit": True}),
    )


def test_harmony_reads_the_call_that_follows_the_analysis_channel() -> None:
    assert HARMONY.parse_tool_call(HARMONY_CALL) == (ToolCall("get_weather", {"city": "Paris"}),)


def test_harmony_reads_the_header_order_the_template_writes() -> None:
    """Generating, the recipient comes after the channel; replaying a history the template
    puts it in the role header instead. Both name the same call."""
    output = (
        "<|start|>assistant to=functions.get_weather"
        '<|channel|>commentary json<|message|>{"city": "Paris"}<|call|>'
    )
    assert HARMONY.parse_tool_call(output) == (ToolCall("get_weather", {"city": "Paris"}),)


def test_harmony_reads_two_calls_in_the_order_they_were_written() -> None:
    output = HARMONY_CALL + (
        "<|start|>assistant<|channel|>commentary to=functions.get_time "
        '<|constrain|>json<|message|>{"zone": "Europe/Paris"}<|call|>'
    )
    assert HARMONY.parse_tool_call(output) == (
        ToolCall("get_weather", {"city": "Paris"}),
        ToolCall("get_time", {"zone": "Europe/Paris"}),
    )


def test_harmony_ignores_a_commentary_message_that_addresses_nobody() -> None:
    """The commentary channel also carries the preamble the model writes before calling;
    without a recipient and without `<|call|>` it is not an invocation."""
    output = "<|channel|>commentary<|message|>I'll check the weather first.<|end|>"
    assert HARMONY.parse_tool_call(output) == ()


def test_harmony_ignores_a_recipient_that_is_not_a_declared_function() -> None:
    """Harmony's builtin namespaces (`browser`, `python`) are never rendered into our
    prompts, and there is nothing on the other side to route them to."""
    output = "<|channel|>commentary to=browser.search<|message|>{}<|call|>"
    assert HARMONY.parse_tool_call(output) == ()


def test_a_harmony_call_cut_before_the_call_token_is_reported() -> None:
    """The same cut as Qwen's unterminated envelope, in the other family's spelling: the
    recipient in the header is what opens it, so a generation that stopped mid-arguments is
    a call in half and not a turn that called nothing."""
    output = (
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city": "Par'
    )
    with pytest.raises(MalformedToolCall, match="never closes"):
        HARMONY.parse_tool_call(output)


def test_harmony_reports_broken_json_in_the_arguments() -> None:
    output = "<|channel|>commentary to=functions.get_weather <|message|>{'city': 'Paris'}<|call|>"
    with pytest.raises(MalformedToolCall, match="did not parse as JSON"):
        HARMONY.parse_tool_call(output)


FAMILIES = [
    ("mlx-community/Qwen3-0.6B-4bit", QWEN),
    ("openai/gpt-oss-20b", HARMONY),
    # Two more spellings, neither of them these two: LFM2.5 writes
    # `<|tool_call_start|>[fn(arg=1)]<|tool_call_end|>`, and Qwen3.6 keeps Qwen's marker
    # while filling it with `<function=...><parameter=...>` XML instead of JSON.
    ("LiquidAI/LFM2.5-8B-A1B", None),
    ("mlx-community/Qwen3.6-35B-A3B-6bit", None),
]


@pytest.mark.parametrize(("repo", "family"), FAMILIES, ids=[repo for repo, _ in FAMILIES])
def test_the_family_comes_off_the_checkpoints_own_template(
    repo: str, family: ToolFamily | None
) -> None:
    """Unknown is an answer: a family picked by resemblance parses an envelope into a call
    that was never made."""
    assert tool_family(template_source(repo)) is family


def test_a_template_that_never_writes_a_call_has_no_family() -> None:
    assert tool_family("{% for message in messages %}{{ message['content'] }}{% endfor %}") is None


def test_the_markers_are_the_ones_the_templates_write() -> None:
    """Pinned against the real sources: these strings are what the stream will hold on to,
    and a marker that drifts from the template stops matching without failing.

    Harmony's opener is the one that cannot be pinned that way, and deliberately: replaying
    a call the template writes the recipient into the role header, while a model generating
    one writes the channel first. The stream only ever sees the generated spelling — and it
    has to carry the recipient, because the commentary channel by itself is also where the
    preamble to the user goes."""
    qwen = template_source("mlx-community/Qwen3-0.6B-4bit")
    assert QWEN.start in qwen and QWEN.end in qwen
    assert HARMONY.end in template_source("openai/gpt-oss-20b")
    # Only the negative carries weight: that the opener is in `HARMONY_CALL` is one constant
    # of this module inside another, and can only fail if the test's own string is edited.
    assert HARMONY.start not in "<|channel|>commentary<|message|>I'll check first.<|end|>"

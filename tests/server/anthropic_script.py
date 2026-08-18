"""The script every `/api/anthropic/v1/*` suite runs against: ids, templates and pieces.

Nothing here is under test. It is what the doubles beside it answer with, and what the
assertions in the suites are written against.
"""

from collections.abc import Mapping

from anthropic.types import ToolParam

from mlx_omnia import ChatTemplate
from mlx_omnia.server.generation.consume import KEEP_ALIVE_SECONDS

ECHO = "echo"
FLAKY = "flaky"
SLOW = "slow"
BASE = "base"
"""A model with no chat template: the conversation has nowhere to go."""

CALLER = "caller"
BIG = "big"
MUTE = "mute"
STRANGER = "stranger"
"""Scripted callers: what a checkpoint writes when it is offered a function is the
checkpoint's own decision, so the generation is pinned and everything around it — the
template, the segmentation, the frames — stays real. `MUTE` writes the call and nothing
else, which is the answer that has to arrive with no text block at all; `STRANGER` writes the
same call behind a template that spells no envelope this server parses."""

GUIDED = "guided"
"""The one model here a grammar can be built over, by fiat in `Recording` below: nothing under
a stand of doubles holds a token table, and what a request carrying `output_config` has to
prove is that the route compiles the schema and hands the walk to the generation. It echoes
like `ECHO` does, so the same answer says which turns reached the prompt."""

COUNTED = "counted"
"""The one model here that holds a tokenizer, which is what `count_tokens` answers with: one
id per character (`Letters`), so the count a test expects is the length of the prompt the
template wrote and not a number read back off the thing under test."""

CATALOGUED = "vendor/tiny"
"""The one entry the fake hub cache holds, so the listing test is about what this file wrote
and not about what the machine happens to have downloaded."""

PRESET = "You are terse."
"""The profile's system prompt, which no request in this file ever sends."""

CACHED = "cached"
"""The echo, reporting a prefix reuse. What the dialect writes about a reuse is not what a
trie does to get one, and pinning the number is what keeps the two apart."""

REUSED = 7
"""Rows `CACHED` reports as covered. Smaller than the shortest render here, so
`input_tokens` never goes negative on the subtraction the dialect does."""

BUDGET = 999
"""Larger than any answer here, so a test that is not about the budget never reaches it."""

CHUNK = 8
"""Characters per piece the echo hands out, and therefore per meter mark: the answer arrives
in several frames, which is what gives the SDK's accumulator something to accumulate."""

SOURCE = (
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>{% endif %}"
    "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}"
    "{% for call in message.tool_calls %}"
    "<call>{{ call.function.name }}{{ call.function.arguments }}</call>"
    "{% endfor %}</{{ message['role'] }}>{% endfor %}"
)
"""One tag per turn with the role in it, and the tools and calls beside them. Nothing a real
checkpoint ships, and that is the point: what these tests read is which turns reached the
render, and in which order.

A call is spelled `<call>`, which is no family's marker, so `parser_of` says nothing about
this template and the tool channel stays shut. That is what the echo needs: it answers with the
prompt it was handed, and a prompt describing a call must not come back read as one.
"""

CALLING = SOURCE.replace("<call>", "<tool_call>").replace("</call>", "</tool_call>")
"""The same template spelling a call the way Qwen does, which is what a checkpoint's own
template does and what says which family it speaks. The callers below are loaded with this one
and the echo is not, and that is the whole difference between them."""

TEMPLATE = ChatTemplate.from_source(SOURCE)
CALLING_TEMPLATE = ChatTemplate.from_source(CALLING)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

DESCRIPTION = "Current weather in a city"

TOOLS: list[ToolParam] = [
    {"name": "get_weather", "description": DESCRIPTION, "input_schema": SCHEMA}
]

ENTRIES: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {"name": "get_weather", "description": DESCRIPTION, "parameters": SCHEMA},
    },
)
"""The same function in `chat/completions`'s nested shape, which is what every template reads
and what this dialect's `input_schema` has to become. Spelled out rather than converted: a
test that asked the route for the shape would agree with whatever the route did."""

CALL_PIECES = (
    "Let me check.",
    "<tool",
    '_call>\n{"name": "get_weather", "arguments": ',
    '{"city": "Paris"}}\n</tool',
    "_call>",
)
"""One generation, cut where a detokenizer would not: both markers straddle two pieces, so a
route that hands a piece out before it can tell hands out half a marker."""

PATCH = "x" * 4096
"""An argument the size an editing tool really carries — the case the old rule made invisible
until the generation ended."""

BIG_PIECES = (
    "Editing.",
    '<tool_call>\n{"name": "apply_patch", "arguments": {"path": "a.py"',
    f', "body": "{PATCH[:2048]}',
    f'{PATCH[2048:]}"',
    "}}\n</tool_call>",
)

PREAMBLE = CALL_PIECES[0]
ENVELOPE = "".join(CALL_PIECES[1:])
ARGUMENTS = '{"city": "Paris"}'
RESULT = "22 C"
ANSWERED = f"It is {RESULT}."

STALL = KEEP_ALIVE_SECONDS * 3
"""Three keep-alives before the first token: the slow model owes the stream a ping long
before it owes it anything else."""


def rendered(*turns: tuple[str, str]) -> str:
    """What the template writes for those turns, spelled out here rather than rendered: a
    test that asked the template what to expect would agree with itself whatever the dialect
    handed it."""
    return "".join(f"<{role}>{content}</{role}>" for role, content in turns)


ASKED = "Weather in Paris?"
"""What the caller is asked, in every suite that offers it a function."""

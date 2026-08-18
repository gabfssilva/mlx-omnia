"""What the Gemini suites call things: the checkpoint ids the stand answers to, the template
they render through, and the fixtures every one of the three files quotes.

Its own module because the doubles in `gemini_doubles.py` are written against these and the
stand in `gemini_stand.py` is written against both.
"""

from collections.abc import Mapping

MODEL = "stand/echo"
"""A `/` in the id, like every Hub repository: the name travels in the path, so the route has
to match a tail that carries slashes."""

BASE = "stand/base"
"""No chat template, like a base checkpoint: a conversation has nowhere to go."""

CACHED = "stand/cached"
"""The echo, reporting a prefix reuse. What the dialect writes about a reuse is not what a
trie does to get one, and pinning the number is what keeps the two apart."""

REUSED = 4
"""Rows `CACHED` reports as covered, smaller than the shortest render here."""

WRITER = "stand/json-writer"
PROSE = "stand/prose"
GUIDED = "stand/guided"
"""The three a structured answer is read off. `WRITER` writes what a model asked for JSON
writes anyway — a line of prose, a fence, and the document across two pieces; `PROSE` writes
no JSON at all; and `GUIDED` is the one model here a grammar can be built over, by fiat in
`Recording` below. Whether a checkpoint obeys a schema is the checkpoint's business, and what
these tests are about is what the route does with the answer either way."""

DOCUMENT: dict[str, object] = {"city": "Paris"}

WRITTEN = ("Sure! ", '```json\n{"city": ', '"Paris"}\n```')

CALLER = "stand/caller"
MUTE = "stand/mute"
STRANGER = "stand/stranger"
FLAKY = "stand/flaky"
"""Scripted callers: what a checkpoint writes when it is offered a function is the
checkpoint's own decision, so the generation is pinned and everything around it — the
template, the segmentation, the frames — stays real. `MUTE` writes the call and nothing else,
which is the answer that has to arrive with no text part at all; `STRANGER` writes the same
call behind a template that spells no envelope this server parses."""

PROFILE = "terse"
SYSTEM = "Be terse."

_TEMPLATE = (
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>\n"
    "{% endif %}"
    "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}\n"
    "{% for call in message.tool_calls %}"
    "<tool_call>{{ call.function.name }}{{ call.function.arguments }}\n"
    "{% endfor %}{% endfor %}"
)
"""One line per turn, the role first, and a line per call after the turn that made it. Not a
checkpoint's — what is being read back is the conversation the dialect built, and a real
template would spell it in special tokens. A call is spelled Qwen's way because that spelling
is also what says which family this checkpoint speaks: `parser_of` reads the template's
source, not the generated text, so a stand whose template spells no envelope has no tool
channel at all."""

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

DESCRIPTION = "Current weather in a city"

ENTRIES: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {"name": "get_weather", "description": DESCRIPTION, "parameters": SCHEMA},
    },
)
"""The same function in `chat/completions`'s nested shape, which is what every template reads
and what a `functionDeclaration` has to become. Spelled out rather than converted: a test that
asked the route for the shape would agree with whatever the route did."""

TOOL_BODY: list[dict[str, object]] = [
    {
        "functionDeclarations": [
            {"name": "get_weather", "description": DESCRIPTION, "parametersJsonSchema": SCHEMA}
        ]
    }
]
"""The same declaration as a body, for the tests that post one directly."""

CALL_PIECES = (
    "Let me check.",
    "<tool",
    '_call>\n{"name": "get_weather", "arguments": ',
    '{"city": "Paris"}}\n</tool',
    "_call>",
)
"""One generation, cut where a detokenizer would not: both markers straddle two pieces, so a
route that hands a piece out before it can tell hands out half a marker."""

PREAMBLE = CALL_PIECES[0]
ENVELOPE = "".join(CALL_PIECES[1:])
ARGUMENTS = '{"city": "Paris"}'
RESULT = "22 C"
ANSWERED = f"It is {RESULT}."

ASKED = "Weather in Paris?"

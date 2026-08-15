"""The call this daemon writes into a history, read back out of the characters the
checkpoint's own template wrote for it.

A template renders the assistant's own turn when it replays a conversation, so the envelope
it spells is the envelope the model emits. That makes the whole dialect layer testable
without a checkpoint loaded, without a GPU and without the network: render a history whose
assistant turn made a call, cut off what the render added, and hand exactly that to the
parser that claims the template.

Two different failures live on the two sides of that round trip and this file exists for
both.

**Render.** What goes into the template is what a dialect built. `arguments` handed over as
the JSON *text* of a mapping raises in eight of the fourteen families here, which is a 500
on the second turn of every tool loop; an id spelled in a shape a template validates is the
same failure one turn later. Neither is reachable from a test that only renders the first
turn of a conversation, which is why the histories below carry a call.

**Parse.** A dialect recognized off a marker it shares with another dialect reads an
envelope into a call that was never made — or, when the reader is strict enough to notice,
into no call at all, which reaches the client as a model that chose to answer in prose. The
second one is the quiet failure: nothing raises, nothing is logged, and the harness draws
the raw envelope as the answer.

A family whose template declares tools and whose calls do not survive this is a **failure
named by family**, never a skip: this file is the map of what is supported, and a skip is
how a hole stops being visible.
"""

import json
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

import pytest

from mlx_omnia.engine.chat import Chat, ChatMessage, ChatTemplate
from mlx_omnia.engine.parsers import Parser, Segmenter, ToolCall


class FamilyEntry(TypedDict):
    model_type: str
    repo: str
    template: str
    special_tokens: dict[str, str]


FIXTURE = Path(__file__).parent / "fixtures" / "tool_templates.json"
FAMILIES: list[FamilyEntry] = json.loads(FIXTURE.read_text(encoding="utf-8"))["families"]

CASES = [pytest.param(entry, id=entry["model_type"]) for entry in FAMILIES]

CALL_ID = "call_9aXbY2c1"
"""Nine alphanumerics because that is the tightest rule any template in circulation puts on
the field, and a shape that satisfies the tightest one satisfies the rest. The dialects mint
their own ids; what this pins is that the shape they mint travels back through a render."""

ARGUMENTS: dict[str, object] = {"city": "Rio", "days": 3}
"""One string and one integer, on purpose. A dialect that spells arguments as text —
`<arg_value>3</arg_value>`, `days=3` — has to give the number back as a number, and a round
trip carrying only strings cannot tell that apart from a reader that stringifies everything.
"""

EXPECTED = ToolCall("get_weather", ARGUMENTS)

TOOLS: tuple[dict[str, object], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city", "days"],
            },
        },
    },
)

ASKED: ChatMessage = {"role": "user", "content": "Weather in Rio for the next three days?"}

CALLED: ChatMessage = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": CALL_ID,
            "type": "function",
            "function": {"name": "get_weather", "arguments": ARGUMENTS},
        }
    ],
}

ANSWERED: ChatMessage = {"role": "tool", "content": "22 C, sunny", "tool_call_id": CALL_ID}


def template_of(entry: FamilyEntry) -> ChatTemplate:
    return ChatTemplate.from_source(entry["template"], entry["special_tokens"])


def rendered(template: ChatTemplate, *messages: ChatMessage) -> str:
    """The conversation as prompt, with no generation prompt on the end: what is being
    compared is the turns themselves, and the generation prompt is what comes after them."""
    return template.render(Chat(messages, tools=TOOLS), add_generation_prompt=False)


def written(template: ChatTemplate) -> str:
    """The characters the template added for the turn that made the call, and nothing else.

    Taken as the difference between the history with that turn and the history without it,
    because there is no other way to find it that does not encode one family's spelling: a
    marker searched for is a marker this file would have to know, and knowing them is the
    parsers' job.
    """
    before = rendered(template, ASKED)
    after = rendered(template, ASKED, CALLED)
    assert after.startswith(before), (
        "adding the assistant's call rewrote the turns before it, so the call cannot be cut "
        "out of the render by difference"
    )
    return after[len(before) :]


@pytest.mark.parametrize("entry", CASES)
def test_history_with_a_call_renders(entry: FamilyEntry) -> None:
    """The whole loop, rendered: the question, the call, and the result answering it.

    This is the turn a tool loop reaches second, and the one nothing rendered before this
    file existed. It carries `tool_call_id` as well as `tool_calls` — a template that reads
    the id back is one that can also refuse the shape of it.
    """
    rendered(template_of(entry), ASKED, CALLED, ANSWERED)


@pytest.mark.parametrize("entry", CASES)
def test_the_template_claims_a_dialect(entry: FamilyEntry) -> None:
    template = template_of(entry)
    parser = template.parser
    assert parser is not None, (
        f"no dialect claims {entry['model_type']} ({entry['repo']}): every call it writes "
        "reaches the client as text"
    )
    assert parser.tools is not None, (
        f"the dialect claiming {entry['model_type']} declares no tool family, so its "
        "envelopes are never read"
    )


@pytest.mark.parametrize("entry", CASES)
def test_the_template_replays_a_call(entry: FamilyEntry) -> None:
    """The turn that called something renders as something.

    Its own failure and not part of the round trip below, because the fix is somewhere else
    entirely: a template with no `tool_calls` branch drops the call however well the dialect
    reads it, and what the model is handed on the next turn is a result answering nothing it
    can see. No parser repairs that.
    """
    template = template_of(entry)
    parser = template.parser
    assert parser is not None and parser.tools is not None, entry["model_type"]
    assert envelopes_in(parser, written(template)), (
        f"{entry['model_type']} ({entry['repo']}): the template writes no call for a turn "
        f"that made one — it never reads `tool_calls` "
        f"({'tool_calls' in entry['template']}), so the assistant's own call is dropped from "
        "every history it renders"
    )


@pytest.mark.parametrize("entry", CASES)
def test_the_call_survives_the_round_trip(entry: FamilyEntry) -> None:
    """Render, cut, segment, parse — and the same call comes back.

    Segmented rather than handed to `parse_tool_call` whole, because the segmenter is what
    decides the tool channel in production: a reader that only ever sees text somebody else
    already cut is not the path the stream takes.
    """
    template = template_of(entry)
    parser = template.parser
    assert parser is not None and parser.tools is not None, entry["model_type"]
    envelopes = envelopes_in(parser, written(template))
    if not envelopes:
        pytest.skip("the template replays no call; test_the_template_replays_a_call owns it")
    made = tuple(
        call for envelope in envelopes for call in parser.tools.parse_tool_call(envelope)
    )
    assert made == (EXPECTED,), (
        f"{entry['model_type']} ({entry['repo']}): the template writes a call this dialect "
        f"does not read back — got {made!r}"
    )


@pytest.mark.parametrize("entry", CASES)
def test_the_call_is_written_once(entry: FamilyEntry) -> None:
    """Exactly one call comes out of a turn that made one.

    The families whose template has no `tool_calls` branch get the call written into their
    content instead (`ChatTemplate._messages`), and the failure that path invites is the
    opposite of the one it fixes: applied to a template that replays calls on its own, the
    model reads its own call twice and answers the second one. Which templates get it is
    decided by reading the source for the key, and this is what holds that decision.
    """
    template = template_of(entry)
    parser = template.parser
    assert parser is not None and parser.tools is not None, entry["model_type"]
    envelopes = envelopes_in(parser, written(template))
    assert len(envelopes) == 1, (
        f"{entry['model_type']} ({entry['repo']}): {len(envelopes)} envelopes for one call — "
        f"the template {'reads' if template.replays_calls else 'does not read'} `tool_calls`"
    )


def envelopes_in(parser: Parser, text: str) -> list[str]:
    segmenter = Segmenter(parser)
    segments = (*segmenter.push(text), *segmenter.flush())
    return [segment.text for segment in segments if segment.channel == "tool"]


def test_every_cached_family_is_in_the_fixture() -> None:
    """The fixture is a floor and this says so out loud: it holds the families whose
    checkpoints were in the cache of whoever regenerated it, and a family nobody has
    downloaded is not a family this suite covers."""
    assert len(FAMILIES) >= 14, (
        "the fixture shrank — regenerating it on a machine with fewer checkpoints cached "
        "drops families silently, and what is dropped is coverage"
    )


def loop(template: ChatTemplate, rounds: int) -> list[str]:
    """A tool loop as it actually grows: ask, call, result, ask again. One rendered prompt per
    turn, each with a generation prompt on the end, which is what the engine is handed."""
    history: list[ChatMessage] = [ASKED]
    prompts: list[str] = []
    for round in range(rounds):
        prompts.append(template.render(Chat(tuple(history), tools=TOOLS)))
        call: ChatMessage = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"{CALL_ID}{round}",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ARGUMENTS},
                }
            ],
        }
        history.append(call)
        history.append({"role": "tool", "content": "22 C", "tool_call_id": f"{CALL_ID}{round}"})
        history.append({"role": "user", "content": f"And the day after, take {round}?"})
    return prompts


@pytest.mark.parametrize("entry", CASES)
def test_a_tool_loop_only_ever_appends_to_its_prompt(entry: FamilyEntry) -> None:
    """What the prefix cache needs, decided without a checkpoint: turn N's prompt is a prefix
    of turn N+1's.

    A tool loop is the longest-running conversation this daemon serves — a harness sends the
    whole history back every turn — so a template that rewrites anything before the new turns
    costs a full prefill per round, and the loop gets slower the longer it runs. Nothing in
    the answer shows it; only `cached_tokens` does, and by then it is a benchmark and not a
    test.

    Three things break it and each is a real template: a date written above the history
    (`strftime_now`), the call id rendered into the prompt when the id is not stable across
    turns, and a `tools` block re-serialized in a different key order. The ids here differ
    per round on purpose — a template that renders them is fine, because the *earlier* rounds
    keep theirs.
    """
    template = template_of(entry)
    prompts = loop(template, rounds=4)
    for before, after in pairwise(prompts):
        head = before
        while head and not after.startswith(head):
            head = head[:-1]
        reused = len(head) / len(before)
        assert reused > 0.95, (
            f"{entry['model_type']} ({entry['repo']}): only {reused:.0%} of the previous "
            "prompt survives into the next turn, so every round of a tool loop re-prefills "
            "what the round before it already paid for"
        )

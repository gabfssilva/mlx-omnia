# The fullwidth bars in the DSML literals below are DeepSeek's own token characters, not
# the ASCII pipe; normalizing them would test a grammar no checkpoint writes.
# ruff: noqa: RUF001
"""What each family does with a call that is not the easy one.

The round-trip in `test_tool_round_trip.py` proves a dialect reads back what its own
template writes, which is the property that matters and is not the whole of it: the template
writes one shape of call, and the model writes whatever it writes. What is pinned here is
the rest — the types the format leaves implicit, several calls in one envelope, a value with
the separator inside it, and an envelope the generation cut in half.

Two properties run over **every** family rather than over one, because they are contracts of
`CallDelta` and not of any dialect:

- writing a call and reading it back gives the same call, which is the only test the readers
  have that does not agree with a literal somebody typed next to them;
- the `arguments` fragments of one index concatenate into the JSON of that call's arguments,
  whatever size the pieces arrive in — which is what the three streaming dialects put on the
  wire, and it has to hold when the pieces are one character each.
"""

import json

import pytest

from mlx_omnia.engine.parsers import (
    CallDelta,
    MalformedToolCall,
    Parser,
    ToolCall,
    ToolFamily,
    arg_key,
    atem,
    dsml,
    harmony,
    python_call,
    qwen,
    qwen_xml,
)


def family_of(parser: Parser) -> ToolFamily:
    assert parser.tools is not None
    return parser.tools


FAMILIES = [
    pytest.param(family_of(parser), id=name)
    for name, parser in [
        ("qwen", qwen.PARSER),
        ("qwen_xml", qwen_xml.PARSER),
        ("arg_key", arg_key.PARSER),
        ("dsml", dsml.PARSER),
        ("python_call", python_call.PARSER),
        ("harmony", harmony.PARSER),
        ("atem", atem.PARSER),
    ]
]

MIXED = ToolCall(
    "edit_file",
    {"path": "src/a.py", "line": 42, "dry_run": True, "tags": ["x", "y"], "note": "a, b"},
)
"""One of every JSON type a tool schema can declare, plus a string carrying the character
that separates arguments in two of these formats. A family that stringifies everything, or
that splits on the separator without watching for quotes, fails on this and passes on a call
made only of short strings."""


def deltas(family: ToolFamily, text: str, size: int) -> tuple[CallDelta, ...]:
    reader = family.reader()
    out: list[CallDelta] = []
    for at in range(0, len(text), size):
        out.extend(reader.push(text[at : at + size]))
    out.extend(reader.finish())
    return tuple(out)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_written_call_reads_back(family: ToolFamily) -> None:
    """`write` is the reader's inverse and this is what says so. Every type survives: a
    reader that hands back `"42"` for `42` breaks a tool whose schema says integer, and
    nothing else here would notice."""
    assert family.parse_tool_call(family.write(MIXED)) == (MIXED,)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("size", [1, 3, 7, 10_000])
def test_the_argument_fragments_concatenate_into_json(family: ToolFamily, size: int) -> None:
    """`CallDelta.arguments` is JSON in pieces, and the pieces are whatever the detokenizer
    handed over — one character included. What a dialect puts on the wire is the
    concatenation, so the concatenation is what has to parse."""
    written = deltas(family, family.write(MIXED), size)
    assembled = "".join(delta.arguments for delta in written)
    assert json.loads(assembled) == dict(MIXED.arguments)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_name_arrives_once(family: ToolFamily) -> None:
    """A client told a call is named cannot be told otherwise: the name is on the first delta
    of an index and on no other. It is what lets a stream open a block with the right name
    before any argument has arrived."""
    written = deltas(family, family.write(MIXED), 1)
    named = [delta for delta in written if delta.name is not None]
    assert len(named) == 1
    assert named[0].name == MIXED.name


@pytest.mark.parametrize("family", FAMILIES)
def test_an_envelope_cut_in_half_is_not_a_call(family: ToolFamily) -> None:
    """The generation stopping inside an envelope is reported, never dropped. A dropped
    envelope reaches the caller as a model that chose to call nothing, which is exactly the
    shape of a correct refusal."""
    whole = family.write(MIXED)
    with pytest.raises(MalformedToolCall):
        family.parse_tool_call(whole[: len(whole) // 2])


ARG_KEY_CALL = (
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key>\n<arg_value>Rio</arg_value>\n"
    "<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n"
    "</tool_call>"
)


def test_arg_key_types_a_bare_number_as_a_number() -> None:
    """The format writes every value as bare text, so the type is only in the characters.
    `3` has to come back as the number, or every tool with an integer parameter refuses the
    call it was handed."""
    (call,) = family_of(arg_key.PARSER).parse_tool_call(ARG_KEY_CALL)
    assert call == ToolCall("get_weather", {"city": "Rio", "days": 3})


def test_arg_key_reads_the_whitespace_of_both_checkpoints() -> None:
    """Ling puts a newline after each element and Laguna runs them together. The reader scans
    for the elements instead of splitting on lines, and this is what says it must keep
    doing that."""
    family = family_of(arg_key.PARSER)
    tight = ARG_KEY_CALL.replace("\n", "")
    assert family.parse_tool_call(tight) == family.parse_tool_call(ARG_KEY_CALL)


def test_arg_key_takes_the_whole_body_as_the_name_when_there_are_no_arguments() -> None:
    (call,) = family_of(arg_key.PARSER).parse_tool_call("<tool_call>ping</tool_call>")
    assert call == ToolCall("ping", {})


DSML_TWO = (
    "<｜DSML｜tool_calls>\n"
    '<｜DSML｜invoke name="get_weather">\n'
    '<｜DSML｜parameter name="city" string="true">Rio</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    '<｜DSML｜invoke name="get_time">\n'
    '<｜DSML｜parameter name="offset" string="false">-3</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>"
)


def test_dsml_reads_two_invocations_from_one_envelope() -> None:
    """The block is not the call — it holds a list of them — so the envelope the segmenter
    suppressed spells two, in the order written."""
    assert family_of(dsml.PARSER).parse_tool_call(DSML_TWO) == (
        ToolCall("get_weather", {"city": "Rio"}),
        ToolCall("get_time", {"offset": -3}),
    )


def test_dsml_obeys_the_type_flag_over_the_characters() -> None:
    """The one family that is told the type instead of inferring it. A city named `3` marked
    `string="true"` stays the string `3`, which is the case every other element format here
    gets wrong and cannot get right."""
    marked = DSML_TWO.replace(
        '<｜DSML｜parameter name="city" string="true">Rio</｜DSML｜parameter>',
        '<｜DSML｜parameter name="city" string="true">3</｜DSML｜parameter>',
    )
    first, _ = family_of(dsml.PARSER).parse_tool_call(marked)
    assert first == ToolCall("get_weather", {"city": "3"})


def test_python_call_keeps_a_separator_that_is_inside_a_value() -> None:
    """`split(",")` reads four arguments where the model wrote two. The comma inside the
    string and the one inside the list are not separators, and only bracket depth and string
    state tell them apart."""
    text = "<|tool_call_start|>[f(a='x, y', b=[1, 2])]<|tool_call_end|>"
    assert family_of(python_call.PARSER).parse_tool_call(text) == (
        ToolCall("f", {"a": "x, y", "b": [1, 2]}),
    )


def test_python_call_reads_several_calls_from_one_list() -> None:
    text = "<|tool_call_start|>[a(x=1), b(y='p')]<|tool_call_end|>"
    assert family_of(python_call.PARSER).parse_tool_call(text) == (
        ToolCall("a", {"x": 1}),
        ToolCall("b", {"y": "p"}),
    )


def test_python_call_refuses_an_argument_that_is_not_a_literal() -> None:
    """`literal_eval` and not `eval`: an argument that is a call raises here instead of
    running, and the envelope goes back as content."""
    text = "<|tool_call_start|>[f(a=open('secrets'))]<|tool_call_end|>"
    with pytest.raises(MalformedToolCall):
        family_of(python_call.PARSER).parse_tool_call(text)


def test_python_call_spells_python_and_not_json() -> None:
    """The written form is Python source, so `True` is not `true` and a string is quoted the
    way Python quotes one. Reading it back through `json.loads` would fail, which is why the
    reader does not."""
    written = family_of(python_call.PARSER).write(ToolCall("f", {"flag": True, "name": "Rio"}))
    assert "True" in written
    assert "true" not in written


WEATHER: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
CLOCK: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"zone": {"type": "string"}},
            "required": ["zone"],
        },
    },
}


def envelope_schema(source: str) -> dict[str, object]:
    """The JSON schema out of the Lark envelope the family built."""
    body = source.split("%json ", 1)[1]
    parsed = json.loads(body)
    assert isinstance(parsed, dict)
    return parsed


def test_the_forced_grammar_ties_each_name_to_its_own_arguments() -> None:
    """One `anyOf` branch per tool, and the branch is the whole point.

    A grammar that pinned the name to *any* offered name and the arguments to *any* offered
    schema would still be a grammar, would still force a call, and would guarantee a
    well-formed call to the wrong function with the other one's arguments. Asserted over the
    grammar rather than over a generation, because a model asked to call something picks a
    coherent pair on its own — the looser grammar passes that test and fails this one.
    """
    family = family_of(qwen.PARSER)
    assert family.grammar is not None
    branches = envelope_schema(family.grammar([WEATHER, CLOCK], json.dumps))["anyOf"]
    assert isinstance(branches, list) and len(branches) == 2
    paired = {
        branch["properties"]["name"]["const"]: sorted(
            branch["properties"]["arguments"]["properties"]
        )
        for branch in branches
    }
    assert paired == {"get_weather": ["city"], "get_time": ["zone"]}


def test_the_forced_grammar_spells_the_marker_the_way_the_vocabulary_does() -> None:
    """The marker is whatever the checkpoint's own table makes it. `<tool_call>` is a single
    added id in Qwen's, and a grammar asking for its eleven bytes stalls on the first token —
    so the family asks how to spell it instead of writing it."""
    family = family_of(qwen.PARSER)
    assert family.grammar is not None
    as_token = family.grammar([WEATHER], lambda text: f"<[{len(text)}]>")
    assert as_token.startswith("start: <[11]> body <[12]>")
    as_bytes = family.grammar([WEATHER], json.dumps)
    assert as_bytes.startswith('start: "<tool_call>" body "</tool_call>"')


def test_only_a_family_whose_body_is_json_declares_a_grammar() -> None:
    """The rule that keeps a guarantee a guarantee. `<arg_value>3</arg_value>` and `days=3`
    are arguments in a syntax the compiler does not know, so those families declare no
    grammar and a request that forces a call is refused with the checkpoint named — rather
    than constrained by something that does not describe what the model writes."""
    declared = {
        name: family_of(parser).grammar is not None
        for name, parser in [
            ("qwen", qwen.PARSER),
            ("qwen_xml", qwen_xml.PARSER),
            ("arg_key", arg_key.PARSER),
            ("dsml", dsml.PARSER),
            ("python_call", python_call.PARSER),
            ("harmony", harmony.PARSER),
            ("atem", atem.PARSER),
        ]
    }
    assert declared["qwen"] is True
    assert not any(value for name, value in declared.items() if name != "qwen"), (
        "a family whose envelope is not a marker around a JSON document cannot promise a "
        f"decode that produces one: {declared}"
    )

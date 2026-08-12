# The fullwidth bars throughout this module are DeepSeek's own token characters, not the
# ASCII pipe. Normalizing them would make every recognizer and marker here match nothing,
# so the ambiguous-character rules are off for the file rather than per line.
# ruff: noqa: RUF001, RUF002
"""DeepSeek V4's markup: one block holding any number of invocations.

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Rio</｜DSML｜parameter>
    <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Two things separate it from the other element-shaped families.

The block is not the call: it holds a **list** of `invoke` elements, so one envelope spells
several calls and `CallDelta.index` moves inside it. The envelope is what the segmenter
suppresses, and the calls are what comes out.

The type is written down. `string="true"` means the characters are the value; anything else
means they are JSON — the template says so in the prompt it builds (*"String parameters
should be specified as is and set `string="true"`. For all other types (numbers, booleans,
arrays, objects), pass the value in JSON format"*) and writes them that way when it replays
a turn. So this family never guesses the way `envelope.encoded` has to: a city named `3`
survives here, because the model was told to mark it.

The markers carry `｜DSML｜` in full-width bars, and the template builds them by
concatenation (`'<' + dsml_token + 'invoke name="'`), which is why the recognizer looks for
the assembled marker in the source rather than for a literal that never appears in it.
"""

import json
from collections.abc import Iterator

from mlx_omnia.engine.parsers.envelope import Body, EnvelopeScanner
from mlx_omnia.engine.parsers.protocol import CallDelta, Parser, ToolCall, ToolFamily
from mlx_omnia.engine.parsers.qwen import REASONING

__all__ = ["PARSER", "DsmlReader"]

_DSML = "｜DSML｜"

_START = f"<{_DSML}tool_calls>"
_END = f"</{_DSML}tool_calls>"

_INVOKE = f"<{_DSML}invoke"
_INVOKE_END = f"</{_DSML}invoke>"
_PARAMETER = f"<{_DSML}parameter"
_PARAMETER_END = f"</{_DSML}parameter>"

_RECOGNIZES = _DSML
"""The one piece of the markup that survives into the template *source*.

Every marker this family writes is assembled at render time (`'<' + dsml_token + 'invoke
name="'`), so none of them appears whole in the source and a recognizer written against one
claims nothing — which is what the first version of this module did. What is written
literally is the token the rest is built from, in one `set` at the top of the template.

It is enough on its own: `｜DSML｜` is in no other template in circulation, and the bars are
full-width characters nobody types by accident."""


def _attribute(head: str, name: str) -> str | None:
    """The value of `name="..."` in an element's head, or None when it is not there."""
    at = head.find(f'{name}="')
    if at < 0:
        return None
    rest = head[at + len(name) + 2 :]
    end = rest.find('"')
    return None if end < 0 else rest[:end]


def _encoded(head: str, value: str) -> str:
    """The JSON text of one parameter, read under the flag its own element carries.

    `string="true"` is the characters as they stand; anything else is already JSON. A value
    flagged as JSON that does not parse is written out as a string rather than dropped — the
    element said what it meant, the model wrote it wrong, and a call cut down to fewer
    arguments is worse than one carrying the text the model produced.
    """
    if _attribute(head, "string") == "true":
        return json.dumps(value)
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return json.dumps(value)
    return value


class DsmlReader:
    def __init__(self) -> None:
        self._scan = EnvelopeScanner(_START, _END)
        self._envelope = -1
        self._held = ""
        self._index = -1
        self._open = False
        self._written = 0

    def push(self, text: str) -> tuple[CallDelta, ...]:
        return self._read(self._scan.push(text))

    def finish(self) -> tuple[CallDelta, ...]:
        out = self._read(self._scan.finish())
        if not self._scan.inside:
            return out
        return (*out, CallDelta(max(self._index, 0), None, "", False))

    def _read(self, bodies: tuple[Body, ...]) -> tuple[CallDelta, ...]:
        out: list[CallDelta] = []
        for body in bodies:
            if body.index != self._envelope:
                self._envelope, self._held = body.index, ""
                self._open, self._written = False, 0
            self._held += body.text
            out.extend(self._drain())
            if body.closed:
                # The envelope closing with an invocation still open is a block the
                # generation cut between two calls: the one in flight never closes, and
                # `_assemble` reports it rather than handing over half a call.
                if self._open:
                    out.append(CallDelta(self._index, None, "", False))
                    self._open = False
                elif self._index < 0:
                    # A block that held no invocation at all. Somebody has to hear that the
                    # envelope existed, or the whole-output path sees a turn that called
                    # nothing — which is the shape of a correct refusal.
                    self._index += 1
                    out.append(CallDelta(self._index, None, "", False))
                self._held = ""
        return tuple(out)

    def _drain(self) -> Iterator[CallDelta]:
        while True:
            if not self._open:
                head = self._element(_INVOKE)
                if head is None:
                    return
                name = _attribute(head, "name")
                if name is None:
                    return
                self._index += 1
                self._open, self._written = True, 0
                yield CallDelta(self._index, name, "{", False)
                continue
            closing = self._held.find(_INVOKE_END)
            head = self._element(_PARAMETER, before=closing)
            if head is None:
                if closing < 0:
                    return
                self._held = self._held[closing + len(_INVOKE_END) :]
                self._open = False
                yield CallDelta(self._index, None, "}", True)
                continue
            value = self._until(_PARAMETER_END)
            if value is None:
                # The head is consumed and the value is not here yet; putting the head back
                # is what keeps the next push from reading a parameter with no name.
                self._held = f"{_PARAMETER}{head}>{self._held}"
                return
            separator = ", " if self._written else ""
            self._written += 1
            key = _attribute(head, "name") or ""
            yield CallDelta(
                self._index,
                None,
                f"{separator}{json.dumps(key)}: {_encoded(head, value)}",
                False,
            )

    def _element(self, opener: str, before: int = -1) -> str | None:
        """The head of the next `opener` element — everything between the marker and the `>`
        that closes it — consumed. `before` bounds the search: a parameter found after the
        invocation's closer belongs to the next invocation, not this one."""
        at = self._held.find(opener)
        if at < 0 or (before >= 0 and at > before):
            return None
        rest = self._held[at + len(opener) :]
        end = rest.find(">")
        if end < 0:
            return None
        self._held = rest[end + 1 :]
        return rest[:end]

    def _until(self, closer: str) -> str | None:
        at = self._held.find(closer)
        if at < 0:
            return None
        value, self._held = self._held[:at], self._held[at + len(closer) :]
        return value


def write(call: ToolCall) -> str:
    """One invocation in a block of its own, each parameter carrying the `string` flag that
    says how to read it — the flag is why this family never has to guess a type."""
    parameters = "".join(
        f'\n<{_DSML}parameter name="{key}" '
        f'string="{"true" if isinstance(value, str) else "false"}">'
        f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
        f"</{_DSML}parameter>"
        for key, value in call.arguments.items()
    )
    return f'{_START}\n<{_DSML}invoke name="{call.name}">{parameters}\n{_INVOKE_END}\n{_END}'


PARSER = Parser(
    recognizes=lambda source: _RECOGNIZES in source,
    reasoning=(REASONING,),
    tools=ToolFamily(start=_START, end=_END, reader=DsmlReader, write=write),
)

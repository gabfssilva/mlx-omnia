"""The chat block: sessions on the left, the thread in the middle, the composer under it.

The numbers under a turn are the daemon's — off the usage frame that closes a stream, or
off the live register while it is still being written — and never timed from out here,
where the load, the queue and the prefill are one wait with no way to tell them apart.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace

import flet as ft
import flet.canvas as cv

from mlx_omnia_application.api import conversation as api
from mlx_omnia_application.api.conversation import (
    EFFORTS,
    Message,
    Metrics,
    Params,
    Session,
    Summary,
)
from mlx_omnia_application.api.engine import Engine, Sample
from mlx_omnia_application.ui import parts, theme
from mlx_omnia_application.ui.forms import area
from mlx_omnia_application.ui.hooks import act
from mlx_omnia_application.ui.theme import t

TITLE_MAX = 48
SIDEBAR = 216.0
COLUMN = 640.0
# 30 a second is what the eye reads; a token a frame is a rebuilt window a frame.
PAINT = 1 / 30
# The clock the reader is left with before the register knows anything.
TICK = 0.25

# The stylesheet's own highlighting palette, which is neither of the two flutter_markdown
# ships with. hljs classes map one to one onto the slots below.
SYN_LIGHT = {"key": "#7B2FA0", "str": "#1B6B3A", "num": "#9A4E14",
             "fn": "#22599F", "com": "#848B8E", "attr": "#B2542C"}
SYN_DARK = {"key": "#C792EA", "str": "#7EE787", "num": "#FFC53D",
            "fn": "#79B8FF", "com": "#6B6E76", "attr": "#FF8A5C"}


def short(identifier: str) -> str:
    """The name without the repository that owns it: the sidebar has 216px and the org is
    the same for every row."""
    return identifier[identifier.rfind("/") + 1 :]


def ago(seconds: float) -> str:
    delta = time.time() - seconds
    if delta < 90:
        return "now"
    if delta < 3600:
        return f"{round(delta / 60)} min"
    if delta < 86400:
        return f"{round(delta / 3600)} h"
    if delta < 2 * 86400:
        return "yesterday"
    return time.strftime("%b %-d", time.localtime(seconds))


def _duration(ms: float) -> str:
    """Seconds under a minute of them, milliseconds under a second: the load of a cold 30B
    and the ttft of a warm one are two orders of magnitude apart and share this line."""
    if ms < 1000:
        return f"{round(ms)} ms"
    return f"{ms / 1000:.2f} s" if ms < 10_000 else f"{ms / 1000:.1f} s"


def _rate(value: float) -> str:
    return f"{value / 1000:.1f}k tok/s" if value >= 1000 else f"{value:.1f} tok/s"


def _decimal(value: float) -> str:
    """One decimal place when one is what the number has: 0.6 and not 0.60."""
    rounded = round(value, 2)
    return str(int(rounded)) if float(rounded).is_integer() else str(rounded)


def _thousands(count: int) -> str:
    return f"{count:,}"


def _title_of(text: str) -> str:
    line = " ".join(text.split())
    return f"{line[:TITLE_MAX].rstrip()}…" if len(line) > TITLE_MAX else line


def _context(tokens: int) -> str:
    return f"{round(tokens / 1000)}k" if tokens >= 1000 else str(tokens)


def _named(label: str, control: ft.Control) -> ft.Control:
    """.param — the same row the sliders wear over them, which is a sentence and not
    an eyebrow: the uppercase mono is for the two-word labels in the forms."""
    return ft.Column(
        [ft.Text(label, style=theme.sans(11, t().fg2), no_wrap=True), control],
        spacing=5,
        tight=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )


@ft.component
def Chat(engine: Engine) -> ft.Control:
    room, _ = ft.use_state(Room)
    return room.tree(engine)


@ft.observable
class Room:
    """What this screen holds: the sessions, the turn being written, and how it is
    sampled."""

    def __init__(self) -> None:
        self.summaries: list[Summary] = []
        self.current: str | None = None
        self.messages: list[Message] = []
        # Per chat, and only for as long as the window is open: the sessions table keeps
        # the conversation, not how it was sampled.
        self.knobs: dict[str, Params] = {}
        self.models: list[str] = []
        self.model = ""
        self.draft = ""
        self.live: tuple[str, str, str | None] | None = None
        self.notice: str | None = None
        self.spent: int | None = None
        self.tab = "chats"
        self.renaming: str | None = None
        self.rename_to = ""
        # The daemon's register as it stands for the turn being written, kept where a
        # generation can read it: what the turn is written with when the stream ends
        # without the frame that carries its totals.
        self._pulse: Sample | None = None
        self.waiting: float | None = None
        self.started = 0.0
        self._run: asyncio.Task[None] | None = None
        self._clock: asyncio.Task[None] | None = None
        self._booted = False
        self._loaded: str | None = None
        # One cell per slider, because the track's width is only known once it is laid
        # out and this pane is rebuilt on every frame.
        self._rails = [[1.0] for _ in range(6)]
        self._opened: set[int] = set()
        self._entry: object = None

    # ── the machine's side ────────────────────────────────────────────────

    def boot(self) -> None:
        if self._booted:
            return
        self._booted = True
        act(self._read())

    async def _read(self) -> None:
        try:
            self.summaries = await api.list_sessions()
            if self.summaries:
                self._open(self.summaries[0].id)
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
        try:
            self.models = await api.served_models()
        except Exception:  # noqa: BLE001 — a daemon with no catalog serves nothing
            self.models = []
        self.notify()

    def _open(self, identifier: str | None) -> None:
        self.current = identifier
        if identifier is not None and self._loaded != identifier:
            act(self._fetch(identifier))

    async def _fetch(self, identifier: str) -> None:
        try:
            session = await api.get_session(identifier)
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
            self.notify()
            return
        if self.current != identifier:
            return
        self._loaded = session.id
        self.messages = session.messages
        self.model = session.model
        self.spent = None
        self.notify()

    def _merge(self, session: Session) -> None:
        """The list is ordered by `updated_at` desc on the server; a write that bumps it
        has to move the row here too, or the order disagrees with the next reload."""
        rest = [row for row in self.summaries if row.id != session.id]
        self.summaries = sorted([session.summary, *rest], key=lambda row: -row.updated_at)

    # ── a turn ────────────────────────────────────────────────────────────

    def _send(self) -> None:
        text = self.draft.strip()
        if text == "" or self.live is not None or self.model == "":
            return
        self.notice = None
        self.draft = ""
        act(self._begin(text))

    async def _begin(self, text: str) -> None:
        try:
            identifier = self.current
            if identifier is None:
                created = await api.create_session(_title_of(text), self.model)
                identifier = created.id
                self._loaded = identifier
                self._merge(created)
                self.current = identifier
            elif not self.messages:
                # The title is the first thing said, truncated — the row was created
                # before anyone had typed.
                self._merge(await api.patch_session(identifier, _title_of(text), self.model))
            history = [*self.messages, Message("user", text)]
            self.messages = history
            self.notify()
            # Its own task: stopping a turn is cancelling the read of its stream, and
            # there is no handle on a coroutine the window merely awaits.
            self._run = asyncio.get_running_loop().create_task(
                self._turn(identifier, history)
            )
            await asyncio.shield(self._run)
        except asyncio.CancelledError:
            pass
        except Exception as error:  # noqa: BLE001
            # Nothing was sent, so the text goes back where it was typed.
            self.draft = text
            self.notice = str(error)
            self.notify()

    def _turns(self, history: list[Message], params: Params) -> list[dict[str, str]]:
        """What goes to the model: the roles and the text, and the system prompt as the
        first turn when this chat carries one. The metrics and the thinking stay here."""
        body = [{"role": turn.role, "content": turn.content} for turn in history]
        prompt = params.system_prompt.strip()
        return body if prompt == "" else [{"role": "system", "content": prompt}, *body]

    async def _turn(self, identifier: str, history: list[Message]) -> None:
        params = self.params_for(identifier)
        content, reasoning = "", ""
        failure: str | None = None
        finish: str | None = None
        timings: api.Timings | None = None
        flushed = 0.0
        self.started = time.monotonic()
        self._pulse = None
        self.live = ("", "", None)
        self._clock = asyncio.get_running_loop().create_task(self._ticking())
        self.notify()

        def paint(force: bool) -> None:
            nonlocal flushed
            now = time.monotonic()
            if not force and now - flushed < PAINT:
                return
            flushed = now
            if self.current == identifier:
                self.live = (content, reasoning, failure)
                self.notify()

        try:
            async for frame in api.completion(self.model, self._turns(history, params), params):
                match frame.kind:
                    case "reasoning":
                        reasoning += frame.text
                    case "content":
                        content += frame.text
                    case "usage":
                        if self.current == identifier:
                            self.spent = frame.total_tokens
                    case "timings":
                        timings = frame.timings
                    case "error":
                        failure = frame.text
                    case "finish":
                        finish = frame.text
                paint(False)
        except asyncio.CancelledError:
            pass
        except Exception as error:  # noqa: BLE001
            failure = str(error)
        finally:
            if self._clock is not None:
                self._clock.cancel()
                self._clock = None
            self.waiting = None

        # The turn's numbers are the daemon's, and the two ways it says them: the frame
        # that closes a stream that reached its end, and the last live entry the register
        # published when it did not — a turn the reader stopped is answered by no frame.
        metrics: Metrics | None = None
        if timings is not None:
            metrics = api.metrics_of(timings, finish)
        elif self._pulse is not None:
            metrics = api.metrics_of_sample(self._pulse)
        answer = Message("assistant", content, reasoning, metrics, failure)
        # A turn stopped before it said anything leaves nothing behind; everything else is
        # persisted, partial or refused.
        whole = (
            history
            if content == "" and reasoning == "" and failure is None
            else [*history, answer]
        )
        if self.current == identifier:
            self.messages = whole
        self.live = None
        self.notify()
        try:
            self._merge(await api.put_messages(identifier, whole))
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
        self.notify()

    async def _ticking(self) -> None:
        """How long this turn has been waiting for the register to know about it: the model
        being read off disk, and the queue in front of it. Both are before the meter."""
        while True:
            self.waiting = (
                None if self._pulse is not None else (time.monotonic() - self.started) * 1000
            )
            self.notify()
            await asyncio.sleep(TICK)

    def _stop(self) -> None:
        if self._run is not None:
            self._run.cancel()

    # ── what the sidebar does ─────────────────────────────────────────────

    def _new(self) -> None:
        self._stop()
        act(self._create())

    async def _create(self) -> None:
        try:
            created = await api.create_session(None, self.model)
            self._loaded = created.id
            self._merge(created)
            self.messages = []
            self.spent = None
            self.current = created.id
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
        self.notify()

    def _select(self, identifier: str) -> None:
        if identifier == self.current:
            return
        self._stop()
        self.live = None
        self._open(identifier)
        self.notify()

    def _rename(self, identifier: str, title: str) -> None:
        self.renaming = None
        act(self._name(identifier, title))

    async def _name(self, identifier: str, title: str) -> None:
        try:
            self._merge(await api.patch_session(identifier, title=title))
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
        self.notify()

    def _delete(self, identifier: str) -> None:
        act(self._drop(identifier))

    async def _drop(self, identifier: str) -> None:
        try:
            await api.delete_session(identifier)
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
            self.notify()
            return
        left = [row for row in self.summaries if row.id != identifier]
        self.summaries = left
        if identifier == self.current:
            self._stop()
            self.live = None
            self._loaded = None
            self.messages = []
            self.spent = None
            self._open(left[0].id if left else None)
        self.notify()

    def _pick_model(self, identifier: str) -> None:
        self.model = identifier
        act(self._move(identifier))

    async def _move(self, identifier: str) -> None:
        current = self.current
        if current is None:
            self.notify()
            return
        # The sampling knobs follow the model: another checkpoint is another set of
        # defaults, and 0.6 carried over from the one before is a value nobody chose for
        # this one. What does not follow are the three the checkpoint has no opinion on —
        # the budget, the effort and the system prompt — which are the reader's and stay.
        self.knobs[current] = api.params_of(self._entry, self.params)
        try:
            self._merge(await api.patch_session(current, model=identifier))
        except Exception as error:  # noqa: BLE001
            self.notice = str(error)
        self.notify()

    # ── drawing ───────────────────────────────────────────────────────────

    @property
    def params(self) -> Params:
        return self.params_for(self.current)

    def params_for(self, identifier: str | None) -> Params:
        held = self.knobs.get(identifier or "")
        return held if held is not None else api.params_of(self._entry)

    def _set_params(self, next_: Params) -> None:
        if self.current is not None:
            self.knobs[self.current] = next_
            self.notify()

    def tree(self, engine: Engine) -> ft.Control:
        self.boot()
        self._entry = engine.entry(self.model)
        live = engine.metrics["live"] if engine.metrics else None
        # The register is the daemon's and not this window's, so only a request on this
        # chat's model is drawn: another client's generation is not this one.
        self._pulse = live if live is not None and live["model"] == self.model else None
        return ft.Container(
            expand=True,
            bgcolor=t().win,
            padding=12,
            content=ft.Row(
                [self._sessions(), self._column()],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )

    # ── the sidebar ───────────────────────────────────────────────────────

    def _sessions(self) -> ft.Control:
        pane = (
            self._chats()
            if self.tab == "chats"
            else ft.Container(
                expand=True,
                padding=ft.Padding(left=11, right=11, top=2, bottom=8),
                content=ft.Column(
                    self._knobs(), spacing=0, scroll=ft.ScrollMode.AUTO, expand=True
                ),
            )
        )
        return ft.Container(
            width=SIDEBAR,
            padding=10,
            bgcolor=t().surface,
            border=theme.hair(),
            border_radius=10,
            content=ft.Column(
                [
                    ft.Container(
                        margin=ft.Margin.only(bottom=12),
                        padding=2,
                        bgcolor=t().sunken,
                        border_radius=9,
                        content=ft.Row(
                            [self._tab("chats", "Chats"), self._tab("params", "Parameters")],
                            spacing=2,
                        ),
                    ),
                    pane,
                ],
                spacing=0,
                expand=True,
            ),
        )

    def _tab(self, key: str, label: str) -> ft.Control:
        on = key == self.tab
        return ft.Container(
            expand=True,
            padding=ft.Padding.symmetric(vertical=5),
            alignment=ft.Alignment.CENTER,
            bgcolor=t().surface if on else None,
            border_radius=7,
            shadow=ft.BoxShadow(
                offset=ft.Offset(0, 1), blur_radius=3, color=theme.alpha("#000000", 0.14)
            )
            if on
            else None,
            content=ft.Text(
                label,
                style=theme.sans(11.5, t().fg if on else t().fg2, ft.FontWeight.W_500),
                no_wrap=True,
            ),
            on_click=lambda _: self._pick_tab(key),
        )

    def _pick_tab(self, key: str) -> None:
        self.tab = key
        self.notify()

    def _chats(self) -> ft.Control:
        rows: list[ft.Control] = [self._row(entry) for entry in self.summaries]
        return ft.Column(
            [
                ft.Container(
                    margin=ft.Margin.only(bottom=6),
                    padding=ft.Padding(left=11, right=11, top=9, bottom=9),
                    border=ft.Border.all(1, t().hair2),
                    border_radius=10,
                    content=ft.Row(
                        [
                            _plus(),
                            ft.Text(
                                "New chat",
                                style=theme.sans(12.5, t().fg2, ft.FontWeight.W_500),
                                no_wrap=True,
                            ),
                        ],
                        spacing=7,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    on_click=lambda _: self._new(),
                ),
                ft.Column(rows, spacing=2, scroll=ft.ScrollMode.AUTO, expand=True),
            ],
            spacing=0,
            expand=True,
        )

    def _row(self, entry: Summary) -> ft.Control:
        on = entry.id == self.current
        parts_ = [short(entry.model) if entry.model else "no model"]
        if entry.message_count > 0:
            parts_.append(f"{entry.message_count} msg")
        parts_.append(ago(entry.updated_at))
        title: ft.Control
        if self.renaming == entry.id:
            title = ft.TextField(
                value=self.rename_to,
                height=22,
                autofocus=True,
                text_style=theme.sans(12.5, weight=ft.FontWeight.W_500),
                border=ft.InputBorder.OUTLINE,
                border_color=t().hair2,
                border_width=1,
                focused_border_color=theme.ACCENT,
                filled=True,
                fill_color=t().sunken,
                border_radius=6,
                content_padding=ft.Padding(left=5, right=5, top=0, bottom=0),
                dense=True,
                cursor_color=theme.ACCENT,
                cursor_width=1,
                on_submit=lambda e, i=entry.id: self._rename(i, (e.control.value or "").strip()),
                on_blur=lambda e, i=entry.id: self._rename(i, (e.control.value or "").strip()),
            )
        else:
            title = ft.Text(
                entry.title,
                style=theme.sans(12.5, weight=ft.FontWeight.W_500, spacing=-0.01 * 12.5),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            )
        armed = self.renaming != entry.id

        # The two buttons are revealed on hover, so the resting row is the one that was
        # approved; the row is rebuilt to show them, never written to on screen.
        def face(over: bool) -> ft.Container:
            return ft.Container(
                padding=ft.Padding(left=11, right=11, top=9, bottom=9),
                bgcolor=t().surface if on else None,
                border_radius=10,
                shadow=ft.BoxShadow(
                    offset=ft.Offset(0, 1), blur_radius=4, color=theme.alpha("#000000", 0.10)
                )
                if on
                else None,
                content=ft.Stack(
                    [
                        ft.Column(
                            [
                                title,
                                ft.Text(
                                    " · ".join(parts_),
                                    style=theme.sans(10.5, t().fg3),
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                            ],
                            spacing=3,
                            tight=True,
                        ),
                        ft.Container(
                            right=0,
                            top=2,
                            visible=over and armed,
                            content=ft.Row(
                                [
                                    _tool(_pencil(), lambda: self._start_rename(entry), "Rename"),
                                    _tool(_trash(), lambda: self._delete(entry.id), "Delete"),
                                ],
                                spacing=1,
                                tight=True,
                            ),
                        ),
                    ]
                ),
            )

        return _Row(face, lambda: self._select(entry.id))

    def _start_rename(self, entry: Summary) -> None:
        self.renaming = entry.id
        self.rename_to = entry.title
        self.notify()

    # ── the parameters pane ───────────────────────────────────────────────

    def _knobs(self) -> list[ft.Control]:
        params = self.params
        limit = None if self._entry is None else self._entry["context"]
        cap = limit or 32768
        budget = min(params.max_tokens, cap)
        drawn: list[ft.Control] = [
            ft.Container(
                padding=ft.Padding.only(bottom=14),
                content=ft.Text(
                    "Applied to this chat from the next message on.",
                    style=theme.sans(10.5, t().fg3),
                ),
            ),
            self._slide("Temperature", _decimal(params.temperature), params.temperature, 0, 2,
                        lambda v: self._set_params(replace(params, temperature=round(v, 2))), 0),
            # From 0.01 and not 0: the dialect refuses a top_p of zero, and a slider that
            # can reach a value the daemon will not take is a 400 nobody asked for.
            self._slide("Top-p", _decimal(params.top_p), params.top_p, 0.01, 1,
                        lambda v: self._set_params(replace(params, top_p=round(v, 2))), 1),
            self._slide(
                "Top-k",
                "—" if params.top_k is None else str(params.top_k),
                float(params.top_k or 0), 0, 100,
                lambda v: self._set_params(
                    replace(params, top_k=None if round(v) == 0 else round(v))
                ),
                2,
            ),
            # To .99 and not 1: min_p is a fraction below 1, and 1.0 keeps nothing.
            self._slide(
                "Min-p",
                "—" if params.min_p == 0 else _decimal(params.min_p),
                params.min_p, 0, 0.99,
                lambda v: self._set_params(replace(params, min_p=round(v, 2))),
                3,
            ),
            self._slide(
                "Repetition penalty",
                "—" if params.repetition_penalty <= 1 else _decimal(params.repetition_penalty),
                params.repetition_penalty, 1, 2,
                lambda v: self._set_params(replace(params, repetition_penalty=round(v, 2))),
                4,
            ),
            self._slide(
                "Max tokens", _thousands(budget), float(budget), 256, cap,
                lambda v: self._set_params(
                    replace(params, max_tokens=max(256, round(v / 256) * 256))
                ),
                5,
            ),
            # How long the model may think has no field in this dialect and is not here: it
            # is a profile knob, under Library. What this sets is how hard it is asked to.
            ft.Container(
                padding=ft.Padding.only(bottom=13),
                content=_named(
                    "Reasoning effort",
                    parts.pick(
                        [(rung, rung) for rung in EFFORTS],
                        params.reasoning_effort,
                        lambda v: self._set_params(replace(params, reasoning_effort=v)),
                    ),
                ),
            ),
            _named(
                "System prompt",
                area(
                    params.system_prompt,
                    lambda v: self._set_params(replace(params, system_prompt=v)),
                    "None — the checkpoint's template stands.",
                ),
            ),
        ]
        return drawn

    def _slide(
        self,
        label: str,
        shown: str,
        value: float,
        low: float,
        high: float,
        on_change: Callable[[float], None],
        rail: int,
    ) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.only(bottom=13),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(label, style=theme.sans(11, t().fg2), no_wrap=True),
                            ft.Container(expand=True),
                            ft.Text(
                                shown,
                                style=theme.sans(11, weight=ft.FontWeight.W_600),
                                no_wrap=True,
                            ),
                        ],
                        spacing=8,
                        vertical_alignment=theme.BASELINE,
                    ),
                    parts.slider(
                        value, low, high, on_change, lambda _: None, self._rails[rail]
                    ),
                ],
                spacing=5,
                tight=True,
            ),
        )

    # ── the thread ────────────────────────────────────────────────────────

    def _column(self) -> ft.Control:
        title = next(
            (row.title for row in self.summaries if row.id == self.current), "New chat"
        )
        tag = " · ".join(
            part
            for part in (
                short(self.model) or "no model",
                None if self._entry is None else self._entry["quantization"],
                f"t {_decimal(self.params.temperature)}",
            )
            if part
        )
        # What the last turn spent against what the checkpoint can hold — either half
        # alone when the other is not known yet.
        spent = [
            None if self.spent is None else _thousands(self.spent),
            None
            if self._entry is None or self._entry["context"] is None
            else _context(self._entry["context"]),
        ]
        tokens = " / ".join(part for part in spent if part is not None)

        head: list[ft.Control] = [
            ft.Text(
                title,
                style=theme.sans(13, weight=ft.FontWeight.W_600, spacing=-0.01 * 13),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
            ft.Container(
                padding=ft.Padding(left=9, right=9, top=3, bottom=3),
                bgcolor=t().sunken,
                border_radius=6,
                content=ft.Text(tag, style=theme.sans(11, t().fg2), no_wrap=True),
            ),
            ft.Container(expand=True),
        ]
        if tokens:
            head.append(ft.Text(f"{tokens} tokens", style=theme.mono(9, t().fg3,
                                                                     spacing=0.11 * 9)))

        return ft.Container(
            expand=True,
            bgcolor=t().surface,
            border=theme.hair(),
            border_radius=10,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Column(
                [
                    ft.Container(
                        height=38,
                        padding=ft.Padding.symmetric(horizontal=13),
                        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                        content=ft.Row(
                            head, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER
                        ),
                    ),
                    self._thread(),
                    self._composer(),
                ],
                spacing=0,
                expand=True,
            ),
        )

    def _thread(self) -> ft.Control:
        turns: list[ft.Control] = []
        for message in self.messages:
            # A system turn is somebody else's prompt — this window sends its own from the
            # parameters pane, and never stores one. Nothing to draw for it.
            if message.role == "system":
                continue
            turns.append(
                self._bubble(message.content)
                if message.role == "user"
                else self._answer(
                    message.content, message.reasoning, message.metrics, None,
                    message.error, False, len(turns)
                )
            )
        if self.live is not None:
            content, reasoning, failure = self.live
            turns.append(
                self._answer(
                    content,
                    reasoning,
                    None if self._pulse is None else api.metrics_of_sample(self._pulse),
                    self.waiting,
                    failure,
                    True,
                    -1,
                )
            )
        if not turns:
            turns = [
                ft.Container(
                    padding=ft.Padding.symmetric(vertical=60),
                    alignment=ft.Alignment.CENTER,
                    content=ft.Text("Nothing said yet.", style=theme.sans(12.5, t().fg3)),
                )
            ]
        return ft.Container(
            expand=True,
            alignment=ft.Alignment.TOP_CENTER,
            content=ft.Container(
                width=COLUMN,
                padding=ft.Padding(left=18, right=18, top=14, bottom=8),
                content=ft.Column(
                    turns,
                    spacing=22,
                    scroll=ft.ScrollMode.AUTO,
                    # Follows the stream. Flutter's own list has no "only while the reader
                    # is at the bottom", so it follows it always.
                    auto_scroll=True,
                    expand=True,
                ),
            ),
        )

    def _bubble(self, text: str) -> ft.Control:
        # max-width: 80%. A Row hands its child unbounded width, so the cap is a loose
        # flex — four fifths at most, and less when the text does not fill it.
        return ft.Row(
            [
                ft.Container(expand=2),
                ft.Container(
                    expand=8,
                    expand_loose=True,
                    padding=ft.Padding(left=15, right=15, top=10, bottom=10),
                    bgcolor=t().surface2,
                    border=theme.hair(),
                    border_radius=ft.BorderRadius(
                        top_left=16, top_right=16, bottom_left=16, bottom_right=4
                    ),
                    content=ft.Text(
                        text,
                        style=theme.sans(12.5, height=1.55),
                        selectable=True,
                    ),
                ),
            ],
            alignment=ft.MainAxisAlignment.END,
            spacing=0,
        )

    def _answer(
        self,
        content: str,
        reasoning: str,
        metrics: Metrics | None,
        waiting: float | None,
        error: str | None,
        streaming: bool,
        index: int,
    ) -> ft.Control:
        stack: list[ft.Control] = []
        if reasoning:
            stack.append(self._reasoning(reasoning, index))
        if content == "" and streaming:
            stack.append(_caret())
        else:
            stack.append(_prose(content))
            if streaming:
                stack.append(_caret())
        if error is not None:
            stack.append(
                ft.Container(
                    padding=ft.Padding.only(top=12),
                    content=ft.Text(error, style=theme.sans(11.5, t().bad, height=1.6)),
                )
            )
        if metrics is not None or waiting is not None:
            stack.append(_metrics(metrics, waiting))
        return ft.Column(stack, spacing=0, tight=True)

    def _reasoning(self, text: str, index: int) -> ft.Control:
        """Closed until the reader opens it, streaming or not: the thinking is where the
        answer comes from and not the answer."""
        open_ = index in self._opened
        body = ft.Container(
            visible=open_,
            margin=ft.Margin.only(bottom=14),
            padding=ft.Padding(left=14, right=0, top=2, bottom=2),
            border=ft.Border.only(left=ft.BorderSide(2, t().hair2)),
            content=ft.Text(text, style=theme.sans(11.5, t().fg3, height=1.6), selectable=True),
        )
        return ft.Column(
            [
                ft.Container(
                    margin=ft.Margin.only(bottom=0 if open_ else 10),
                    padding=ft.Padding(left=6, right=6, top=2, bottom=2),
                    border_radius=6,
                    content=ft.Row(
                        [
                            _chevron(open_),
                            ft.Text("Reasoning", style=theme.sans(11.5, t().fg3), no_wrap=True),
                        ],
                        spacing=5,
                        tight=True,
                    ),
                    on_click=lambda _: self._toggle(index),
                ),
                body,
            ],
            spacing=0,
            tight=True,
        )

    def _toggle(self, index: int) -> None:
        self._opened.symmetric_difference_update({index})
        self.notify()

    # ── the composer ──────────────────────────────────────────────────────

    def _composer(self) -> ft.Control:
        streaming = self.live is not None
        # A model the catalog does not list — the one this chat was started on, gone since
        # — is still what the chat says it uses, so it stays selectable.
        offered = (
            self.models
            if self.model in self.models or self.model == ""
            else [self.model, *self.models]
        )
        stack: list[ft.Control] = []
        if self.notice is not None:
            stack.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=4),
                    content=ft.Text(self.notice, style=theme.sans(11, t().bad)),
                )
            )
        stack += [
            ft.TextField(
                value=self.draft,
                multiline=True,
                min_lines=1,
                max_lines=7,
                text_style=theme.sans(12.5, height=1.5),
                hint_text="Ask anything…",
                hint_style=theme.sans(12.5, t().fg3),
                border=ft.InputBorder.NONE,
                content_padding=ft.Padding(left=4, right=4, top=1, bottom=1),
                cursor_color=theme.ACCENT,
                cursor_width=1,
                dense=True,
                shift_enter=True,
                on_change=self._type,
                on_submit=lambda _: self._send(),
            ),
            ft.Row(
                [
                    _model_pick(
                        [(one, short(one)) for one in offered] or [("", "no models")],
                        self.model,
                        self._pick_model,
                    ),
                    ft.Container(expand=True),
                    ft.Text(
                        "generating…" if streaming else "⏎ send",
                        style=theme.sans(11, t().fg3),
                        no_wrap=True,
                    ),
                    _send_button(
                        streaming,
                        self._stop if streaming else self._send,
                        streaming or (self.draft.strip() != "" and self.model != ""),
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]
        return ft.Container(
            padding=ft.Padding(left=18, right=18, top=8, bottom=12),
            alignment=ft.Alignment.TOP_CENTER,
            content=ft.Container(
                width=COLUMN,
                padding=ft.Padding(left=11, right=11, top=9, bottom=9),
                bgcolor=t().field,
                border=theme.hair(t().hair2),
                border_radius=10,
                # Stretched: a column left on its default alignment gives each child its
                # own width, and a text field asked how wide it wants to be answers with
                # something near half the box — the draft wraps early and the rest of the
                # line is dead space.
                content=ft.Column(
                    stack,
                    spacing=10,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ),
        )

    def _type(self, event: ft.ControlEvent) -> None:
        """The field keeps its own text; the window is only rebuilt when the send button
        has to change, which is when the draft goes from empty to not."""
        was = self.draft.strip() != ""
        self.draft = event.control.value or ""
        if was != (self.draft.strip() != ""):
            self.notify()


# ── the pieces below the class ────────────────────────────────────────────


def _syntax() -> ft.MarkdownCustomCodeTheme:
    palette = SYN_DARK if t() is theme.DARK else SYN_LIGHT

    def style(color: str, italic: bool = False, weight: ft.FontWeight | None = None):
        return ft.TextStyle(
            font_family=theme.MONO,
            font_family_fallback=theme.MONO_FALLBACK,
            size=12,
            color=color,
            italic=italic,
            weight=weight,
        )

    key, string, number = style(palette["key"]), style(palette["str"]), style(palette["num"])
    name, attr = style(palette["fn"]), style(palette["attr"])
    comment = style(palette["com"], italic=True)
    return ft.MarkdownCustomCodeTheme(
        root=style(t().fg2),
        code=style(t().fg2),
        comment=comment,
        quote=comment,
        keyword=key,
        selector_tag=key,
        built_in=key,
        literal=key,
        meta=key,
        string=string,
        regexp=string,
        symbol=string,
        addition=string,
        number=number,
        deletion=number,
        title=name,
        class_name=name,
        function=name,
        section=name,
        attr=attr,
        attribute=attr,
        variable=attr,
        type=attr,
        params=attr,
        selector_attr=attr,
        emphasis=style(t().fg2, italic=True),
        strong=style(t().fg2, weight=ft.FontWeight.W_600),
    )


def _inline_code() -> ft.TextStyle:
    style = theme.mono(12, t().fg)
    style.bgcolor = t().sunken
    return style


def _italic() -> ft.TextStyle:
    style = theme.sans(12.5, t().fg, height=1.62)
    style.italic = True
    return style


def _prose(text: str) -> ft.Control:
    """.msg.model — the model's own text. `pre-wrap` in the stylesheet is what makes a
    list written with single newlines a list; `soft_line_break` is the same thing here."""
    body = theme.sans(12.5, t().fg, height=1.62)
    return ft.Markdown(
        text,
        selectable=True,
        soft_line_break=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme=_syntax(),
        md_style_sheet=ft.MarkdownStyleSheet(
            p_text_style=body,
            a_text_style=theme.sans(12.5, t().mat(0), height=1.62),
            em_text_style=_italic(),
            strong_text_style=theme.sans(12.5, t().fg, ft.FontWeight.W_600, height=1.62),
            h1_text_style=theme.sans(14, t().fg, ft.FontWeight.W_600),
            h2_text_style=theme.sans(14, t().fg, ft.FontWeight.W_600),
            h3_text_style=theme.sans(14, t().fg, ft.FontWeight.W_600),
            h4_text_style=theme.sans(14, t().fg, ft.FontWeight.W_600),
            h1_padding=ft.Padding.only(top=16, bottom=8),
            h2_padding=ft.Padding.only(top=16, bottom=8),
            h3_padding=ft.Padding.only(top=16, bottom=8),
            code_text_style=_inline_code(),
            blockquote_text_style=theme.sans(12.5, t().fg2, height=1.62),
            blockquote_padding=ft.Padding(left=12, right=12, top=1, bottom=1),
            blockquote_decoration=ft.BoxDecoration(
                border=ft.Border.only(left=ft.BorderSide(2, t().hair2))
            ),
            codeblock_padding=ft.Padding(left=12, right=12, top=10, bottom=10),
            codeblock_decoration=ft.BoxDecoration(bgcolor=t().sunken, border_radius=8),
            horizontal_rule_decoration=ft.BoxDecoration(
                border=ft.Border.only(top=ft.BorderSide(1, t().hair2))
            ),
            list_indent=22,
            list_bullet_text_style=theme.sans(12.5, t().fg2, height=1.62),
            block_spacing=12,
            table_head_text_style=theme.sans(12.5, t().fg, ft.FontWeight.W_600),
            table_body_text_style=theme.sans(12.5, t().fg),
            table_cells_padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            table_cells_decoration=ft.BoxDecoration(border=ft.Border.all(1, t().hair2)),
        ),
    )


def _metrics(metrics: Metrics | None, waiting: float | None) -> ft.Control:
    """What the turn cost, drawn the same while it is being written and once it is done —
    the numbers come from the daemon either way, so the line does not change shape."""
    said: list[ft.Control] = []

    def say(text: str) -> None:
        said.append(ft.Text(text, style=theme.sans(11, t().fg3), no_wrap=True))

    if waiting is not None:
        say(f"waiting {_duration(waiting)}")
    if metrics is not None:
        if metrics.load_ms is not None:
            say(f"load {_duration(metrics.load_ms)}")
        if metrics.ttft_ms is not None:
            say(f"ttft {_duration(metrics.ttft_ms)}")
        if metrics.prefill_tokens_per_second is not None:
            say(f"prefill {_rate(metrics.prefill_tokens_per_second)}")
        if metrics.tokens_per_second is not None:
            say(f"decode {_rate(metrics.tokens_per_second)}")
        if metrics.ceiling_fraction is not None:
            said.append(
                ft.Row(
                    [
                        ft.Text(
                            f"{round(metrics.ceiling_fraction * 100)}%",
                            style=theme.sans(11, t().mat(0), ft.FontWeight.W_600),
                        ),
                        ft.Text("of ceiling", style=theme.sans(11, t().fg3)),
                    ],
                    spacing=4,
                    tight=True,
                )
            )
        # The one thing on this line that is not a cost: whether a drafter wrote any of it.
        # Only drawn when one did — "no speculation" is what every turn on every other
        # model looks like, and a chip saying so on all of them says nothing.
        spec = metrics.speculation
        if spec is not None and spec["rounds"] > 0:
            # Per round and not as a share of what was proposed: the share falls with the
            # block length whatever the drafter does, so two settings cannot be read
            # against each other by it.
            landed = spec["accepted"] / spec["rounds"]
            width = spec["proposed"] / spec["rounds"]
            said.append(
                ft.Row(
                    [
                        ft.Text("dflash", style=theme.sans(11, t().fg3)),
                        ft.Text(
                            f"{landed:.1f}",
                            style=theme.sans(11, t().mat(0), ft.FontWeight.W_600),
                        ),
                        ft.Text(f"of {width:.0f} a round", style=theme.sans(11, t().fg3)),
                    ],
                    spacing=4,
                    tight=True,
                )
            )
        if metrics.finish == "length":
            say("cut at the token budget")

    woven: list[ft.Control] = []
    for index, one in enumerate(said):
        if index:
            woven.append(ft.Text("·", style=theme.sans(11, t().hair2)))
        woven.append(one)
    return ft.Container(
        padding=ft.Padding.only(top=14),
        content=ft.Row(woven, spacing=8, wrap=True, run_spacing=4),
    )


def _caret() -> ft.Control:
    return ft.Container(width=7, height=15, border_radius=2, bgcolor=t().mat(0))


def _stroke(width: float = 2.0) -> ft.Paint:
    return ft.Paint(
        color=t().fg3,
        stroke_width=width,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )


def _chevron(open_: bool) -> ft.Control:
    s = 12 / 24
    points = [(6, 9), (12, 15), (18, 9)] if open_ else [(9, 6), (15, 12), (9, 18)]
    return cv.Canvas(
        [
            cv.Path(
                [cv.Path.MoveTo(points[0][0] * s, points[0][1] * s)]
                + [cv.Path.LineTo(x * s, y * s) for x, y in points[1:]],
                paint=_stroke(),
            )
        ],
        width=12,
        height=12,
    )


def _plus() -> ft.Control:
    s = 13 / 24
    return cv.Canvas(
        [
            cv.Line(12 * s, 5 * s, 12 * s, 19 * s, paint=_stroke()),
            cv.Line(5 * s, 12 * s, 19 * s, 12 * s, paint=_stroke()),
        ],
        width=13,
        height=13,
    )


def _pencil() -> list[cv.Shape]:
    s = 12 / 24
    return [
        cv.Path(
            [
                cv.Path.MoveTo(17 * s, 3 * s),
                cv.Path.LineTo(21 * s, 7 * s),
                cv.Path.LineTo(7.5 * s, 20.5 * s),
                cv.Path.LineTo(2 * s, 22 * s),
                cv.Path.LineTo(3.5 * s, 16.5 * s),
                cv.Path.Close(),
            ],
            paint=_stroke(1.8 * s * 2),
        )
    ]


def _trash() -> list[cv.Shape]:
    s = 12 / 24
    paint = _stroke(1.8 * s * 2)
    return [
        cv.Line(3 * s, 6 * s, 21 * s, 6 * s, paint=paint),
        cv.Path(
            [
                cv.Path.MoveTo(19 * s, 6 * s),
                cv.Path.LineTo(19 * s, 20 * s),
                cv.Path.LineTo(5 * s, 20 * s),
                cv.Path.LineTo(5 * s, 6 * s),
            ],
            paint=paint,
        ),
        cv.Path(
            [
                cv.Path.MoveTo(9 * s, 6 * s),
                cv.Path.LineTo(9 * s, 3 * s),
                cv.Path.LineTo(15 * s, 3 * s),
                cv.Path.LineTo(15 * s, 6 * s),
            ],
            paint=paint,
        ),
    ]


def _tool(draw: list[cv.Shape], on_click: Callable[[], None], hint: str) -> ft.Control:
    return ft.Container(
        width=22,
        height=22,
        border_radius=6,
        alignment=ft.Alignment.CENTER,
        bgcolor=t().surface2,
        border=theme.hair(),
        tooltip=hint,
        content=cv.Canvas(draw, width=12, height=12),
        on_click=lambda _: on_click(),
    )


@ft.component
def _Row(
    face: Callable[[bool], ft.Container], on_click: Callable[[], None]
) -> ft.Control:
    """A session row, which is under the pointer or is not. State and a rebuild, because a
    control that has been patched into a component's tree is frozen."""
    over, set_over = ft.use_state(False)
    return ft.Container(
        content=face(over),
        on_hover=lambda event: set_over(bool(event.data)),
        on_click=lambda _: on_click(),
    )


def _model_pick(
    options: list[tuple[str, str]], value: str, on_pick: Callable[[str], None]
) -> ft.Control:
    """.modelpick — the composer's pop-up button, sunken and unbordered, wearing the same
    well the ones in the forms do: without it the control reads as a label."""
    label = next((text for key, text in options if key == value), options[0][1])
    return ft.Container(
        height=26,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        content=ft.PopupMenuButton(
            height=26,
            padding=0,
            tooltip="",
            # The menu's constraints, not the button's — see parts._pick.
            menu_padding=ft.Padding.symmetric(vertical=4),
            bgcolor=t().surface2,
            elevation=8,
            shape=ft.RoundedRectangleBorder(radius=7),
            content=ft.Container(
                height=26,
                padding=ft.Padding(left=10, right=3, top=0, bottom=0),
                bgcolor=t().sunken,
                border_radius=7,
                content=ft.Row(
                    [
                        ft.Container(
                            content=ft.Text(
                                label,
                                style=theme.sans(11.5, t().fg2),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            width=min(260, 8 + 6.4 * len(label)),
                        ),
                        parts.well(),
                    ],
                    spacing=6,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
            items=[
                ft.PopupMenuItem(
                    height=24,
                    padding=ft.Padding.symmetric(horizontal=11),
                    content=ft.Text(
                        text,
                        style=theme.sans(
                            11.5, t().fg, ft.FontWeight.W_500 if key == value else None
                        ),
                        no_wrap=True,
                    ),
                    on_click=lambda _, key=key: on_pick(key),
                )
                for key, text in options
            ],
        ),
    )


def _send_button(streaming: bool, on_click: Callable[[], None], enabled: bool) -> ft.Control:
    """.send — a 30pt disc in the foreground colour, which is the one control in this
    window that is not a rectangle."""
    s = 15 / 24
    paint = ft.Paint(
        color=t().win,
        stroke_width=2.2 * s,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    glyph: list[cv.Shape] = (
        [cv.Rect(7 * s, 7 * s, 10 * s, 10 * s, border_radius=1.5 * s,
                 paint=ft.Paint(color=t().win))]
        if streaming
        else [
            cv.Line(12 * s, 19 * s, 12 * s, 5 * s, paint=paint),
            cv.Path(
                [
                    cv.Path.MoveTo(5 * s, 12 * s),
                    cv.Path.LineTo(12 * s, 5 * s),
                    cv.Path.LineTo(19 * s, 12 * s),
                ],
                paint=paint,
            ),
        ]
    )
    face = ft.Container(
        width=30,
        height=30,
        border_radius=15,
        bgcolor=t().fg,
        alignment=ft.Alignment.CENTER,
        opacity=1.0 if enabled else 0.4,
        content=cv.Canvas(glyph, width=15, height=15),
    )
    if not enabled:
        return face
    return ft.GestureDetector(content=face, on_tap=lambda _: on_click())

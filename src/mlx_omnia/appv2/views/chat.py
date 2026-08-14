"""Chat: the model as a pill on top — a picker — the thread, the composer under it.

The thread is a session in the daemon's store; the reply is the OpenAI dialect's own
stream, drawn frame by frame, and the numbers under a turn are the daemon's, never
timed from here. A fresh conversation is not a void: it names what is resident and
offers somewhere to start.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace

import flet as ft
import flet.canvas as cv

from mlx_omnia.app.api import catalog as catalog_api
from mlx_omnia.app.api import conversation
from mlx_omnia.app.api.engine import Engine
from mlx_omnia.app.ui.format import display_name, tokens
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

MEASURE = 620
"""How wide prose runs before it wraps — the proposal's 65ch at 14 pt."""

SUGGESTIONS: tuple[str, ...] = (
    "Why is decode sitting below the 610 GB/s ceiling?",
    "Draft release notes from the last five commits.",
    "How much KV does 262k of context actually cost?",
)


def _meta_line(metrics: conversation.Metrics) -> str | None:
    cells: list[str] = []
    if metrics.tokens_per_second is not None:
        cells.append(f"{metrics.tokens_per_second:.1f} tok/s")
    if metrics.prefill_tokens_per_second is not None:
        cells.append(f"{metrics.prefill_tokens_per_second:,.0f} prefill")
    if metrics.ttft_ms is not None:
        cells.append(f"{metrics.ttft_ms:.0f} ms TTFT")
    return " · ".join(cells) if cells else None


def _pill(name: str, material: str, flip: Callable[[], None]) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=14, right=12, top=6, bottom=6),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=999,
        on_click=lambda _: flip(),
        content=ft.Row(
            [
                ft.Container(width=9, height=9, border_radius=3, bgcolor=material),
                ft.Text(name, style=theme.sans(13, weight=ft.FontWeight.W_500), no_wrap=True),
                ft.Text("▾", style=theme.sans(9, t().fg3), no_wrap=True),
            ],
            spacing=8,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _menu(engine: Engine, current: str, pick: Callable[[str], None]) -> ft.Control:
    """The picker: residents first, then what a pick would have to load. Profiles are
    not here — they are the tuning panel's, with the rest of the knobs."""
    materials = engine.materials

    def entry(identifier: str, resident: bool) -> ft.Control:
        lead = (
            ft.Container(
                width=8, height=8, border_radius=3,
                bgcolor=t().mat(materials.get(identifier, 0)),
            )
            if resident
            else ft.Container(width=8, height=8, border_radius=3, border=theme.hair(t().hair2))
        )
        return ft.Container(
            padding=ft.Padding(left=12, right=12, top=7, bottom=7),
            border_radius=7,
            bgcolor=t().accent_soft if identifier == current else None,
            on_click=lambda _: pick(identifier),
            content=ft.Row(
                [
                    lead,
                    ft.Text(
                        display_name(identifier),
                        style=theme.sans(12.5, t().fg if resident else t().fg2),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                    ft.Text(
                        "resident" if resident else "loads on pick",
                        style=theme.mono(10, t().fg3),
                        no_wrap=True,
                    ),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    loaded = [slot["id"] for slot in engine.models]
    # Only what this engine can actually serve: the catalog lists every checkpoint on
    # disk, and offering one with no loader is a picker that refuses after the pick.
    on_disk = [
        e["id"] for e in engine.catalog if e["supported"] and e["id"] not in set(loaded)
    ]
    rows: list[ft.Control] = [entry(identifier, True) for identifier in loaded]
    rows.extend(entry(identifier, False) for identifier in on_disk)
    if not rows:
        rows.append(
            ft.Container(
                padding=ft.Padding(left=12, right=12, top=7, bottom=7),
                content=ft.Text(
                    "Nothing on disk — fetch a model first.",
                    style=theme.sans(12.5, t().fg3),
                ),
            )
        )

    # The list scrolls once it outgrows the window's lap; a Column only scrolls inside a
    # bounded height, so the cap is what buys the scrollbar.
    tall = 12 + 31.5 * len(rows)
    return ft.Container(
        width=340,
        height=430 if tall > 430 else None,
        padding=6,
        bgcolor=t().elev,
        border=theme.hair(t().hair2),
        border_radius=11,
        shadow=ft.BoxShadow(offset=ft.Offset(0, 10), blur_radius=30, color=t().shade),
        content=ft.Column(rows, spacing=1, tight=True, scroll=ft.ScrollMode.AUTO),
    )


def _topbar(
    engine: Engine, current: str, used: int, flip: Callable[[], None]
) -> ft.Control:
    material = t().mat(engine.materials.get(current, 0)) if current else t().sel
    entry = engine.entry(current) if current else None
    window = None if entry is None else entry["context"]
    if used and window:
        context = f"{tokens(used)} · {used / window:.1%} of {tokens(window)}"
    elif window:
        context = f"0 · {tokens(window)} ctx"
    else:
        context = ""
    middle = (
        _pill(display_name(current), material, flip)
        if current
        else ft.Text("No model", style=theme.sans(13, t().fg3), no_wrap=True)
    )
    return ft.WindowDragArea(
        maximizable=False,
        content=ft.Container(
            height=52,
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=ft.Stack(
                [
                    ft.Container(alignment=ft.Alignment.CENTER, content=middle),
                    ft.Container(
                        right=16,
                        top=0,
                        bottom=0,
                        alignment=ft.Alignment.CENTER_RIGHT,
                        content=ft.Text(
                            context, style=theme.mono(10.5, t().fg3), no_wrap=True
                        ),
                    ),
                ]
            ),
        ),
    )


# ── the tuning panel ──────────────────────────────────────────────────────

_KNOBS: tuple[tuple[str, str, float, float, bool], ...] = (
    ("Temperature", "temperature", 0.0, 2.0, False),
    # From 0.01 and not 0: the dialect refuses a top_p of zero.
    ("Top-p", "top_p", 0.01, 1.0, False),
    ("Top-k", "top_k", 0.0, 100.0, True),
    # To .99 and not 1: min_p is a fraction below 1, and 1.0 keeps nothing.
    ("Min-p", "min_p", 0.0, 0.99, False),
    ("Repetition penalty", "repetition_penalty", 1.0, 2.0, False),
)

# The dialect's two words the profile store does not keep — the same collapse the
# server applies when a request's effort reaches the engine.
_PROFILE_EFFORT = {"none": "off", "minimal": "low"}

_RAIL = 236
"""How wide a knob's slider runs — the panel minus its inset."""


def _num(value: object) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


@ft.component
def _Tuning(
    engine: Engine,
    current: str,
    profile: str,
    knobs: conversation.Params,
    set_knobs: Callable[[conversation.Params], None],
    set_profile: Callable[[str], None],
) -> ft.Control:
    """This conversation's tuning: the profile chips, then every knob with its
    effective value and where it comes from. Touching a slider is what names a knob;
    the ✕ hands the word back to the profile."""
    profiles, set_profiles = ft.use_state(list[str]())
    preset, set_preset = ft.use_state(None)

    def load_names() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                set_profiles(await catalog_api.profile_names(current))
            except Exception:  # noqa: BLE001 — a daemon that is down has no profiles
                set_profiles([])

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load_names, [current])

    def load_preset() -> Callable[[], None] | None:
        if not profile:
            set_preset(None)
            return None

        async def fetch() -> None:
            try:
                set_preset(await catalog_api.get_profile(current, profile))
            except Exception:  # noqa: BLE001 — a profile that is gone sets nothing
                set_preset(None)

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load_preset, [current, profile])

    entry = engine.entry(current)

    def of_profile(field: str) -> float | None:
        return None if preset is None else _num(dict(preset["sampling"]).get(field))

    def declared(field: str) -> float | None:
        return None if entry is None else _num(dict(entry["defaults"]).get(field))

    def eyebrow(text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.only(top=15, bottom=2), content=theme.eyebrow(text)
        )

    def chip_of(name: str) -> ft.Control:
        on = (name == "default" and not profile) or name == profile
        return ft.Container(
            padding=ft.Padding(left=11, right=11, top=3.5, bottom=3.5),
            border=theme.hair(t().accent if on else t().hair),
            bgcolor=t().accent_soft if on else None,
            border_radius=999,
            on_click=lambda _, name=name: set_profile("" if name == "default" else name),
            content=ft.Text(
                name, style=theme.sans(12, t().fg if on else t().fg2), no_wrap=True
            ),
        )

    def knob_row(label: str, field: str, low: float, high: float, whole: bool) -> ft.Control:
        named = _num(getattr(knobs, field))
        if named is not None:
            value, source = named, ""
        elif (preset_value := of_profile(field)) is not None:
            value, source = preset_value, profile
        elif (declared_value := declared(field)) is not None:
            value, source = declared_value, "checkpoint"
        else:
            value, source = None, ""

        def fmt(number: float) -> str:
            return str(round(number)) if whole else f"{number:g}"

        tail: list[ft.Control]
        if named is not None:
            tail = [
                ft.Text(fmt(named), style=theme.mono(11.5, t().accent, ft.FontWeight.W_600),
                        no_wrap=True),
                ft.Container(
                    padding=ft.Padding(left=4, right=2, top=0, bottom=0),
                    on_click=lambda _: set_knobs(replace(knobs, **{field: None})),
                    content=ft.Text("✕", style=theme.sans(11, t().fg3), no_wrap=True),
                ),
            ]
        else:
            tail = [
                ft.Text(
                    "—" if value is None else fmt(value),
                    style=theme.mono(11.5, t().fg3),
                    no_wrap=True,
                ),
            ]
            if source:
                tail.append(
                    ft.Text(f"· {source}", style=theme.sans(10.5, t().fg3), no_wrap=True)
                )

        fraction = 0.0 if value is None else (value - low) / (high - low)

        def slide(part: float) -> None:
            moved = low + part * (high - low)
            if whole:
                count = round(moved)
                set_knobs(replace(knobs, **{field: None if count == 0 else count}))
            else:
                set_knobs(replace(knobs, **{field: round(moved, 2)}))

        return ft.Container(
            padding=ft.Padding(left=0, right=0, top=9, bottom=10),
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(label, style=theme.sans(12.5, weight=ft.FontWeight.W_500),
                                    no_wrap=True),
                            ft.Container(expand=True),
                            *tail,
                        ],
                        spacing=7,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    parts.Slider(_RAIL, min(1.0, max(0.0, fraction)), slide),
                ],
                spacing=4,
                tight=True,
            ),
        )

    # ── reasoning effort, a dropdown ──────────────────────────────────────
    effort_named = knobs.reasoning_effort != "default"
    preset_effort = None if preset is None else preset["sampling"]["reasoning_effort"]
    shown_effort = knobs.reasoning_effort if effort_named else (preset_effort or "default")
    effort_source = "" if effort_named or not preset_effort else profile

    def pick_effort(option: str) -> None:
        set_knobs(replace(knobs, reasoning_effort=option))

    # A real popup: the options float over the panel instead of pushing it down.
    effort_row = ft.Row(
        [
            ft.Text("Reasoning effort", style=theme.sans(12.5, weight=ft.FontWeight.W_500),
                    no_wrap=True),
            ft.Container(expand=True),
            ft.Text(f"· {effort_source}" if effort_source else "",
                    style=theme.sans(10.5, t().fg3), no_wrap=True),
            ft.PopupMenuButton(
                bgcolor=t().elev,
                items=[
                    ft.PopupMenuItem(
                        content=ft.Text(
                            option,
                            style=theme.mono(
                                11.5,
                                t().accent if option == knobs.reasoning_effort else t().fg2,
                            ),
                        ),
                        on_click=lambda _, option=option: pick_effort(option),
                    )
                    for option in conversation.EFFORTS
                ],
                content=ft.Container(
                    padding=ft.Padding(left=9, right=9, top=3, bottom=3),
                    bgcolor=t().field,
                    border=theme.hair(t().hair2),
                    border_radius=7,
                    content=ft.Row(
                        [
                            ft.Text(
                                shown_effort,
                                style=theme.mono(
                                    11.5, t().accent if effort_named else t().fg2
                                ),
                                no_wrap=True,
                            ),
                            ft.Text("▾", style=theme.sans(9, t().fg3), no_wrap=True),
                        ],
                        spacing=7,
                        tight=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
            ),
        ],
        spacing=7,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ── system prompt ─────────────────────────────────────────────────────
    preset_prompt = None if preset is None else preset["system_prompt"]
    hint = preset_prompt or "None — the checkpoint's template stands."
    prompt_box = ft.Container(
        padding=ft.Padding(left=10, right=10, top=8, bottom=8),
        bgcolor=t().field,
        border=theme.hair(),
        border_radius=8,
        content=ft.TextField(
            value=knobs.system_prompt,
            multiline=True,
            min_lines=2,
            max_lines=5,
            text_style=theme.sans(12, height=1.45),
            hint_text=hint,
            hint_style=theme.sans(12, t().fg3, height=1.45),
            border=ft.InputBorder.NONE,
            content_padding=ft.Padding.all(0),
            cursor_color=t().accent,
            cursor_width=1,
            dense=True,
            on_change=lambda event: set_knobs(
                replace(knobs, system_prompt=event.control.value or "")
            ),
        ),
    )

    def save_as() -> None:
        async def write() -> None:
            sampling: dict[str, object] = dict(preset["sampling"]) if preset is not None else {}
            for _, field, _, _, _ in _KNOBS:
                value = getattr(knobs, field)
                if value is not None:
                    sampling[field] = value
            if effort_named:
                sampling["reasoning_effort"] = _PROFILE_EFFORT.get(
                    knobs.reasoning_effort, knobs.reasoning_effort
                )
            prompt = knobs.system_prompt.strip() or preset_prompt
            name = f"profile-{len(profiles) + 1}"
            await catalog_api.save_profile(
                current, name, {"sampling": sampling, "system_prompt": prompt or None}
            )
            # The new profile now carries the panel's state, so the overrides fold
            # into it and the conversation follows the name.
            set_profiles([*profiles, name])
            set_profile(name)
            set_knobs(conversation.DEFAULTS)

        runtime.act(write())

    body: list[ft.Control] = [
        # CENTER and not BASELINE: a baseline row without an explicit text baseline is
        # a Flutter layout exception, and the whole panel vanishes with it.
        ft.Row(
            [
                ft.Text("Tuning", style=theme.sans(13.5, weight=ft.FontWeight.W_600),
                        no_wrap=True),
                ft.Text("this conversation", style=theme.sans(11, t().fg3), no_wrap=True),
            ],
            spacing=8,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        eyebrow("Profile"),
        ft.Row(
            [chip_of(name) for name in ["default", *profiles]],
            wrap=True,
            spacing=7,
            run_spacing=7,
        ),
        eyebrow("Sampling"),
        *(knob_row(*shape) for shape in _KNOBS),
        ft.Container(padding=ft.Padding.only(top=10, bottom=2), content=effort_row),
        eyebrow("System prompt"),
        prompt_box,
        ft.Container(height=10),
        # In a Row and not bare: the panel's column scrolls, and a flex child inside a
        # scrollable has no bounded height to expand into — the row is what bounds it.
        ft.Row([parts.button("Save as new profile…", save_as, expand=True)]),
        ft.Container(
            padding=ft.Padding.only(top=8),
            alignment=ft.Alignment.CENTER,
            content=ft.Text(
                "named › profile › checkpoint",
                style=theme.mono(10, t().fg3),
                no_wrap=True,
            ),
        ),
    ]
    return ft.Container(
        width=288,
        bgcolor=t().elev,
        border=ft.Border.only(left=ft.BorderSide(1, t().hair)),
        padding=ft.Padding(left=18, right=18, top=16, bottom=14),
        content=ft.Column(body, spacing=0, tight=True, scroll=ft.ScrollMode.AUTO),
    )


def _sheet(muted: bool = False) -> ft.MarkdownStyleSheet:
    """The markdown styles, derived from the tokens at render time — which is what
    makes the rendered prose follow the theme without asking. `muted` is the
    reasoning's register: a step smaller and a step dimmer than the answer."""
    size = 13 if muted else 14
    color = t().fg2 if muted else None
    body = theme.sans(size, color, height=1.55)
    return ft.MarkdownStyleSheet(
        p_text_style=body,
        a_text_style=theme.sans(size, t().accent, height=1.55),
        strong_text_style=theme.sans(size, color, ft.FontWeight.W_600, height=1.55),
        code_text_style=theme.mono(12 if muted else 12.5, color if muted else t().fg),
        h1_text_style=theme.display(21),
        h2_text_style=theme.display(18),
        h3_text_style=theme.sans(15.5, weight=ft.FontWeight.W_600),
        h4_text_style=theme.sans(14, weight=ft.FontWeight.W_600),
        blockquote_text_style=theme.sans(14, t().fg2, height=1.55),
        blockquote_decoration=ft.BoxDecoration(
            bgcolor=t().sel,
            border_radius=ft.BorderRadius.all(8),
        ),
        blockquote_padding=ft.Padding.all(12),
        codeblock_decoration=ft.BoxDecoration(
            bgcolor=t().field,
            border=theme.hair(),
            border_radius=ft.BorderRadius.all(8),
        ),
        codeblock_padding=ft.Padding.all(12),
        list_bullet_text_style=body,
        table_head_text_style=theme.sans(13, weight=ft.FontWeight.W_600),
        table_body_text_style=theme.sans(13),
        block_spacing=10,
    )


def _prose(text: str, muted: bool = False) -> ft.Control:
    return ft.Markdown(
        value=text,
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK
        if t() is theme.DARK
        else ft.MarkdownCodeTheme.ATOM_ONE_LIGHT,
        md_style_sheet=_sheet(muted),
        soft_line_break=True,
    )


@dataclass(frozen=True)
class _Piece:
    code: bool
    lang: str = ""
    text: str = ""
    closed: bool = True


def _pieces(text: str) -> list[_Piece]:
    """The reply split at its fences, so code can fold while prose stays prose. An
    unclosed fence at the end is a block still being written."""
    pieces: list[_Piece] = []
    prose: list[str] = []
    code: list[str] | None = None
    lang = ""

    def flush() -> None:
        joined = "\n".join(prose)
        if joined.strip():
            pieces.append(_Piece(False, text=joined))
        prose.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if code is None and stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            code = []
        elif code is not None and stripped.startswith("```"):
            pieces.append(_Piece(True, lang, "\n".join(code)))
            code = None
        elif code is not None:
            code.append(line)
        else:
            prose.append(line)
    if code is not None:
        pieces.append(_Piece(True, lang, "\n".join(code), closed=False))
    else:
        flush()
    return pieces


def _spinner() -> ft.Control:
    return ft.ProgressRing(width=11, height=11, stroke_width=1.5, color=t().accent)


def _chevron(opened: bool) -> ft.Control:
    return ft.Text("▾" if opened else "▸", style=theme.sans(9, t().fg3), no_wrap=True)


@ft.component
def _Code(lang: str, body: str, writing: bool) -> ft.Control:
    """A fenced block, folded to one line until asked — and while the model is still
    writing it, the line is the activity itself."""
    opened, set_opened = ft.use_state(False)
    lines = body.count("\n") + 1
    if writing:
        lead: ft.Control = _spinner()
        label = "Coding…"
    else:
        lead = ft.Text("‹›", style=theme.mono(10, t().fg3), no_wrap=True)
        label = f"{lang or 'code'} · {lines} line{'s' if lines != 1 else ''}"
    header = ft.Container(
        padding=ft.Padding(left=12, right=12, top=7, bottom=7),
        bgcolor=t().field,
        border=theme.hair(),
        border_radius=8,
        on_click=lambda _: set_opened(not opened),
        content=ft.Row(
            [
                lead,
                ft.Text(label, style=theme.mono(11, t().fg2), no_wrap=True),
                ft.Container(expand=True),
                _chevron(opened),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    if not opened:
        return header
    return ft.Column(
        [header, _prose(f"```{lang}\n{body}\n```")],
        spacing=6,
        tight=True,
    )


@ft.component
def _Reasoning(text: str, thinking: bool) -> ft.Control:
    """The model's thinking, folded to a word until asked; opened, it reads as
    markdown in the muted register."""
    opened, set_opened = ft.use_state(False)
    lead: ft.Control = (
        _spinner()
        if thinking
        else ft.Container(width=6, height=6, border_radius=2, bgcolor=t().fg3)
    )
    header = ft.Container(
        padding=ft.Padding(left=0, right=0, top=2, bottom=2),
        on_click=lambda _: set_opened(not opened),
        content=ft.Row(
            [
                lead,
                ft.Text(
                    "Thinking…" if thinking else "Reasoning",
                    style=theme.sans(12, t().fg3, ft.FontWeight.W_500),
                    no_wrap=True,
                ),
                _chevron(opened),
            ],
            spacing=7,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    if not opened:
        return header
    return ft.Column([header, _prose(text, muted=True)], spacing=8, tight=True)


def _turn(
    message: conversation.Message,
    live: bool = False,
    spot: conversation.Metrics | None = None,
) -> ft.Control:
    """One message. `live` marks the reply still being streamed; `spot` is the live
    register's numbers for it, shown until the closing frame brings the turn's own."""
    if message.role == "user":
        # An Align, not a Row: a Row hands its children unbounded width and the text
        # never wraps; the aligned box bounds it at the column minus the inset.
        return ft.Container(
            padding=ft.Padding.only(left=120),
            alignment=ft.Alignment.CENTER_RIGHT,
            content=ft.Container(
                padding=ft.Padding(left=16, right=16, top=9, bottom=9),
                bgcolor=t().accent_soft,
                border_radius=ft.BorderRadius(
                    top_left=16, top_right=16, bottom_left=16, bottom_right=4
                ),
                content=ft.Text(
                    message.content, style=theme.sans(14, height=1.45), selectable=True
                ),
            ),
        )
    body: list[ft.Control] = []
    if message.reasoning:
        body.append(
            ft.Container(
                width=MEASURE,
                content=_Reasoning(message.reasoning, live and not message.content),
            )
        )
    if message.content:
        for piece in _pieces(message.content):
            inner = (
                _Code(piece.lang, piece.text, live and not piece.closed)
                if piece.code
                else _prose(piece.text)
            )
            body.append(ft.Container(width=MEASURE, content=inner))
    elif not message.reasoning:
        body.append(
            ft.Container(
                width=MEASURE,
                content=ft.Text("…", style=theme.sans(14, height=1.55)),
            )
        )
    if message.error is not None:
        body.append(ft.Text(message.error, style=theme.sans(12.5, t().bad)))
    metrics = message.metrics if message.metrics is not None else spot
    if metrics is not None:
        line = _meta_line(metrics)
        if line is not None:
            body.append(ft.Text(line, style=theme.mono(11, t().fg3), no_wrap=True))
    return ft.Column(body, spacing=8, tight=True)


def _send() -> ft.Control:
    arrow = ft.Paint(
        color=t().on_accent,
        stroke_width=1.6,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    return ft.Container(
        width=30,
        height=30,
        border_radius=9,
        bgcolor=t().accent,
        alignment=ft.Alignment.CENTER,
        content=cv.Canvas(
            [
                cv.Line(7, 10.5, 7, 3.5, paint=arrow),
                cv.Path(
                    [cv.Path.MoveTo(3.8, 6.6), cv.Path.LineTo(7, 3.4), cv.Path.LineTo(10.2, 6.6)],
                    paint=arrow,
                ),
            ],
            width=14,
            height=14,
        ),
    )


def _attach() -> ft.Control:
    """The multimodal door, honest: no family in this engine takes an image yet, so it
    stays shut and says why."""
    return ft.Container(
        width=24,
        height=24,
        border_radius=12,
        border=theme.hair(t().hair2),
        alignment=ft.Alignment.CENTER,
        content=ft.Text("+", style=theme.sans(14, t().fg3, ft.FontWeight.W_500)),
        tooltip="Vision is not served by the engine yet",
    )


def _hello(
    engine: Engine, current: str, say: Callable[[str], None], pick: Callable[[str], None]
) -> ft.Control:
    def suggestion(text: str) -> ft.Control:
        return ft.Container(
            width=190,
            padding=ft.Padding(left=14, right=14, top=12, bottom=12),
            bgcolor=t().elev,
            border=theme.hair(),
            border_radius=10,
            on_click=lambda _: say(text),
            content=ft.Text(text, style=theme.sans(12.5, t().fg2, height=1.45)),
        )

    materials = engine.materials

    def resident_chip(identifier: str) -> ft.Control:
        on = identifier == current
        return ft.Container(
            padding=ft.Padding(left=11, right=11, top=4, bottom=4),
            border=theme.hair(t().accent if on else t().hair),
            bgcolor=t().accent_soft if on else None,
            border_radius=999,
            on_click=lambda _: pick(identifier),
            content=ft.Row(
                [
                    ft.Container(
                        width=7,
                        height=7,
                        border_radius=2.5,
                        bgcolor=t().mat(materials.get(identifier, 0)),
                    ),
                    ft.Text(
                        display_name(identifier),
                        style=theme.sans(12, t().fg if on else t().fg2),
                        no_wrap=True,
                    ),
                ],
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    residents = [slot["id"] for slot in engine.models]
    if current:
        state = "resident" if current in residents else "on disk"
        title = f"{display_name(current)} is {state}."
        sub = "Ask, or start from one of these."
    elif engine.health is None:
        title = "The engine is starting."
        sub = "The dot in the sidebar goes green when it answers."
    else:
        title = "Nothing on disk yet."
        sub = "Fetch a model under Models, then come back."

    column: list[ft.Control] = [
        ft.Text(title, style=theme.display(19)),
        ft.Text(sub, style=theme.sans(12.5, t().fg3)),
    ]
    if current:
        column.append(ft.Container(height=8))
        column.append(
            ft.Row(
                [suggestion(text) for text in SUGGESTIONS],
                spacing=10,
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )
    if len(residents) > 1:
        column.append(ft.Container(height=6))
        column.append(
            ft.Row(
                [resident_chip(identifier) for identifier in residents],
                spacing=8,
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )
    return ft.Container(
        expand=True,
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            column,
            spacing=10,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


@ft.component
def Chat(shell: Shell, engine: Engine) -> ft.Control:
    model_id, set_model_id = ft.use_state("")
    profile, set_profile = ft.use_state("")
    picker, set_picker = ft.use_state(False)
    # This conversation's named knobs, in window memory: closing the app falls back to
    # the profile, which is the place for what should last.
    knobs, set_knobs = ft.use_state(conversation.DEFAULTS)
    draft, set_draft = ft.use_state("")
    turns, set_turns = ft.use_state(list[conversation.Message]())
    busy, set_busy = ft.use_state(False)
    used, set_used = ft.use_state(0)
    owned = ft.use_ref(False)

    slots = engine.models
    runnable = [e["id"] for e in engine.catalog if e["supported"]]
    current = model_id or (
        slots[0]["id"] if slots else (runnable[0] if runnable else "")
    )

    def load() -> Callable[[], None] | None:
        # A session this window just created is already on screen; reloading it would
        # clobber the stream writing into it.
        if owned.current:
            owned.current = False
            return None

        async def fetch() -> None:
            set_knobs(conversation.DEFAULTS)
            if shell.conversation is None:
                set_turns([])
                set_used(0)
                return
            try:
                session = await conversation.get_session(shell.conversation)
            except Exception:  # noqa: BLE001 — a session that is gone opens as fresh
                set_turns([])
                return
            set_turns(list(session.messages))
            # The session's model may carry the profile it was opened with — the same
            # `model:name` the daemon serves it under.
            base, _, chosen = session.model.partition(":")
            if base:
                set_model_id(base)
            set_profile(chosen)

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [shell.conversation])

    def pick(identifier: str) -> None:
        set_model_id(identifier)
        # An override belongs to the checkpoint it was tuned against, and so does a
        # profile name: another model, a clean panel.
        set_profile("")
        set_knobs(conversation.DEFAULTS)
        set_picker(False)

    def say(text: str) -> None:
        spoken_text = text.strip()
        if not spoken_text or busy or not current:
            return
        set_busy(True)

        async def run() -> None:
            target = f"{current}:{profile}" if profile else current
            try:
                session_id = shell.conversation
                if session_id is None:
                    created = await conversation.create_session(
                        title=spoken_text[:48], model=target
                    )
                    session_id = created.id
                    owned.current = True
                    shell.conversation = session_id
                else:
                    # The stored model follows the turn, so reopening restores the
                    # model and profile the conversation last actually used.
                    await conversation.patch_session(session_id, model=target)
                before = [*turns, conversation.Message("user", spoken_text)]
                reply = conversation.Message("assistant", "")
                set_turns([*before, reply])
                wire = [{"role": m.role, "content": m.content} for m in before]
                # Only what the panel named rides the request and beats the profile;
                # the rest stays out, and the daemon's preset decides. A named system
                # prompt goes as the system turn — the daemon only prepends the
                # profile's when the client sent none.
                prompt = knobs.system_prompt.strip()
                if prompt:
                    wire = [{"role": "system", "content": prompt}, *wire]
                finish: str | None = None
                timings: conversation.Timings | None = None
                async for frame in conversation.completion(target, wire, knobs):
                    if frame.kind == "content":
                        reply.content += frame.text
                    elif frame.kind == "reasoning":
                        reply.reasoning += frame.text
                    elif frame.kind == "usage":
                        set_used(frame.total_tokens)
                    elif frame.kind == "finish":
                        finish = frame.text
                    elif frame.kind == "timings":
                        timings = frame.timings
                    elif frame.kind == "error":
                        reply.error = frame.text
                    if timings is not None:
                        reply.metrics = conversation.metrics_of(timings, finish)
                    set_turns([*before, reply])
                await conversation.put_messages(session_id, [*before, reply])
            except Exception as error:  # noqa: BLE001 — the refusal is the message shown
                refused = conversation.Message("assistant", "", error=str(error))
                set_turns([*turns, conversation.Message("user", spoken_text), refused])
            finally:
                set_busy(False)

        ft.context.page.run_task(run)

    def submit(_event: ft.Event[ft.TextField]) -> None:
        # The event's control is frozen in this Flet; the state is the field's only
        # writable face, so clearing means re-rendering with an empty value.
        text = draft
        set_draft("")
        say(text)

    composer = ft.Container(
        margin=ft.Margin(left=44, right=44, top=8, bottom=24),
        padding=ft.Padding(left=10, right=8, top=7, bottom=7),
        bgcolor=t().field,
        border=theme.hair(t().hair2),
        border_radius=14,
        shadow=ft.BoxShadow(
            offset=ft.Offset(0, 1), blur_radius=3, color=theme.alpha("#000000", 0.10)
        ),
        content=ft.Row(
            [
                _attach(),
                ft.TextField(
                    expand=True,
                    value=draft,
                    text_style=theme.sans(14),
                    hint_text="Generating…" if busy else "Ask anything…",
                    hint_style=theme.sans(14, t().fg3),
                    border=ft.InputBorder.NONE,
                    content_padding=ft.Padding.all(0),
                    cursor_color=t().accent,
                    cursor_width=1,
                    dense=True,
                    on_change=lambda event: set_draft(event.control.value or ""),
                    on_submit=submit,
                ),
                _send(),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    def spot_of(index: int, message: conversation.Message) -> conversation.Metrics | None:
        # The live register's numbers, only for the reply still being written and only
        # until the closing frame writes the turn's own.
        streaming = busy and index == len(turns) - 1 and message.role == "assistant"
        if not streaming or message.metrics is not None:
            return None
        sample = engine.live(current)
        return None if sample is None else conversation.metrics_of_sample(sample)

    body: ft.Control
    if turns:
        body = ft.Container(
            expand=True,
            padding=ft.Padding(left=44, right=44, top=28, bottom=0),
            content=ft.Column(
                [
                    _turn(
                        message,
                        busy
                        and index == len(turns) - 1
                        and message.role == "assistant",
                        spot_of(index, message),
                    )
                    for index, message in enumerate(turns)
                ],
                spacing=22,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )
    else:
        body = _hello(engine, current, say, pick)

    lap: list[ft.Control] = [
        ft.Column([body, composer], spacing=0, expand=True),
    ]
    if current:
        lap.append(_Tuning(engine, current, profile, knobs, set_knobs, set_profile))
    screen = ft.Column(
        [
            _topbar(engine, current, used, lambda: set_picker(not picker)),
            ft.Row(
                lap,
                spacing=0,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        ],
        spacing=0,
        expand=True,
    )

    layers: list[ft.Control] = [screen]
    if picker:
        layers.append(
            ft.Container(
                top=58,
                left=0,
                right=0,
                alignment=ft.Alignment.TOP_CENTER,
                content=_menu(engine, current, pick),
            )
        )
    return ft.Stack(layers, expand=True)

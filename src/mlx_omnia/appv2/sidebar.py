"""The sidebar: the traffic lights, four places, the current view's own list, and the
gauge at the foot — the band in miniature, which is the door back to the Server.

Each place wears a thin-stroke glyph and a quiet second reading on the right: the
Server carries the engine's dot and port, Models its count, Chat and Benchmark nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from mlx_omnia.app.api import conversation
from mlx_omnia.app.api.engine import GIB, Engine
from mlx_omnia.app.ui.format import display_name, gb
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

WIDTH = 228
LIGHT_SIZE = 13.5
LIGHT_STEP = 23.0
CLOSE, MINIMIZE = "#FF5F57", "#FEBC2E"
IDLE_LIGHT = ("#DCDCDC", "#4F5052")

VIEWS = [
    ("server", "Server", "pulse"),
    ("chat", "Chat", "bubble"),
    ("models", "Models", "stack"),
    ("quantize", "Quantize", "grain"),
    ("benchmark", "Benchmark", "dial"),
]

# The gauge's own track: the sidebar's 228 minus its 12 of padding and the gauge's 12.
TRACK = WIDTH - 2 * 12 - 2 * 12


def port_of(engine: Engine) -> str:
    if engine.system is not None:
        return engine.system["constants"].get("port", "8642")
    return "8642"


def _glyph(kind: str) -> ft.Control:
    paint = ft.Paint(
        color=theme.alpha("#000000", 0.58),
        stroke_width=1.1,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
    )
    shapes: list[cv.Shape] = (
        [cv.Line(4.4, 4.4, 8.6, 8.6, paint=paint), cv.Line(8.6, 4.4, 4.4, 8.6, paint=paint)]
        if kind == "close"
        else [cv.Line(3.5, 6.5, 9.5, 6.5, paint=paint)]
    )
    return cv.Canvas(shapes, width=LIGHT_SIZE, height=LIGHT_SIZE)


@ft.component
def Lights(focused: bool) -> ft.Control:
    """The v1 traffic lights, drawn where AppKit would put them: (16, 14) off the
    window's corner, which against the sidebar's 12 of padding is left 4, top 2."""
    over, set_over = ft.use_state(False)
    page = ft.context.page
    idle = IDLE_LIGHT[1] if t() is theme.DARK else IDLE_LIGHT[0]

    def close() -> None:
        page.run_task(page.window.close)

    def minimize() -> None:
        page.window.minimized = True
        page.update()

    buttons: list[ft.Control] = []
    for kind, color, act in (
        ("close", CLOSE, close),
        ("minimize", MINIMIZE, minimize),
        ("zoom", None, None),
    ):
        buttons.append(
            ft.Container(
                width=LIGHT_SIZE,
                height=LIGHT_SIZE,
                border_radius=LIGHT_SIZE / 2,
                bgcolor=color if color is not None and focused else idle,
                border=theme.hair(theme.alpha("#000000", 0.10)),
                content=_glyph(kind) if over and focused and color is not None else None,
                on_click=None if act is None else (lambda _, act=act: act()),
            )
        )

    return ft.Container(
        padding=ft.Padding(left=4, right=0, top=2, bottom=18),
        content=ft.Row(buttons, spacing=LIGHT_STEP - LIGHT_SIZE, tight=True),
        on_hover=(lambda event: set_over(bool(event.data))) if focused else None,
    )


def _trailing(engine: Engine, identifier: str) -> ft.Control | None:
    """The quiet second reading on a place's right edge."""
    if identifier == "server":
        return ft.Row(
            [
                parts.dot("bad" if engine.health is None else "ok"),
                ft.Text(port_of(engine), style=theme.mono(10.5, t().fg3), no_wrap=True),
            ],
            spacing=5,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    if identifier == "models" and engine.catalog:
        return ft.Text(str(len(engine.catalog)), style=theme.mono(10.5, t().fg3), no_wrap=True)
    return None


@ft.component
def NavItem(shell: Shell, engine: Engine, identifier: str, title: str, mark: str) -> ft.Control:
    on = shell.view == identifier

    def body(tint: str | None) -> ft.Container:
        lead = t().fg if on else t().fg2
        row: list[ft.Control] = [
            parts.glyph(mark, lead),
            ft.Text(
                title,
                style=theme.sans(13.5, lead, ft.FontWeight.W_600 if on else ft.FontWeight.W_500),
                no_wrap=True,
            ),
            ft.Container(expand=True),
        ]
        tail = _trailing(engine, identifier)
        if tail is not None:
            row.append(tail)
        return ft.Container(
            padding=ft.Padding(left=11, right=11, top=7, bottom=7),
            bgcolor=t().accent_soft if on else tint,
            border_radius=8,
            content=ft.Row(row, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    return parts.press(
        body,
        lambda: setattr(shell, "view", identifier),
        None,
        None if on else t().sel,
        t().sel,
    )


@ft.component
def Conversations(shell: Shell) -> ft.Control:
    """The chat list, off `/admin/sessions`. The + starts a fresh conversation — which is
    the empty state — and the ✕, revealed by the pointer, deletes one from the store."""
    chats, set_chats = ft.use_state(list[conversation.Summary]())

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                set_chats(await conversation.list_sessions())
            except Exception:  # noqa: BLE001 — a daemon that is down has no sessions
                set_chats([])

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [shell.conversation])

    def start() -> None:
        shell.conversation = None

    def forget(identifier: str) -> None:
        async def drop() -> None:
            await conversation.delete_session(identifier)
            set_chats([chat for chat in chats if chat.id != identifier])
            if shell.conversation == identifier:
                shell.conversation = None

        runtime.act(drop())

    plus = ft.Container(
        width=17,
        height=17,
        border_radius=5,
        bgcolor=t().sel,
        alignment=ft.Alignment.CENTER,
        content=ft.Text("+", style=theme.sans(12, t().fg2, ft.FontWeight.W_500)),
        on_click=lambda _: start(),
    )

    rows: list[ft.Control] = [
        ft.Container(
            padding=ft.Padding(left=11, right=11, top=0, bottom=6),
            content=ft.Row(
                [theme.eyebrow("Conversations"), ft.Container(expand=True), plus],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )
    ]
    rows.extend(_ConvoRow(shell, chat.id, chat.title, forget) for chat in chats)
    return ft.Column(rows, spacing=2, tight=True)


@ft.component
def _ConvoRow(
    shell: Shell, identifier: str, title: str, forget: Callable[[str], None]
) -> ft.Control:
    on = shell.conversation == identifier
    over, set_over = ft.use_state(False)

    inner: list[ft.Control] = [
        ft.Text(
            title,
            style=theme.sans(13, t().fg if on else t().fg2),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
    ]
    if over:
        inner.append(
            ft.Container(
                content=ft.Text("✕", style=theme.sans(10.5, t().fg3)),
                padding=ft.Padding(left=4, right=2, top=0, bottom=0),
                on_click=lambda _: forget(identifier),
            )
        )

    def choose() -> None:
        shell.conversation = identifier
        shell.view = "chat"

    return ft.Container(
        content=ft.Container(
            padding=ft.Padding(left=11, right=7, top=6, bottom=6),
            bgcolor=t().sel if on or over else None,
            border_radius=7,
            content=ft.Row(inner, spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            on_click=lambda _: choose(),
        ),
        on_hover=lambda event: set_over(bool(event.data)),
    )


def _quantizing_mini(engine: Engine) -> ft.Control:
    """The quantize jobs still moving — name, progress and the leaf of the moment.
    Shown from any view: the job does not hold the screen it was born on."""
    rows: list[ft.Control] = [
        ft.Container(
            padding=ft.Padding(left=11, right=11, top=0, bottom=6),
            content=theme.eyebrow("Quantizing"),
        )
    ]
    for job in engine.jobs:
        if job["kind"] != "quantize" or job["state"] not in ("pending", "running"):
            continue
        model = job["subject"].get("model")
        target = job["subject"].get("target")
        source = display_name(model) if isinstance(model, str) else "?"
        landing = display_name(target) if isinstance(target, str) else "?"
        suffix = landing.removeprefix(f"{source}-") or landing
        total = job["progress"]["total"]
        done = job["progress"]["completed"]
        fraction = 0.0 if not total else min(1.0, done / total)
        message = (
            f"{round(fraction * 100)}% · {job['progress']['message']}"
            if total
            else job["progress"]["message"]
        )
        rows.append(
            ft.Container(
                padding=ft.Padding(left=11, right=11, top=4, bottom=8),
                content=ft.Column(
                    [
                        ft.Text(
                            f"{source} → {suffix}",
                            style=theme.sans(12.5, t().fg2),
                            no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                        ft.Container(
                            height=3,
                            border_radius=2,
                            bgcolor=t().sel,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                            content=ft.Row(
                                [
                                    ft.Container(
                                        expand=max(1, round(fraction * 100)),
                                        bgcolor=t().accent,
                                    ),
                                    ft.Container(expand=max(1, round((1 - fraction) * 100))),
                                ],
                                spacing=0,
                                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                            ),
                        ),
                        ft.Text(
                            message,
                            style=theme.mono(10, t().fg3),
                            no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                    ],
                    spacing=5,
                    tight=True,
                ),
            )
        )
    return ft.Column(rows, spacing=0, tight=True)


def _residents_mini(engine: Engine) -> ft.Control:
    """The residents in miniature — dot, name, GB — so Models' sidebar says something."""
    rows: list[ft.Control] = [
        ft.Container(
            padding=ft.Padding(left=11, right=11, top=0, bottom=6),
            content=theme.eyebrow("Resident"),
        )
    ]
    materials = engine.materials
    for slot in engine.models:
        rows.append(
            ft.Container(
                padding=ft.Padding(left=11, right=11, top=4, bottom=4),
                content=ft.Row(
                    [
                        ft.Container(
                            width=7,
                            height=7,
                            border_radius=2.5,
                            bgcolor=t().mat(materials.get(slot["id"], 0)),
                        ),
                        ft.Text(
                            display_name(slot["id"]),
                            style=theme.sans(12.5, t().fg2),
                            no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS,
                            expand=True,
                        ),
                        ft.Text(
                            gb(slot["weights_bytes"] + slot["kv_bytes"]),
                            style=theme.mono(10, t().fg3),
                            no_wrap=True,
                        ),
                    ],
                    spacing=7,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        )
    if not engine.models:
        rows.append(
            ft.Container(
                padding=ft.Padding(left=11, right=11, top=4, bottom=4),
                content=ft.Text("Nothing resident.", style=theme.sans(12, t().fg3)),
            )
        )
    return ft.Column(rows, spacing=0, tight=True)


def _segments(engine: Engine, total: int) -> list[ft.Control]:
    """The residents drawn to the gauge's scale — weights solid, KV the same material
    thinned, as on the band."""
    bars: list[ft.Control] = []
    materials = engine.materials
    for slot in engine.models:
        material = t().mat(materials.get(slot["id"], 0))
        bars.append(ft.Container(width=slot["weights_bytes"] / total * TRACK, bgcolor=material))
        if slot["kv_bytes"] > 0:
            bars.append(
                ft.Container(
                    width=slot["kv_bytes"] / total * TRACK,
                    bgcolor=theme.mix(material, 0.40, theme.TRANSPARENT),
                )
            )
    return bars


@ft.component
def Gauge(shell: Shell, engine: Engine) -> ft.Control:
    total = engine.system["memory_bytes"] if engine.system is not None else 128 * GIB
    used = 0 if engine.state is None else engine.state["resident_bytes"]
    ceiling = engine.ceiling
    down = engine.health is None

    def body(tint: str | None) -> ft.Container:
        layers: list[ft.Control] = [
            ft.Row(
                _segments(engine, total),
                spacing=1,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        ]
        if ceiling is not None:
            layers.append(
                ft.Container(
                    left=ceiling / total * TRACK, top=0, bottom=0, width=1.5, bgcolor=t().fg2
                )
            )
        return ft.Container(
            padding=12,
            bgcolor=tint,
            border=theme.hair(),
            border_radius=10,
            content=ft.Column(
                [
                    ft.Container(
                        height=7,
                        border_radius=4,
                        bgcolor=t().sel,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        content=ft.Stack(layers),
                    ),
                    ft.Row(
                        [
                            ft.Text(
                                f"{gb(used, 0)} of {gb(total, 0)} GB",
                                style=theme.mono(10.5, t().fg2),
                                no_wrap=True,
                            ),
                            ft.Container(expand=True),
                            parts.dot("bad" if down else "ok"),
                            ft.Text(
                                f":{port_of(engine)}",
                                style=theme.mono(10.5, t().fg2),
                                no_wrap=True,
                            ),
                        ],
                        spacing=5,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=8,
                tight=True,
            ),
        )

    return parts.press(
        body,
        lambda: setattr(shell, "view", "server"),
        t().elev,
        theme.mix(t().fg, 0.04, t().elev),
        theme.mix(t().fg, 0.08, t().elev),
    )


@ft.component
def Sidebar(shell: Shell, engine: Engine) -> ft.Control:
    column: list[ft.Control] = [
        ft.WindowDragArea(content=Lights(shell.focused), maximizable=False),
        ft.Column(
            [
                NavItem(shell, engine, identifier, title, mark)
                for identifier, title, mark in VIEWS
            ],
            spacing=2,
            tight=True,
        ),
    ]
    if shell.view == "chat":
        column.append(ft.Container(height=20))
        column.append(Conversations(shell))
    elif shell.view == "models":
        column.append(ft.Container(height=20))
        column.append(_residents_mini(engine))
    if any(
        job["kind"] == "quantize" and job["state"] in ("pending", "running")
        for job in engine.jobs
    ):
        column.append(ft.Container(height=20))
        column.append(_quantizing_mini(engine))
    column.append(ft.Container(expand=True))
    column.append(Gauge(shell, engine))

    return ft.Container(
        width=WIDTH,
        padding=12,
        bgcolor=t().side,
        border=ft.Border.only(right=ft.BorderSide(1, t().hair)),
        content=ft.Column(column, spacing=0, expand=True),
    )

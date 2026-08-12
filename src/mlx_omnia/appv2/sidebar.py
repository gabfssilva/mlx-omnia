"""The sidebar: the traffic lights, three places, the current view's own list, and the
gauge at the foot — the band in miniature, which is the door to the Now panel.

The gauge is the v1 title-bar meter moved home: same scale, same materials, plus the
total and the engine's dot. It is the one thing on screen in every view.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv

from mlx_omnia.appv2 import data, parts, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

WIDTH = 228
LIGHT_SIZE = 13.5
LIGHT_STEP = 23.0
CLOSE, MINIMIZE = "#FF5F57", "#FEBC2E"
IDLE_LIGHT = ("#DCDCDC", "#4F5052")

VIEWS = [
    ("chat", "Chat"),
    ("models", "Models"),
    ("settings", "Settings"),
]

# The gauge's own track: the sidebar's 228 minus its 12 of padding and the gauge's 12.
TRACK = WIDTH - 2 * 12 - 2 * 12


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


@ft.component
def NavItem(shell: Shell, identifier: str, title: str) -> ft.Control:
    on = shell.view == identifier

    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=11, right=11, top=7, bottom=7),
            bgcolor=t().accent_soft if on else tint,
            border_radius=8,
            content=ft.Text(
                title,
                style=theme.sans(
                    13.5,
                    t().fg if on else t().fg2,
                    ft.FontWeight.W_600 if on else ft.FontWeight.W_500,
                ),
                no_wrap=True,
            ),
        )

    return parts.press(
        body,
        lambda: setattr(shell, "view", identifier),
        None,
        None if on else t().sel,
        t().sel,
    )


@ft.component
def Conversations() -> ft.Control:
    """The chat list, mocked: the first one is the open one."""
    chosen, set_chosen = ft.use_state(0)

    def row(index: int, title: str) -> ft.Control:
        on = index == chosen

        def body(tint: str | None) -> ft.Container:
            return ft.Container(
                padding=ft.Padding(left=11, right=11, top=6, bottom=6),
                bgcolor=t().sel if on else tint,
                border_radius=7,
                content=ft.Text(
                    title,
                    style=theme.sans(13, t().fg if on else t().fg2),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
            )

        return parts.press(body, lambda: set_chosen(index), None, None if on else t().sel, t().sel)

    return ft.Column(
        [
            ft.Container(
                padding=ft.Padding(left=11, right=11, top=0, bottom=6),
                content=theme.eyebrow("Conversations"),
            ),
            *[row(index, title) for index, title in enumerate(data.CHATS)],
        ],
        spacing=2,
        tight=True,
    )


def _segments() -> list[ft.Control]:
    """The residents drawn to the gauge's scale — weights solid, KV the same material
    thinned, as on the band."""
    bars: list[ft.Control] = []
    for model in data.residents():
        material = t().mat(model.material or 0)
        bars.append(ft.Container(width=model.size_gb / data.TOTAL_GB * TRACK, bgcolor=material))
        if model.kv_gb > 0:
            bars.append(
                ft.Container(
                    width=model.kv_gb / data.TOTAL_GB * TRACK,
                    bgcolor=theme.mix(material, 0.40, theme.TRANSPARENT),
                )
            )
    return bars


@ft.component
def Gauge(shell: Shell) -> ft.Control:
    used = data.used_gb()

    def body(tint: str | None) -> ft.Container:
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
                        content=ft.Stack(
                            [
                                ft.Row(
                                    _segments(),
                                    spacing=1,
                                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                                ),
                                ft.Container(
                                    left=data.CEILING_GB / data.TOTAL_GB * TRACK,
                                    top=0,
                                    bottom=0,
                                    width=1.5,
                                    bgcolor=t().fg2,
                                ),
                            ]
                        ),
                    ),
                    ft.Row(
                        [
                            ft.Text(
                                f"{used:.0f} of {data.TOTAL_GB:.0f} GB",
                                style=theme.mono(10.5, t().fg2),
                                no_wrap=True,
                            ),
                            ft.Container(expand=True),
                            parts.dot("ok"),
                            ft.Text(f":{data.PORT}", style=theme.mono(10.5, t().fg2), no_wrap=True),
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
        lambda: setattr(shell, "now_open", True),
        t().elev,
        theme.mix(t().fg, 0.04, t().elev),
        theme.mix(t().fg, 0.08, t().elev),
    )


@ft.component
def Sidebar(shell: Shell) -> ft.Control:
    column: list[ft.Control] = [
        ft.WindowDragArea(content=Lights(shell.focused), maximizable=False),
        ft.Column(
            [NavItem(shell, identifier, title) for identifier, title in VIEWS],
            spacing=2,
            tight=True,
        ),
    ]
    if shell.view == "chat":
        column.append(ft.Container(height=20))
        column.append(Conversations())
    column.append(ft.Container(expand=True))
    column.append(Gauge(shell))

    return ft.Container(
        width=WIDTH,
        padding=12,
        bgcolor=t().side,
        border=ft.Border.only(right=ft.BorderSide(1, t().hair)),
        content=ft.Column(column, spacing=0, expand=True),
    )

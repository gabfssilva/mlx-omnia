"""Chat: the model as a pill on top, the thread, the composer under it.

The parameters — temperature, max tokens, reasoning, system prompt — are not on this
screen by design; they open in an inspector later (⌘I in the proposal). The composer
appends a canned turn, which is the mockup saying out loud that nothing answered.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv

from mlx_omnia.appv2 import data, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

MEASURE = 620
"""How wide prose runs before it wraps — the proposal's 65ch at 14 pt."""


def _topbar() -> ft.Control:
    model = data.residents()[0]
    pill = ft.Container(
        padding=ft.Padding(left=14, right=14, top=6, bottom=6),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=999,
        content=ft.Row(
            [
                ft.Container(
                    width=9,
                    height=9,
                    border_radius=3,
                    bgcolor=t().mat(model.material or 0),
                ),
                ft.Text(model.name, style=theme.sans(13, weight=ft.FontWeight.W_500), no_wrap=True),
                ft.Text("· default", style=theme.sans(13, t().fg3), no_wrap=True),
            ],
            spacing=8,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    return ft.WindowDragArea(
        maximizable=False,
        content=ft.Container(
            height=52,
            alignment=ft.Alignment.CENTER,
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=pill,
        ),
    )


def _turn(turn: data.Turn) -> ft.Control:
    if turn.role == "user":
        return ft.Row(
            [
                ft.Container(
                    padding=ft.Padding(left=16, right=16, top=9, bottom=9),
                    bgcolor=t().accent_soft,
                    border_radius=ft.BorderRadius(
                        top_left=16, top_right=16, bottom_left=16, bottom_right=4
                    ),
                    content=ft.Text(turn.text, style=theme.sans(14, height=1.45), width=None),
                )
            ],
            alignment=ft.MainAxisAlignment.END,
        )
    body: list[ft.Control] = [
        ft.Container(
            width=MEASURE,
            content=ft.Text(turn.text, style=theme.sans(14, height=1.55)),
        )
    ]
    if turn.meta is not None:
        body.append(ft.Text(turn.meta, style=theme.mono(11, t().fg3), no_wrap=True))
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


@ft.component
def Chat(shell: Shell) -> ft.Control:
    turns, set_turns = ft.use_state(list(data.THREAD))

    def submit(event: ft.Event[ft.TextField]) -> None:
        text = (event.control.value or "").strip()
        if not text:
            return
        event.control.value = ""
        set_turns([*turns, data.Turn("user", text), data.CANNED_REPLY])

    composer = ft.Container(
        margin=ft.Margin(left=44, right=44, top=8, bottom=24),
        padding=ft.Padding(left=16, right=8, top=7, bottom=7),
        bgcolor=t().field,
        border=theme.hair(t().hair2),
        border_radius=14,
        shadow=ft.BoxShadow(
            offset=ft.Offset(0, 1), blur_radius=3, color=theme.alpha("#000000", 0.10)
        ),
        content=ft.Row(
            [
                ft.TextField(
                    expand=True,
                    text_style=theme.sans(14),
                    hint_text="Ask anything…",
                    hint_style=theme.sans(14, t().fg3),
                    border=ft.InputBorder.NONE,
                    content_padding=ft.Padding.all(0),
                    cursor_color=t().accent,
                    cursor_width=1,
                    dense=True,
                    on_submit=submit,
                ),
                _send(),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    return ft.Column(
        [
            _topbar(),
            ft.Container(
                expand=True,
                padding=ft.Padding(left=44, right=44, top=28, bottom=0),
                content=ft.Column(
                    [_turn(turn) for turn in turns],
                    spacing=22,
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                ),
            ),
            composer,
        ],
        spacing=0,
        expand=True,
    )

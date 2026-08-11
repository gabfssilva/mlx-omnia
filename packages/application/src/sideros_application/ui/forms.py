"""The pieces a form is written with: an eyebrow over a control, a read-only line, a
textarea, the daemon's refusal, and the two-column grid the knobs sit in.

Apart from `parts` because these are the shapes of a *form* rather than of a control, and
because three screens reach for them.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from sideros_application.ui import theme
from sideros_application.ui.theme import t


def line(label: str, value: str, unset: bool = False) -> ft.Control:
    """.kvline"""
    return ft.Container(
        padding=ft.Padding(left=0, right=0, top=5.5, bottom=5.5),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                theme.eyebrow(label),
                ft.Container(expand=True),
                ft.Text(
                    value,
                    style=theme.mono(
                        11, t().fg3 if unset else t().fg, None if unset else ft.FontWeight.W_500
                    ),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                    text_align=ft.TextAlign.RIGHT,
                ),
            ],
            spacing=10,
            vertical_alignment=theme.BASELINE,
        ),
    )


def labelled(
    label: str, control: ft.Control, hint: ft.Control | None = None, stretch: bool = True
) -> ft.Control:
    """.field — an eyebrow over the control it names.

    A field fills its column; a segmented control does not (`flex: none` on the strip), so
    the two are told apart here rather than each caller wrapping its own.
    """
    stack: list[ft.Control] = [theme.eyebrow(label), control]
    if hint is not None:
        stack.append(hint)
    return ft.Column(
        stack,
        spacing=4,
        tight=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        if stretch
        else ft.CrossAxisAlignment.START,
    )


def area(value: str, on_commit: Callable[[str], None], hint: str) -> ft.Control:
    """textarea.input — three lines of the body face, and the same box the fields wear."""
    return ft.TextField(
        value=value,
        multiline=True,
        min_lines=3,
        max_lines=8,
        text_style=theme.sans(11.5),
        hint_text=hint,
        hint_style=theme.sans(11.5, t().fg3),
        border=ft.InputBorder.OUTLINE,
        border_color=t().hair2,
        border_width=1,
        focused_border_color=theme.ACCENT,
        focused_border_width=1,
        filled=True,
        fill_color=t().field,
        border_radius=7,
        content_padding=ft.Padding(left=9, right=9, top=7, bottom=7),
        cursor_color=theme.ACCENT,
        cursor_width=1,
        dense=True,
        on_blur=lambda e: on_commit(e.control.value or ""),
    )


def notice(message: str, dismiss: Callable[[], None] | None = None) -> ft.Control:
    """.notice — the daemon's own sentence, kept where the action was asked for."""
    row: list[ft.Control] = [
        ft.Text(message, style=theme.sans(11, t().bad, height=1.5), expand=True)
    ]
    if dismiss is not None:
        row.append(
            ft.Container(
                content=ft.Text("✕", style=theme.sans(11, t().bad)),
                padding=4,
                on_click=lambda _: dismiss(),
            )
        )
    return ft.Container(
        margin=ft.Margin.symmetric(vertical=9),
        padding=ft.Padding(left=10, right=10, top=8, bottom=8),
        bgcolor=theme.mix(t().bad, 0.10, theme.TRANSPARENT),
        border=theme.hair(theme.mix(t().bad, 0.28, theme.TRANSPARENT)),
        border_radius=8,
        content=ft.Row(row, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
    )


def grid(cells: list[ft.Control], gap: float = 10, run: float = 8) -> ft.Control:
    """.knobs — two equal columns, filled row by row."""
    rows: list[ft.Control] = []
    for start in range(0, len(cells), 2):
        pair = cells[start : start + 2]
        pair += [ft.Container() for _ in range(2 - len(pair))]
        rows.append(
            ft.Row(
                [ft.Container(content=cell, expand=True) for cell in pair],
                spacing=gap,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
    return ft.Column(rows, spacing=run, tight=True)


def h3(title: str, meta: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.only(bottom=8),
        content=ft.Row(
            [
                ft.Text(title, style=theme.sans(11.5, weight=ft.FontWeight.W_600), no_wrap=True),
                ft.Text(meta, style=theme.sans(10.5, t().fg3), no_wrap=True),
            ],
            spacing=7,
            vertical_alignment=theme.BASELINE,
        ),
    )

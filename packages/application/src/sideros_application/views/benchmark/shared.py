"""The pieces more than one of the benchmark's panels is written with."""

from __future__ import annotations

import flet as ft

from sideros_application.ui import theme
from sideros_application.ui.theme import t

CORPUS = "wikitext103"


def knob(label: str, control: ft.Control) -> ft.Control:
    return ft.Row(
        [theme.eyebrow(label), control],
        spacing=6,
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def group(label: str, tail: str = "", first: bool = False) -> ft.Control:
    row: list[ft.Control] = [theme.eyebrow(label)]
    if tail:
        row.append(ft.Text(tail, style=theme.mono(9, t().fg3), no_wrap=True))
    return ft.Container(
        padding=ft.Padding(left=0, right=0, top=0 if first else 13, bottom=4),
        content=ft.Row(row, spacing=5, tight=True),
    )


def keyline(key: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.symmetric(horizontal=10, vertical=7),
        bgcolor=t().sunken,
        border_radius=7,
        content=ft.Row(
            [
                ft.Text("key", style=theme.mono(10, t().fg2)),
                ft.Text(
                    key,
                    style=theme.mono(10, t().fg, ft.FontWeight.W_500),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                ),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def cost(label: str, count: str, tail: str, right: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.symmetric(horizontal=10, vertical=8),
        bgcolor=t().field,
        border=theme.hair(),
        border_radius=7,
        content=ft.Row(
            [
                ft.Text(label, style=theme.sans(10.5, t().fg2), no_wrap=True),
                ft.Text(count, style=theme.mono(11.5, t().fg, ft.FontWeight.W_500),
                        no_wrap=True),
                ft.Text(tail, style=theme.sans(10.5, t().fg2), no_wrap=True),
                ft.Container(expand=True),
                ft.Text(right, style=theme.mono(10, t().fg3), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
            ],
            spacing=8,
            vertical_alignment=theme.BASELINE,
        ),
    )

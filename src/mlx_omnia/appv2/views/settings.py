"""Settings: a native grouped form, one column, nothing tabbed.

The theme control is the one thing here that already works — it drives the same
mode/system rule the v1 window uses. Everything else is drawn at rest until the
screen is wired to the daemon.
"""

from __future__ import annotations

import flet as ft

from mlx_omnia.appv2 import data, parts, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

WIDTH = 560


def _row(label: str, value: ft.Control, first: bool = False) -> ft.Container:
    return ft.Container(
        padding=ft.Padding(left=16, right=16, top=11, bottom=11),
        border=None if first else ft.Border.only(top=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                ft.Text(label, style=theme.sans(13.5), no_wrap=True),
                ft.Container(expand=True),
                value,
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _measure(text: str) -> ft.Text:
    return ft.Text(text, style=theme.mono(12.5, t().fg2), no_wrap=True)


def _group(title: str, rows: list[ft.Container]) -> ft.Control:
    return ft.Column(
        [
            ft.Container(
                padding=ft.Padding(left=16, right=16, top=0, bottom=6),
                content=theme.eyebrow(title),
            ),
            ft.Container(
                bgcolor=t().elev,
                border=theme.hair(),
                border_radius=11,
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                content=ft.Column(rows, spacing=0, tight=True),
            ),
        ],
        spacing=0,
        tight=True,
    )


@ft.component
def Settings(shell: Shell) -> ft.Control:
    groups = ft.Column(
        [
            _group(
                "Engine",
                [
                    _row("Port", _measure(data.PORT), first=True),
                    _row("Memory ceiling", _measure(f"{data.CEILING_GB:.0f} GB")),
                    _row(
                        "Unload after idle",
                        ft.Row(
                            [_measure("30 min"), parts.toggle(True)],
                            spacing=10,
                            tight=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                ],
            ),
            _group(
                "Endpoints",
                [
                    _row("OpenAI", _measure("/v1 · active"), first=True),
                    _row("Anthropic", _measure("/anthropic · active")),
                    _row("Claude Code", parts.button("Copy configuration", lambda: None)),
                ],
            ),
            _group(
                "Storage",
                [
                    _row("Checkpoints", _measure("~/.cache/huggingface"), first=True),
                    _row("Disk free", _measure("214 GB")),
                ],
            ),
            _group(
                "Appearance",
                [
                    _row(
                        "Theme",
                        parts.scope(
                            [("light", "Light"), ("system", "System"), ("dark", "Dark")],
                            shell.mode,
                            shell.choose,
                        ),
                        first=True,
                    ),
                ],
            ),
            _group(
                "Daemon",
                [
                    _row(
                        "This window's process",
                        ft.Row(
                            [
                                parts.button("Restart", lambda: None),
                                parts.button("Stop", lambda: None, "danger"),
                            ],
                            spacing=8,
                            tight=True,
                        ),
                        first=True,
                    ),
                ],
            ),
        ],
        spacing=20,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    return ft.Column(
        [
            ft.WindowDragArea(content=ft.Container(height=28), maximizable=False),
            ft.Container(
                expand=True,
                alignment=ft.Alignment.TOP_CENTER,
                padding=ft.Padding(left=24, right=24, top=8, bottom=24),
                content=ft.Container(width=WIDTH, content=groups),
            ),
        ],
        spacing=0,
        expand=True,
    )

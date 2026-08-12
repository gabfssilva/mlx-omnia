"""Now: the v1 Resident, whole, as a panel over whatever screen is on.

The band is drawn to the same scale rules as v1 — weights in the block's material,
KV shaded against its right edge, the ceiling dashed, the headroom hatched. Esc and
the close button put it away; the gauge in the sidebar brings it back.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from mlx_omnia.appv2 import data, parts, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

PANEL = 760
TRACK = PANEL - 2 * 22 - 2
TRACK_HEIGHT = 56


def _slab(model: data.Model, on: bool, pick: Callable[[], None]) -> ft.Control:
    material = t().mat(model.material or 0)
    width = model.resident_gb / data.TOTAL_GB * TRACK
    size = model.resident_gb

    face: list[ft.Control] = []
    if model.kv_gb > 0:
        face.append(
            ft.Container(
                right=0,
                top=0,
                bottom=0,
                width=model.kv_gb / size * width,
                bgcolor=theme.mix(material, 0.30, theme.TRANSPARENT),
                border=ft.Border.only(
                    left=ft.BorderSide(1, theme.mix(material, 0.50, theme.TRANSPARENT))
                ),
            )
        )
    face.append(ft.Container(left=0, top=0, bottom=0, width=3, bgcolor=material))
    face.append(
        ft.Container(
            left=10,
            right=5,
            top=8,
            content=ft.Text(
                model.name,
                style=theme.sans(11, weight=ft.FontWeight.W_600, height=1.2),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
        )
    )
    face.append(
        ft.Container(
            left=10,
            right=5,
            top=24,
            content=ft.Text(
                f"{model.size_gb:.1f} GB + {model.kv_gb:.1f} KV",
                style=theme.mono(10, t().fg2),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
        )
    )

    return ft.Container(
        content=ft.Stack(face),
        width=width,
        bgcolor=theme.mix(material, 0.24, t().surface),
        border=theme.hair(material if on else theme.mix(material, 0.45, theme.TRANSPARENT)),
        border_radius=6,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        on_click=lambda _: pick(),
    )


def _dashed(height: float, color: str) -> ft.Control:
    count = int(height // 6)
    return ft.Column(
        [ft.Container(width=1.5, height=3, bgcolor=color) for _ in range(count)],
        spacing=3,
        tight=True,
    )


def _band(chosen: str, pick: Callable[[str], None]) -> ft.Control:
    headroom = data.TOTAL_GB - data.CEILING_GB
    layers: list[ft.Control] = [
        ft.Row(
            [
                _slab(model, model.id == chosen, lambda model=model: pick(model.id))
                for model in data.residents()
            ],
            spacing=2,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        ft.Container(
            right=0,
            top=0,
            bottom=0,
            width=headroom / data.TOTAL_GB * TRACK,
            gradient=theme.hatch(t().hair, 5, TRACK, duty=0.2),
        ),
        ft.Container(
            left=data.CEILING_GB / data.TOTAL_GB * TRACK,
            top=-3,
            bottom=-3,
            content=ft.Stack(
                [
                    _dashed(TRACK_HEIGHT + 6, t().fg2),
                    ft.Container(
                        right=7,
                        bottom=5,
                        content=ft.Text(
                            f"ceiling · {data.CEILING_GB:.0f} GB",
                            style=theme.mono(10, t().fg2),
                            no_wrap=True,
                        ),
                    ),
                ],
                clip_behavior=ft.ClipBehavior.NONE,
            ),
        ),
    ]
    return ft.Container(
        height=TRACK_HEIGHT,
        border_radius=8,
        bgcolor=theme.mix(t().sel, 0.55, t().surface),
        border=theme.hair(),
        content=ft.Stack(layers, clip_behavior=ft.ClipBehavior.NONE),
    )


def _close(shell: Shell) -> ft.Control:
    mark = ft.Paint(
        color=t().fg2,
        stroke_width=1.4,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
    )
    return ft.Container(
        width=26,
        height=26,
        border_radius=13,
        bgcolor=t().sel,
        alignment=ft.Alignment.CENTER,
        content=cv.Canvas(
            [cv.Line(3.5, 3.5, 8.5, 8.5, paint=mark), cv.Line(8.5, 3.5, 3.5, 8.5, paint=mark)],
            width=12,
            height=12,
        ),
        on_click=lambda _: setattr(shell, "now_open", False),
    )


@ft.component
def Now(shell: Shell) -> ft.Control:
    chosen, set_chosen = ft.use_state(data.residents()[0].id)
    selected = next(model for model in data.residents() if model.id == chosen)

    stats = ft.Container(
        padding=ft.Padding(left=18, right=18, top=14, bottom=14),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=11,
        content=ft.Row(
            [
                ft.Container(
                    content=parts.fact("Weights", f"{selected.size_gb:.1f} GB"),
                    expand=True,
                ),
                ft.Container(
                    content=parts.fact("KV", f"{selected.kv_gb:.1f} GB"),
                    expand=True,
                ),
                ft.Container(content=parts.fact("Decode", selected.decode or "—"), expand=True),
                ft.Container(content=parts.fact("Queue", "0"), expand=True),
                parts.button("Unload", lambda: None, "danger"),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    trace = ft.Column(
        [
            ft.Container(padding=ft.Padding.only(bottom=4), content=theme.eyebrow("Recent")),
            *[
                ft.Container(
                    padding=ft.Padding(left=2, right=2, top=7, bottom=7),
                    border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            ft.Container(
                                width=46,
                                content=ft.Text(
                                    when, style=theme.mono(11, t().fg3), no_wrap=True
                                ),
                            ),
                            ft.Text(what, style=theme.sans(12.5, t().fg2), no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                            ft.Text(measure, style=theme.mono(11, t().fg3), no_wrap=True),
                        ],
                        spacing=14,
                    ),
                )
                for when, what, measure in data.TRACE
            ],
        ],
        spacing=0,
        tight=True,
    )

    used = data.used_gb()
    panel = ft.Container(
        width=PANEL,
        padding=22,
        bgcolor=t().surface,
        border=theme.hair(t().hair2),
        border_radius=14,
        shadow=ft.BoxShadow(offset=ft.Offset(0, 22), blur_radius=60, color=t().shade),
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Now", style=theme.display(18)),
                        ft.Container(expand=True),
                        ft.Text(
                            f"{used:.1f} of {data.TOTAL_GB:.0f} GB"
                            f" · ceiling {data.CEILING_GB:.0f} GB",
                            style=theme.mono(11.5, t().fg3),
                            no_wrap=True,
                        ),
                        _close(shell),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Container(height=10),
                _band(chosen, set_chosen),
                ft.Container(height=14),
                stats,
                ft.Container(height=16),
                trace,
            ],
            spacing=0,
            tight=True,
        ),
    )

    return ft.Container(
        expand=True,
        alignment=ft.Alignment.CENTER,
        padding=24,
        bgcolor=theme.mix(t().win, 0.62, theme.TRANSPARENT),
        blur=3,
        content=panel,
    )

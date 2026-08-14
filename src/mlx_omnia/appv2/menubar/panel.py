"""The panel's own state, its measures, and the shell every tab is drawn inside.

The window has no sidebar — 228 pt do not fit in 380 — so what the sidebar carried is
split in two. The band, the dot and the port move into a head that never leaves the
screen, and the five places become four tabs. That head is the management view: what is
looked at ten times a day is already open before a tab is chosen.

The measures are the spike's. 380 is the narrowest width that still fits a model's name
beside the Bench verdict; 560 is a height that clears the menu bar on a laptop screen with
room under it.
"""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from mlx_omnia.app.api.engine import GIB, Engine
from mlx_omnia.app.ui.format import gb
from mlx_omnia.appv2 import parts, theme
from mlx_omnia.appv2.sidebar import port_of
from mlx_omnia.appv2.theme import t

WIDTH, HEIGHT = 380.0, 560.0
PAD = 14.0
TRACK = WIDTH - 2 * PAD
# Under the menu bar, not against it.
GAP = 6.0
# What the panel keeps between itself and the edge of the screen when the icon sits too
# near one for the panel to be centred under it.
EDGE = 8.0

TABS = (("manage", "Manage"), ("models", "Models"), ("quantize", "Quantize"), ("bench", "Bench"))


@ft.observable
@dataclass
class Panel:
    """Observable rather than component state: the status item's clicks and the system
    going dark both arrive from outside any render."""

    tab: str = "manage"
    opened: str = ""
    """The model pushed over the list, whole-panel. Empty is the list itself."""
    mode: str = "system"
    system_dark: bool = False
    dark: bool = False

    def choose(self, mode: str) -> None:
        self.mode = mode
        self.dark = self.system_dark if mode == "system" else mode == "dark"

    def go(self, tab: str) -> None:
        self.tab = tab
        self.opened = ""


def place(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """Where the panel goes for an icon at `x, y, width, height`, top-left coordinates.

    Centred under the icon, and pulled back from the left edge when centring would put it
    off-screen. The right edge is not clamped here: it needs the width of the screen the
    icon is on, and the item reports the icon, not the screen. Being a few points off the
    centre is a better failure than reading the wrong display's width.
    """
    return max(EDGE, x + width / 2 - WIDTH / 2), y + height + GAP


# ── the head ─────────────────────────────────────────────────────────────


def _segments(engine: Engine, total: int) -> list[ft.Control]:
    """The residents drawn to the band's scale — weights solid, KV the same material
    thinned, exactly as the sidebar's gauge does it."""
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


def head(panel: Panel, engine: Engine, overflow: ft.Control) -> ft.Control:
    total = engine.system["memory_bytes"] if engine.system is not None else 128 * GIB
    used = 0 if engine.state is None else engine.state["resident_bytes"]
    ceiling = engine.ceiling
    down = engine.health is None

    layers: list[ft.Control] = [
        ft.Row(
            _segments(engine, total),
            spacing=1,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
    ]
    # The tick is the daemon's number. With the daemon gone the band is empty, and a mark
    # left standing on it is the last answer pretending to be the current one.
    if ceiling is not None and not down:
        layers.append(
            ft.Container(left=ceiling / total * TRACK, top=0, bottom=0, width=1.5, bgcolor=t().fg2)
        )

    reading: list[ft.Control] = [
        ft.Text(
            "— of 128 GB" if down else f"{gb(used)} of {gb(total, 0)} GB",
            style=theme.mono(10.5, t().fg2),
            no_wrap=True,
        )
    ]
    if ceiling is not None and not down:
        reading.append(
            ft.Text(f"· ceiling {gb(ceiling, 0)}", style=theme.mono(10.5, t().fg3), no_wrap=True)
        )
    reading.append(ft.Container(expand=True))
    reading.append(parts.dot("bad" if down else "ok"))
    reading.append(
        ft.Text(f":{port_of(engine)}", style=theme.mono(10.5, t().fg2), no_wrap=True)
    )

    return ft.Container(
        padding=ft.Padding(left=PAD, right=PAD, top=12, bottom=11),
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Omnia", style=theme.display(15)),
                        ft.Container(expand=True),
                        overflow,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Container(
                    height=7,
                    margin=ft.Margin.only(top=11),
                    border_radius=4,
                    bgcolor=t().sel,
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    content=ft.Stack(layers),
                ),
                ft.Container(
                    margin=ft.Margin.only(top=7),
                    content=ft.Row(
                        reading, spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER
                    ),
                ),
            ],
            spacing=0,
            tight=True,
        ),
    )


def tabs(panel: Panel) -> ft.Control:
    return ft.Container(
        margin=ft.Margin.only(left=PAD, right=PAD, top=0),
        padding=2,
        bgcolor=t().sel,
        border_radius=9,
        content=ft.Row(
            [
                ft.Container(
                    expand=True,
                    padding=ft.Padding(left=0, right=0, top=4, bottom=4),
                    alignment=ft.Alignment.CENTER,
                    bgcolor=t().elev if panel.tab == key else None,
                    border_radius=7,
                    shadow=ft.BoxShadow(
                        offset=ft.Offset(0, 1),
                        blur_radius=2,
                        color=theme.alpha("#000000", 0.16),
                    )
                    if panel.tab == key
                    else None,
                    content=ft.Text(
                        label,
                        style=theme.sans(
                            12.5,
                            t().fg if panel.tab == key else t().fg2,
                            ft.FontWeight.W_500 if panel.tab == key else ft.FontWeight.W_400,
                        ),
                        no_wrap=True,
                    ),
                    on_click=lambda _, key=key: panel.go(key),
                )
                for key, label in TABS
            ],
            spacing=2,
        ),
    )


def empty(message: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=2, right=2, top=26, bottom=0),
        alignment=ft.Alignment.CENTER,
        content=ft.Text(message, style=theme.sans(12.5, t().fg3), text_align=ft.TextAlign.CENTER),
    )

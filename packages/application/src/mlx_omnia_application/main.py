"""The window: a fixed 1160x760 canvas, a 40pt title bar that owns which view is on, and
one stream feeding every screen.

Three things AppKit gives a native window have no Flet equivalent and are named where
they are approximated: the title bar's vibrancy material, the traffic lights placed at
(16, 14), and the system accent.

The screens are components and the state they read is observable, so a resource landing
redraws what reads it and nothing else. Rebuilding a whole tree instead is what Flet
answers with a full remount — which loses, among other things, every scroll position.
"""

from __future__ import annotations

import asyncio

import flet as ft
import flet.canvas as cv

from mlx_omnia_application.api import engine as engine_api
from mlx_omnia_application.api.daemon import Daemon
from mlx_omnia_application.api.downloads import Downloads
from mlx_omnia_application.ui import hooks, memory, parts, theme
from mlx_omnia_application.ui.chrome import Chrome
from mlx_omnia_application.ui.theme import t
from mlx_omnia_application.views import resident
from mlx_omnia_application.views.benchmark import Benchmark
from mlx_omnia_application.views.chat import Chat
from mlx_omnia_application.views.library import Library
from mlx_omnia_application.views.resident import Resident
from mlx_omnia_application.views.settings import Settings

WIDTH, HEIGHT = 1160, 760
TITLE_BAR = 40
# .lights — the 72pt the stylesheet reserves at the head of the bar. AppKit places the
# traffic lights inside it at (16, 14), centring them on the 40pt row; the prebuilt Flet
# client owns that placement and offers no equivalent, so measured they land at
# (16.5, 15.5) — 7pt left and 5pt high. Moving them needs the macOS runner, which is
# `flet build --template-dir` and not anything reachable from here.
LIGHTS = 72

# Drawn here rather than placed, because the prebuilt client offers no placement. Apple's
# own colours and the measure taken off a native window: the group's top-left corner at
# (16, 14), 13pt across, 23pt between centres. Zoom is the grey of a button that is
# not there to be pressed — the window has one size, so it is disabled as it is in AppKit.
CLOSE, MINIMIZE = "#FF5F57", "#FEBC2E"
IDLE_LIGHT = ("#DCDCDC", "#4F5052")
LIGHT_SIZE = 13.5
LIGHT_STEP = 23.0

VIEWS = [
    ("resident", "Resident"),
    ("library", "Library"),
    ("benchmark", "Benchmark"),
    ("chat", "Chat"),
    ("settings", "Settings"),
]


def _glyph(kind: str) -> ft.Control:
    """What a traffic light shows once the pointer is over the group."""
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
    """.lights — the traffic lights, and the only place this app draws AppKit itself.

    The glyphs answer the pointer over the group and not over the button under it, which
    is what every Mac window does; an inactive window shows all three grey.
    """
    over, set_over = ft.use_state(False)
    page = ft.context.page
    idle = IDLE_LIGHT[1] if theme.t() is theme.DARK else IDLE_LIGHT[0]

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
                # AppKit rings each button in a hairline of its own colour, darkened.
                border=theme.hair(theme.alpha("#000000", 0.10)),
                content=_glyph(kind) if over and focused and color is not None else None,
                on_click=None if act is None else (lambda _, act=act: act()),
            )
        )

    # The box is the whole bar so that `top` is the stylesheet's own 14pt and not an
    # offset from wherever a centred row happened to put it.
    return ft.Container(
        width=LIGHTS,
        height=TITLE_BAR,
        content=ft.Stack(
            [
                ft.Container(
                    # x=16, against the bar's own 14pt of padding.
                    left=2,
                    top=14,
                    content=ft.Row(buttons, spacing=LIGHT_STEP - LIGHT_SIZE, tight=True),
                    on_hover=(lambda event: set_over(bool(event.data))) if focused else None,
                )
            ]
        ),
    )


@ft.component
def Tabs(chrome: Chrome) -> ft.Control:
    def tab(identifier: str, title: str) -> ft.Control:
        on = identifier == chrome.view
        return ft.Container(
            content=ft.Text(
                title,
                style=theme.sans(11.5, t().fg if on else t().fg2, ft.FontWeight.W_500),
                no_wrap=True,
            ),
            padding=ft.Padding(left=11, right=11, top=3, bottom=3),
            bgcolor=t().surface2 if on else None,
            border_radius=5,
            shadow=ft.BoxShadow(
                offset=ft.Offset(0, 1), blur_radius=1.5, color=theme.alpha("#000000", 0.16)
            )
            if on
            else None,
            on_click=lambda _, identifier=identifier: setattr(chrome, "view", identifier),
        )

    return ft.Container(
        padding=2,
        bgcolor=t().sunken,
        border=theme.hair(),
        border_radius=7,
        content=ft.Row(
            [tab(identifier, title) for identifier, title in VIEWS], spacing=1, tight=True
        ),
    )


@ft.component
def TitleBar(engine: engine_api.Engine, chrome: Chrome) -> ft.Control:
    down = engine.health is None
    port = "8642"
    if engine.system is not None:
        port = engine.system["constants"].get("port", port)

    right: list[ft.Control] = []
    if chrome.view != "resident":
        right.extend(memory.meter(engine, chrome.ghost if chrome.view == "library" else None))
    right.append(
        parts.chip(
            [
                parts.dot("bad" if down else "ok"),
                parts.chip_text("Engine"),
                ft.Text(f":{port}", style=theme.mono(11, t().fg), no_wrap=True),
            ],
            lambda: setattr(chrome, "view", "settings"),
        )
    )

    bar = ft.Container(
        height=TITLE_BAR,
        padding=ft.Padding(left=14, right=12, top=0, bottom=0),
        # No NSVisualEffectView here: --surface is what the header material resolves to
        # over a plain desktop on both sides of the theme.
        bgcolor=t().surface,
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        # An inactive window's chrome goes quiet, like every AppKit title bar does.
        opacity=1.0 if chrome.focused else 0.72,
        color_filter=None
        if chrome.focused
        else ft.ColorFilter(color="#FF808080", blend_mode=ft.BlendMode.SATURATION),
        content=ft.Row(
            [
                Lights(chrome.focused),
                Tabs(chrome),
                ft.Container(expand=True),
                ft.Row(right, spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ],
            spacing=14,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    return ft.WindowDragArea(content=bar, maximizable=False)


@ft.component
def App(
    engine: engine_api.Engine, chrome: Chrome, downloads: Downloads, daemon: Daemon
) -> ft.Control:
    theme.use(chrome.dark)
    ft.context.page.bgcolor = t().win

    def jump(view: str) -> None:
        chrome.view = view

    if chrome.view == "resident":
        body = Resident(engine, jump)
    elif chrome.view == "library":
        body = Library(engine, chrome, downloads, jump)
    elif chrome.view == "benchmark":
        body = Benchmark(engine)
    elif chrome.view == "chat":
        body = Chat(engine)
    else:
        body = Settings(engine, chrome, daemon)

    return ft.Column([TitleBar(engine, chrome), body], spacing=0, expand=True)


async def _size(page: ft.Page) -> None:
    """One size, both axes: the layout is three measured columns and a memory band drawn
    to scale, and a fixed canvas is what makes the band a real ruler.

    Window properties are pushed to the Flutter side on update and applied there, so they
    do not compose within one batch: a size set beside `resizable = False` is clamped by
    the size the window already had. Hiding the title bar shrinks the window by its
    height, which is why it goes first and the measure after it is the content's.
    """
    window = page.window

    async def settle() -> None:
        page.update()
        await asyncio.sleep(0.2)

    window.title_bar_hidden = True
    # Hidden because this app draws its own: see Lights.
    window.title_bar_buttons_hidden = True
    await settle()

    window.width, window.height = WIDTH, HEIGHT
    await settle()

    window.min_width, window.max_width = WIDTH, WIDTH
    window.min_height, window.max_height = HEIGHT, HEIGHT
    window.resizable = False
    window.maximizable = False
    # Revealed once there is something to show, so the app arrives already drawn rather
    # than as an empty rectangle. Opacity and not `visible`: the client is waited on with
    # `open -W`, and a window that is never presented ends the process.
    window.opacity = 0
    await settle()

    await window.center()


async def main(page: ft.Page) -> None:
    dark = page.platform_brightness is ft.Brightness.DARK
    chrome = Chrome(system_dark=dark, dark=dark)
    theme.use(chrome.dark)

    page.title = "Omnia"
    page.padding = 0
    page.spacing = 0
    page.bgcolor = t().win
    page.theme_mode = ft.ThemeMode.SYSTEM
    # Roboto is what Flutter reaches for when a style names no family, and a bundled face
    # is the first thing that reads as somebody else's toolkit.
    page.theme = ft.Theme(font_family=theme.SANS)
    page.dark_theme = ft.Theme(font_family=theme.SANS)
    page.services.append(parts.CLIPBOARD)

    await _size(page)

    engine = engine_api.Engine()
    downloads = Downloads()
    daemon = Daemon()

    def brightness(_event: object) -> None:
        chrome.system_dark = page.platform_brightness is ft.Brightness.DARK
        if chrome.mode == "system":
            chrome.dark = chrome.system_dark

    page.on_platform_brightness_change = brightness

    # Two signals for one fact, because neither is reliable on its own here: the window's
    # own FOCUS/BLUR is not emitted for a window that opens behind another app, and the
    # lifecycle state is what the platform reports when the app itself moves in and out
    # of the foreground.
    def attend(focused: bool) -> None:
        if chrome.focused != focused:
            chrome.focused = focused

    page.window.on_event = lambda event: (
        attend(event.type is ft.WindowEventType.FOCUS)
        if event.type in (ft.WindowEventType.FOCUS, ft.WindowEventType.BLUR)
        else None
    )
    # Escape closes the sheet on top, which is what a window does and a page does not.
    # `hooks.escape` is where the stack is.
    page.on_keyboard_event = lambda event: (
        hooks.escape() if event.key == "Escape" else None
    )
    page.on_app_lifecycle_state_change = lambda event: (
        attend(event.state is ft.AppLifecycleState.RESUME)
        if event.state in (ft.AppLifecycleState.RESUME, ft.AppLifecycleState.INACTIVE)
        else None
    )

    page.render(App, engine, chrome, downloads, daemon)
    # The reveal is its own batch: the window's properties are applied on the Flutter
    # side, and one set beside the first render is clamped by the state before it.
    await asyncio.sleep(0.2)
    page.window.opacity = 1
    page.update()

    # Not awaited: a daemon that is slow to come up shows on the rail's dot, and the
    # window is drawn either way.
    page.run_task(daemon.boot)
    page.run_task(engine_api.follow, engine)
    page.run_task(downloads.boot)
    page.run_task(resident.trace, engine)


def run() -> None:
    ft.run(main)

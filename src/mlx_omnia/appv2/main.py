"""The Quieta window: the same fixed 1160x760 canvas as v1, with the sidebar owning
which view is on and the Now panel stacked over whatever that is.

The sizing dance is v1's, kept verbatim: window properties are applied on the Flutter
side and do not compose within one batch, so the title bar goes first, the measure
after it, and the reveal is its own batch.
"""

from __future__ import annotations

import asyncio

import flet as ft

from mlx_omnia.appv2 import theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.sidebar import Sidebar
from mlx_omnia.appv2.theme import t
from mlx_omnia.appv2.views.chat import Chat
from mlx_omnia.appv2.views.models import Models
from mlx_omnia.appv2.views.now import Now
from mlx_omnia.appv2.views.settings import Settings

WIDTH, HEIGHT = 1160, 760


@ft.component
def App(shell: Shell) -> ft.Control:
    theme.use(shell.dark)
    ft.context.page.bgcolor = t().win

    if shell.view == "models":
        body: ft.Control = Models(shell)
    elif shell.view == "settings":
        body = Settings(shell)
    else:
        body = Chat(shell)

    window = ft.Row(
        [Sidebar(shell), ft.Container(content=body, expand=True, bgcolor=t().surface)],
        spacing=0,
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )
    layers: list[ft.Control] = [window]
    if shell.now_open:
        layers.append(Now(shell))
    return ft.Stack(layers, expand=True)


async def _size(page: ft.Page) -> None:
    window = page.window

    async def settle() -> None:
        page.update()
        await asyncio.sleep(0.2)

    window.title_bar_hidden = True
    window.title_bar_buttons_hidden = True
    await settle()

    window.width, window.height = WIDTH, HEIGHT
    await settle()

    window.min_width, window.max_width = WIDTH, WIDTH
    window.min_height, window.max_height = HEIGHT, HEIGHT
    window.resizable = False
    window.maximizable = False
    window.opacity = 0
    await settle()

    await window.center()


async def main(page: ft.Page) -> None:
    dark = page.platform_brightness is ft.Brightness.DARK
    shell = Shell(system_dark=dark, dark=dark)
    theme.use(shell.dark)

    page.title = "Omnia"
    page.padding = 0
    page.spacing = 0
    page.bgcolor = t().win
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.theme = ft.Theme(font_family=theme.SANS)
    page.dark_theme = ft.Theme(font_family=theme.SANS)

    await _size(page)

    def brightness(_event: object) -> None:
        shell.system_dark = page.platform_brightness is ft.Brightness.DARK
        if shell.mode == "system":
            shell.dark = shell.system_dark

    page.on_platform_brightness_change = brightness

    def attend(focused: bool) -> None:
        if shell.focused != focused:
            shell.focused = focused

    page.window.on_event = lambda event: (
        attend(event.type is ft.WindowEventType.FOCUS)
        if event.type in (ft.WindowEventType.FOCUS, ft.WindowEventType.BLUR)
        else None
    )
    page.on_app_lifecycle_state_change = lambda event: (
        attend(event.state is ft.AppLifecycleState.RESUME)
        if event.state in (ft.AppLifecycleState.RESUME, ft.AppLifecycleState.INACTIVE)
        else None
    )

    page.on_keyboard_event = lambda event: (
        setattr(shell, "now_open", False) if event.key == "Escape" else None
    )

    page.render(App, shell)
    await asyncio.sleep(0.2)
    page.window.opacity = 1
    page.update()


def run() -> None:
    ft.run(main)

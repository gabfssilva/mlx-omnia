"""The Quieta window: the same fixed 1160x760 canvas as v1, with the sidebar owning
which view is on. The app opens on the Server — Omnia is a server with a chat beside
it, not the other way around.

The wiring is v1's: one Engine filled by the daemon's event stream, a Daemon that
starts the server when nobody answers on the port and owns only the one it started,
and Downloads folding the job streams. The sizing dance is v1's too, kept verbatim.
"""

from __future__ import annotations

import asyncio

import flet as ft

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.daemon import Daemon
from mlx_omnia.app.api.downloads import Downloads
from mlx_omnia.appv2 import runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.sidebar import Sidebar
from mlx_omnia.appv2.theme import t
from mlx_omnia.appv2.views.benchmark import Benchmark
from mlx_omnia.appv2.views.chat import Chat
from mlx_omnia.appv2.views.models import Models
from mlx_omnia.appv2.views.quantize import Quantize
from mlx_omnia.appv2.views.server import Server

WIDTH, HEIGHT = 1160, 760


@ft.component
def App(
    shell: Shell, engine: engine_api.Engine, downloads: Downloads, daemon: Daemon
) -> ft.Control:
    theme.use(shell.dark)
    ft.context.page.bgcolor = t().win

    if shell.view == "chat":
        body: ft.Control = Chat(shell, engine)
    elif shell.view == "models":
        body = Models(shell, engine, downloads)
    elif shell.view == "quantize":
        body = Quantize(shell, engine)
    elif shell.view == "benchmark":
        body = Benchmark(shell, engine)
    else:
        body = Server(shell, engine, daemon)

    return ft.Row(
        [Sidebar(shell, engine), ft.Container(content=body, expand=True, bgcolor=t().surface)],
        spacing=0,
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )


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

    engine = engine_api.Engine()
    downloads = Downloads()
    daemon = Daemon()

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

    page.render(App, shell, engine, downloads, daemon)
    await asyncio.sleep(0.2)
    page.window.opacity = 1
    page.update()

    # Not awaited: a daemon that is slow to come up shows on the sidebar's dot, and the
    # window is drawn either way.
    page.run_task(daemon.boot)
    page.run_task(engine_api.follow, engine)
    page.run_task(downloads.boot)
    page.run_task(runtime.trace, engine)


def run() -> None:
    ft.run(main)

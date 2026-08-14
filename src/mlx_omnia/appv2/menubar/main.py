"""The panel window: a frameless Flet window that the status item moves and shows.

The boot order is the spike's, and every step of it was measured rather than guessed:

- `window.visible = False` is not how this hides. The template answers `true` to
  `applicationShouldTerminateAfterLastWindowClosed`, so a window that is never presented
  ends the process; `opacity` hides without that.
- An invisible window is still a window: at opacity 0 it goes on swallowing every click
  inside its 380×560, and whatever is under it cannot be reached. `ignore_mouse_events`
  travels with the opacity, so the panel is transparent to the pointer exactly while it
  is transparent to the eye.
- `window.skip_task_bar = True` blocks positioning outright — the window then refuses every
  `left`/`top` it is given. `LSUIElement` in Info.plist is what keeps it out of the Dock.
- `window.resizable = False` freezes the window after its first placement, so the panel
  would open under the icon once and never again.
- `always_on_top` is swallowed when it shares a batch with another property. It gets its
  own `page.update()`, once, at boot.

Opening and closing is one batch and no sleeps — position and opacity together — which is
what makes the toggle land in 5–15 ms instead of the ~600 four batches cost.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import flet as ft

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.daemon import ROOT, Daemon, bundled_python
from mlx_omnia.app.api.downloads import Downloads
from mlx_omnia.appv2 import runtime, theme
from mlx_omnia.appv2.menubar import client
from mlx_omnia.appv2.menubar import panel as shell
from mlx_omnia.appv2.menubar.bar import Anchor, Bar
from mlx_omnia.appv2.menubar.panel import HEIGHT, WIDTH, Panel
from mlx_omnia.appv2.menubar.views.bench import Bench
from mlx_omnia.appv2.menubar.views.manage import Manage
from mlx_omnia.appv2.menubar.views.model import ModelScreen
from mlx_omnia.appv2.menubar.views.models import Models
from mlx_omnia.appv2.menubar.views.quantize import Quantize
from mlx_omnia.appv2.theme import t

# A click on the icon that lands within this of the panel losing focus is the click that
# closed it, and must not open it again.
REOPEN_GUARD = 0.25


def open_window() -> None:
    """Bring up the full window, the way the daemon module brings up the engine.

    Outside a bundle this is the console script beside this interpreter, or `uv run` in the
    checkout. Inside one it has nowhere to go yet: the window is the bundle, and a second
    face of the same .app needs a switch in the bundle's own entry point that `mise run dmg`
    does not build. Rather than exec something that is not there, this says so.
    """
    if bundled_python() is not None:
        print("open window: not wired inside the bundle yet")
        return
    beside = pathlib.Path(sys.executable).parent / "omnia-appv2"
    command = (
        [str(beside)] if beside.exists() else ["uv", "run", "--project", str(ROOT), "omnia-appv2"]
    )
    asyncio.get_running_loop().create_task(
        asyncio.create_subprocess_exec(*command, cwd=str(ROOT))
    )


@ft.component
def App(
    panel: Panel,
    engine: engine_api.Engine,
    downloads: Downloads,
    daemon: Daemon,
    on_quit: ft.OptionalControlEventHandler,
) -> ft.Control:
    theme.use(panel.dark)

    def choice(
        label: str, on_click: ft.OptionalControlEventHandler, color: str | None = None
    ) -> ft.PopupMenuItem:
        return ft.PopupMenuItem(
            height=34,
            content=ft.Text(label, style=theme.sans(12.5, color), no_wrap=True),
            on_click=on_click,
        )

    # Every app-level verb lives here and nowhere else. The window is one of them: a link
    # to it on every screen is a link nobody reads by the third screen, and the panel's
    # own job is the four tabs.
    overflow = ft.PopupMenuButton(
        bgcolor=t().elev,
        items=[
            choice("Open Omnia ↗", lambda _: open_window(), t().accent),
            choice("Restart engine", lambda _: runtime.act(daemon.restart())),
            choice("Stop engine", lambda _: runtime.act(daemon.stop()), t().bad),
            *(
                choice(
                    f"Theme: {label}",
                    lambda _, key=key: panel.choose(key),
                    t().accent if panel.mode == key else t().fg2,
                )
                for key, label in (("light", "Light"), ("system", "System"), ("dark", "Dark"))
            ),
            choice("Quit Omnia", on_quit, t().fg3),
        ],
        content=ft.Container(
            width=22,
            height=22,
            border_radius=6,
            bgcolor=t().sel,
            alignment=ft.Alignment.CENTER,
            content=ft.Text("⋯", style=theme.sans(13, t().fg3), no_wrap=True),
        ),
    )

    if panel.opened:
        entry = engine.entry(panel.opened)
        if entry is not None:
            body: ft.Control = ModelScreen(panel, engine, entry)
        else:
            panel.opened = ""
            body = Models(panel, engine, downloads)
    elif panel.tab == "models":
        body = Models(panel, engine, downloads)
    elif panel.tab == "quantize":
        body = Quantize(panel, engine)
    elif panel.tab == "bench":
        body = Bench(panel, engine)
    else:
        body = Manage(panel, engine)

    return ft.Container(
        expand=True,
        bgcolor=t().surface,
        border=theme.hair(t().hair2),
        border_radius=12,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        content=ft.Column(
            [
                shell.head(panel, engine, overflow),
                shell.tabs(panel),
                ft.Container(
                    expand=True,
                    padding=ft.Padding(left=shell.PAD, right=shell.PAD, top=14, bottom=12),
                    content=body,
                ),
            ],
            spacing=0,
            expand=True,
        ),
    )


async def _shape(page: ft.Page) -> None:
    window = page.window

    async def settle() -> None:
        page.update()
        await asyncio.sleep(0.2)

    # First, and in the same breath as the frame: any later and the window is presented
    # at its default size in the middle of the screen before it is hidden, which reads as
    # a slow black blink every launch.
    window.opacity = 0
    window.ignore_mouse_events = True
    window.title_bar_hidden = True
    window.title_bar_buttons_hidden = True
    window.frameless = True
    await settle()

    window.width, window.height = WIDTH, HEIGHT
    await settle()

    window.bgcolor = theme.TRANSPARENT
    window.shadow = True
    await settle()

    # Alone, and once: batched with anything else it is silently dropped and the panel
    # opens behind whatever was in front.
    window.always_on_top = True
    page.update()


async def main(page: ft.Page) -> None:
    dark = page.platform_brightness is ft.Brightness.DARK
    panel = Panel(system_dark=dark, dark=dark)
    theme.use(panel.dark)

    page.title = "Omnia Panel"
    page.padding = 0
    page.spacing = 0
    page.bgcolor = theme.TRANSPARENT
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.theme = ft.Theme(font_family=theme.SANS)
    page.dark_theme = ft.Theme(font_family=theme.SANS)

    await _shape(page)

    engine = engine_api.Engine()
    downloads = Downloads()
    daemon = Daemon()
    hidden_at = 0.0
    # Whether the panel is up is the window's own state and not the view's, and it is set
    # from the status item's thread — writing it on the observable would notify a component
    # from outside any render, which Flet answers by raising rather than redrawing.
    shown = False

    def show(spot: Anchor) -> None:
        nonlocal shown
        left, top = shell.place(*spot)
        window = page.window
        window.left, window.top = left, top
        window.opacity = 1
        window.ignore_mouse_events = False
        window.focused = True
        page.update()
        shown = True

    def hide() -> None:
        nonlocal hidden_at, shown
        if not shown:
            return
        hidden_at = time.monotonic()
        page.window.opacity = 0
        page.window.ignore_mouse_events = True
        page.update()
        shown = False

    def toggle(spot: Anchor) -> None:
        if shown:
            hide()
        elif time.monotonic() - hidden_at > REOPEN_GUARD:
            show(spot)

    bar = Bar(toggle, toggle)

    def quit_all(_event: object = None) -> None:
        bar.stop()
        page.run_task(page.window.close)

    def brightness(_event: object) -> None:
        panel.system_dark = page.platform_brightness is ft.Brightness.DARK
        if panel.mode == "system":
            panel.dark = panel.system_dark

    page.on_platform_brightness_change = brightness

    # Clicking away closes the panel, which is what a menu bar panel does. The guard above
    # is what keeps that from fighting the icon: the click that blurs is often the same one
    # that would reopen.
    page.window.on_event = lambda event: (
        hide() if event.type is ft.WindowEventType.BLUR else None
    )

    page.render(App, panel, engine, downloads, daemon, quit_all)

    async def watch() -> None:
        """Tell the icon how the daemon is, for as long as the panel runs."""
        while True:
            bar.state(engine.health is not None)
            await asyncio.sleep(1.0)

    bar.boot()
    page.run_task(daemon.boot)
    page.run_task(engine_api.follow, engine)
    page.run_task(downloads.boot)
    page.run_task(runtime.trace, engine)
    page.run_task(watch)


def run() -> None:
    client.use()
    ft.run(main)

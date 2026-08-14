"""The window's side of the status item: start it, read it, tell it how the daemon is.

`item.py` is a process because AppKit needs a main thread this one does not have; what
crosses between them is four kinds of line and nothing else, so the panel never links
against AppKit and the item never learns what a model is.

The pipes are `subprocess` and a reader thread, not `asyncio.create_subprocess_exec`. Under
Flet the loop is not the one asyncio assumes, and a spawn asking it for pipe transports
never returns — the child is not started and nothing is raised, so the icon simply never
appears. `daemon.py` never hit this because the engine's output goes to a file and it asks
for no pipes at all. A thread and `call_soon_threadsafe` need nothing from the loop but a
callback.

The interpreter is the same one `mlx_omnia.app.api.daemon` starts the engine under — the
bundle lays a CPython beside the window, and outside a bundle this process's own is right.
`item.py` imports the standard library and nothing more, so either will do.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import subprocess
import sys
import threading
from collections.abc import Callable

from mlx_omnia import paths
from mlx_omnia.app.api.daemon import bundled_python, child_environment

MODULE = "mlx_omnia.appv2.menubar.item"

Anchor = tuple[float, float, float, float]


class Bar:
    """The status item, as the panel sees it.

    `on_toggle` and `on_menu` are handed where the icon is — x, y, width, height, in
    top-left screen coordinates. They are called on the window's own loop, never on the
    reader thread, because what they touch is the window.
    """

    def __init__(
        self,
        on_toggle: Callable[[Anchor], None],
        on_menu: Callable[[Anchor], None],
    ) -> None:
        self.on_toggle = on_toggle
        self.on_menu = on_menu
        self.process: subprocess.Popen[bytes] | None = None
        self.ready = asyncio.Event()
        self._said: bool | None = None

    def boot(self) -> None:
        """Start the item and read it for as long as it runs.

        The item's own stderr goes to the app log rather than to this process's, which has
        nowhere to put it: Flet swallows the window's output, and an icon that did not come
        up is the one failure the panel cannot report on screen — the screen is what the
        icon opens.
        """
        loop = asyncio.get_running_loop()
        interpreter = bundled_python()
        bundled = interpreter is not None
        with paths.app_log().open("ab") as handle:
            self.process = subprocess.Popen(
                [
                    str(interpreter) if interpreter is not None else sys.executable,
                    "-m",
                    MODULE,
                    "--parent-pid",
                    str(os.getpid()),
                ],
                cwd=None if bundled else str(pathlib.Path(__file__).parents[4]),
                # Always, not only in the bundle. Flet's own launcher exports PYTHON* too,
                # and a child interpreter that inherits them dies before it has a frame to
                # blame — `Py_Initialize` fails and nothing reaches stderr.
                env=child_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=handle,
            )
        threading.Thread(target=self._read, args=(loop, self.process), daemon=True).start()

    def _read(self, loop: asyncio.AbstractEventLoop, child: subprocess.Popen[bytes]) -> None:
        assert child.stdout is not None
        for raw in child.stdout:
            line = raw.decode().strip()
            loop.call_soon_threadsafe(self._heard, line)
        # The item outliving the window is the normal end, and it is reached through stdin
        # closing rather than here. Anything else is the icon having gone while the panel
        # still wants it, and the log is the only place that can say so — Flet keeps the
        # window's own output.
        code = child.wait()
        if self.process is child:
            with paths.app_log().open("a") as log:
                log.write(f"the status item stopped on its own, exit {code}\n")

    def _heard(self, line: str) -> None:
        word, _, rest = line.partition(" ")
        if word == "ready":
            self.ready.set()
            return
        if word not in ("toggle", "menu"):
            return
        parts = rest.split()
        if len(parts) != 4:
            return
        spot = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        # One line per click. Flet keeps the window's own output, so without this there is
        # no way to tell a click that never arrived from a panel that opened out of sight.
        with contextlib.suppress(OSError), paths.app_log().open("a") as log:
            log.write(f"{word} at {spot}\n")
        (self.on_toggle if word == "toggle" else self.on_menu)(spot)

    def state(self, ok: bool) -> None:
        """Dim the icon while the daemon is not answering.

        Said only when it changes: the health poll runs on a timer and the item has no
        reason to hear the same thing twice a second.
        """
        if ok == self._said:
            return
        child = self.process
        if child is None or child.stdin is None:
            return
        self._said = ok
        with contextlib.suppress(OSError, ValueError):
            child.stdin.write(f"state {'ok' if ok else 'down'}\n".encode())
            child.stdin.flush()

    def stop(self) -> None:
        """Take the icon out of the bar. Closing stdin is the whole protocol — the item
        reads EOF and terminates itself, which is also what happens if this process dies
        without getting here."""
        child = self.process
        if child is None:
            return
        self.process = None
        with contextlib.suppress(OSError, ValueError):
            if child.stdin is not None:
                child.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(3)

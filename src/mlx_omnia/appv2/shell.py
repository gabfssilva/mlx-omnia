"""What the window knows about itself — the v1 Chrome, with the Agora panel on it.

Its own module because every view reads it and `main` mounts them all: a view importing
`main` for it would be a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft


@ft.observable
@dataclass
class Shell:
    """Observable rather than component state, because the window's own events — focus,
    the system going dark — arrive from outside any render."""

    view: str = "chat"
    focused: bool = True
    mode: str = "system"
    system_dark: bool = False
    dark: bool = False
    now_open: bool = False

    def choose(self, mode: str) -> None:
        self.mode = mode
        self.dark = self.system_dark if mode == "system" else mode == "dark"

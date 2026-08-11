"""What the window knows about itself, and the two things a view tells it.

Its own module because every view reads it and `main` mounts them all: a view importing
`main` for it would be a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft


@ft.observable
@dataclass
class Chrome:
    """Observable rather than component state, because the window's own events — focus,
    the system going dark — arrive from outside any render."""

    view: str = "resident"
    focused: bool = True
    # Nothing stamped is the system following itself, which is what the platform
    # brightness is for; a choice overrides it until it is set back to "system".
    mode: str = "system"
    # What the platform says right now, whether or not the app is following it.
    system_dark: bool = False
    dark: bool = False
    # What Library is pricing against the ceiling, drawn as a ghost on the meter.
    ghost: float | None = None

    def choose(self, mode: str) -> None:
        self.mode = mode
        self.dark = self.system_dark if mode == "system" else mode == "dark"

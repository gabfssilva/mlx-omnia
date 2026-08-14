"""Where Omnia keeps its files.

macOS draws the line between state and logs, and so does this:

- `~/Library/Application Support/mlx-omnia/` holds state — `server.db` with the bench
  history, the job rows and the config, and `daemon.lock`. It is what a user backs up.
- `~/Library/Logs/mlx-omnia/` holds logs, and Console.app lists it under Log Reports, so
  "open Console, look under Log Reports" is an instruction that needs no path in it. It is
  what a user deletes.

One path per file, wherever it is read from. The CLI reads `daemon.log` to say why a daemon
that did not answer stopped and the panel's Settings card names it on screen; a second
definition drifting from the first leaves someone reading a file nobody wrote. The panel is
Swift and spells these itself — `app/Sources/OmniaPanel/Daemon.swift` — so the agreement
between the two is the directory names above, and nothing else.

This module sits at the roof because the CLI and the server both need these and neither may
import the other.
"""

from __future__ import annotations

import os
import pathlib
import shutil

LOGS = pathlib.Path.home() / "Library" / "Logs" / "mlx-omnia"
_SUPPORT = pathlib.Path.home() / "Library" / "Application Support" / "mlx-omnia"
# Where the state lived while the layout followed XDG rather than macOS.
_FORMER = pathlib.Path.home() / ".config" / "mlx_omnia"


def state_dir() -> pathlib.Path:
    """`~/Library/Application Support/mlx-omnia`, or `OMNIA_STATE_DIR` when it is set.

    The override is what tests point at a temp directory; it replaces `XDG_CONFIG_HOME`,
    which named a Linux convention this no longer follows.
    """
    override = os.environ.get("OMNIA_STATE_DIR")
    return pathlib.Path(override) if override else _SUPPORT


def server_db() -> pathlib.Path:
    return state_dir() / "server.db"


def daemon_lock() -> pathlib.Path:
    return state_dir() / "daemon.lock"


def _log(name: str) -> pathlib.Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / name


def daemon_log() -> pathlib.Path:
    """The engine's output, whoever started it."""
    return _log("daemon.log")


def adopt_former_state() -> pathlib.Path | None:
    """Move state left behind by the XDG layout, and answer where it came from.

    `server.db` is bench history and job rows — a user's own measurements, not a cache. A
    silent change of address would read as having lost them. Only ever moves into a
    directory that is not there yet, so a run that has already written the new one is never
    overwritten by the old.
    """
    destination = state_dir()
    if destination.exists() or not _FORMER.is_dir():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_FORMER), str(destination))
    return _FORMER

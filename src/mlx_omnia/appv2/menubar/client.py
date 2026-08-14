"""The Flet client, marked as an accessory.

A menu bar app has no Dock tile, no application menu, and no place in ⌘-Tab. On macOS that
is one Info.plist key, `LSUIElement`, and it belongs to the bundle that owns the process —
which for a Flet window is the client, not this interpreter. Run from a checkout the client
is the shared one under `~/.flet/client`, and patching that would make every Flet app on the
machine an accessory. So the client is cloned once into the app's own state directory,
patched there, and pointed at with `FLET_VIEW_PATH`.

The clone is `cp -c`: on APFS that is copy-on-write, so the 134 MB costs no disk until one
of the two changes. Re-signing is required and ad-hoc is enough — the signature breaks the
moment Info.plist does, and an unsigned bundle will not launch.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

from mlx_omnia import paths

CACHE = pathlib.Path.home() / ".flet" / "client"
PLIST = "Contents/Info.plist"


def _newest_cached() -> pathlib.Path | None:
    """The newest `Flet.app` flet has cached. Whichever version this interpreter would have
    launched is already down there — `ft.run` puts it there on first use."""
    found = sorted(CACHE.glob("flet-desktop-*/Flet.app"))
    return found[-1] if found else None


def accessory() -> pathlib.Path | None:
    """The directory to hand `FLET_VIEW_PATH`, or None when there is no client to clone.

    None is not a failure to hide: it means flet has not cached a client yet, and the run
    that follows will fetch one and come up as an ordinary app. The next run finds it.
    """
    source = _newest_cached()
    if source is None:
        return None
    ours = paths.state_dir() / "menubar-client"
    app = ours / "Flet.app"
    if app.is_dir() and app.stat().st_mtime >= source.stat().st_mtime:
        return ours

    ours.mkdir(parents=True, exist_ok=True)
    if app.exists():
        subprocess.run(["rm", "-rf", str(app)], check=True)
    subprocess.run(["cp", "-Rc", str(source), str(app)], check=True)
    plist = app / PLIST
    # `Add` fails when the key is already there, which a re-clone never has.
    subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", "Add :LSUIElement bool true", str(plist)], check=True
    )
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
    return ours


def use() -> None:
    """Point this process at the accessory client, and step out of the checkout.

    The second half is not decoration. `flet_desktop` looks for `build/<platform>/*.app`
    under the *current working directory* before it reads `FLET_VIEW_PATH`, so a panel
    started from a checkout that has ever run `flet build` silently gets the packaged app
    as its client — which is a whole second Omnia, with a Dock tile of its own. Nothing here
    resolves a path against the working directory; the child processes are all given theirs.
    """
    ours = accessory()
    if ours is not None:
        os.environ["FLET_VIEW_PATH"] = str(ours)
    os.chdir(paths.state_dir())

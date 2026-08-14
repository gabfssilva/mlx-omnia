"""Sparkle, started by hand from the window's own process.

`flet build` has no way to say "copy this framework into the bundle", so `Sparkle.framework`
is laid into `Contents/Frameworks` after the build and signed with everything else —
`mise run dmg:sparkle`. What is left here is to load it and hold on to an updater. The
framework reads the rest off `Info.plist`, the feed URL and the public key among them, and
this module never names them.

The bridge is ctypes rather than pyobjc, and that is a build constraint and not a taste:
`pyobjc-core` ships a `PyObjCTest` package with a `.dSYM` inside it, and Xcode's strip
phase over the packaged site-packages dies on the DWARF file — *string table not at the
end of the file* — taking `flet build` with it. Four selectors are cheaper than a wheel the
build cannot swallow, and the window's dependencies stay the two it declares.

Two more things shape the code:

- serious_python runs this interpreter on a thread that is not the main one, and
  `SPUStandardUpdaterController` reaches AppKit while it initialises. The construction goes
  to the main queue, so `start` returns before the updater exists.
- Nothing here may be collected. `dispatch_async_f` does not retain the function it is
  handed, and an updater released along with the last reference to it stops checking
  without saying so — which is why both are held in module globals.

Outside a bundle there is nothing to load and nothing to replace: run from a checkout the
window is a process `uv` started, and an updater there would be offering to overwrite a
working tree.
"""

from __future__ import annotations

import ctypes
import pathlib
from collections.abc import Callable

from mlx_omnia import paths

FRAMEWORK = pathlib.PurePath("Contents/Frameworks/Sparkle.framework")
# The Mach-O inside the bundle. Opening it is what registers Sparkle's classes with the
# ObjC runtime; `NSBundle -load` would do the same thing through one more layer.
BINARY = pathlib.PurePath("Versions/Current/Sparkle")

_OBJC = "/usr/lib/libobjc.dylib"
_SYSTEM = "/usr/lib/libSystem.B.dylib"

_controller: int | None = None
_blocks: list[object] = []


def framework() -> pathlib.Path | None:
    """The Sparkle laid into the running bundle, or None when there is none to load."""
    app = paths.bundle()
    if app is None:
        return None
    embedded = app / FRAMEWORK
    return embedded if embedded.is_dir() else None


def _selector(name: bytes) -> int:
    objc = ctypes.CDLL(_OBJC)
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    return objc.sel_registerName(name)


def _message(restype: type | None, *argtypes: type) -> Callable[..., int | None]:
    """An `objc_msgSend` for one signature, and one only.

    A message's ABI is its method's, so the prototype cannot be shared: setting argtypes
    on a single function object and calling it again with different arguments is how
    ctypes turns a mismatched signature into a crash instead of an error.
    """
    objc = ctypes.CDLL(_OBJC)
    return ctypes.CFUNCTYPE(restype, *argtypes)(("objc_msgSend", objc))


def _on_main(work: Callable[[], None]) -> None:
    """Run `work` where AppKit is."""
    system = ctypes.CDLL(_SYSTEM)
    block = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _context: work())
    _blocks.append(block)

    # `_dispatch_main_q` is the queue itself and not a pointer to it, so what is passed is
    # the address of the symbol.
    queue = ctypes.c_void_p.in_dll(system, "_dispatch_main_q")
    system.dispatch_async_f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    system.dispatch_async_f.restype = None
    system.dispatch_async_f(ctypes.addressof(queue), None, ctypes.cast(block, ctypes.c_void_p))


def start() -> bool:
    """Load Sparkle and put a standard updater on the main queue.

    Answers whether there was a framework to load, and not whether the updater came up:
    the construction runs later, on another thread. What it has to say goes to the app
    log, which is where a bundled window's output goes anyway.
    """
    embedded = framework()
    if embedded is None:
        return False

    try:
        ctypes.CDLL(str(embedded / BINARY))

        objc = ctypes.CDLL(_OBJC)
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        updater = objc.objc_getClass(b"SPUStandardUpdaterController")
        if not updater:
            raise LookupError("the framework loaded without SPUStandardUpdaterController")

        alloc = _message(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        init = _message(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        # `startingUpdater:` is what schedules the first check; the two delegates are the
        # customisation this does not take, so the standard UI answers for both.
        started = _selector(b"initWithStartingUpdater:updaterDelegate:userDriverDelegate:")

        def raise_updater() -> None:
            global _controller
            _controller = init(alloc(updater, _selector(b"alloc")), started, True, None, None)

        _on_main(raise_updater)
    except Exception as error:  # noqa: BLE001 — no updater is not a reason not to open
        # A framework that refuses to load is a window with no updater behind it, not a
        # window that fails to open — dyld turning one down over a signature is exactly the
        # case, and the app is still the app. The message goes to the log with the rest.
        print(f"sparkle did not start: {error}")
        return False
    return True


def check() -> bool:
    """Ask for a check now, the way a Check for Updates… menu item would.

    False when no updater is running, which is every case where `start` answered False and
    the window between that call and the main queue draining it.
    """
    controller = _controller
    if controller is None:
        return False
    ask = _message(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    _on_main(lambda: ask(controller, _selector(b"checkForUpdates:"), None))
    return True

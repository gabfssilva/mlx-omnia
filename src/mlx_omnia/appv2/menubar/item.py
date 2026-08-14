"""The status item, and nothing else. A process whose main thread is a run loop.

It draws no panel and knows no daemon: every click it prints where the icon is, in
top-left screen coordinates, which is what the window side needs to anchor to. Whoever
reads it is the Python process that owns the panel.

Why a process of its own, rather than the window's. `NSStatusItem` needs AppKit on a main
thread with a run loop, and the window has neither to lend:

- from a checkout, `ft.run` puts asyncio on the main thread and the Flutter client is a
  separate process entirely — there is no run loop here to attach to;
- from the bundle, serious_python runs the interpreter on a thread that is not the main
  one, as `mlx_omnia.app.updates` documents, so AppKit is reachable only by hopping to the
  main queue of a process this code does not drive.

Neither shape is the same as the other, and a status item that only works in one of them
is worse than one process that works in both. Here the main thread is ours, `[NSApp run]`
is the whole program, and the two environments differ by nothing.

The bridge is ctypes rather than pyobjc for the reason `updates.py` gives — `pyobjc-core`
ships a `.dSYM` that kills Xcode's strip phase during `flet build` — and here it buys more
than it costs anyway: this module imports the standard library and nothing else, so it runs
under any interpreter the bundle happens to have beside it.

Spoken on stdout, one per line:

    ready                      the icon is up
    toggle <x> <y> <w> <h>     left mouse down
    menu <x> <y> <w> <h>       right mouse up
    gone                       the run loop ended

Heard on stdin:

    state ok | down            whether the daemon is answering; the icon dims when it is not
    click                      press the button as a mouse would, for tests
    quit                       stand down

The window is watched by pid — `--parent-pid`, as the engine is — and not by stdin reaching
EOF. Under Flet the child's stdin is at EOF from the start, and an item that reads that as
"the window has gone" leaves the bar milliseconds after arriving, exit 0, saying nothing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
import time

# ── the objc bridge ──────────────────────────────────────────────────────

_OBJC = ctypes.CDLL("/usr/lib/libobjc.dylib")
_SYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
# Without this every AppKit class below is nil, and messaging nil in Objective-C is a
# silent no-op returning nil: the item reports itself ready, draws nothing, and the run
# loop it never started returns at once. Loading the framework is what registers the
# classes with the runtime — `_class` raising is the second half of the same lesson.
_APPKIT = ctypes.util.find_library("AppKit")
ctypes.CDLL(_APPKIT or "/System/Library/Frameworks/AppKit.framework/AppKit")

_OBJC.sel_registerName.argtypes = [ctypes.c_char_p]
_OBJC.sel_registerName.restype = ctypes.c_void_p
_OBJC.objc_getClass.argtypes = [ctypes.c_char_p]
_OBJC.objc_getClass.restype = ctypes.c_void_p
_OBJC.objc_allocateClassPair.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_OBJC.objc_allocateClassPair.restype = ctypes.c_void_p
_OBJC.class_addMethod.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
]
_OBJC.class_addMethod.restype = ctypes.c_bool
_OBJC.objc_registerClassPair.argtypes = [ctypes.c_void_p]
_OBJC.objc_registerClassPair.restype = None


class NSPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class NSSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class NSRect(ctypes.Structure):
    _fields_ = [("origin", NSPoint), ("size", NSSize)]


def _sel(name: bytes) -> ctypes.c_void_p:
    return ctypes.c_void_p(_OBJC.sel_registerName(name))


def _class(name: bytes) -> ctypes.c_void_p:
    """A class, or a refusal. Never a nil that goes on to swallow every message sent to
    it — that failure looks exactly like success from Python."""
    found = _OBJC.objc_getClass(name)
    if not found:
        raise LookupError(f"{name.decode()} is not registered — was its framework loaded?")
    return ctypes.c_void_p(found)


def _message(restype: type | None, *argtypes: type):
    """An `objc_msgSend` for one signature, and one only — `updates.py`'s rule.

    A message's ABI is its method's, so setting argtypes on a shared function object and
    calling it with a different signature is how ctypes turns a mismatch into a crash
    instead of an error.
    """
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
        ("objc_msgSend", _OBJC)
    )


_VOID = ctypes.c_void_p

_send = _message(_VOID)
_send_double = _message(_VOID, ctypes.c_double)
_send_object = _message(_VOID, _VOID)
_send_two = _message(_VOID, _VOID, _VOID)
_send_index = _message(_VOID, ctypes.c_ulong)
_void_object = _message(None, _VOID)
_void_bool = _message(None, ctypes.c_bool)
_void_long = _message(None, ctypes.c_long)
_void_double = _message(None, ctypes.c_double)
_send_long = _message(ctypes.c_long)
_send_ulong = _message(ctypes.c_ulong)
_send_rect = _message(NSRect)


def _string(text: str) -> ctypes.c_void_p:
    return _send_object(_class(b"NSString"), _sel(b"stringWithUTF8String:"), text.encode())


# Nothing constructed here may be collected: `class_addMethod` keeps the address of the
# thunk and not the object that owns it, and a dispatched block is not retained either.
_held: list[object] = []



def _on_main(work) -> None:
    """Run `work` on the main queue, which is where this process's run loop is."""
    block = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _context: work())
    _held.append(block)
    queue = ctypes.c_void_p.in_dll(_SYSTEM, "_dispatch_main_q")
    _SYSTEM.dispatch_async_f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _SYSTEM.dispatch_async_f.restype = None
    _SYSTEM.dispatch_async_f(
        ctypes.addressof(queue), None, ctypes.cast(block, ctypes.c_void_p)
    )


# ── geometry ─────────────────────────────────────────────────────────────

SYMBOL = "waveform.path.ecg"
# NSEventMaskLeftMouseDown | NSEventMaskRightMouseUp. Down and not up on the left, because
# down is when every menu in the bar opens; waiting for the up spends whatever the finger
# takes to lift, and that reads as the app being slow.
MASK = (1 << 1) | (1 << 4)
RIGHT_MOUSE_UP = 4
ACCESSORY = 1  # NSApplicationActivationPolicyAccessory
VARIABLE_LENGTH = -1.0
DIMMED = 0.45


def _primary_height() -> float:
    """The height of the screen whose origin is (0,0).

    Cocoa's global space has its origin at the bottom-left of that screen with y going up;
    the window side places in top-left coordinates with y going down. This is the number
    that converts between them, and it is the primary's height even when the menu bar is
    on another display — where the answer comes out negative, correctly, because that
    display sits above the primary.
    """
    screens = _send(_class(b"NSScreen"), _sel(b"screens"))
    count = _send_long(screens, _sel(b"count"))
    for index in range(count):
        frame = _send_rect(_send_index(screens, _sel(b"objectAtIndex:"), index), _sel(b"frame"))
        if frame.origin.x == 0.0 and frame.origin.y == 0.0:
            return frame.size.height
    if count:
        return _send_rect(
            _send_index(screens, _sel(b"objectAtIndex:"), 0), _sel(b"frame")
        ).size.height
    return 0.0


def _anchor(button: ctypes.c_void_p) -> tuple[float, float, float, float] | None:
    """Where the icon is, top-left, or None before AppKit has laid the button out.

    Inside `applicationDidFinishLaunching` the button measures 32x0 at the origin; at click
    time it is always right. Anyone reading this outside a click has to allow for that.
    """
    window = _send(button, _sel(b"window"))
    if not window:
        return None
    frame = _send_rect(window, _sel(b"frame"))
    if frame.size.height == 0.0:
        return None
    top = _primary_height() - frame.origin.y - frame.size.height
    return frame.origin.x, top, frame.size.width, frame.size.height


def _parent_pid() -> int | None:
    """`--parent-pid <n>`, the way the engine is told the same thing."""
    if "--parent-pid" in sys.argv:
        spot = sys.argv.index("--parent-pid") + 1
        if spot < len(sys.argv):
            return int(sys.argv[spot])
    return None


def _say(line: str) -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


# ── the item ─────────────────────────────────────────────────────────────


def run() -> None:
    application = _send(_class(b"NSApplication"), _sel(b"sharedApplication"))
    _void_long(application, _sel(b"setActivationPolicy:"), ACCESSORY)
    # Before the status item and not after: AppKit hands out a menu bar slot to an
    # application that has finished launching, and the Swift item this replaces got that
    # for free by building itself inside `applicationDidFinishLaunching`.
    _send(application, _sel(b"finishLaunching"))

    bar = _send(_class(b"NSStatusBar"), _sel(b"systemStatusBar"))
    item = _send_double(bar, _sel(b"statusItemWithLength:"), VARIABLE_LENGTH)
    button = _send(item, _sel(b"button"))

    image = _send_two(
        _class(b"NSImage"),
        _sel(b"imageWithSystemSymbolName:accessibilityDescription:"),
        _string(SYMBOL),
        _string("Omnia"),
    )
    # A template image is the one macOS recolours for the bar it lands on — light, dark,
    # and the inverse it wears while a menu is open.
    _void_bool(image, _sel(b"setTemplate:"), True)
    _void_object(button, _sel(b"setImage:"), image)

    def clicked(_self: int, _cmd: int, _sender: int) -> None:
        event = _send(application, _sel(b"currentEvent"))
        kind = _send_ulong(event, _sel(b"type")) if event else 0
        spot = _anchor(button)
        if spot is None:
            return
        verb = "menu" if kind == RIGHT_MOUSE_UP else "toggle"
        x, y, width, height = spot
        _say(f"{verb} {x:.1f} {y:.1f} {width:.1f} {height:.1f}")

    thunk = ctypes.CFUNCTYPE(None, _VOID, _VOID, _VOID)(clicked)
    _held.append(thunk)
    target_class = _OBJC.objc_allocateClassPair(_class(b"NSObject"), b"OmniaBarTarget", 0)
    if not target_class:
        raise RuntimeError("objc_allocateClassPair refused OmniaBarTarget")
    if not _OBJC.class_addMethod(
        target_class, _sel(b"clicked:"), ctypes.cast(thunk, _VOID), b"v@:@"
    ):
        raise RuntimeError("class_addMethod refused clicked:")
    _OBJC.objc_registerClassPair(target_class)
    target = _send(_send(ctypes.c_void_p(target_class), _sel(b"alloc")), _sel(b"init"))
    _held.append(target)

    _void_object(button, _sel(b"setTarget:"), target)
    _void_object(button, _sel(b"setAction:"), _sel(b"clicked:"))
    _void_long(button, _sel(b"sendActionOn:"), MASK)

    def stand_down() -> None:
        _on_main(lambda: _void_object(application, _sel(b"terminate:"), None))

    def listen() -> None:
        """stdin, on a thread of its own. Every line it acts on hops to the main queue,
        because AppKit is not this thread's to touch."""
        for line in sys.stdin:
            word, _, rest = line.strip().partition(" ")
            if word == "state":
                alpha = 1.0 if rest == "ok" else DIMMED
                _on_main(lambda alpha=alpha: _void_double(button, _sel(b"setAlphaValue:"), alpha))
            elif word == "click":
                # The button's own action path, driven without a mouse. This is how the
                # wiring is tested: everything after `performClick:` is what a real click
                # takes, target and action and all.
                _on_main(lambda: _void_object(button, _sel(b"performClick:"), None))
            elif word == "quit":
                stand_down()
                return

    def watch(parent: int) -> None:
        """The window's pid, checked once a second — the daemon's own rule.

        The lifetime does not hang off stdin reaching EOF. Under Flet it reaches EOF at
        once, and an item that treats that as "the window has gone" takes itself out of the
        bar a few milliseconds after arriving, cleanly and with nothing to show for it.
        """
        while True:
            try:
                os.kill(parent, 0)
            except OSError:
                stand_down()
                return
            time.sleep(1.0)

    threading.Thread(target=listen, daemon=True).start()
    parent = _parent_pid()
    if parent is not None:
        threading.Thread(target=watch, args=(parent,), daemon=True).start()

    _say("ready")

    # `[NSApp run]` and not a hand-driven NSRunLoop. The two look interchangeable and are
    # not: the run loop services timers and sources, while draining the AppKit event queue —
    # `nextEventMatchingMask:` then `sendEvent:` — is what `run` adds. Pumping the run loop
    # alone keeps the process up and the icon drawn, and delivers no clicks at all, which
    # reads as an item that is there and dead. `performClick:` still works under it, because
    # that path never touches the queue; that is the trap, and it is why the test hook above
    # is not evidence on its own.
    _send(application, _sel(b"run"))
    _say("gone")


if __name__ == "__main__":
    run()

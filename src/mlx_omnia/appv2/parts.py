"""The Quieta controls: push buttons, state pills, facts, the scope and the search box.

Material's widgets are not used, for the v1 reason — a ripple and Roboto read as
somebody else's toolkit. Everything is a Container with the proposal's own measures:
controls are 30 pt tall against v1's 26, and no face is smaller than 11 pt.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from mlx_omnia.appv2 import theme
from mlx_omnia.appv2.theme import t


def press(
    build: Callable[[str | None], ft.Container],
    on_click: Callable[[], None] | None,
    resting: str | None,
    hovered: str | None,
    pressed: str | None,
) -> ft.Control:
    if on_click is None:
        return build(resting)
    return _Press(build, on_click, resting, hovered, pressed)


@ft.component
def _Press(
    build: Callable[[str | None], ft.Container],
    on_click: Callable[[], None],
    resting: str | None,
    hovered: str | None,
    pressed: str | None,
) -> ft.Control:
    """The v1 pressable: which of the three phases the pointer is in, held as state and
    spent on a rebuild, because a control already patched into the tree is frozen."""
    phase, set_phase = ft.use_state("rest")
    body = build(pressed if phase == "press" else (hovered if phase == "hover" else resting))

    hold = ft.GestureDetector(
        content=body,
        expand=body.expand,
        on_tap_down=lambda _: set_phase("press"),
        on_tap_up=lambda _: (set_phase("rest"), on_click()),
        on_tap_cancel=lambda _: set_phase("rest"),
    )
    if hovered is None:
        return hold
    return ft.Container(
        content=hold,
        expand=body.expand,
        on_hover=lambda e: set_phase("hover" if e.data else "rest"),
    )


def button(
    label: str,
    on_click: Callable[[], None] | None = None,
    kind: str = "",
    expand: bool = False,
) -> ft.Control:
    """A push button at the new scale: height 30, radius 8, 13 pt."""
    classes = kind.split()
    primary = "primary" in classes
    danger = "danger" in classes
    face = t().on_accent if primary else (t().bad if danger else t().fg)

    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            content=ft.Text(
                label,
                style=theme.sans(13, face, ft.FontWeight.W_600 if primary else ft.FontWeight.W_500),
                no_wrap=True,
            ),
            height=30,
            padding=ft.Padding.symmetric(horizontal=14),
            alignment=ft.Alignment.CENTER,
            bgcolor=tint,
            border=None if primary else theme.hair(t().hair2),
            border_radius=8,
            expand=expand,
        )

    resting = t().accent if primary else t().elev
    return press(
        body,
        on_click,
        resting,
        None,
        theme.mix("#000000", 0.12, t().accent) if primary else theme.mix(t().fg, 0.09, t().elev),
    )


def pill(label: str, kind: str) -> ft.Container:
    """A state, worn as a pill: identity never — the materials do that."""
    if kind == "resident":
        fg, bg, border = t().accent, t().accent_soft, None
    elif kind == "hub":
        fg, bg, border = t().fg3, None, theme.hair()
    else:
        fg, bg, border = t().fg2, t().sel, None
    return ft.Container(
        content=ft.Text(label, style=theme.sans(11.5, fg, ft.FontWeight.W_500), no_wrap=True),
        padding=ft.Padding(left=10, right=10, top=3, bottom=3),
        bgcolor=bg,
        border=border,
        border_radius=999,
    )


def fact(label: str, value: str) -> ft.Column:
    """An eyebrow over a measure — the key/value of the proposal."""
    return ft.Column(
        [theme.eyebrow(label), ft.Text(value, style=theme.mono(13), no_wrap=True)],
        spacing=3,
        tight=True,
    )


def dot(state: str = "ok") -> ft.Container:
    colors = {"ok": t().ok, "warn": t().warn, "bad": t().bad}
    return ft.Container(width=7, height=7, border_radius=4, bgcolor=colors.get(state, t().fg3))


def scope(
    options: list[tuple[str, str]],
    chosen: str,
    on_pick: Callable[[str], None],
) -> ft.Container:
    """The segmented control, at 12.5 pt."""

    def one(key: str, label: str) -> ft.Control:
        on = key == chosen
        return ft.Container(
            padding=ft.Padding(left=13, right=13, top=4, bottom=4),
            alignment=ft.Alignment.CENTER,
            bgcolor=t().elev if on else None,
            border_radius=7,
            shadow=ft.BoxShadow(
                offset=ft.Offset(0, 1), blur_radius=2, color=theme.alpha("#000000", 0.16)
            )
            if on
            else None,
            content=ft.Text(
                label,
                style=theme.sans(
                    12.5,
                    t().fg if on else t().fg2,
                    ft.FontWeight.W_500 if on else ft.FontWeight.W_400,
                ),
                no_wrap=True,
            ),
            on_click=lambda _, key=key: on_pick(key),
        )

    return ft.Container(
        padding=2,
        bgcolor=t().sel,
        border_radius=9,
        content=ft.Row([one(key, label) for key, label in options], spacing=2, tight=True),
    )


def searchbox(placeholder: str, value: str, on_change: Callable[[str], None]) -> ft.Control:
    """The v1 search box, one size up: a field with a magnifier, 32 pt tall.

    The field is controlled: every render hands the current value back, because this
    Flet's controls are frozen — the state is the only place the text lives, and a
    rebuild without it erases what was typed."""
    glass = cv.Canvas(
        [
            cv.Circle(
                5.2,
                5.2,
                3.8,
                paint=ft.Paint(color=t().fg3, stroke_width=1.3, style=ft.PaintingStyle.STROKE),
            ),
            cv.Line(
                8.2,
                8.2,
                11.2,
                11.2,
                paint=ft.Paint(
                    color=t().fg3,
                    stroke_width=1.3,
                    style=ft.PaintingStyle.STROKE,
                    stroke_cap=ft.StrokeCap.ROUND,
                ),
            ),
        ],
        width=12,
        height=12,
    )
    return ft.Container(
        height=32,
        padding=ft.Padding(left=11, right=11, top=0, bottom=0),
        bgcolor=t().field,
        border=theme.hair(t().hair2),
        border_radius=9,
        content=ft.Row(
            [
                glass,
                ft.TextField(
                    expand=True,
                    value=value,
                    text_style=theme.sans(13),
                    hint_text=placeholder,
                    hint_style=theme.sans(13, t().fg3),
                    border=ft.InputBorder.NONE,
                    content_padding=ft.Padding.all(0),
                    cursor_color=t().accent,
                    cursor_width=1,
                    dense=True,
                    on_change=lambda e: on_change(e.control.value or ""),
                ),
            ],
            spacing=9,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def toggle(on: bool, on_flip: Callable[[], None] | None = None) -> ft.Container:
    """A switch; hand it on_flip and it answers the pointer."""
    knob = ft.Container(width=16, height=16, border_radius=8, bgcolor="#FFFFFF")
    return ft.Container(
        width=36,
        height=20,
        border_radius=10,
        bgcolor=t().accent if on else t().sel,
        padding=2,
        alignment=ft.Alignment.CENTER_RIGHT if on else ft.Alignment.CENTER_LEFT,
        content=knob,
        on_click=None if on_flip is None else (lambda _: on_flip()),
    )


def chip(text: str) -> ft.Container:
    """A quiet measure worn inline — the decode chip on a model's row."""
    return ft.Container(
        content=ft.Text(text, style=theme.mono(11, t().fg2), no_wrap=True),
        padding=ft.Padding(left=8, right=8, top=2, bottom=2),
        bgcolor=t().sel,
        border_radius=6,
    )


def glyph(kind: str, color: str, size: float = 13.0) -> ft.Control:
    """The sidebar's thin-stroke marks: pulse, bubble, stack, grain, dial."""
    paint = ft.Paint(
        color=color,
        stroke_width=1.2,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    shapes: list[cv.Shape]
    if kind == "pulse":
        shapes = [
            cv.Path(
                [
                    cv.Path.MoveTo(1.5, 8),
                    cv.Path.LineTo(4, 8),
                    cv.Path.LineTo(5.5, 4),
                    cv.Path.LineTo(7.5, 11),
                    cv.Path.LineTo(9, 8),
                    cv.Path.LineTo(11.5, 8),
                ],
                paint=paint,
            )
        ]
    elif kind == "bubble":
        shapes = [
            cv.Path(
                [
                    cv.Path.MoveTo(2, 2.5),
                    cv.Path.LineTo(11, 2.5),
                    cv.Path.LineTo(11, 9),
                    cv.Path.LineTo(5.5, 9),
                    cv.Path.LineTo(3, 11.5),
                    cv.Path.LineTo(3, 9),
                    cv.Path.LineTo(2, 9),
                    cv.Path.Close(),
                ],
                paint=paint,
            )
        ]
    elif kind == "stack":
        shapes = [
            cv.Path(
                [
                    cv.Path.MoveTo(2, 2),
                    cv.Path.LineTo(11, 2),
                    cv.Path.LineTo(11, 11),
                    cv.Path.LineTo(2, 11),
                    cv.Path.Close(),
                ],
                paint=paint,
            ),
            cv.Line(2, 5, 11, 5, paint=paint),
            cv.Line(2, 8, 11, 8, paint=paint),
        ]
    elif kind == "grain":
        shapes = [
            cv.Path(
                [
                    cv.Path.MoveTo(2, 2),
                    cv.Path.LineTo(11, 2),
                    cv.Path.LineTo(11, 11),
                    cv.Path.LineTo(2, 11),
                    cv.Path.Close(),
                ],
                paint=paint,
            ),
            cv.Path(
                [
                    cv.Path.MoveTo(5.1, 5.1),
                    cv.Path.LineTo(7.9, 5.1),
                    cv.Path.LineTo(7.9, 7.9),
                    cv.Path.LineTo(5.1, 7.9),
                    cv.Path.Close(),
                ],
                paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL),
            ),
        ]
    else:  # dial
        shapes = [
            cv.Arc(2, 5, 9, 9, math.pi, math.pi, paint=paint),
            cv.Line(6.5, 9.5, 9.2, 6.2, paint=paint),
        ]
    return cv.Canvas(shapes, width=size, height=size)


@ft.component
def Slider(
    width: float,
    fraction: float,
    on_fraction: Callable[[float], None],
) -> ft.Control:
    """A thin native-looking slider: a filled track and a knob, dragged by fraction."""

    def place(x: float) -> None:
        on_fraction(min(1.0, max(0.0, x / width)))

    knob = ft.Container(
        width=14,
        height=14,
        border_radius=7,
        bgcolor="#FFFFFF",
        border=theme.hair(t().hair2),
        shadow=ft.BoxShadow(
            offset=ft.Offset(0, 1), blur_radius=2, color=theme.alpha("#000000", 0.25)
        ),
    )
    track = ft.Stack(
        [
            ft.Container(
                top=5, left=0, width=width, height=4, border_radius=2, bgcolor=t().sel
            ),
            ft.Container(
                top=5,
                left=0,
                width=max(2.0, fraction * width),
                height=4,
                border_radius=2,
                bgcolor=t().accent,
            ),
            ft.Container(top=0, left=min(width - 14, max(0.0, fraction * width - 7)), content=knob),
        ],
        width=width,
        height=14,
    )
    return ft.GestureDetector(
        content=track,
        on_tap_down=lambda event: place(event.local_position.x),
        on_pan_update=lambda event: place(event.local_position.x),
    )


def stepper(on_minus: Callable[[], None], on_plus: Callable[[], None]) -> ft.Container:
    """The − / + pair, joined the way AppKit joins them."""

    def half(mark: str, act: Callable[[], None], left: bool) -> ft.Container:
        return ft.Container(
            width=26,
            height=22,
            alignment=ft.Alignment.CENTER,
            border=ft.Border.only(right=ft.BorderSide(1, t().hair)) if left else None,
            content=ft.Text(mark, style=theme.sans(13, t().fg2, ft.FontWeight.W_500)),
            on_click=lambda _: act(),
        )

    return ft.Container(
        bgcolor=t().elev,
        border=theme.hair(t().hair2),
        border_radius=6,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        content=ft.Row(
            [half("−", on_minus, True), half("+", on_plus, False)], spacing=0, tight=True
        ),
    )

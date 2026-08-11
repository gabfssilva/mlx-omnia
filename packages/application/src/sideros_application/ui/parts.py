"""The controls theme.css draws by hand: chips, push buttons, panes, key/value rows.

Material's own widgets are not used anywhere in this app. An ElevatedButton arrives with
a ripple, a hover elevation and Roboto, which is the same failure a bundled face is — it
reads as somebody else's toolkit. Everything below is a Container with the stylesheet's
own measures on it.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from sideros_application.ui import theme
from sideros_application.ui.theme import lift, t


def _pressable(
    build: Callable[[str | None], ft.Container],
    on_click: Callable[[], None] | None,
    resting: str | None,
    hovered: str | None,
    pressed: str | None,
) -> ft.Control:
    """Push buttons answer the click, not the mouse passing by: the borderless ones are
    toolbar items and keep their hover tint, which is what AppKit does too."""
    if on_click is None:
        return build(resting)
    return _Pressable(build, on_click, resting, hovered, pressed)


@ft.component
def _Pressable(
    build: Callable[[str | None], ft.Container],
    on_click: Callable[[], None],
    resting: str | None,
    hovered: str | None,
    pressed: str | None,
) -> ft.Control:
    """Which of the three the pointer is in, held as state and spent on a rebuild.

    The body is built here rather than handed in already made: once a control has been
    patched into a component's tree it is frozen, so the tint is a render and never a
    write to the control that is already on screen.
    """
    phase, set_phase = ft.use_state("rest")
    body = build(pressed if phase == "press" else (hovered if phase == "hover" else resting))

    # Whatever wraps the body has to carry its flex: the parent lays out the wrapper, and
    # a body that expands inside a wrapper that does not is a control with no width.
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


def btn(
    label: str,
    on_click: Callable[[], None] | None = None,
    kind: str = "",
    enabled: bool = True,
    expand: bool = False,
) -> ft.Control:
    """.btn / .btn.pri / .btn.quiet / .btn.danger — height 26, radius 7, 11.5px.

    `kind` is the class list, so "quiet danger" is both of them, as it is in the markup.
    """
    classes = kind.split()
    primary, quiet = "pri" in classes, "quiet" in classes
    face = theme.ON_ACCENT if primary else (t().fg2 if quiet else t().fg)
    if "danger" in classes:
        face = t().bad
    resting = None if quiet else (theme.ACCENT if primary else t().surface2)

    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            content=ft.Text(
                label,
                style=theme.sans(
                    11.5,
                    face,
                    ft.FontWeight.W_600 if primary else ft.FontWeight.W_500,
                ),
                no_wrap=True,
            ),
            height=26,
            padding=ft.Padding.symmetric(horizontal=12),
            alignment=ft.Alignment.CENTER,
            bgcolor=tint,
            border=None if primary or quiet else theme.hair(t().hair2),
            border_radius=7,
            opacity=1.0 if enabled else 0.45,
            expand=expand,
        )

    if not enabled:
        return body(resting)
    return _pressable(
        body,
        on_click,
        resting,
        t().sel if quiet else None,
        theme.mix(t().fg, 0.09, theme.TRANSPARENT if quiet else t().surface2)
        if not primary
        else theme.mix("#000000", 0.12, theme.ACCENT),
    )


def chip(
    content: list[ft.Control],
    on_click: Callable[[], None] | None = None,
    gap: float = 6,
    left: float = 9,
) -> ft.Control:
    """.chip — height 23, radius 6, on --surface."""
    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            content=ft.Row(content, spacing=gap, tight=True),
            height=23,
            padding=ft.Padding(left=left, right=9, top=0, bottom=0),
            bgcolor=tint,
            border=theme.hair(),
            border_radius=6,
        )

    return _pressable(body, on_click, t().surface, t().surface2, t().sel)


def chip_text(text: str, strong: bool = False) -> ft.Text:
    return ft.Text(
        text,
        style=theme.sans(
            11,
            t().fg if strong else t().fg2,
            ft.FontWeight.W_500 if strong else ft.FontWeight.W_400,
        ),
        no_wrap=True,
    )


def dot(state: str = "") -> ft.Container:
    colors = {"ok": t().ok, "warn": t().warn, "bad": t().bad}
    return ft.Container(
        width=6, height=6, border_radius=3, bgcolor=colors.get(state, t().fg3)
    )


def pane(content: list[ft.Control], width: float | None = None) -> ft.Container:
    """.pane — the surface every column of the work area sits on."""
    return ft.Container(
        content=ft.Column(content, spacing=0, expand=True),
        width=width,
        expand=width is None,
        bgcolor=t().surface,
        border=theme.hair(),
        border_radius=10,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )


def head(title: str, meta: str | None = None) -> ft.Container:
    """.ph — a pane's own 38px header."""
    row: list[ft.Control] = [
        ft.Text(title, style=theme.sans(13, weight=ft.FontWeight.W_600), no_wrap=True)
    ]
    if meta is not None:
        row.append(ft.Container(expand=True))
        row.append(ft.Text(meta, style=theme.mono(10, t().fg3), no_wrap=True))
    return ft.Container(
        content=ft.Row(row, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        height=38,
        padding=ft.Padding.symmetric(horizontal=13),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
    )


def foot(left: str, right: str | None = None) -> ft.Container:
    """.pfoot"""
    row: list[ft.Control] = [ft.Text(left, style=theme.sans(11, t().fg2), no_wrap=True)]
    if right is not None:
        row.append(ft.Container(expand=True))
        row.append(ft.Text(right, style=theme.mono(11, t().fg2), no_wrap=True))
    return ft.Container(
        content=ft.Row(row, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.Padding(left=13, right=13, top=8, bottom=8),
        border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
    )


def body(content: list[ft.Control], padding: float = 9) -> ft.Container:
    """.pbody.scroll — only the panes scroll, and only downwards."""
    return ft.Container(
        content=ft.Column(content, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
        padding=padding,
        expand=True,
    )


def group(label: str, first: bool = False) -> ft.Container:
    """.grouplab — 13px of air above it, except against the top of the body."""
    return ft.Container(
        content=theme.eyebrow(label),
        padding=ft.Padding(left=0, right=0, top=0 if first else 13, bottom=4),
    )


def kvr(key: str, value: str, em: str | None = None, last: bool = False) -> ft.Container:
    """.kvr — the key/value line the inspector is written in."""
    measure: list[ft.Control] = [ft.Text(value, style=theme.mono(11), no_wrap=True)]
    if em is not None:
        measure.append(ft.Text(em, style=theme.mono(11, t().fg3), no_wrap=True))
    return ft.Container(
        content=ft.Row(
            [
                ft.Text(key, style=theme.sans(11, t().fg3), no_wrap=True),
                ft.Container(expand=True),
                ft.Row(measure, spacing=4, tight=True),
            ],
            spacing=10,
            vertical_alignment=theme.BASELINE,
        ),
        padding=ft.Padding(left=0, right=0, top=5.5, bottom=5.5),
        border=None if last else ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
    )


def empty(title: str, prose: str, action: ft.Control | None = None) -> ft.Container:
    """.empty — full height, so anything under it starts below the fold."""
    stack: list[ft.Control] = [
        ft.Text(title, style=theme.sans(13, weight=ft.FontWeight.W_600)),
        ft.Container(
            content=ft.Text(
                prose,
                style=theme.sans(11.5, t().fg3, height=1.5),
                text_align=ft.TextAlign.CENTER,
            ),
            # 44ch of the body face, the stylesheet's own measure.
            width=280,
        ),
    ]
    if action is not None:
        stack.append(action)
    return ft.Container(
        content=ft.Column(
            stack,
            spacing=9,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=24,
        expand=True,
        alignment=ft.Alignment.CENTER,
    )


# ── prose and blocks ──────────────────────────────────────────────────────


def note(text: str, bad: bool = False) -> ft.Text:
    """.note"""
    return ft.Text(text, style=theme.sans(10.5, t().bad if bad else t().fg3, height=1.45))


def err(message: str) -> ft.Container:
    """.err — a refusal, in the one colour that is state and never identity."""
    return ft.Container(
        content=ft.Text(message, style=theme.sans(11, t().bad)),
        padding=ft.Padding(left=11, right=11, top=8, bottom=8),
        bgcolor=theme.mix(t().bad, 0.12, theme.TRANSPARENT),
        border=theme.hair(theme.mix(t().bad, 0.30, theme.TRANSPARENT)),
        border_radius=8,
    )


def blk(title: str, content: list[ft.Control]) -> ft.Container:
    """.blk — a titled block on --win, which is what a section of a form is drawn as."""
    return ft.Container(
        padding=ft.Padding(left=12, right=12, top=11, bottom=11),
        bgcolor=t().win,
        border=theme.hair(),
        border_radius=9,
        content=ft.Column(
            [ft.Text(title, style=theme.sans(11.5, weight=ft.FontWeight.W_600)), *content],
            spacing=8,
            tight=True,
        ),
    )


def cols(left: list[ft.Control], right: list[ft.Control]) -> ft.Row:
    """.cols + .colstack — two equal columns that start at the top."""
    # STRETCH is `align-items: stretch` on the stack: a block is as wide as its column and
    # not as wide as whatever it happens to hold.
    return ft.Row(
        [
            ft.Column(
                side,
                spacing=9,
                tight=True,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
            for side in (left, right)
        ],
        spacing=9,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def urlrow(url: str, action: ft.Control) -> ft.Container:
    """.urlrow — a copyable line, set in the mono face because it is a literal."""
    return ft.Container(
        padding=ft.Padding(left=9, right=9, top=6, bottom=6),
        bgcolor=t().field,
        border=theme.hair(),
        border_radius=7,
        content=ft.Row(
            [
                ft.Text(
                    url,
                    style=theme.mono(10.5),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    selectable=True,
                    expand=True,
                ),
                action,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def sublab(left: str, right: str) -> ft.Row:
    """.sublab"""
    return ft.Row(
        [
            ft.Text(left, style=theme.mono(9, t().fg3)),
            ft.Container(expand=True),
            ft.Text(right, style=theme.mono(9, t().fg3)),
        ],
        spacing=8,
    )


def sheet(header: list[ft.Control], content: ft.Control, width: float = 868) -> ft.Control:
    """.sheet — a modal over the whole work area, on a blurred wash of the window colour.

    Flutter blurs what is behind a container rather than what is behind the page, which is
    the same picture here: the sheet is the only thing over the view.
    """
    return ft.Container(
        expand=True,
        alignment=ft.Alignment.CENTER,
        padding=24,
        bgcolor=theme.mix(t().win, 0.62, theme.TRANSPARENT),
        blur=3,
        content=ft.Container(
            width=width,
            bgcolor=t().surface,
            border=theme.hair(t().hair2),
            border_radius=13,
            shadow=ft.BoxShadow(offset=ft.Offset(0, 22), blur_radius=60, color=t().shade),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Column(
                [
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=15, vertical=12),
                        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                        content=ft.Row(
                            header,
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                    content,
                ],
                spacing=0,
                tight=True,
            ),
        ),
    )


# ── choosers ──────────────────────────────────────────────────────────────


CLIPBOARD = ft.Clipboard()
"""The one clipboard the window owns — a service, registered on the page at startup."""


def advance(label: str) -> float:
    """Roughly how wide `label` sets at 11pt in the body face."""
    return sum(8.0 if c.isupper() else 6.6 if c.isdigit() else 6.2 for c in label)


# Every glyph of the mono face is the same width; measured off a rendered key, at 11pt it
# is 7pt and not the 6.6 the metrics promise — the client is not always setting SF Mono.
STRIDE = 7.0


def fits(labels: list[str], cap: float) -> float:
    """What a pop-up button has to be to show its longest option whole.

    A `<select>` is as wide as the option it has to hold, and the stylesheet lets it be:
    a fixed width here is either slack in the row or a clipped label, and the row this
    sizes has no slack to give.
    """
    widest = max((STRIDE * len(label) for label in labels), default=0.0)
    # 9 of padding, 6 of gap, the 17pt well, and 3 the other side.
    return min(cap, widest + 35)


def seg(
    options: list[tuple[str, str]],
    chosen: str,
    on_pick: Callable[[str], None],
    mono: bool = False,
    tight: bool = False,
) -> ft.Container:
    """.seg — a segmented control: every option visible, one of them on.

    Sized by its widest label rather than by its container, which is what `flex: none` on
    the strip and `flex: 1` on the buttons add up to. `tight` is the strip the band header
    carries: `min-width: 22px` over `padding: 2px 7px`, against the 9pt the rest of the
    app sets, because that row has no room to give away.
    """
    # Flet cannot measure a string, so the equal width is estimated off the longest label,
    # counted by class rather than by length: an all-caps GPTQ is half again as wide as
    # four lowercase letters, and taking one advance for both is what clips it.
    span = max(advance(label) for _, label in options)
    side = 7.0 if tight else 9.0
    width = max(22.0, span + 2 * side) if tight else span + 20

    def one(key: str, label: str) -> ft.Control:
        on = key == chosen
        face = theme.mono if mono else theme.sans
        return ft.Container(
            width=width,
            padding=ft.Padding(left=side, right=side, top=3 if not tight else 2,
                               bottom=3 if not tight else 2),
            alignment=ft.Alignment.CENTER,
            bgcolor=t().surface2 if on else None,
            border_radius=5,
            shadow=ft.BoxShadow(
                offset=ft.Offset(0, 1), blur_radius=1.5, color=theme.alpha("#000000", 0.14)
            )
            if on
            else None,
            content=ft.Text(
                label,
                style=face(11, t().fg if on else t().fg2,
                           ft.FontWeight.W_500 if on else ft.FontWeight.W_400),
                no_wrap=True,
            ),
            on_click=lambda _, key=key: on_pick(key),
        )

    return ft.Container(
        padding=2,
        bgcolor=t().sunken,
        border=theme.hair(),
        border_radius=7,
        content=ft.Row([one(k, label) for k, label in options], spacing=3, tight=True),
    )


def fchip(label: str, on: bool, on_click: Callable[[], None]) -> ft.Control:
    """.fchip — a filter, which is a pill and not a push button."""
    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            height=23,
            padding=ft.Padding.symmetric(horizontal=10),
            alignment=ft.Alignment.CENTER,
            bgcolor=tint,
            border=theme.hair(t().hair2 if on else t().hair),
            border_radius=12,
            content=ft.Text(
                label,
                style=theme.sans(11, t().fg if on else t().fg2,
                                 ft.FontWeight.W_500 if on else ft.FontWeight.W_400),
                no_wrap=True,
            ),
        )

    return _pressable(body, on_click, t().surface2 if on else t().win, t().surface2, t().sel)


def field(
    value: str,
    on_commit: Callable[[str], None],
    width: float | None = None,
    mono: bool = False,
    hint: str = "",
) -> ft.Control:
    """.input — a text field with none of Material's decoration on it.

    Committed on blur and on Enter, which is what an AppKit field does; a form that wrote
    on every keystroke would PATCH the daemon once per character.
    """
    style = theme.mono(11) if mono else theme.sans(11.5)
    return ft.TextField(
        value=value,
        width=width,
        height=26,
        text_style=style,
        hint_text=hint,
        hint_style=theme.mono(11, t().fg3) if mono else theme.sans(11.5, t().fg3),
        border=ft.InputBorder.OUTLINE,
        border_color=t().hair2,
        border_width=1,
        focused_border_color=theme.ACCENT,
        focused_border_width=1,
        filled=True,
        fill_color=t().field,
        border_radius=7,
        content_padding=ft.Padding(left=9, right=9, top=0, bottom=0),
        cursor_color=theme.ACCENT,
        cursor_width=1,
        selection_color=theme.mix(theme.ACCENT, 0.30, theme.TRANSPARENT),
        dense=True,
        on_blur=lambda e: on_commit(e.control.value or ""),
        on_submit=lambda e: on_commit(e.control.value or ""),
    )


def well() -> ft.Control:
    """The chevron well of a pop-up button: AppKit marks it with two chevrons pointing
    apart, because the menu opens over the control in both directions."""
    return cv.Canvas(
        [
            cv.Rect(0, 0, 17, 18, border_radius=5, paint=ft.Paint(color=theme.ACCENT)),
            cv.Path(
                [cv.Path.MoveTo(5.6, 7.4), cv.Path.LineTo(8.5, 4.6), cv.Path.LineTo(11.4, 7.4)],
                paint=_chevron(),
            ),
            cv.Path(
                [cv.Path.MoveTo(5.6, 10.6), cv.Path.LineTo(8.5, 13.4), cv.Path.LineTo(11.4, 10.6)],
                paint=_chevron(),
            ),
        ],
        width=17,
        height=18,
    )


def _chevron() -> ft.Paint:
    return ft.Paint(
        color=theme.ON_ACCENT,
        stroke_width=1.5,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )


def pick(
    options: list[tuple[str, str]],
    value: str,
    on_pick: Callable[[str], None],
    width: float | None = None,
) -> ft.Control:
    """select.input — a pop-up button, filled like a button rather than a text field."""
    label = next((text for key, text in options if key == value), "")
    # Boxed, because the button keeps a few points of its own around the content that no
    # padding or constraint on it gives back.
    return ft.Container(
        height=26,
        width=width,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        content=_pick(options, value, on_pick, width, label),
    )


def _pick(
    options: list[tuple[str, str]],
    value: str,
    on_pick: Callable[[str], None],
    width: float | None,
    label: str,
) -> ft.Control:
    return ft.PopupMenuButton(
        width=width,
        height=26,
        padding=0,
        # Flutter labels its own button "Show menu" and sizes it for a touch target; a
        # pop-up button is 26pt tall and says what is chosen, not what it does.
        tooltip="",
        # No size_constraints: they are the *menu's*, not the button's, and a max_height of
        # 26 is what left the list one row tall with everything else behind a scroll. The
        # button's own 26 is its `height` and its content's. Flutter caps the menu at the
        # window and scrolls only past that, which is what a `select` does.
        menu_padding=ft.Padding.symmetric(vertical=4),
        bgcolor=t().surface2,
        elevation=8,
        shape=ft.RoundedRectangleBorder(radius=7),
        content=ft.Container(
            height=26,
            width=width,
            padding=ft.Padding(left=9, right=3, top=0, bottom=0),
            bgcolor=t().surface2,
            border=theme.hair(t().hair2),
            border_radius=7,
            content=ft.Row(
                [
                    ft.Text(
                        label,
                        style=theme.mono(11),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                    well(),
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ),
        items=[
            ft.PopupMenuItem(
                height=24,
                padding=ft.Padding.symmetric(horizontal=11),
                content=ft.Text(
                    text,
                    style=theme.mono(11, t().fg, ft.FontWeight.W_500 if key == value else None),
                    no_wrap=True,
                ),
                on_click=lambda _, key=key: on_pick(key),
            )
            for key, text in options
        ],
    )


def slider(
    value: float,
    low: float,
    high: float,
    on_change: Callable[[float], None],
    on_commit: Callable[[float], None],
    measured: list[float],
) -> ft.Control:
    """input[type=range] — a 4px track with a 13px thumb, and no Material anywhere.

    The track's width is not known until it is laid out, and this control is rebuilt on
    every frame, so the measure is the caller's one-cell list and outlives the rebuild.
    """
    THUMB = 13.0

    def at(x: float) -> float:
        span = max(measured[0] - THUMB, 1.0)
        return low + min(max((x - THUMB / 2) / span, 0.0), 1.0) * (high - low)

    fraction = 0.0 if high <= low else min(max((value - low) / (high - low), 0.0), 1.0)
    track = ft.Container(
        height=4,
        border_radius=3,
        bgcolor=t().sunken,
        alignment=ft.Alignment.CENTER_LEFT,
    )
    thumb = ft.Container(
        width=THUMB,
        height=THUMB,
        border_radius=THUMB / 2,
        bgcolor=t().surface2,
        shadow=[
            ft.BoxShadow(spread_radius=1, blur_radius=0, color=t().hair2),
            ft.BoxShadow(offset=ft.Offset(0, 1), blur_radius=3, color=theme.alpha("#000000", 0.22)),
        ],
    )
    rail = ft.Container(
        height=THUMB,
        on_size_change=lambda e: measured.__setitem__(0, e.width),
        content=ft.Stack(
            [
                ft.Container(left=0, right=0, top=(THUMB - 4) / 2, content=track),
                ft.Container(
                    left=fraction * max(measured[0] - THUMB, 1.0)
                    if measured[0] > 1
                    else None,
                    top=0,
                    content=thumb,
                ),
            ]
        ),
    )
    return ft.GestureDetector(
        content=rail,
        mouse_cursor=ft.MouseCursor.CLICK,
        drag_interval=30,
        on_tap_down=lambda e: on_change(at(e.local_position.x)),
        on_pan_update=lambda e: on_change(at(e.local_position.x)),
        on_pan_end=lambda _: on_commit(value),
    )


# ── the shelf ─────────────────────────────────────────────────────────────


def search(value: str, on_change: Callable[[str], None], placeholder: str) -> ft.Control:
    """.search — a field with a magnifier in it, which is a search box and not a form."""
    glass = cv.Canvas(
        [
            cv.Circle(4.6, 4.6, 3.4, paint=ft.Paint(
                color=t().fg3, stroke_width=1.2, style=ft.PaintingStyle.STROKE)),
            cv.Line(7.3, 7.3, 10, 10, paint=ft.Paint(
                color=t().fg3, stroke_width=1.2, style=ft.PaintingStyle.STROKE,
                stroke_cap=ft.StrokeCap.ROUND)),
        ],
        width=11,
        height=11,
    )
    return ft.Container(
        expand=True,
        height=27,
        padding=ft.Padding(left=10, right=10, top=0, bottom=0),
        bgcolor=t().field,
        border=theme.hair(t().hair2),
        border_radius=7,
        content=ft.Row(
            [
                glass,
                ft.TextField(
                    value=value,
                    expand=True,
                    # No height of its own: a field told to be 27pt tall with no padding
                    # lays its text out against the top of that box rather than down the
                    # middle of it, and the placeholder ends up a few points above the
                    # magnifier. The box around it is what is 27pt tall.
                    text_style=theme.sans(11.5),
                    hint_text=placeholder,
                    hint_style=theme.sans(11.5, t().fg3),
                    border=ft.InputBorder.NONE,
                    content_padding=ft.Padding.all(0),
                    cursor_color=theme.ACCENT,
                    cursor_width=1,
                    dense=True,
                    on_change=lambda e: on_change(e.control.value or ""),
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def iconbtn(draw: list[cv.Shape], on_click: Callable[[], None], hint: str = "") -> ft.Control:
    """.iconbtn — a 24x23 toolbar item, which keeps its hover tint."""
    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            width=24,
            height=23,
            alignment=ft.Alignment.CENTER,
            bgcolor=tint,
            border=theme.hair(),
            border_radius=6,
            tooltip=hint or None,
            content=cv.Canvas(draw, width=13, height=13),
        )

    return _pressable(body, on_click, t().surface2, t().sel, theme.mix(t().fg, 0.09, t().surface2))


def cards(tiles: list[ft.Control], columns: int = 4, gap: float = 8) -> ft.Control:
    """.cards — a four-up grid, which Flutter has no word for at this size: rows of four,
    with the last one padded so its tiles keep the column width."""
    rows: list[ft.Control] = []
    for start in range(0, len(tiles), columns):
        run = tiles[start : start + columns]
        run += [ft.Container(expand=True) for _ in range(columns - len(run))]
        # Not STRETCH: a row that stretches inside a tight column has no height to
        # stretch into, and Flutter answers an unbounded cross axis by drawing nothing.
        # The tiles carry their own 104pt.
        rows.append(ft.Row(run, spacing=gap))
    return ft.Column(rows, spacing=gap, tight=True)


def card(
    name: str,
    org: str,
    measure: str,
    meta: str,
    tag: str,
    material: str | None,
    on: bool,
    on_pick: Callable[[], None],
    progress: float | None = None,
    tag_color: str | None = None,
    hatched: bool = False,
) -> ft.Control:
    """.card — one checkpoint on the shelf.

    A checkpoint that is not on disk yet wears no material: the colour is identity, and
    bytes that do not exist have none to wear. The hatch is what "not yet" is drawn with
    everywhere else in this window.
    """
    face: list[ft.Control] = [
        ft.Container(
            expand=True,
            padding=ft.Padding(left=11, right=11, top=10, bottom=9),
            content=ft.Column(
                [
                    ft.Text(
                        name,
                        style=theme.sans(12, weight=ft.FontWeight.W_600, height=1.25),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Text(
                        org,
                        style=theme.mono(9, t().fg3),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Container(expand=True),
                    ft.Row(
                        [
                            ft.Text(measure, style=theme.mono(12.5), no_wrap=True),
                            ft.Text(
                                meta,
                                style=theme.sans(9.5, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                                expand=True,
                            ),
                            ft.Text(
                                tag.upper(),
                                style=theme.sans(
                                    8.5,
                                    tag_color or (material or t().fg3),
                                    ft.FontWeight.W_600,
                                    spacing=0.08 * 8.5,
                                ),
                                no_wrap=True,
                            )
                            if tag
                            else ft.Container(),
                        ],
                        spacing=7,
                        vertical_alignment=theme.BASELINE,
                    ),
                ],
                spacing=2,
                expand=True,
            ),
        )
    ]
    if material is not None:
        face.append(ft.Container(left=0, top=0, bottom=0, width=3, bgcolor=material))
    elif hatched:
        face.append(
            ft.Container(
                left=0,
                top=0,
                bottom=0,
                width=3,
                gradient=theme.hatch(theme.mix(t().fg2, 0.45, theme.TRANSPARENT), 6, 3),
            )
        )
    if progress is not None:
        face.append(
            ft.Container(
                left=0,
                right=0,
                bottom=0,
                height=3,
                bgcolor=t().sunken,
                content=ft.Row(
                    [
                        ft.Container(
                            expand=max(1, round(progress * 1000)),
                            bgcolor=material or t().fg2,
                        ),
                        ft.Container(expand=max(1, round((1 - progress) * 1000))),
                    ],
                    spacing=0,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            )
        )
    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            expand=True,
            height=104,
            bgcolor=tint,
            border=theme.hair(t().hair2 if on else t().hair),
            border_radius=9,
            shadow=lift() if on else None,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Stack(face),
        )

    return _pressable(body, on_pick, t().surface2 if on else t().win, None, t().sel)

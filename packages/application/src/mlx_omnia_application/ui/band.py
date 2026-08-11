"""Parallel coordinates: one axis per measurement, one line per checkpoint.

Speed and quality never share a chart — a line crossing from tok/s into accuracy joins
two things that don't trade against each other on the same run, and context and
concurrency only exist on the speed side.

Load and TTFT are drawn upside down, because lower is better there and "up" has to mean the
same thing on every axis. A dashed segment means that checkpoint has no run under this key.

The renderer draws into an SVG viewBox 1112 wide that CSS stretches to the pane; a Flutter
canvas has no viewBox, so the geometry below is scaled by the width the box reports and the
lane hit-targets are laid over it as containers rather than as fat invisible strokes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import flet as ft
import flet.canvas as cv

from mlx_omnia_application.ui import theme
from mlx_omnia_application.ui.theme import t

# The stylesheet's own viewBox, and the three heights inside it.
W = 1112.0
H = 152.0
TOP = 56.0
BOT = 118.0
MIDDLE = (TOP + BOT) / 2


@dataclass
class Axis:
    id: str
    # What it is, over the rail.
    nm: str
    # What scopes it, under the name — the part of the key this axis was read at.
    key: str
    unit: str = ""
    fix: int = 1
    # Lower is better: the top of the rail is the smallest value, and the name carries ↓.
    down: bool = False


@dataclass
class Line:
    model: str
    material: int
    values: dict[str, float | None] = field(default_factory=dict)


def _bounds(lines: list[Line], axis: Axis) -> tuple[float, float] | None:
    found = [
        value
        for value in (line.values.get(axis.id) for line in lines)
        if value is not None
    ]
    if not found:
        return None
    low, high = min(found), max(found)
    pad = (high - low) * 0.22 or abs(high) * 0.1 or 1.0
    return low - pad, high + pad


Segment = tuple[float, float, float, float]


def _dashes(
    x1: float, y1: float, x2: float, y2: float, on: float, off: float
) -> list[Segment]:
    """Flutter strokes have no dash array, so a dashed line is drawn as its segments."""
    span = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if span <= 0:
        return []
    ux, uy = (x2 - x1) / span, (y2 - y1) / span
    out: list[Segment] = []
    walked = 0.0
    while walked < span:
        end = min(walked + on, span)
        out.append((x1 + ux * walked, y1 + uy * walked, x1 + ux * end, y1 + uy * end))
        walked = end + off
    return out


def band(
    axes: list[Axis],
    lines: list[Line],
    selected: str | None,
    on_select: Callable[[str], None],
    width: float,
) -> ft.Control:
    """.track2 — the chart and the invisible lanes that pick a line off it."""
    scale = width / W
    height = H * scale
    column = W / max(1, len(axes))
    ranges = {axis.id: _bounds(lines, axis) for axis in axes}

    def at(axis: Axis, value: float) -> float:
        span = ranges.get(axis.id)
        if span is None:
            return MIDDLE
        low, high = span
        share = (value - low) / (high - low) if high != low else 0.5
        return BOT - ((1 - share) if axis.down else share) * (BOT - TOP)

    shapes: list[cv.Shape] = []
    labels: list[ft.Control] = []

    for index, axis in enumerate(axes):
        x = column * (index + 0.5)
        span = ranges.get(axis.id)
        rail = ft.Paint(
            color=t().hair2 if span is not None else t().hair, stroke_width=1 * scale
        )
        if span is None:
            # .railghost — 2 on, 3 off.
            shapes += [
                cv.Line(ax * scale, ay * scale, bx * scale, by * scale, paint=rail)
                for ax, ay, bx, by in _dashes(x, TOP - 8, x, BOT + 8, 2, 3)
            ]
        else:
            shapes.append(
                cv.Line(x * scale, (TOP - 8) * scale, x * scale, (BOT + 8) * scale, paint=rail)
            )
        labels.append(
            _text(
                axis.nm + (" ↓" if axis.down else ""),
                x * scale,
                22 * scale,
                theme.sans(10.5 * scale, t().fg, ft.FontWeight.W_600),
                ft.TextAlign.CENTER,
                column * scale,
            )
        )
        labels.append(
            _text(
                axis.key,
                x * scale,
                35 * scale,
                theme.mono(8.5 * scale, t().fg3, spacing=0.03 * 8.5 * scale),
                ft.TextAlign.CENTER,
                column * scale,
            )
        )
        if span is not None:
            top = span[0] if axis.down else span[1]
            bottom = span[1] if axis.down else span[0]
            labels.append(
                _text(
                    f"{top:.{axis.fix}f}{axis.unit}",
                    (x + 8) * scale,
                    (TOP - 10) * scale,
                    theme.mono(8.5 * scale, t().fg3),
                    ft.TextAlign.LEFT,
                    column * scale / 2,
                )
            )
            labels.append(
                _text(
                    f"{bottom:.{axis.fix}f}{axis.unit}",
                    (x + 8) * scale,
                    (BOT + 4) * scale,
                    theme.mono(8.5 * scale, t().fg3),
                    ft.TextAlign.LEFT,
                    column * scale / 2,
                )
            )

    lanes: list[ft.Control] = []
    for line in lines:
        on = line.model == selected
        colour = t().mat(line.material)
        points = [
            (
                column * (index + 0.5),
                None
                if ranges.get(axis.id) is None or line.values.get(axis.id) is None
                else at(axis, line.values[axis.id] or 0.0),
                line.values.get(axis.id),
            )
            for index, axis in enumerate(axes)
        ]
        stroke = ft.Paint(
            color=theme.alpha(colour, 1.0 if on else 0.34),
            stroke_width=(2.4 if on else 1.5) * scale,
            style=ft.PaintingStyle.STROKE,
            stroke_cap=ft.StrokeCap.ROUND,
        )
        broken_paint = ft.Paint(
            color=theme.alpha(colour, 0.28),
            stroke_width=1.5 * scale,
            style=ft.PaintingStyle.STROKE,
        )
        for index in range(len(points) - 1):
            fx, fy, _ = points[index]
            tx, ty, _ = points[index + 1]
            if fy is None and ty is None:
                continue
            broken = fy is None or ty is None
            x1, y1 = fx, fy if fy is not None else MIDDLE
            x2, y2 = tx, ty if ty is not None else MIDDLE
            if broken:
                shapes += [
                    cv.Line(ax * scale, ay * scale, bx * scale, by * scale, paint=broken_paint)
                    for ax, ay, bx, by in _dashes(x1, y1, x2, y2, 3, 3)
                ]
            else:
                shapes.append(
                    cv.Line(x1 * scale, y1 * scale, x2 * scale, y2 * scale, paint=stroke)
                )
        for px, py, _ in points:
            if py is None:
                shapes.append(
                    cv.Circle(
                        px * scale,
                        MIDDLE * scale,
                        3.4 * scale,
                        paint=ft.Paint(color=t().surface),
                    )
                )
                shapes.append(
                    cv.Circle(
                        px * scale,
                        MIDDLE * scale,
                        3.4 * scale,
                        paint=ft.Paint(
                            color=t().fg3, stroke_width=1.4 * scale, style=ft.PaintingStyle.STROKE
                        ),
                    )
                )
                continue
            radius = (4.6 if on else 3.4) * scale
            shapes.append(
                cv.Circle(px * scale, py * scale, radius, paint=ft.Paint(color=colour))
            )
            shapes.append(
                cv.Circle(
                    px * scale,
                    py * scale,
                    radius,
                    paint=ft.Paint(
                        color=t().surface, stroke_width=1.4 * scale, style=ft.PaintingStyle.STROKE
                    ),
                )
            )
        if on:
            for index, (px, py, value) in enumerate(points):
                if py is None or value is None:
                    continue
                labels.append(
                    _text(
                        f"{value:.{axes[index].fix}f}",
                        (px - 9) * scale,
                        (py - 6) * scale,
                        theme.mono(9.5 * scale, t().fg, ft.FontWeight.W_500),
                        ft.TextAlign.RIGHT,
                        column * scale / 2,
                        right=True,
                    )
                )
        # The lane: 16pt of transparent stroke in the renderer, a box over each segment
        # here, because a Flutter canvas answers no pointer of its own.
        for index in range(len(points) - 1):
            fx, fy, _ = points[index]
            tx, ty, _ = points[index + 1]
            if fy is None or ty is None:
                continue
            top = min(fy, ty) - 8
            lanes.append(
                ft.Container(
                    left=fx * scale,
                    top=top * scale,
                    width=(tx - fx) * scale,
                    height=(abs(ty - fy) + 16) * scale,
                    on_click=lambda _, model=line.model: on_select(model),
                )
            )

    return ft.Container(
        height=height,
        border_radius=6,
        bgcolor=theme.mix(t().sunken, 0.55, t().surface),
        border=theme.hair(),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        content=ft.Stack([cv.Canvas(shapes, expand=True), *labels, *lanes]),
    )


def _text(
    value: str,
    x: float,
    y: float,
    style: ft.TextStyle,
    align: ft.TextAlign,
    width: float,
    right: bool = False,
) -> ft.Control:
    """SVG anchors text at a point; Flutter lays it out in a box, so the box is centred on
    that point (or hung off it) and the alignment inside does the rest."""
    left = x - width / 2 if align is ft.TextAlign.CENTER else (x - width if right else x)
    return ft.Container(
        left=left,
        top=y - (style.size or 10) * 0.85,
        width=width,
        content=ft.Text(value, style=style, text_align=align, no_wrap=True),
    )


def autofill(models: list[str], score: Callable[[str], float | None]) -> list[str]:
    """Best, worst and four spread by rank between them — six, one per material. The
    automatic fill is a starting point and then it is the user's: `auto ∪ pinned − dropped`."""
    scored = [(model, score(model)) for model in models]
    ranked = [
        model
        for model, value in sorted(
            ((m, v) for m, v in scored if v is not None), key=lambda pair: -pair[1]
        )
    ]
    ordered = ranked + [model for model in models if model not in ranked]
    if len(ordered) <= 6:
        return ordered
    picked = {ordered[0], ordered[-1]}
    for step in range(1, 5):
        picked.add(ordered[round(step * (len(ordered) - 1) / 5)])
    return [model for model in ordered if model in picked]

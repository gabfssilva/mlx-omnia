"""The rows and the inspector of the three lists."""

from __future__ import annotations

import json
from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from mlx_omnia_application.api import benchmarks as api
from mlx_omnia_application.api.benchmarks import (
    Run,
    human,
)
from mlx_omnia_application.api.engine import Job
from mlx_omnia_application.ui import parts, theme
from mlx_omnia_application.ui.theme import t
from mlx_omnia_application.views.benchmark.shared import group, keyline

TABS = [("result", "Result"), ("setup", "Setup"), ("raw", "Raw")]
TRACK = 1160 - 24 - 22 - 2
INSPECTOR = 334.0


def nb(value: float | None) -> str:
    return "—" if value is None else f"{round(value):,}"


def fix(value: float | None, places: int) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def finished(job: Job | None) -> bool:
    return job is None or job["state"] in ("ok", "error", "cancelled")


def fidelity_result(run: Run, reference: str) -> list[ft.Control]:
    body = run["fidelity"]
    bars = api.numbers((body or {}).get("histogram") or "[]")
    top1 = None if body is None else body["top1"]
    flipped = None if top1 is None else (1 - top1) * 100
    drawn: list[ft.Control] = [
        group(f"Against {reference}", first=True),
        kv("KL mean", fix((body or {}).get("kl_mean"), 4)),
        kv("KL p95", fix((body or {}).get("kl_p95"), 4)),
        kv("Top-1 agreement", "—" if top1 is None else f"{top1 * 100:.2f}%"),
        kv(
            "Top-5 overlap",
            "—" if body is None or body["top5"] is None else f"{body['top5'] * 100:.2f}%",
        ),
        kv("Δ perplexity", fix((body or {}).get("ppl_delta"), 4), last=True),
        group("Comparison key"),
        keyline(run["key"]),
        ft.Container(
            padding=ft.Padding.only(top=6),
            content=parts.note(
                "The reference is part of the key. Measured against a different checkpoint, "
                "these are different numbers and do not compare."
            ),
        ),
    ]
    if run["state"] != "ok":
        drawn += [
            group("Did not run"),
            parts.err(
                "The two vocabularies differ, so there is no vector to subtract. This pair "
                "has no comparison, not a missing one."
                if run["reason"] == "vocabulary_mismatch"
                else "The teacher-forced pass is task 59.9. The pair is recorded so nothing "
                "about it has to be typed twice."
                if run["reason"] == "not_implemented"
                else (run["reason"] or "unknown")
            ),
        ]
    if bars:
        # The tail is what matters: a low mean with a fat tail is a model that agrees
        # almost always and diverges hard when it does not.
        drawn += [
            group("KL per token"),
            bars(bars),
            parts.sublab("kl per token, bucketed", "tail on the right"),
        ]
    if flipped is not None:
        drawn.append(
            ft.Container(
                padding=ft.Padding.only(top=8),
                content=parts.note(
                    f"{flipped:.1f} tokens in every hundred come out with a different first "
                    "choice, and a 1400-item benchmark does not see that in the aggregate."
                ),
            )
        )
    return drawn


def setup(run: Run) -> list[ft.Control]:
    body = run["speed"]
    try:
        residents = json.loads(run["residents"])
    except ValueError:
        residents = []
    return [
        group("Run", first=True),
        kv("key", run["key"]),
        kv(
            "context → generate",
            "—" if body is None else f"{human(body['context'])} → {body['generate']}",
        ),
        kv("rounds", str((body or {}).get("rounds") or "—")),
        kv("streams from", (body or {}).get("stream_source") or "—", last=True),
        group("Provenance"),
        kv("engine", run["engine_version"]),
        kv("mlx", run["mlx_version"]),
        kv(
            "gate",
            "none" if body is None or body["gate_c"] is None else f"{body['gate_c']} °C",
        ),
        kv(
            "temp at start",
            "—" if run["temp_c_start"] is None else f"{run['temp_c_start']:.1f} °C",
        ),
        kv("other residents", "none" if len(residents) <= 1 else str(len(residents) - 1)),
        kv("queue", "exclusive", last=True),
        ft.Container(
            padding=ft.Padding.only(top=9),
            content=parts.note(
                "Every line here is stored with the number. Without it, next week's "
                "comparison isn't one."
            ),
        ),
    ]


def qhead(title: str, meta: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=3, right=3, top=14, bottom=5),
        content=ft.Row(
            [
                theme.eyebrow(title),
                ft.Container(expand=True),
                ft.Text(meta, style=theme.mono(9.5, t().fg3), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
            ],
            spacing=9,
            vertical_alignment=theme.BASELINE,
        ),
    )


def num(value: str, label: str, dim: bool) -> ft.Control:
    return ft.Column(
        [
            ft.Text(
                value,
                style=theme.mono(16, t().fg3 if dim else t().fg, spacing=-0.02 * 16),
                no_wrap=True,
                text_align=ft.TextAlign.RIGHT,
            ),
            ft.Text(
                label.upper(),
                style=theme.sans(8.5, t().fg3, spacing=0.07 * 8.5),
                no_wrap=True,
                text_align=ft.TextAlign.RIGHT,
            ),
        ],
        spacing=3,
        tight=True,
        horizontal_alignment=ft.CrossAxisAlignment.END,
    )


def spark(samples: list[float], width: float = 84, height: float = 24) -> ft.Control:
    if len(samples) < 2:
        return ft.Container(width=width, height=height)
    low, high = min(samples), max(samples)
    span = high - low or 1
    paint = ft.Paint(
        color=t().fg2,
        stroke_width=1.2,
        style=ft.PaintingStyle.STROKE,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    points = [
        (index / (len(samples) - 1) * width, height - 2 - (value - low) / span * (height - 4))
        for index, value in enumerate(samples)
    ]
    return cv.Canvas(
        [
            cv.Path(
                [cv.Path.MoveTo(*points[0])] + [cv.Path.LineTo(x, y) for x, y in points[1:]],
                paint=paint,
            )
        ],
        width=width,
        height=height,
    )


def run_row(
    model: str,
    key: str,
    material: int | None,
    on: bool,
    ok: bool,
    numbers: list[tuple[str, str, bool]],
    samples: list[float],
    dot: str,
    state: str,
    under: str,
    on_click: Callable[[], None] | None,
) -> ft.Control:
    """.run — the grid the three views share, with the columns each one needs."""
    head: list[ft.Control] = []
    if material is not None:
        head.append(
            ft.Container(width=9, height=2.5, border_radius=2, bgcolor=t().mat(material))
        )
    head.append(
        ft.Text(
            model,
            style=theme.sans(12.5, weight=ft.FontWeight.W_600, height=1.25),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
    )
    cells: list[ft.Control] = [
        ft.Column(
            [
                ft.Row(head, spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Text(
                    key,
                    style=theme.mono(9.5, t().fg3),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
            ],
            spacing=2,
            tight=True,
            expand=True,
        )
    ]
    cells += [
        ft.Container(width=width, content=num(value, label, dim))
        for (value, label, dim), width in zip(
            # .run.s and .run.q, which differ by a column: 68/60/66 against 78/64.
            numbers, (68, 60, 66) if len(numbers) == 3 else (78, 64), strict=False
        )
    ]
    if samples or len(numbers) == 3:
        cells.append(spark(samples))
    cells.append(
        ft.Container(
            width=120,
            content=ft.Column(
                [
                    ft.Row(
                        [parts.dot(dot),
                         ft.Text(state, style=theme.sans(10.5, t().fg2), no_wrap=True,
                                 overflow=ft.TextOverflow.ELLIPSIS)],
                        spacing=6,
                        tight=True,
                        alignment=ft.MainAxisAlignment.END,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Text(under, style=theme.sans(9, t().fg3), no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS,
                            text_align=ft.TextAlign.RIGHT),
                ],
                spacing=2,
                tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.END,
            ),
        )
    )
    body = ft.Container(
        margin=ft.Margin.only(bottom=5),
        padding=ft.Padding(left=15, right=12, top=9, bottom=9),
        bgcolor=t().surface2 if on else t().win,
        border=theme.hair(t().hair2 if on else t().hair),
        border_radius=8,
        shadow=theme.lift() if on else None,
        opacity=1.0 if ok else 0.5,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        # The rule hangs into the padding, and a Stack clips to its own box by default.
        content=ft.Stack(
            clip_behavior=ft.ClipBehavior.NONE,
            controls=[
                ft.Row(cells, spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                *(
                    [
                        ft.Container(
                            left=-15,
                            top=6,
                            bottom=6,
                            width=3,
                            bgcolor=t().mat(material),
                            border_radius=ft.BorderRadius(
                                top_left=0, top_right=3, bottom_left=0, bottom_right=3
                            ),
                        )
                    ]
                    if material is not None
                    else []
                ),
            ],
        ),
    )
    if on_click is None:
        return body
    return ft.Container(content=body, on_click=lambda _: on_click())


def kv(key: str, value: str, last: bool = False) -> ft.Control:
    return parts.kvr(key, value, last=last)


def bars(values: list[float], height: float = 38) -> ft.Control:
    """.rounds — one bar per sample, bottom-aligned."""
    tall = max([1.0, *values])
    return ft.Container(
        height=height,
        content=ft.Row(
            [
                ft.Column(
                    [
                        ft.Container(expand=True),
                        ft.Container(
                            height=max(1.0, value / tall * height),
                            bgcolor=theme.mix(t().mat(0), 0.45, t().surface),
                            border_radius=ft.BorderRadius(
                                top_left=2, top_right=2, bottom_left=0, bottom_right=0
                            ),
                        ),
                    ],
                    spacing=0,
                    expand=True,
                )
                for value in values
            ],
            spacing=3,
            vertical_alignment=ft.CrossAxisAlignment.END,
        ),
    )


def curve(
    series: list[tuple[int, float | None]],
    chosen: int,
    high: float,
    on_pick: Callable[[int], None],
) -> ft.Control:
    """.curve — one bar per concurrency level, the current one solid, a dead one hatched."""
    height = 52.0
    columns: list[ft.Control] = []
    for value, decode in series:
        on = value == chosen
        share = (decode if decode is not None else high * 0.12) / high
        bar = ft.Container(
            height=max(2.0, share * (height - 14)),
            bgcolor=t().mat(0) if on else (None if decode is None else
                                           theme.mix(t().mat(0), 0.26, t().surface)),
            gradient=theme.hatch(t().hair2, 6, 40) if decode is None and not on else None,
            border=None
            if on or decode is None
            else theme.hair(theme.mix(t().mat(0), 0.40, theme.TRANSPARENT)),
            border_radius=ft.BorderRadius(
                top_left=2, top_right=2, bottom_left=0, bottom_right=0
            ),
        )
        columns.append(
            ft.Container(
                expand=True,
                content=ft.Column(
                    [
                        ft.Container(expand=True),
                        bar,
                        ft.Text(
                            str(value),
                            style=theme.mono(8.5, t().fg if on else t().fg3),
                            text_align=ft.TextAlign.CENTER,
                            no_wrap=True,
                        ),
                    ],
                    spacing=3,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                on_click=lambda _, value=value: on_pick(value),
            )
        )
    return ft.Container(
        height=height,
        content=ft.Row(columns, spacing=4, vertical_alignment=ft.CrossAxisAlignment.STRETCH),
    )


def insph(name: str, subtitle: str, rule: str) -> ft.Control:
    return ft.Container(
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Stack(
            [
                ft.Container(
                    left=0,
                    top=11,
                    width=3,
                    height=26,
                    bgcolor=rule,
                    border_radius=ft.BorderRadius(
                        top_left=0, top_right=3, bottom_left=0, bottom_right=3
                    ),
                ),
                ft.Container(
                    width=INSPECTOR - 2,
                    padding=ft.Padding(left=13, right=13, top=11, bottom=9),
                    content=ft.Column(
                        [
                            ft.Text(
                                name,
                                style=theme.sans(13, weight=ft.FontWeight.W_600, height=1.25),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                subtitle,
                                style=theme.mono(9.5, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=3,
                        tight=True,
                    ),
                ),
            ]
        ),
    )


def itabs(chosen: str, on_pick: Callable[[str], None]) -> ft.Control:
    def one(key: str, label: str) -> ft.Control:
        on = key == chosen
        return ft.Container(
            padding=ft.Padding(left=9, right=9, top=4, bottom=7),
            border=ft.Border.only(
                bottom=ft.BorderSide(1.5, t().mat(0) if on else theme.TRANSPARENT)
            ),
            content=ft.Text(
                label,
                style=theme.sans(11, t().fg if on else t().fg3, ft.FontWeight.W_500),
                no_wrap=True,
            ),
            on_click=lambda _, key=key: on_pick(key),
        )

    return ft.Container(
        padding=ft.Padding(left=9, right=9, top=7, bottom=0),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row([one(key, label) for key, label in TABS], spacing=2, tight=True),
    )

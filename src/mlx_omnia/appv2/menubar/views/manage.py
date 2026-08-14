"""Manage: the tab the panel opens on, and the reason the panel exists.

What is resident and what it costs, whatever is still moving, and what the server answered
last. Nothing here is a form — every control is one verb over something already on screen.
"""

from __future__ import annotations

import time

import flet as ft
import flet.canvas as cv

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.engine import Engine, Job
from mlx_omnia.app.ui.format import display_name, gb
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.menubar.panel import Panel, empty
from mlx_omnia.appv2.sidebar import port_of
from mlx_omnia.appv2.theme import t

SPARK = (62.0, 18.0)


def _spark(model: str) -> ft.Control:
    """This model's share of its ceiling over the last minute; the dot is now. The
    Server card's pulse at the panel's scale."""
    values = runtime.points(model)
    width, height = SPARK
    if len(values) < 2:
        return ft.Container(width=width, height=height)
    step = width / (runtime.TRACE - 1)
    start = runtime.TRACE - len(values)
    coords = [
        ((start + index) * step, 3 + (height - 8) * (1 - min(value, 100.0) / 100.0))
        for index, value in enumerate(values)
    ]
    line = ft.Paint(
        color=t().accent,
        stroke_width=1.6,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    steps: list[cv.Path.MoveTo | cv.Path.LineTo] = [cv.Path.MoveTo(*coords[0])]
    steps.extend(cv.Path.LineTo(x, y) for x, y in coords[1:])
    return cv.Canvas(
        [cv.Path(steps, paint=line), cv.Circle(*coords[-1], 2.4, paint=ft.Paint(color=t().accent))],
        width=width,
        height=height,
    )


def _card(inner: list[ft.Control]) -> ft.Container:
    return ft.Container(
        padding=ft.Padding(left=12, right=12, top=10, bottom=10),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=12,
        content=ft.Column(inner, spacing=0, tight=True),
    )


def _resident(engine: Engine, panel: Panel, slot: engine_api.Slot) -> ft.Control:
    identifier = slot["id"]
    material = t().mat(engine.materials.get(identifier, 0))
    measures: list[ft.Control] = []
    decode = runtime.decode_of(engine, identifier)
    if decode is not None:
        measures.append(parts.chip(decode))
    pair = runtime.prefill_ttft_of(engine, identifier)
    if pair is not None:
        measures.append(ft.Text(pair, style=theme.mono(10, t().fg3), no_wrap=True))

    top = ft.Row(
        [
            ft.Container(width=4, height=30, border_radius=2, bgcolor=material),
            ft.Column(
                [
                    ft.Text(
                        display_name(identifier),
                        style=theme.sans(13.5, weight=ft.FontWeight.W_600),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Text(
                        f"{gb(slot['weights_bytes'])} GB + {gb(slot['kv_bytes'])} KV",
                        style=theme.mono(10, t().fg3),
                        no_wrap=True,
                    ),
                ],
                spacing=2,
                tight=True,
                expand=True,
            ),
            _spark(identifier),
            ft.Container(
                content=ft.Text("⏏", style=theme.sans(12, t().fg3), no_wrap=True),
                padding=ft.Padding(left=6, right=2, top=0, bottom=0),
                on_click=lambda _: runtime.act(engine_api.unload(identifier)),
            ),
        ],
        spacing=9,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    inner: list[ft.Control] = [top]
    if measures:
        inner.append(
            ft.Container(
                margin=ft.Margin.only(top=7),
                content=ft.Row(
                    measures, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER
                ),
            )
        )
    return ft.Container(
        content=_card(inner),
        on_click=lambda _: (setattr(panel, "tab", "models"), setattr(panel, "opened", identifier)),
    )


def _job(job: Job) -> ft.Control:
    subject = job["subject"]
    model = subject.get("model") or subject.get("models")
    if isinstance(model, list):
        model = model[0] if model else None
    name = display_name(model) if isinstance(model, str) else "?"
    target = subject.get("target")
    if job["kind"] == "quantize" and isinstance(target, str):
        landing = display_name(target)
        name = f"{name} → {landing.removeprefix(f'{name}-') or landing}"

    total = job["progress"]["total"]
    done = job["progress"]["completed"]
    fraction = 0.0 if not total else min(1.0, done / total)
    message = job["progress"]["message"] or "running…"
    return _card(
        [
            ft.Row(
                [
                    ft.Text(
                        name,
                        style=theme.sans(12.5, t().fg2),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Text("✕", style=theme.sans(10.5, t().fg3), no_wrap=True),
                        on_click=lambda _: runtime.act(engine_api.cancel_job(job["id"])),
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Container(
                height=3,
                margin=ft.Margin.only(top=6, bottom=5),
                border_radius=2,
                bgcolor=t().sel,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                content=ft.Row(
                    [
                        ft.Container(expand=max(1, round(fraction * 100)), bgcolor=t().accent),
                        ft.Container(expand=max(1, round((1 - fraction) * 100))),
                    ],
                    spacing=0,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ),
            ft.Text(
                (f"{round(fraction * 100)}% · " if total else "") + message,
                style=theme.mono(10, t().fg3),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
        ]
    )


def _section(title: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=2, right=2, top=6, bottom=0), content=theme.eyebrow(title)
    )


@ft.component
def Manage(panel: Panel, engine: Engine) -> ft.Control:
    # The pulse ages with nothing else to redraw it.
    runtime.use_tick()

    rows: list[ft.Control] = []

    if engine.health is None:
        rows.append(
            _card(
                [
                    ft.Container(
                        padding=ft.Padding(left=4, right=4, top=12, bottom=4),
                        alignment=ft.Alignment.CENTER,
                        content=ft.Text(
                            f"The engine is not answering on :{port_of(engine)} — starting it…",
                            style=theme.sans(12.5, t().fg3),
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ),
                    # Indeterminate on purpose: the daemon reports nothing until it answers,
                    # and a fraction invented here would be a number about nothing.
                    ft.Container(
                        margin=ft.Margin.only(left=40, right=40, top=14, bottom=6),
                        content=ft.ProgressBar(
                            bar_height=3, border_radius=2, bgcolor=t().sel, color=t().accent
                        ),
                    ),
                ]
            )
        )
    elif engine.models:
        rows.append(_section(f"Resident · {len(engine.models)}"))
        rows.extend(_resident(engine, panel, slot) for slot in engine.models)
    else:
        rows.append(_section("Resident"))
        rows.append(empty("Nothing resident. Load a model under Models."))

    moving = [job for job in engine.jobs if job["state"] in ("pending", "running")]
    if moving:
        rows.append(_section("Running"))
        rows.extend(_job(job) for job in moving)

    requests = [] if engine.metrics is None else engine.metrics["requests"]
    if requests:
        rows.append(_section("Recent"))
        for sample in requests[:4]:
            when = time.strftime("%H:%M", time.localtime(sample["started_at"]))
            speed = sample["tokens_per_second"]
            rows.append(
                ft.Container(
                    padding=ft.Padding(left=2, right=2, top=6, bottom=6),
                    border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            ft.Container(
                                width=38,
                                content=ft.Text(
                                    when, style=theme.mono(10, t().fg3), no_wrap=True
                                ),
                            ),
                            ft.Text(
                                f"{display_name(sample['model'])}"
                                f" · {sample['completion_tokens']} tokens",
                                style=theme.sans(12.5, t().fg2),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                                expand=True,
                            ),
                            ft.Text(
                                "—" if speed is None else f"{speed:.1f} tok/s",
                                style=theme.mono(10, t().fg3),
                                no_wrap=True,
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
            )

    return ft.Column(rows, spacing=7, scroll=ft.ScrollMode.AUTO, expand=True)

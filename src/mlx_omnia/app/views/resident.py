"""What the machine is doing, in the two dimensions that bound it: the band is space, the
list below it is time.

Nothing here is a form — the only control on the screen is the ceiling, and it lives on
the band.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.engine import GIB, Aggregate, CatalogEntry, Engine, Job, Sample, Slot
from mlx_omnia.app.ui import memory, parts, theme
from mlx_omnia.app.ui.format import display_name, gb, tokens
from mlx_omnia.app.ui.hooks import act, use_tick
from mlx_omnia.app.ui.theme import t

# One point per second, so the trace is what actually happened while the window was open —
# the daemon keeps no history to ask for. It is kept outside the view because the trace
# does not stop when the screen is not the one on: a model generating while somebody reads
# Library still has a shape when they come back.
TRACE = 56
_TRACE: dict[str, list[float]] = {}


async def trace(engine: Engine) -> None:
    """Sample every model's share of the ceiling, once a second, for as long as the app
    runs."""
    while True:
        snapshot = engine.metrics
        if snapshot is not None:
            for aggregate in snapshot["models"]:
                model = aggregate["model"]
                live = engine.live(model)
                point = (live["ceiling_fraction"] or 0.0) if live is not None else 0.0
                _TRACE[model] = [*_TRACE.get(model, []), point * 100][-TRACE:]
        await asyncio.sleep(1.0)


SPARK = (84.0, 24.0)

INSPECTOR = 334.0

def seconds(value: float) -> str:
    if value < 60:
        return f"{round(value)} s"
    if value < 3600:
        return f"{round(value / 60)} min"
    return f"{value / 3600:.1f} h"


def clock(epoch: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def ceiling_rate(bytes_per_token: float, sustained: float) -> float:
    """tok/s a decode step can reach at the sustained bandwidth: measured, never
    published."""
    return sustained * 1e9 / bytes_per_token


@ft.component
def Resident(engine: Engine, jump: Callable[[str], None]) -> ft.Control:
    """The view, drawn from whole snapshots — which is what the stream carries."""
    chosen, set_chosen = ft.use_state(None)
    # The grip moves in whole GiB, which is also the step the arrow keys take: a pan that
    # has not crossed one is not a write, and what it has not crossed is remembered here.
    written = ft.use_ref(None)
    # "Idle for 12 s" and the queue's clocks age with nothing to redraw them.
    use_tick()

    models = engine.models
    selected = next((m for m in models if m["id"] == chosen), models[0] if models else None)

    def ceiling(byte_count: int) -> None:
        whole = max(GIB, round(byte_count / GIB) * GIB)
        if whole != written.current:
            written.current = whole
            act(engine_api.patch_config({"memory_limit_bytes": whole}))

    return ft.Container(
        expand=True,
        bgcolor=t().win,
        content=ft.Column(
            [
                memory.band(engine, ceiling),
                ft.Container(
                    expand=True,
                    padding=12,
                    content=ft.Row(
                        [
                            _running(engine, models, selected, jump, set_chosen),
                            _inspector(engine, selected, jump),
                        ],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                ),
            ],
            spacing=0,
            expand=True,
        ),
    )


# ── the roster ────────────────────────────────────────────────────────


def _running(
    engine: Engine,
    models: list[Slot],
    selected: Slot | None,
    jump: Callable[[str], None],
    on_pick: Callable[[str], None],
) -> ft.Control:
    waiting = 0 if engine.state is None else engine.state["queue"]["waiting"]
    snapshot = engine.metrics
    requests = [] if snapshot is None else snapshot["requests"]
    queued = [s for s in requests if s["state"] in ("queued", "running")]
    past = [s for s in requests if s["state"] not in ("queued", "running")]

    stack: list[ft.Control] = []
    if not models and not engine.jobs:
        # A job in flight is the answer to "nothing is loaded" — and it is also what
        # the prose would be telling the person to go and do.
        stack.append(
            parts.empty(
                "Nothing is loaded",
                "Pick a checkpoint in Library and load it, or point a client at the "
                "engine — the first request loads what it asks for.",
                parts.btn("Open Library", lambda: jump("library")),
            )
        )
    for model in models:
        stack.append(
            _run(
                model,
                engine.materials.get(model["id"], 0),
                engine.entry(model["id"]),
                engine.live(model["id"]),
                _TRACE.get(model["id"], []),
                None if engine.system is None else engine.system["bandwidth_sustained_gbs"],
                selected is not None and model["id"] == selected["id"],
                on_pick,
            )
        )
        stack.append(ft.Container(height=5))
    stack.extend(_job(job) for job in engine.jobs)

    if queued:
        stack.append(_qhead("Queue", f"first come, first served · {len(queued)} in flight"))
        stack.extend(
            _qrow(index + 1, sample, engine.materials.get(sample["model"], 0))
            for index, sample in enumerate(queued)
        )
    if past:
        stack.append(_qhead("Recent", "this session"))
        stack.extend(_logrow(sample) for sample in past[:8])

    pane: list[ft.Control] = [
        parts.head("Running", f"{len(models)} resident · {waiting} waiting"),
        parts.body(stack),
    ]
    system = engine.system
    if system is not None:
        pane.append(
            parts.foot(
                f"{system['chip']} · {system['gpu_cores']} GPU cores · "
                f"{gb(system['memory_bytes'], 0)} GB",
                f"{system['bandwidth_sustained_gbs']} GB/s",
            )
        )
    return parts.pane(pane)


def _run(
    model: Slot,
    material: int,
    entry: CatalogEntry | None,
    live: Sample | None,
    points: list[float],
    sustained: float | None,
    on: bool,
    on_pick: Callable[[str], None],
) -> ft.Control:
    rate = None if live is None else live["tokens_per_second"]
    fraction = None if live is None else live["ceiling_fraction"]
    per_token = None if entry is None else entry["bytes_per_token"]
    ceiling = (
        ceiling_rate(per_token, sustained)
        if per_token and sustained is not None
        else None
    )
    idle_for = None if model["last_used"] is None else time.time() - model["last_used"]

    meta = (entry["quantization"] or "dense") if entry is not None else "dense"
    if entry is not None and entry["context"]:
        meta += f" · {tokens(entry['context'])} ctx"
    if ceiling is not None:
        meta += f" · ceiling {round(ceiling)}"

    state: list[ft.Control] = [
        ft.Row(
            [parts.dot("" if live is None else "ok"),
             ft.Text("Idle" if live is None else "Generating",
                     style=theme.sans(10.5, t().fg2), no_wrap=True)],
            spacing=6,
            tight=True,
        )
    ]
    if live is None and idle_for is not None:
        state.append(
            ft.Text(f"for {seconds(idle_for)}", style=theme.sans(9, t().fg3), no_wrap=True)
        )

    columns = ft.Row(
        [
            ft.Column(
                [
                    ft.Text(
                        display_name(model["id"]),
                        style=theme.sans(12.5, weight=ft.FontWeight.W_600, height=1.25),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Text(
                        meta,
                        style=theme.mono(9.5, t().fg3),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                spacing=2,
                tight=True,
                expand=True,
            ),
            _num("—" if rate is None else f"{rate:.1f}", "tok/s", rate is None, 72),
            _num(
                "—" if fraction is None else f"{round(fraction * 100)} %",
                "of ceiling",
                fraction is None,
                60,
                fraction,
                t().mat(material),
            ),
            ft.Container(
                width=SPARK[0], height=SPARK[1], content=_spark(points, t().mat(material))
            ),
            ft.Container(
                width=116,
                content=ft.Column(
                    state,
                    spacing=2,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.END,
                ),
            ),
        ],
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # A row of a single-select roster: the material rules the left edge, and the
    # selected one lifts onto --surface2.
    return ft.Container(
        content=ft.Stack(
            [
                ft.Container(
                    content=columns,
                    padding=ft.Padding(left=15, right=12, top=9, bottom=9),
                ),
                ft.Container(left=0, top=6, bottom=6, width=3, bgcolor=t().mat(material),
                             border_radius=ft.BorderRadius(
                                 top_left=0, top_right=3, bottom_left=0, bottom_right=3
                             )),
            ]
        ),
        bgcolor=t().surface2 if on else t().win,
        border=theme.hair(t().hair2 if on else t().hair),
        border_radius=8,
        shadow=theme.lift() if on else None,
        on_click=lambda _: on_pick(model["id"]),
    )


def _job(job: Job) -> ft.Control:
    subject = job["subject"]
    model = subject.get("model")
    models = subject.get("models")
    name = (
        display_name(model)
        if isinstance(model, str)
        else ", ".join(display_name(m) for m in models)
        if isinstance(models, list)
        else "—"
    )
    progress = job["progress"]
    total = progress["total"]
    pull = job["kind"] == "download"
    color = t().fg2 if pull else t().mat(5)
    return ft.Container(
        margin=ft.Margin.only(top=9),
        padding=ft.Padding(left=12, right=12, top=10, bottom=10),
        bgcolor=t().win,
        border=theme.hair(),
        border_radius=8,
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Row(
                            [
                                ft.Text(name, style=theme.sans(12, weight=ft.FontWeight.W_500)),
                                ft.Text(
                                    f"· {progress['message']}",
                                    style=theme.sans(12, t().fg3),
                                ),
                            ],
                            spacing=4,
                            tight=True,
                            expand=True,
                        ),
                        ft.Text(
                            f"{gb(progress['completed'])} GB"
                            if total is None
                            else f"{gb(progress['completed'])} of {gb(total)} GB",
                            style=theme.mono(10.5, t().fg2),
                            no_wrap=True,
                        ),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                _bar(
                    0.3 if not total else min(1.0, progress["completed"] / total),
                    color,
                    4,
                ),
            ],
            spacing=8,
            tight=True,
        ),
    )

# ── the inspector ─────────────────────────────────────────────────────


def _inspector(
    engine: Engine, selected: Slot | None, jump: Callable[[str], None]
) -> ft.Control:
    if selected is None or engine.system is None:
        return parts.pane(
            [parts.empty("No block selected", "Load a model to see what it costs and "
                         "what it can reach.")],
            INSPECTOR,
        )

    material = t().mat(engine.materials.get(selected["id"], 0))
    entry = engine.entry(selected["id"])
    aggregate = engine.aggregate(selected["id"])
    sustained = engine.system["bandwidth_sustained_gbs"]
    size = selected["weights_bytes"] + selected["kv_bytes"]

    rows: list[ft.Control] = [
        parts.group("Throughput", first=True),
        *_kvs(_throughput(aggregate, entry, sustained)),
        parts.group("Memory"),
        *_kvs(
            [
                ("Weights resident", f"{gb(selected['weights_bytes'])} GB", None),
                ("KV cache", f"{gb(selected['kv_bytes'])} GB", None),
                (
                    "Share of the machine",
                    f"{size / engine.system['memory_bytes'] * 100:.1f} %",
                    None,
                ),
                (
                    "Last used",
                    "never"
                    if selected["last_used"] is None
                    else f"{seconds(time.time() - selected['last_used'])} ago",
                    None,
                ),
            ]
        ),
    ]
    if entry is not None:
        rows.append(parts.group("Checkpoint"))
        rows.extend(
            _kvs(
                [
                    ("Architecture", entry["architecture"], None),
                    (
                        "Quantization",
                        entry["quantization"] or "dense",
                        f"· {entry['dtype']}" if entry["dtype"] else None,
                    ),
                    (
                        "Context",
                        "—"
                        if entry["context"] is None
                        else f"{tokens(entry['context'])} tokens",
                        None,
                    ),
                    ("On disk", f"{gb(entry['bytes_on_disk'])} GB", None),
                ]
            )
        )

    return parts.pane(
        [
            # .insph — the material rules the head, which is where the identity is
            # declared for everything below it.
            ft.Container(
                border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                content=ft.Stack(
                    [
                        # The rule is against the pane's edge, not the head's padding,
                        # so it sits outside the padded box rather than inside it.
                        ft.Container(
                            left=0,
                            top=11,
                            width=3,
                            height=26,
                            bgcolor=material,
                            border_radius=ft.BorderRadius(
                                top_left=0, top_right=3, bottom_left=0, bottom_right=3
                            ),
                        ),
                        # A Stack takes the width of its widest unpositioned child,
                        # and this one would be as wide as the name in it — the rule
                        # under the head has to reach the pane's other edge.
                        ft.Container(
                          width=INSPECTOR - 2,
                          padding=ft.Padding(left=13, right=13, top=11, bottom=9),
                          content=ft.Column(
                            [
                                ft.Text(
                                    display_name(selected["id"]),
                                    style=theme.sans(
                                        13, weight=ft.FontWeight.W_600, height=1.25
                                    ),
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                                ft.Text(
                                    selected["id"],
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
            ),
            ft.Container(
                expand=True,
                padding=ft.Padding.symmetric(horizontal=13, vertical=11),
                content=ft.Column(rows, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
            ),
            ft.Container(
                padding=ft.Padding.symmetric(horizontal=11, vertical=9),
                border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                content=ft.Row(
                    [
                        parts.btn("Chat", lambda: jump("chat"), "pri"),
                        parts.btn("Unload", lambda: act(engine_api.unload(selected["id"]))),
                        ft.Container(expand=True),
                        parts.btn("In Library", lambda: jump("library"), "quiet"),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
        ],
        INSPECTOR,
    )


def _throughput(
    aggregate: Aggregate | None, entry: CatalogEntry | None, sustained: float
) -> list[tuple[str, str, str | None]]:
    rate = None if aggregate is None else aggregate["tokens_per_second"]
    fraction = None if aggregate is None else aggregate["ceiling_fraction"]
    per_token = None if entry is None else entry["bytes_per_token"]
    return [
        ("Decode", "—" if rate is None else f"{rate:.1f} tok/s", None),
        (
            "Ceiling",
            f"{round(ceiling_rate(per_token, sustained))} tok/s" if per_token else "—",
            f"· {round(fraction * 100)} %" if fraction else None,
        ),
        ("Bytes per token", f"{per_token / 1e9:.2f} GB" if per_token else "—", None),
        ("Requests", "0" if aggregate is None else str(aggregate["requests"]), None),
    ]


def _kvs(rows: list[tuple[str, str, str | None]]) -> list[ft.Control]:
    return [
        parts.kvr(key, value, em, last=index == len(rows) - 1)
        for index, (key, value, em) in enumerate(rows)
    ]


def _num(
    value: str,
    label: str,
    dim: bool,
    width: float,
    fraction: float | None = None,
    material: str = "",
) -> ft.Control:
    """.num — a measure right-aligned over its unit, with the ceiling bar under it."""
    stack: list[ft.Control] = [
        ft.Text(
            value,
            style=theme.mono(16, t().fg3 if dim else t().fg, spacing=-0.02 * 16),
            text_align=ft.TextAlign.RIGHT,
            no_wrap=True,
        )
    ]
    if fraction is not None:
        stack.append(ft.Container(height=5))
        stack.append(_bar(min(1.0, fraction), material, 3))
    stack.append(
        ft.Text(
            label.upper(),
            style=theme.sans(8.5, t().fg3, spacing=0.07 * 8.5),
            text_align=ft.TextAlign.RIGHT,
            no_wrap=True,
        )
    )
    return ft.Container(
        width=width,
        content=ft.Column(
            stack, spacing=0, tight=True, horizontal_alignment=ft.CrossAxisAlignment.END
        ),
    )


def _bar(fraction: float, color: str, height: float) -> ft.Control:
    return ft.Container(
        height=height,
        border_radius=height / 2,
        bgcolor=t().sunken,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        content=ft.Row(
            [ft.Container(expand=max(1, round(fraction * 1000)), bgcolor=color),
             ft.Container(expand=max(1, round((1 - fraction) * 1000)))],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
    )


def _spark(points: list[float], color: str) -> ft.Control | None:
    """20–75 % of the ceiling is the window a decode run lives in; outside it the trace
    says nothing worth the pixels."""
    if len(points) < 2:
        return None
    width, height = SPARK
    path: list[cv.Path.PathElement] = []
    for index, value in enumerate(points):
        x = 1 + index / (len(points) - 1) * (width - 2)
        y = height - 2 - min(max((value - 20) / 55, 0.0), 1.0) * (height - 4)
        path.append(cv.Path.MoveTo(x, y) if index == 0 else cv.Path.LineTo(x, y))
    return cv.Canvas(
        [cv.Path(path, paint=ft.Paint(color=color, stroke_width=1.2,
                                      style=ft.PaintingStyle.STROKE,
                                      stroke_cap=ft.StrokeCap.ROUND,
                                      stroke_join=ft.StrokeJoin.ROUND))],
        width=width,
        height=height,
    )


def _qhead(title: str, note: str) -> ft.Control:
    return ft.Container(
        margin=ft.Margin.only(left=3, right=3, top=14, bottom=5),
        content=ft.Row(
            [
                theme.eyebrow(title),
                ft.Container(expand=True),
                ft.Text(note, style=theme.mono(9.5, t().fg3), no_wrap=True),
            ],
            spacing=9,
            vertical_alignment=theme.BASELINE,
        ),
    )


def _qrow(index: int, sample: Sample, material: int) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=3, right=12, top=5.5, bottom=5.5),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                ft.Container(
                    width=16,
                    content=ft.Text(
                        str(index),
                        style=theme.mono(9.5, t().fg3),
                        text_align=ft.TextAlign.RIGHT,
                    ),
                ),
                ft.Container(
                    expand=True,
                    content=ft.Text(
                        "generating" if sample["state"] == "running" else "waiting",
                        style=theme.mono(10.5, t().fg2),
                        no_wrap=True,
                    ),
                ),
                ft.Container(
                    width=132,
                    content=ft.Row(
                        [
                            ft.Container(width=7, height=7, border_radius=2,
                                         bgcolor=t().mat(material)),
                            ft.Text(
                                display_name(sample["model"]),
                                style=theme.sans(10.5),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=6,
                    ),
                ),
                ft.Container(
                    width=78,
                    content=ft.Text(
                        f"{sample['prompt_tokens']} in",
                        style=theme.mono(10.5, t().fg2),
                        text_align=ft.TextAlign.RIGHT,
                    ),
                ),
                ft.Container(
                    width=54,
                    content=ft.Text(
                        seconds(time.time() - sample["started_at"]),
                        style=theme.mono(10.5, t().fg2),
                        text_align=ft.TextAlign.RIGHT,
                    ),
                ),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _logrow(sample: Sample) -> ft.Control:
    failed = sample["state"] == "error"
    tail = " · cancelled" if sample["state"] == "cancelled" else (" · failed" if failed else "")
    rate = sample["tokens_per_second"]
    return ft.Container(
        padding=ft.Padding(left=3, right=12, top=4.5, bottom=4.5),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                ft.Container(
                    width=54,
                    content=ft.Text(clock(sample["started_at"]), style=theme.mono(9.5, t().fg3)),
                ),
                ft.Container(
                    expand=True,
                    content=ft.Row(
                        [
                            ft.Text(
                                display_name(sample["model"]),
                                style=theme.sans(
                                    11,
                                    t().bad if failed else t().fg,
                                    ft.FontWeight.W_500,
                                ),
                                no_wrap=True,
                            ),
                            ft.Text(
                                f"· {sample['prompt_tokens']} in, "
                                f"{sample['completion_tokens']} out{tail}",
                                style=theme.sans(11, t().bad if failed else t().fg2),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=4,
                        tight=True,
                    ),
                ),
                ft.Text(
                    "—" if rate is None else f"{rate:.1f} tok/s",
                    style=theme.mono(9.5, t().fg3),
                    no_wrap=True,
                ),
            ],
            spacing=11,
            vertical_alignment=theme.BASELINE,
        ),
    )

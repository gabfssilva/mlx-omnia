"""Server: the first place. The band at full scale, the chosen block's numbers with
the decode pulse, the recent requests — and Settings as the same scroll's footer,
because the daemon's knobs belong to the daemon's screen.

Everything drawn here is the daemon's own: slots and bytes off the state stream,
numbers off the metrics register, knobs off `/admin/config` and written back to it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import flet as ft
import flet.canvas as cv

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.daemon import Daemon
from mlx_omnia.app.api.engine import GIB, Engine, Slot
from mlx_omnia.app.ui.format import display_name, gb
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.sidebar import WIDTH as SIDEBAR
from mlx_omnia.appv2.sidebar import port_of
from mlx_omnia.appv2.theme import t

PAD = 24
TRACK = 1160 - SIDEBAR - 2 * PAD - 2
TRACK_HEIGHT = 56


def _slab(engine: Engine, slot: Slot, total: int, on: bool, pick: Callable[[], None]) -> ft.Control:
    material = t().mat(engine.materials.get(slot["id"], 0))
    size = slot["weights_bytes"] + slot["kv_bytes"]
    width = size / total * TRACK

    face: list[ft.Control] = []
    if slot["kv_bytes"] > 0:
        face.append(
            ft.Container(
                right=0,
                top=0,
                bottom=0,
                width=slot["kv_bytes"] / size * width,
                bgcolor=theme.mix(material, 0.30, theme.TRANSPARENT),
                border=ft.Border.only(
                    left=ft.BorderSide(1, theme.mix(material, 0.50, theme.TRANSPARENT))
                ),
            )
        )
    face.append(ft.Container(left=0, top=0, bottom=0, width=3, bgcolor=material))
    face.append(
        ft.Container(
            left=10,
            right=5,
            top=8,
            content=ft.Text(
                display_name(slot["id"]),
                style=theme.sans(11, weight=ft.FontWeight.W_600, height=1.2),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
        )
    )
    face.append(
        ft.Container(
            left=10,
            right=5,
            top=24,
            content=ft.Text(
                f"{gb(slot['weights_bytes'])} GB + {gb(slot['kv_bytes'])} KV",
                style=theme.mono(10, t().fg2),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
        )
    )

    return ft.Container(
        content=ft.Stack(face),
        width=width,
        bgcolor=theme.mix(material, 0.24, t().surface),
        border=theme.hair(material if on else theme.mix(material, 0.45, theme.TRANSPARENT)),
        border_radius=6,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        on_click=lambda _: pick(),
    )


def _dashed(height: float, color: str) -> ft.Control:
    count = int(height // 6)
    return ft.Column(
        [ft.Container(width=1.5, height=3, bgcolor=color) for _ in range(count)],
        spacing=3,
        tight=True,
    )


def _band(engine: Engine, total: int, chosen: str, pick: Callable[[str], None]) -> ft.Control:
    ceiling = engine.ceiling
    layers: list[ft.Control] = [
        ft.Row(
            [
                _slab(engine, slot, total, slot["id"] == chosen,
                      lambda slot=slot: pick(slot["id"]))
                for slot in engine.models
            ],
            spacing=2,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
    ]
    if ceiling is not None:
        headroom = total - ceiling
        if headroom > 0:
            layers.append(
                ft.Container(
                    right=0,
                    top=0,
                    bottom=0,
                    width=headroom / total * TRACK,
                    gradient=theme.hatch(t().hair, 5, TRACK, duty=0.2),
                )
            )
        layers.append(
            ft.Container(
                left=ceiling / total * TRACK,
                top=-3,
                bottom=-3,
                content=ft.Stack(
                    [
                        _dashed(TRACK_HEIGHT + 6, t().fg2),
                        ft.Container(
                            right=7,
                            bottom=5,
                            content=ft.Text(
                                f"ceiling · {gb(ceiling, 0)} GB",
                                style=theme.mono(10, t().fg2),
                                no_wrap=True,
                            ),
                        ),
                    ],
                    clip_behavior=ft.ClipBehavior.NONE,
                ),
            )
        )
    return ft.Container(
        height=TRACK_HEIGHT,
        border_radius=8,
        bgcolor=theme.mix(t().sel, 0.55, t().surface),
        border=theme.hair(),
        content=ft.Stack(layers, clip_behavior=ft.ClipBehavior.NONE),
    )


def _spark(model: str, width: float = 120, height: float = 30) -> ft.Control:
    """This model's share of its ceiling over the last minute; the dot is now."""
    values = runtime.points(model)
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
        [
            cv.Path(steps, paint=line),
            cv.Circle(*coords[-1], 2.4, paint=ft.Paint(color=t().accent)),
        ],
        width=width,
        height=height,
    )


def _stats(engine: Engine, slot: Slot) -> ft.Control:
    sample = runtime.sample_of(engine, slot["id"])
    prefill = None if sample is None else sample["prefill_tokens_per_second"]
    ttft = None if sample is None else sample["ttft"]
    cells: list[ft.Control] = [
        ft.Container(content=parts.fact("Weights", f"{gb(slot['weights_bytes'])} GB"), expand=True),
        ft.Container(content=parts.fact("KV", f"{gb(slot['kv_bytes'])} GB"), expand=True),
        ft.Container(
            content=parts.fact("Decode", runtime.decode_of(engine, slot["id"]) or "—"),
            expand=True,
        ),
        ft.Container(
            content=parts.fact("Prefill", "—" if prefill is None else f"{prefill:,.0f}"),
            expand=True,
        ),
        ft.Container(
            content=parts.fact("TTFT", "—" if ttft is None else f"{ttft * 1000:.0f} ms"),
            expand=True,
        ),
        _spark(slot["id"]),
        parts.button(
            "Unload", lambda: runtime.act(engine_api.unload(slot["id"])), "danger"
        ),
    ]
    return ft.Container(
        padding=ft.Padding(left=18, right=18, top=14, bottom=14),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=11,
        content=ft.Row(cells, spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
    )


def _recent(engine: Engine) -> ft.Control:
    rows: list[ft.Control] = [
        ft.Container(padding=ft.Padding.only(bottom=4), content=theme.eyebrow("Recent"))
    ]
    # The wire carries requests newest first, which is also the order a "Recent" reads in.
    requests = [] if engine.metrics is None else engine.metrics["requests"]
    for sample in requests[:6]:
        when = time.strftime("%H:%M", time.localtime(sample["started_at"]))
        speed = sample["tokens_per_second"]
        rows.append(
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=7, bottom=7),
                border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                content=ft.Row(
                    [
                        ft.Container(
                            width=46,
                            content=ft.Text(when, style=theme.mono(11, t().fg3), no_wrap=True),
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
                            style=theme.mono(11, t().fg3),
                            no_wrap=True,
                        ),
                    ],
                    spacing=14,
                ),
            )
        )
    if not requests:
        rows.append(
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=7, bottom=7),
                border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                content=ft.Text("Nothing served yet.", style=theme.sans(12.5, t().fg3)),
            )
        )
    return ft.Column(rows, spacing=0, tight=True)


def _row(label: str, value: ft.Control, first: bool = False) -> ft.Container:
    return ft.Container(
        padding=ft.Padding(left=16, right=16, top=11, bottom=11),
        border=None if first else ft.Border.only(top=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                ft.Text(label, style=theme.sans(13.5), no_wrap=True),
                ft.Container(expand=True),
                value,
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _measure(text: str) -> ft.Text:
    return ft.Text(text, style=theme.mono(12.5, t().fg2), no_wrap=True)


def _group(title: str, rows: list[ft.Container]) -> ft.Control:
    return ft.Column(
        [
            ft.Container(
                padding=ft.Padding(left=16, right=16, top=0, bottom=6),
                content=theme.eyebrow(title),
            ),
            ft.Container(
                bgcolor=t().elev,
                border=theme.hair(),
                border_radius=11,
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                content=ft.Column(rows, spacing=0, tight=True),
            ),
        ],
        spacing=0,
        tight=True,
    )


@ft.component
def _Settings(shell: Shell, engine: Engine, daemon: Daemon) -> ft.Control:
    # The grip writes in whole GiB; what it has not crossed yet is remembered here so a
    # drag is not a request per pixel.
    written = ft.use_ref(None)

    total = engine.system["memory_bytes"] if engine.system is not None else 128 * GIB
    ceiling = engine.ceiling or total

    def slide(fraction: float) -> None:
        whole = max(GIB, round(fraction * total / GIB) * GIB)
        if whole != written.current:
            written.current = whole
            runtime.act(engine_api.patch_config({"memory_limit_bytes": whole}))

    idle_setting = engine.config.get("idle_ttl_seconds")
    idle_value = None if idle_setting is None else idle_setting["value"]
    idle_seconds = int(idle_value) if isinstance(idle_value, int | float) else None

    def set_idle(seconds: int | None) -> None:
        runtime.act(engine_api.patch_config({"idle_ttl_seconds": seconds}))

    idle_controls: list[ft.Control] = []
    if idle_seconds is not None:
        idle_controls.append(_measure(f"{idle_seconds // 60} min"))
        idle_controls.append(
            parts.stepper(
                lambda: set_idle(max(300, idle_seconds - 300)),
                lambda: set_idle(idle_seconds + 300),
            )
        )
    else:
        idle_controls.append(_measure("off"))
    idle_controls.append(
        parts.toggle(
            idle_seconds is not None,
            lambda: set_idle(None if idle_seconds is not None else 1800),
        )
    )

    engine_group = _group(
        "Engine",
        [
            _row("Port", _measure(port_of(engine)), first=True),
            _row(
                "Memory ceiling",
                ft.Row(
                    [
                        parts.Slider(160, ceiling / total, slide),
                        _measure(f"{gb(ceiling, 0)} GB"),
                    ],
                    spacing=12,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
            _row(
                "Unload after idle",
                ft.Row(
                    idle_controls,
                    spacing=10,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
        ],
    )
    endpoints = _group(
        "Endpoints",
        [
            _row("OpenAI", _measure("/api/openai/v1 · active"), first=True),
            _row("Anthropic", _measure("/api/anthropic · active")),
        ],
    )
    storage_rows = [
        _row(
            "Checkpoints",
            _measure(engine.system["catalog"] if engine.system is not None else "—"),
            first=True,
        ),
        _row(
            "Disk free",
            _measure(
                f"{gb(engine.system['disk_free_bytes'], 0)} GB"
                if engine.system is not None
                else "—"
            ),
        ),
    ]
    appearance = _group(
        "Appearance",
        [
            _row(
                "Theme",
                parts.scope(
                    [("light", "Light"), ("system", "System"), ("dark", "Dark")],
                    shell.mode,
                    shell.choose,
                ),
                first=True,
            ),
        ],
    )
    ours = daemon.owned
    daemon_group = _group(
        "Daemon",
        [
            _row(
                "This window's process" if ours else "Started elsewhere",
                ft.Row(
                    [
                        parts.button("Restart", lambda: runtime.act(daemon.restart())),
                        parts.button("Stop", lambda: runtime.act(daemon.stop()), "danger"),
                    ],
                    spacing=8,
                    tight=True,
                ),
                first=True,
            ),
        ],
    )
    return ft.Column(
        [engine_group, endpoints, _group("Storage", storage_rows), appearance, daemon_group],
        spacing=20,
        tight=True,
    )


@ft.component
def Server(shell: Shell, engine: Engine, daemon: Daemon) -> ft.Control:
    chosen, set_chosen = ft.use_state("")
    # The pulse ages with nothing else to redraw it.
    runtime.use_tick()

    slots = engine.models
    selected = next((s for s in slots if s["id"] == chosen), slots[0] if slots else None)
    total = engine.system["memory_bytes"] if engine.system is not None else 128 * GIB
    used = 0 if engine.state is None else engine.state["resident_bytes"]
    ceiling = engine.ceiling

    header = ft.Row(
        [
            ft.Text("Server", style=theme.display(20)),
            ft.Container(expand=True),
            ft.Text(
                f"{gb(used)} of {gb(total, 0)} GB"
                + (f" · ceiling {gb(ceiling, 0)} GB" if ceiling is not None else ""),
                style=theme.mono(11.5, t().fg3),
                no_wrap=True,
            ),
        ],
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    body: list[ft.Control] = [header, ft.Container(height=12)]
    if engine.health is None:
        body.append(
            ft.Container(
                height=TRACK_HEIGHT,
                border_radius=8,
                bgcolor=theme.mix(t().sel, 0.55, t().surface),
                border=theme.hair(),
                alignment=ft.Alignment.CENTER,
                content=ft.Text(
                    f"The engine is not answering on :{port_of(engine)} — starting it…",
                    style=theme.sans(12.5, t().fg3),
                ),
            )
        )
    else:
        body.append(_band(engine, total, chosen, set_chosen))
        if selected is not None:
            body.append(ft.Container(height=14))
            body.append(_stats(engine, selected))
    body.append(ft.Container(height=16))
    body.append(_recent(engine))
    body.append(ft.Container(height=24))
    body.append(_Settings(shell, engine, daemon))

    return ft.Column(
        [
            ft.WindowDragArea(content=ft.Container(height=28), maximizable=False),
            ft.Container(
                expand=True,
                padding=ft.Padding(left=PAD, right=PAD, top=0, bottom=PAD),
                content=ft.Column(body, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
            ),
        ],
        spacing=0,
        expand=True,
    )

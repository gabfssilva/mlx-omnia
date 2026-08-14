"""Benchmark: the fourth place. One row per checkpoint, the three axes as columns —
Speed off the newest speed run, Quality and Fidelity off theirs. Nothing here compares
engines, and nothing recomputes what the server decided: every number is the run's own.

Run starts the daemon's speed benchmark for that model — interleaved rounds, thermally
gated on the server's side — and the jobs stream is what draws the progress.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import flet as ft

from mlx_omnia.app.api import benchmarks
from mlx_omnia.app.api.engine import Engine, Job
from mlx_omnia.app.ui.format import display_name
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

COLUMNS = (("Decode", 74), ("Prefill", 64), ("TTFT", 56), ("Quality", 76), ("Fidelity", 86))

RUN_SHAPE: dict[str, object] = {
    "kind": "speed",
    "contexts": [4096],
    "generates": [256],
    "concurrencies": [1],
    "rounds": 3,
    "thermal_gate_c": 50,
    "skip_if_measured": False,
}


def _latest(rows: list[benchmarks.Run]) -> dict[str, benchmarks.Run]:
    """The newest ok run per model."""
    newest: dict[str, benchmarks.Run] = {}
    for run in rows:
        if run["state"] != "ok":
            continue
        held = newest.get(run["model"])
        if held is None or run["created_at"] > held["created_at"]:
            newest[run["model"]] = run
    return newest


def _bench_job(engine: Engine, model: str) -> Job | None:
    for job in engine.jobs:
        if job["kind"] not in ("bench", "benchmark"):
            continue
        subject = job["subject"].get("models") or job["subject"].get("model")
        names = subject if isinstance(subject, list) else [subject]
        if model in names and job["state"] in ("pending", "running"):
            return job
    return None


def _verdict(text: str | None) -> ft.Container:
    known = text is not None
    return ft.Container(
        padding=ft.Padding(left=9, right=9, top=2, bottom=2),
        bgcolor=t().accent_soft if known else t().sel,
        border_radius=999,
        content=ft.Text(
            text if known else "—",
            style=theme.sans(10.5, t().accent if known else t().fg3, ft.FontWeight.W_500),
            no_wrap=True,
        ),
    )


def _cell(width: float, content: ft.Control) -> ft.Container:
    return ft.Container(width=width, content=content)


def _head() -> ft.Control:
    cells: list[ft.Control] = [ft.Container(content=theme.eyebrow("Model"), expand=True)]
    cells.extend(_cell(width, theme.eyebrow(label)) for label, width in COLUMNS)
    return ft.Container(
        padding=ft.Padding(left=14, right=18, top=4, bottom=2),
        content=ft.Row(cells, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
    )


def _mono(text: str, sub: str | None = None) -> ft.Control:
    if sub is None:
        return ft.Text(text, style=theme.mono(11.5, t().fg2), no_wrap=True)
    return ft.Column(
        [
            ft.Text(text, style=theme.mono(11.5, t().fg2), no_wrap=True),
            ft.Text(sub, style=theme.mono(9, t().fg3), no_wrap=True),
        ],
        spacing=1,
        tight=True,
    )


def _name(engine: Engine, model: str, resident: bool) -> ft.Control:
    color = t().mat(engine.materials.get(model, 0)) if resident else t().sel
    return ft.Row(
        [
            ft.Container(width=3.5, height=30, border_radius=2, bgcolor=color),
            ft.Text(
                display_name(model),
                style=theme.sans(13.5, weight=ft.FontWeight.W_600),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
                expand=True,
            ),
        ],
        spacing=10,
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _quality_pill(run: benchmarks.Run | None) -> ft.Container:
    if run is None or run["quality"] is None or run["quality"]["accuracy"] is None:
        return _verdict(None)
    return _verdict(f"{run['quality']['accuracy']:.1%}")


def _fidelity_pill(run: benchmarks.Run | None) -> ft.Container:
    if run is None or run["fidelity"] is None:
        return _verdict(None)
    body = run["fidelity"]
    if body["top1"] is not None:
        return _verdict(f"top1 {body['top1']:.1%}")
    if body["kl_mean"] is not None:
        return _verdict(f"KL {body['kl_mean']:.2e}")
    return _verdict(None)


def _axes(
    speed: benchmarks.Run | None,
    quality: benchmarks.Run | None,
    fidelity: benchmarks.Run | None,
) -> ft.Control:
    """The row, opened: one line per axis, the run's key beside its numbers."""

    def axis(
        label: str, value: str, verdict: ft.Container, first: bool = False
    ) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=2, right=2, top=8, bottom=8),
            border=None if first else ft.Border.only(top=ft.BorderSide(1, t().hair)),
            content=ft.Row(
                [
                    ft.Container(
                        width=64,
                        content=ft.Text(
                            label, style=theme.sans(12.5, weight=ft.FontWeight.W_600),
                            no_wrap=True,
                        ),
                    ),
                    ft.Text(value, style=theme.mono(11, t().fg2), no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                    verdict,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def described(run: benchmarks.Run | None) -> str:
        return "not measured" if run is None else run["key"]

    body = None if speed is None else speed["speed"]
    share = None if body is None else body["ceiling_fraction"]
    speed_pill = _verdict(None if share is None else f"{share:.0%} of ceiling")
    return ft.Container(
        margin=ft.Margin.only(top=10),
        border=ft.Border.only(top=ft.BorderSide(1, t().hair2)),
        content=ft.Column(
            [
                axis("Speed", described(speed), speed_pill, first=True),
                axis("Quality", described(quality), _quality_pill(quality)),
                axis("Fidelity", described(fidelity), _fidelity_pill(fidelity)),
            ],
            spacing=0,
            tight=True,
        ),
    )


@ft.component
def _Row(
    engine: Engine,
    model: str,
    resident: bool,
    supported: bool,
    speed: benchmarks.Run | None,
    quality: benchmarks.Run | None,
    fidelity: benchmarks.Run | None,
    on: bool,
    pick: Callable[[], None],
) -> ft.Control:
    job = _bench_job(engine, model)
    body = None if speed is None else speed["speed"]

    cells: list[ft.Control] = [_name(engine, model, resident)]
    if job is not None:
        span = sum(width for _, width in COLUMNS) + 4 * 10
        message = job["progress"]["message"] or "running…"
        cells.append(
            _cell(span, ft.Text(message, style=theme.mono(11.5, t().fg3), no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS))
        )
    elif body is not None:
        decode = body["decode_tps"]
        share = body["ceiling_fraction"]
        prefill = body["prefill_tps"]
        ttft = body["ttft_p50_ms"]
        cells.append(
            _cell(
                COLUMNS[0][1],
                _mono(
                    "—" if decode is None else f"{decode:.1f}",
                    None if share is None else f"{share:.0%} ceil",
                ),
            )
        )
        cells.append(
            _cell(COLUMNS[1][1], _mono("—" if prefill is None else f"{prefill:,.0f}"))
        )
        cells.append(_cell(COLUMNS[2][1], _mono("—" if ttft is None else f"{ttft:.0f} ms")))
        cells.append(_cell(COLUMNS[3][1], _quality_pill(quality)))
        cells.append(_cell(COLUMNS[4][1], _fidelity_pill(fidelity)))
    else:
        span = sum(width for _, width in COLUMNS) + 4 * 10

        def start() -> None:
            runtime.act(_start(model))

        cells.append(
            _cell(
                span - COLUMNS[4][1] - 10,
                ft.Text(
                    "not measured" if supported else "no loader for this architecture",
                    style=theme.mono(11.5, t().fg3),
                    no_wrap=True,
                ),
            )
        )
        cells.append(
            _cell(
                COLUMNS[4][1],
                parts.button("Run", start) if supported else ft.Container(),
            )
        )

    inner: list[ft.Control] = [
        ft.Row(cells, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    ]
    measured = body is not None
    if on and measured:
        inner.append(_axes(speed, quality, fidelity))

    def face(tint: str | None) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=14, right=18, top=11, bottom=11),
            bgcolor=tint,
            border=theme.hair(t().accent if on else t().hair),
            border_radius=10,
            content=ft.Column(inner, spacing=0, tight=True),
        )

    return parts.press(
        face,
        pick if measured else None,
        t().elev,
        theme.mix(t().fg, 0.03, t().elev),
        t().sel,
    )


async def _start(model: str) -> None:
    await benchmarks.start({**RUN_SHAPE, "models": [model]})


@ft.component
def Benchmark(shell: Shell, engine: Engine) -> ft.Control:
    opened, set_opened = ft.use_state("")
    speed, set_speed = ft.use_state(dict[str, benchmarks.Run]())
    quality, set_quality = ft.use_state(dict[str, benchmarks.Run]())
    fidelity, set_fidelity = ft.use_state(dict[str, benchmarks.Run]())

    active = sum(
        1
        for job in engine.jobs
        if job["kind"] in ("bench", "benchmark") and job["state"] in ("pending", "running")
    )

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                set_speed(_latest(await benchmarks.runs("speed")))
                set_quality(_latest(await benchmarks.runs("quality")))
                set_fidelity(_latest(await benchmarks.runs("fidelity")))
            except Exception:  # noqa: BLE001 — a daemon that is down has no runs to show
                pass

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    # Refetched when the bench queue drains, so a finished run lands without a reopen.
    ft.use_effect(load, [active])

    rows: list[ft.Control] = [
        ft.WindowDragArea(content=ft.Container(height=24), maximizable=False),
        ft.Container(
            padding=ft.Padding(left=2, right=2, top=0, bottom=8),
            content=ft.Text("Benchmark", style=theme.display(20)),
        ),
    ]
    if engine.catalog:
        rows.append(_head())
        loaded = runtime.resident_ids(engine)
        rows.extend(
            _Row(
                engine,
                entry["id"],
                entry["id"] in loaded,
                entry["supported"],
                speed.get(entry["id"]),
                quality.get(entry["id"]),
                fidelity.get(entry["id"]),
                opened == entry["id"],
                lambda entry=entry: set_opened(
                    "" if opened == entry["id"] else entry["id"]
                ),
            )
            for entry in engine.catalog
        )
    else:
        rows.append(
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=24, bottom=0),
                alignment=ft.Alignment.CENTER,
                content=ft.Text(
                    "Nothing on disk to measure — fetch a model under Models.",
                    style=theme.sans(12.5, t().fg3),
                ),
            )
        )

    return ft.Container(
        expand=True,
        padding=ft.Padding(left=16, right=16, top=0, bottom=16),
        content=ft.Column(rows, spacing=8, scroll=ft.ScrollMode.AUTO, expand=True),
    )

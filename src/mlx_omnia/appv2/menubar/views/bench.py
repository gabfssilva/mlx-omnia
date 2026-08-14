"""Bench: the window's five columns, folded into two lines.

Decode, prefill and TTFT do not fit across 380 pt beside a name, so the numbers move under
it and the verdict — the share of the ceiling — keeps the right edge, which is the column
that is actually scanned. Tapping a measured row opens the three axes and the key of the
run each came from. Nothing here recomputes what the run decided.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import flet as ft

from mlx_omnia.app.api import benchmarks
from mlx_omnia.app.api.engine import CatalogEntry, Engine, Job
from mlx_omnia.app.ui.format import display_name
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.menubar.panel import Panel, empty
from mlx_omnia.appv2.theme import t

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
    def described(run: benchmarks.Run | None) -> str:
        return "not measured" if run is None else run["key"]

    def axis(label: str, value: str, verdict: ft.Container, first: bool = False) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=2, right=2, top=7, bottom=7),
            border=None if first else ft.Border.only(top=ft.BorderSide(1, t().hair)),
            content=ft.Row(
                [
                    ft.Container(
                        width=56,
                        content=ft.Text(
                            label, style=theme.sans(12.5, weight=ft.FontWeight.W_600), no_wrap=True
                        ),
                    ),
                    ft.Text(
                        value,
                        style=theme.mono(10, t().fg2),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                    verdict,
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    body = None if speed is None else speed["speed"]
    share = None if body is None else body["ceiling_fraction"]
    return ft.Container(
        margin=ft.Margin.only(top=9),
        border=ft.Border.only(top=ft.BorderSide(1, t().hair2)),
        content=ft.Column(
            [
                axis(
                    "Speed",
                    described(speed),
                    _verdict(None if share is None else f"{share:.0%} of ceiling"),
                    first=True,
                ),
                axis("Quality", described(quality), _quality_pill(quality)),
                axis("Fidelity", described(fidelity), _fidelity_pill(fidelity)),
            ],
            spacing=0,
            tight=True,
        ),
    )


def _row(
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
    measured = body is not None

    tail: ft.Control
    second: ft.Control
    track: ft.Control | None = None
    if job is not None:
        total = job["progress"]["total"]
        done = job["progress"]["completed"]
        fraction = 0.0 if not total else min(1.0, done / total)
        tail = ft.Text(
            f"{round(done / total * 100)}%" if total else "running",
            style=theme.mono(10, t().fg3),
            no_wrap=True,
        )
        track = ft.Container(
            height=3,
            margin=ft.Margin.only(top=8, bottom=5),
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
        )
        second = ft.Text(
            job["progress"]["message"] or "running…",
            style=theme.mono(10, t().fg3),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    elif body is not None:
        share = body["ceiling_fraction"]
        decode, prefill, ttft = body["decode_tps"], body["prefill_tps"], body["ttft_p50_ms"]
        tail = _verdict(None if share is None else f"{share:.0%} of ceiling")
        second = ft.Text(
            " · ".join(
                [
                    "—" if decode is None else f"{decode:.1f} decode",
                    "—" if prefill is None else f"{prefill:,.0f} prefill",
                    "—" if ttft is None else f"{ttft:.0f} ms TTFT",
                ]
            ),
            style=theme.mono(10, t().fg2),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    elif supported:
        tail = parts.button("Run", lambda: runtime.act(_start(model)))
        second = ft.Text("not measured", style=theme.mono(10, t().fg3), no_wrap=True)
    else:
        tail = parts.pill("Unsupported", "hub")
        second = ft.Text(
            "no loader for this architecture", style=theme.mono(10, t().fg3), no_wrap=True
        )

    inner: list[ft.Control] = [
        ft.Row(
            [
                ft.Container(
                    width=3.5,
                    height=28,
                    border_radius=2,
                    bgcolor=t().mat(engine.materials.get(model, 0)) if resident else t().sel,
                ),
                ft.Text(
                    display_name(model),
                    style=theme.sans(13.5, weight=ft.FontWeight.W_600),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                ),
                tail,
            ],
            spacing=9,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    ]
    if track is not None:
        inner.extend((track, second))
    else:
        inner.append(ft.Container(margin=ft.Margin.only(top=6), content=second))
    if on and measured:
        inner.append(_axes(speed, quality, fidelity))

    def face(tint: str | None) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=12, right=12, top=10, bottom=10),
            bgcolor=tint,
            border=theme.hair(t().accent if on else t().hair),
            border_radius=12,
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
def Bench(panel: Panel, engine: Engine) -> ft.Control:
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

    if not engine.catalog:
        return ft.Column(
            [empty("Nothing on disk to measure — fetch a model under Models.")],
            spacing=0,
            expand=True,
        )

    loaded = runtime.resident_ids(engine)
    rows: list[ft.Control] = [
        ft.Container(
            padding=ft.Padding(left=2, right=2, top=0, bottom=4),
            content=ft.Text(
                "Interleaved and thermally gated on the daemon's side. Nothing here "
                "recomputes what the run decided.",
                style=theme.sans(12, t().fg3),
            ),
        )
    ]

    def rank(entry: CatalogEntry) -> int:
        """Measured, then moving, then the rest. A disk with 78 checkpoints buries the
        four that have numbers if the catalog's own order is kept."""
        if speed.get(entry["id"]) is not None:
            return 0
        if _bench_job(engine, entry["id"]) is not None:
            return 1
        return 2 if entry["supported"] else 3

    rows.extend(
        _row(
            engine,
            entry["id"],
            entry["id"] in loaded,
            entry["supported"],
            speed.get(entry["id"]),
            quality.get(entry["id"]),
            fidelity.get(entry["id"]),
            opened == entry["id"],
            lambda entry=entry: set_opened("" if opened == entry["id"] else entry["id"]),
        )
        for entry in sorted(engine.catalog, key=rank)
    )
    return ft.Column(rows, spacing=7, scroll=ft.ScrollMode.AUTO, expand=True)

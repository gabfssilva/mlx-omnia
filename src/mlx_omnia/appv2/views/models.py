"""Models: the one shelf, in sections instead of a filter — what the catalog stream
carries, plus what the Hub answers when the search asks it.

Each row's tail says something: the measured trio on a resident, whether an on-disk
one fits under the ceiling, Download right on a Hub row, cancel on what's coming down.
Clicking a catalog row pushes the model over the list, whole-screen — `model.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import flet as ft

from mlx_omnia.app.api import catalog as catalog_api
from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.downloads import Downloads, Pull
from mlx_omnia.app.api.engine import CatalogEntry, Engine
from mlx_omnia.app.ui.format import display_name, gb, rate, tokens
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t
from mlx_omnia.appv2.views.model import ModelScreen


def _bar(color: str) -> ft.Control:
    return ft.Container(width=4, height=52, border_radius=2, bgcolor=color)


def _meta(entry: CatalogEntry) -> str:
    quant = entry["quantization"] or entry["dtype"] or "—"
    ctx = "—" if entry["context"] is None else tokens(entry["context"])
    return f"{entry['architecture']} · {quant} · {gb(entry['bytes_on_disk'])} GB · ctx {ctx}"


def _tail(engine: Engine, entry: CatalogEntry) -> list[ft.Control]:
    if entry["id"] in runtime.resident_ids(engine):
        cells: list[ft.Control] = []
        decode = runtime.decode_of(engine, entry["id"])
        if decode is not None:
            cells.append(parts.chip(decode))
        pair = runtime.prefill_ttft_of(engine, entry["id"])
        if pair is not None:
            cells.append(ft.Text(pair, style=theme.mono(10, t().fg3), no_wrap=True))
        cells.append(parts.pill("Resident", "resident"))
        return cells
    if not entry["supported"]:
        return [
            ft.Text(
                f"no loader for {entry['architecture']}",
                style=theme.mono(10, t().fg3),
                no_wrap=True,
            ),
            parts.pill("Unsupported", "hub"),
        ]
    ceiling = engine.ceiling
    used = 0 if engine.state is None else engine.state["resident_bytes"]
    if ceiling is not None:
        free = ceiling - used
        verdict = (
            f"fits · {gb(free, 0)} GB free"
            if entry["bytes_on_disk"] <= free
            else f"needs {gb(entry['bytes_on_disk'] - free, 0)} GB more"
        )
        return [
            ft.Text(verdict, style=theme.mono(10, t().fg3), no_wrap=True),
            parts.pill("On disk", "disk"),
        ]
    return [parts.pill("On disk", "disk")]


def _row(
    lead: ft.Control,
    name: str,
    identifier: str,
    meta: list[ft.Control],
    tail: list[ft.Control],
    open_model: Callable[[], None] | None,
) -> ft.Control:
    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=14, right=18, top=17, bottom=17),
            bgcolor=tint,
            border=theme.hair(t().hair),
            border_radius=12,
            content=ft.Row(
                [
                    lead,
                    ft.Column(
                        [
                            ft.Text(
                                name,
                                style=theme.sans(16, weight=ft.FontWeight.W_600),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                identifier,
                                style=theme.mono(11, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            *meta,
                        ],
                        spacing=3,
                        tight=True,
                        expand=True,
                    ),
                    ft.Column(
                        tail,
                        spacing=5,
                        tight=True,
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                    ),
                ],
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    return parts.press(body, open_model, t().elev, theme.mix(t().fg, 0.03, t().elev), t().sel)


def _catalog_row(engine: Engine, entry: CatalogEntry, open_model: Callable[[], None]) -> ft.Control:
    material = (
        t().mat(engine.materials.get(entry["id"], 0))
        if entry["id"] in runtime.resident_ids(engine)
        else t().sel
    )
    meta: list[ft.Control] = [
        ft.Text(
            _meta(entry),
            style=theme.sans(12.5, t().fg2),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    ]
    return _row(
        _bar(material),
        display_name(entry["id"]),
        entry["id"],
        meta,
        _tail(engine, entry),
        open_model,
    )


def _download_row(pull: Pull) -> ft.Control:
    fraction = pull.fraction
    meta: list[ft.Control] = [
        ft.Text(
            pull.error if pull.error is not None else pull.message,
            style=theme.sans(12.5, t().bad if pull.error is not None else t().fg2),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    ]
    if fraction is not None:
        meta.append(
            ft.Container(
                height=4,
                border_radius=2,
                bgcolor=t().sel,
                margin=ft.Margin.only(top=6),
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
        )
    speed = "" if pull.rate is None else f" · {rate(pull.rate)}"
    tail: list[ft.Control] = [
        ft.Text(f"{pull.share}{speed}", style=theme.mono(10, t().fg3), no_wrap=True),
        ft.Container(
            content=ft.Text("✕ Cancel", style=theme.sans(11.5, t().fg2), no_wrap=True),
            on_click=lambda _: runtime.act(engine_api.cancel_job(pull.job)),
        ),
    ]
    return _row(_bar(t().sel), display_name(pull.repo), pull.repo, meta, tail, None)


def _hub_row(found: catalog_api.HubModel, downloads: Downloads) -> ft.Control:
    def start() -> None:
        async def pull() -> None:
            downloads.follow(await catalog_api.pull(found["id"]))

        runtime.act(pull())

    counts: list[str] = []
    if found["downloads"] is not None:
        counts.append(f"{found['downloads']:,} downloads")
    if found["likes"] is not None:
        counts.append(f"{found['likes']:,} likes")
    meta: list[ft.Control] = [
        ft.Text(
            " · ".join(counts) if counts else "on the Hub",
            style=theme.sans(12.5, t().fg2),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    ]
    tail: list[ft.Control] = [parts.button("Download", start, "primary")]
    return _row(_bar(t().sel), display_name(found["id"]), found["id"], meta, tail, None)


@ft.component
def Models(shell: Shell, engine: Engine, downloads: Downloads) -> ft.Control:
    query, set_query = ft.use_state("")
    opened, set_opened = ft.use_state("")
    hub, set_hub = ft.use_state(list[catalog_api.HubModel]())

    def look() -> Callable[[], None]:
        async def search() -> None:
            if len(query) < 2:
                set_hub([])
                return
            # A breath after the keystroke, so the Hub is asked about words and not letters.
            await asyncio.sleep(0.35)
            try:
                set_hub(await catalog_api.search_hub(query))
            except Exception:  # noqa: BLE001 — the Hub not answering leaves the section out
                set_hub([])

        task = asyncio.get_running_loop().create_task(search())
        return task.cancel

    ft.use_effect(look, [query])

    if opened:
        entry = engine.entry(opened)
        if entry is not None:
            return ModelScreen(shell, engine, entry, lambda: set_opened(""))
        set_opened("")

    needle = query.lower()

    def matches(identifier: str) -> bool:
        return needle in identifier.lower()

    loaded = runtime.resident_ids(engine)
    on_disk = [e for e in engine.catalog if matches(e["id"])]
    resident = [e for e in on_disk if e["id"] in loaded]
    idle = [e for e in on_disk if e["id"] not in loaded]
    pulls = [p for p in downloads.active.values() if matches(p.repo)]
    known = {e["id"] for e in engine.catalog} | {p.repo for p in pulls}
    found = [h for h in hub if h["id"] not in known]

    rows: list[ft.Control] = [
        ft.WindowDragArea(
            content=parts.searchbox("Search disk and the Hub…", query, set_query),
            maximizable=False,
        )
    ]

    def section(title: str, count: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding(left=2, right=2, top=8, bottom=0),
            content=theme.eyebrow(f"{title}{count}"),
        )

    if resident:
        rows.append(section("Resident", f" · {len(resident)}"))
        rows.extend(
            _catalog_row(engine, entry, lambda entry=entry: set_opened(entry["id"]))
            for entry in resident
        )
    if idle or pulls:
        rows.append(section("On this Mac", f" · {len(idle) + len(pulls)}"))
        rows.extend(_download_row(pull) for pull in pulls)
        rows.extend(
            _catalog_row(engine, entry, lambda entry=entry: set_opened(entry["id"]))
            for entry in idle
        )
    if found:
        rows.append(section("From the Hub", ""))
        rows.extend(_hub_row(one, downloads) for one in found)

    if not (resident or idle or pulls or found):
        message = (
            "Nothing on disk matches — search the Hub."
            if query
            else "Nothing on disk yet — search the Hub above."
        )
        rows.append(
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=24, bottom=0),
                alignment=ft.Alignment.CENTER,
                content=ft.Text(message, style=theme.sans(12.5, t().fg3)),
            )
        )

    return ft.Container(
        expand=True,
        padding=16,
        content=ft.Column(rows, spacing=10, scroll=ft.ScrollMode.AUTO, expand=True),
    )

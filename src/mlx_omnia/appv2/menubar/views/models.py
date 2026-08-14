"""Models: the window's shelf at the panel's width.

The window gives a row three lines — name, id, meta — and 86 pt. Here a row is two lines
and 52: the id goes, because the name and the meta together already say which checkpoint
this is, and the id is what the screen behind this one is for.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import flet as ft

from mlx_omnia.app.api import catalog as catalog_api
from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.downloads import Downloads, Pull
from mlx_omnia.app.api.engine import CatalogEntry, Engine
from mlx_omnia.app.ui.format import display_name, gb, left, rate, size_pair, tokens
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.menubar.panel import Panel, empty
from mlx_omnia.appv2.theme import t


def _meta(entry: CatalogEntry) -> str:
    quant = entry["quantization"] or entry["dtype"] or "—"
    context = "—" if entry["context"] is None else tokens(entry["context"])
    return f"{entry['architecture']} · {quant} · {gb(entry['bytes_on_disk'])} GB · ctx {context}"


def _row(
    material: str,
    name: str,
    meta: str,
    meta_color: str | None,
    tail: list[ft.Control],
    open_model: Callable[[], None] | None,
    extra: list[ft.Control] | None = None,
) -> ft.Control:
    def body(tint: str | None) -> ft.Container:
        inner: list[ft.Control] = [
            ft.Row(
                [
                    ft.Container(width=4, height=32, border_radius=2, bgcolor=material),
                    ft.Column(
                        [
                            ft.Text(
                                name,
                                style=theme.sans(13.5, weight=ft.FontWeight.W_600),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                meta,
                                style=theme.sans(11.5, meta_color or t().fg2),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=2,
                        tight=True,
                        expand=True,
                    ),
                    ft.Column(
                        tail, spacing=4, tight=True, horizontal_alignment=ft.CrossAxisAlignment.END
                    ),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        ]
        if extra:
            inner.extend(extra)
        return ft.Container(
            padding=ft.Padding(left=12, right=12, top=10, bottom=10),
            bgcolor=tint,
            border=theme.hair(),
            border_radius=12,
            content=ft.Column(inner, spacing=0, tight=True),
        )

    return parts.press(
        body, open_model, t().elev, theme.mix(t().fg, 0.03, t().elev), t().sel
    )


def _catalog_row(engine: Engine, entry: CatalogEntry, open_model: Callable[[], None]) -> ft.Control:
    identifier = entry["id"]
    resident = identifier in runtime.resident_ids(engine)
    material = t().mat(engine.materials.get(identifier, 0)) if resident else t().sel

    tail: list[ft.Control]
    if resident:
        tail = [parts.pill("Resident", "resident")]
    elif not entry["supported"]:
        tail = [parts.pill("Unsupported", "hub")]
    else:
        tail = [parts.pill("On disk", "disk")]
        ceiling = engine.ceiling
        used = 0 if engine.state is None else engine.state["resident_bytes"]
        if ceiling is not None:
            free = ceiling - used
            verdict = (
                f"fits · {gb(free, 0)} GB free"
                if entry["bytes_on_disk"] <= free
                else f"needs {gb(entry['bytes_on_disk'] - free, 0)} GB more"
            )
            tail.insert(0, ft.Text(verdict, style=theme.mono(10, t().fg3), no_wrap=True))

    return _row(material, display_name(identifier), _meta(entry), None, tail, open_model)


def _download_row(pull: Pull) -> ft.Control:
    fraction = pull.fraction
    extra: list[ft.Control] = []
    if fraction is not None:
        extra.append(
            ft.Container(
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
        )
    said = [pull.share if pull.total is None else size_pair(pull.completed, pull.total)]
    if pull.rate is not None:
        said.append(rate(pull.rate))
    remaining = pull.eta
    if remaining is not None:
        said.append(left(remaining))
    extra.append(
        ft.Text(" · ".join(said), style=theme.mono(10, t().fg3), no_wrap=True)
    )
    tail: list[ft.Control] = [
        ft.Container(
            content=ft.Text("✕", style=theme.sans(11, t().fg3), no_wrap=True),
            on_click=lambda _: runtime.act(engine_api.cancel_job(pull.job)),
        )
    ]
    return _row(
        t().sel,
        display_name(pull.repo),
        pull.error if pull.error is not None else pull.message,
        t().bad if pull.error is not None else None,
        tail,
        None,
        extra,
    )


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
    return _row(
        t().sel,
        display_name(found["id"]),
        " · ".join(counts) if counts else "on the Hub",
        None,
        [parts.button("Download", start, "primary")],
        None,
    )


@ft.component
def Models(panel: Panel, engine: Engine, downloads: Downloads) -> ft.Control:
    query, set_query = ft.use_state("")
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

    needle = query.lower()

    def matches(identifier: str) -> bool:
        return needle in identifier.lower()

    loaded = runtime.resident_ids(engine)
    on_disk = [entry for entry in engine.catalog if matches(entry["id"])]
    resident = [entry for entry in on_disk if entry["id"] in loaded]
    idle = [entry for entry in on_disk if entry["id"] not in loaded]
    pulls = [pull for pull in downloads.active.values() if matches(pull.repo)]
    known = {entry["id"] for entry in engine.catalog} | {pull.repo for pull in pulls}
    found = [one for one in hub if one["id"] not in known]

    def section(title: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding(left=2, right=2, top=6, bottom=0), content=theme.eyebrow(title)
        )

    def open_model(identifier: str) -> Callable[[], None]:
        return lambda: setattr(panel, "opened", identifier)

    rows: list[ft.Control] = [parts.searchbox("Search disk and the Hub…", query, set_query)]
    if resident:
        rows.append(section(f"Resident · {len(resident)}"))
        rows.extend(
            _catalog_row(engine, entry, open_model(entry["id"])) for entry in resident
        )
    if idle or pulls:
        rows.append(section(f"On this Mac · {len(idle) + len(pulls)}"))
        rows.extend(_download_row(pull) for pull in pulls)
        rows.extend(_catalog_row(engine, entry, open_model(entry["id"])) for entry in idle)
    if found:
        rows.append(section("From the Hub"))
        rows.extend(_hub_row(one, downloads) for one in found)

    if not (resident or idle or pulls or found):
        rows.append(
            empty(
                "Nothing on disk matches — search the Hub."
                if query
                else "Nothing on disk yet — search the Hub above."
            )
        )

    return ft.Column(rows, spacing=7, scroll=ft.ScrollMode.AUTO, expand=True)

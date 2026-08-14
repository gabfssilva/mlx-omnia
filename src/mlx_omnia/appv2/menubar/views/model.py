"""One model, pushed over the shelf. What the checkpoint declares, what it measures, the
speculation the daemon can build for it, and the three verbs.

The window's Files tab and its profile editor are not here: both are reading and writing at
a length a 380 pt panel makes worse, and the window is one item down the ⋯ menu. What a
profile sets does show — knowing a model is served under `strict` at temp 0 is the kind of
thing that is looked up, not edited.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine

import flet as ft

from mlx_omnia.app.api import catalog as catalog_api
from mlx_omnia.app.api.engine import CatalogEntry, Engine
from mlx_omnia.app.ui.format import display_name, gb, tokens
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.menubar.panel import Panel
from mlx_omnia.appv2.theme import t


async def _fire(work: Coroutine[None, None, object]) -> None:
    """Await a coroutine whose answer — a job's first frame — this screen does not keep;
    the jobs stream is what draws it from here."""
    await work


def _pillbox(name: str, on: bool) -> ft.Container:
    return ft.Container(
        padding=ft.Padding(left=9, right=9, top=2, bottom=2),
        bgcolor=t().accent_soft if on else t().sel,
        border_radius=999,
        content=ft.Text(
            name,
            style=theme.sans(11, t().accent if on else t().fg3, ft.FontWeight.W_500),
            no_wrap=True,
        ),
    )


def _setting(label: str, value: ft.Control, dim: bool = False, first: bool = False) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=2, right=2, top=9, bottom=9),
        border=None if first else ft.Border.only(top=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                ft.Text(label, style=theme.sans(13, t().fg3 if dim else t().fg), no_wrap=True),
                ft.Container(expand=True),
                value,
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


@ft.component
def _Features(entry: CatalogEntry) -> ft.Control:
    identifier = entry["id"]
    settings, set_settings = ft.use_state(None)

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                set_settings(await catalog_api.get_settings(identifier))
            except Exception:  # noqa: BLE001 — a daemon that is down settles for nothing
                set_settings(None)

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [identifier])

    if settings is None:
        return _setting(
            "Speculation",
            ft.Text("asking the daemon…", style=theme.mono(10.5, t().fg3), no_wrap=True),
            dim=True,
            first=True,
        )

    speculation = settings["features"]["speculation"]
    kind = None if speculation is None else speculation["kind"]
    drafter = None if speculation is None else speculation["drafter"]

    def choose(chosen_kind: str | None, chosen_drafter: str | None) -> None:
        async def write() -> None:
            body: catalog_api.Features = {
                "speculation": None
                if chosen_kind is None
                else {"kind": chosen_kind, "drafter": chosen_drafter, "block_size": None}
            }
            set_settings(await catalog_api.save_settings(identifier, body))

        runtime.act(write())

    def option(label: str, on: bool, pick: Callable[[], None]) -> ft.Control:
        return ft.Container(
            padding=ft.Padding(left=10, right=10, top=3, bottom=3),
            border=theme.hair(t().accent if on else t().hair),
            bgcolor=t().accent_soft if on else None,
            border_radius=999,
            on_click=lambda _: pick(),
            content=ft.Text(
                label, style=theme.sans(11.5, t().fg if on else t().fg2), no_wrap=True
            ),
        )

    choices: list[ft.Control] = [option("off", kind is None, lambda: choose(None, None))]
    if settings["mtp_available"]:
        choices.append(option("mtp", kind == "mtp", lambda: choose("mtp", None)))
    choices.extend(
        option(
            display_name(candidate),
            kind == "dflash" and drafter == candidate,
            lambda candidate=candidate: choose("dflash", candidate),
        )
        for candidate in settings["available"]
    )

    rows: list[ft.Control] = [
        _setting(
            "Speculation",
            ft.Row(
                choices,
                spacing=6,
                tight=True,
                wrap=True,
                alignment=ft.MainAxisAlignment.END,
            ),
            first=True,
        )
    ]
    if settings["unavailable_reason"] is not None:
        rows.append(
            ft.Container(
                padding=ft.Padding.only(top=2),
                alignment=ft.Alignment.CENTER_RIGHT,
                content=ft.Text(
                    settings["unavailable_reason"],
                    style=theme.mono(10.5, t().fg3),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
            )
        )
    return ft.Column(rows, spacing=0, tight=True)


def _knobs(sampling: catalog_api.Sampling) -> str:
    """What this profile actually sets, in the order the window's editor asks for it.
    A profile that sets nothing is served at the engine's defaults, and says so."""
    said: list[str] = []
    if sampling["temperature"] is not None:
        said.append(f"temp {sampling['temperature']:.2f}")
    if sampling["top_p"] is not None:
        said.append(f"top-p {sampling['top_p']:.2f}")
    if sampling["top_k"] is not None:
        said.append(f"top-k {sampling['top_k']}")
    return " · ".join(said) if said else "engine defaults"


@ft.component
def _Profiles(identifier: str) -> ft.Control:
    profiles, set_profiles = ft.use_state(list[tuple[str, str]]())

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                found: list[tuple[str, str]] = []
                for name in await catalog_api.profile_names(identifier):
                    profile = await catalog_api.get_profile(identifier, name)
                    found.append((name, _knobs(profile["sampling"])))
                set_profiles(found)
            except Exception:  # noqa: BLE001
                set_profiles([])

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [identifier])

    if not profiles:
        return ft.Container()
    return ft.Column(
        [
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=12, bottom=4),
                content=theme.eyebrow("Profiles"),
            ),
            *(
                ft.Container(
                    padding=ft.Padding(left=2, right=2, top=7, bottom=7),
                    border=None if index == 0 else ft.Border.only(top=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            ft.Text(
                                name,
                                style=theme.mono(12.5, t().fg, ft.FontWeight.W_500),
                                no_wrap=True,
                            ),
                            ft.Container(expand=True),
                            ft.Text(
                                knobs,
                                style=theme.mono(10, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=9,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
                for index, (name, knobs) in enumerate(profiles)
            ),
        ],
        spacing=0,
        tight=True,
    )


@ft.component
def ModelScreen(panel: Panel, engine: Engine, entry: CatalogEntry) -> ft.Control:
    identifier = entry["id"]
    resident = identifier in runtime.resident_ids(engine)
    sample = runtime.sample_of(engine, identifier)
    prefill = None if sample is None else sample["prefill_tokens_per_second"]
    ttft = None if sample is None else sample["ttft"]

    def back() -> None:
        panel.opened = ""

    def delete() -> None:
        async def drop() -> None:
            await catalog_api.remove_model(identifier)
            back()

        runtime.act(drop())

    if resident:
        state = _pillbox("resident", True)
    elif entry["supported"]:
        state = _pillbox("on disk", False)
    else:
        state = _pillbox(f"no loader for {entry['architecture']}", False)

    facts = [
        ("Quantization", entry["quantization"] or entry["dtype"] or "—"),
        ("Size", f"{gb(entry['bytes_on_disk'])} GB"),
        ("Context", "—" if entry["context"] is None else tokens(entry["context"])),
        ("Decode", (runtime.decode_of(engine, identifier) or "—").removesuffix(" tok/s")),
        ("Prefill", "—" if prefill is None else f"{prefill:,.0f}"),
        ("TTFT", "—" if ttft is None else f"{ttft * 1000:.0f} ms"),
    ]

    def cell(index: int) -> ft.Control:
        label, value = facts[index]
        return ft.Container(expand=True, content=parts.fact(label, value))

    actions: list[ft.Control] = []
    if resident:
        actions.append(
            parts.button("Unload", lambda: runtime.act(catalog_api.unload_model(identifier)))
        )
    elif entry["supported"]:
        actions.append(
            parts.button(
                "Load",
                lambda: runtime.act(_fire(catalog_api.load_model(identifier))),
                "primary",
            )
        )
    actions.append(ft.Container(expand=True))
    actions.append(parts.button("Delete", delete, "danger"))

    return ft.Column(
        [
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=0, bottom=10),
                on_click=lambda _: back(),
                content=ft.Text(
                    "‹ Models",
                    style=theme.sans(12.5, t().accent, ft.FontWeight.W_600),
                    no_wrap=True,
                ),
            ),
            ft.Row(
                [
                    ft.Container(
                        width=4,
                        height=44,
                        border_radius=2,
                        bgcolor=t().mat(engine.materials.get(identifier, 0))
                        if resident
                        else t().sel,
                    ),
                    ft.Column(
                        [
                            ft.Text(
                                display_name(identifier),
                                style=theme.display(17),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                identifier,
                                style=theme.mono(10, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Row([state], spacing=6, tight=True),
                        ],
                        spacing=3,
                        tight=True,
                        expand=True,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Container(
                margin=ft.Margin.only(top=12),
                padding=ft.Padding(left=13, right=13, top=11, bottom=11),
                bgcolor=t().elev,
                border=theme.hair(),
                border_radius=12,
                content=ft.Column(
                    [
                        ft.Row([cell(0), cell(1), cell(2)], spacing=10),
                        ft.Row([cell(3), cell(4), cell(5)], spacing=10),
                    ],
                    spacing=11,
                    tight=True,
                ),
            ),
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=14, bottom=4),
                content=theme.eyebrow("Features"),
            ),
            _Features(entry),
            _Profiles(identifier),
            ft.Container(
                margin=ft.Margin.only(top=16),
                content=ft.Row(actions, spacing=8),
            ),
        ],
        spacing=0,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

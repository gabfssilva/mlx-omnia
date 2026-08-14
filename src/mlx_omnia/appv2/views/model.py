"""One model, whole-screen: pushed over the list, ‹ Models brings it back.

Three tabs, one verb each. Overview reads — what the checkpoint declares. Configure
writes — the speculation the daemon can build for it, and the sampling profiles it is
served under. Files lists, with the total at the foot. Every write goes to the daemon
and what the screen shows after is what the daemon answered.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine

import flet as ft

from mlx_omnia.app.api import catalog as catalog_api
from mlx_omnia.app.api.engine import CatalogEntry, Engine
from mlx_omnia.app.ui.format import display_name, gb, size, tokens
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

TABS = ("Overview", "Configure", "Files")


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


def _header(
    shell: Shell, engine: Engine, entry: CatalogEntry, back: Callable[[], None]
) -> ft.Control:
    identifier = entry["id"]
    resident = identifier in runtime.resident_ids(engine)
    actions: list[ft.Control] = []
    if resident:
        actions.append(
            parts.button("Open in Chat", lambda: setattr(shell, "view", "chat"), "primary")
        )
        actions.append(
            parts.button(
                "Unload", lambda: runtime.act(catalog_api.unload_model(identifier))
            )
        )
    elif entry["supported"]:
        actions.append(
            parts.button(
                "Load",
                lambda: runtime.act(_fire(catalog_api.load_model(identifier))),
                "primary",
            )
        )

    def delete() -> None:
        async def drop() -> None:
            await catalog_api.remove_model(identifier)
            back()

        runtime.act(drop())

    actions.append(parts.button("Delete", delete, "danger"))

    color = t().mat(engine.materials.get(identifier, 0)) if resident else t().sel
    if resident:
        state = _pillbox("resident", True)
    elif entry["supported"]:
        state = _pillbox("on disk", False)
    else:
        state = _pillbox(f"no loader for {entry['architecture']}", False)
    return ft.Row(
        [
            ft.Container(width=4, height=48, border_radius=2, bgcolor=color),
            ft.Column(
                [
                    ft.Text(display_name(identifier), style=theme.display(20), no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Text(identifier, style=theme.mono(11, t().fg3), no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Row([state], spacing=6, tight=True),
                ],
                spacing=4,
                tight=True,
                expand=True,
            ),
            ft.Row(actions, spacing=8, tight=True),
        ],
        spacing=14,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


async def _fire(work: Coroutine[None, None, object]) -> None:
    """Await a coroutine whose answer (a job's first frame) the screen does not keep —
    the jobs stream is what draws it from here."""
    await work


def _facts(engine: Engine, entry: CatalogEntry) -> ft.Control:
    sample = runtime.sample_of(engine, entry["id"])
    prefill = None if sample is None else sample["prefill_tokens_per_second"]
    ttft = None if sample is None else sample["ttft"]
    cells = [
        ("Quantization", entry["quantization"] or entry["dtype"] or "—"),
        ("Size", f"{gb(entry['bytes_on_disk'])} GB"),
        ("Context", "—" if entry["context"] is None else tokens(entry["context"])),
        ("Decode", runtime.decode_of(engine, entry["id"]) or "—"),
        ("Prefill", "—" if prefill is None else f"{prefill:,.0f}"),
        ("TTFT", "—" if ttft is None else f"{ttft * 1000:.0f} ms"),
    ]
    return ft.Container(
        padding=ft.Padding(left=18, right=18, top=14, bottom=14),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=11,
        content=ft.Row(
            [ft.Container(content=parts.fact(label, value), expand=True) for label, value in cells],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _tabs(tab: str, set_tab: Callable[[str], None]) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.only(bottom=9),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                # A Text has no on_click in this Flet, and handing it one fails the whole
                # render silently — the box around it is what answers the pointer.
                ft.Container(
                    content=ft.Text(
                        name,
                        style=theme.sans(
                            13,
                            t().fg if name == tab else t().fg3,
                            ft.FontWeight.W_600 if name == tab else ft.FontWeight.W_400,
                        ),
                        no_wrap=True,
                    ),
                    on_click=lambda _, name=name: set_tab(name),
                )
                for name in TABS
            ],
            spacing=16,
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


def _hint(text: str) -> ft.Text:
    return ft.Text(text, style=theme.mono(10.5, t().fg3), no_wrap=True)


def _overview(entry: CatalogEntry) -> ft.Control:
    defaults = entry["defaults"]
    declared = " · ".join(
        f"{name} {value}"
        for name, value in (
            ("temp", defaults["temperature"]),
            ("top-p", defaults["top_p"]),
            ("top-k", defaults["top_k"]),
            ("min-p", defaults["min_p"]),
            ("rep", defaults["repetition_penalty"]),
        )
        if value is not None
    )
    rows: list[ft.Control] = [
        parts.fact("Architecture", entry["architecture"]),
        parts.fact(
            "Active per token",
            "—" if entry["bytes_per_token"] is None else f"{gb(entry['bytes_per_token'], 2)} GB",
        ),
        parts.fact(
            "KV per token",
            "—"
            if entry["kv_bytes_per_token"] is None
            else size(entry["kv_bytes_per_token"]),
        ),
        parts.fact("Declared sampling", declared or "none — the checkpoint says nothing"),
        parts.fact("Store", entry["store"]),
    ]
    return ft.Column(rows, spacing=14, tight=True)


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

    rows: list[ft.Control] = [
        ft.Container(padding=ft.Padding.only(bottom=4), content=theme.eyebrow("Features"))
    ]
    if settings is None:
        rows.append(_setting("Speculation", _hint("asking the daemon…"), dim=True, first=True))
        rows.append(_setting("Vision", _hint("not served by this engine yet"), dim=True))
        return ft.Column(rows, spacing=0, tight=True)

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
                label,
                style=theme.sans(11.5, t().fg if on else t().fg2),
                no_wrap=True,
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

    rows.append(
        _setting(
            "Speculation",
            ft.Row(choices, spacing=6, tight=True, wrap=True),
            first=True,
        )
    )
    if settings["unavailable_reason"] is not None:
        rows.append(
            ft.Container(
                padding=ft.Padding.only(top=2),
                alignment=ft.Alignment.CENTER_RIGHT,
                content=_hint(settings["unavailable_reason"]),
            )
        )
    rows.append(_setting("Vision", _hint("not served by this engine yet"), dim=True))
    return ft.Column(rows, spacing=0, tight=True)


@ft.component
def _Profiles(entry: CatalogEntry) -> ft.Control:
    identifier = entry["id"]
    names, set_names = ft.use_state(list[str]())
    editing, set_editing = ft.use_state("")

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                set_names(await catalog_api.profile_names(identifier))
            except Exception:  # noqa: BLE001
                set_names([])

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [identifier, editing])

    def create() -> None:
        async def write() -> None:
            name = f"profile-{len(names) + 1}"
            await catalog_api.save_profile(identifier, name, {"sampling": {}})
            set_editing(name)

        runtime.act(write())

    if editing:
        return _Editor(identifier, editing, lambda: set_editing(""))

    rows: list[ft.Control] = [
        ft.Container(padding=ft.Padding.only(bottom=4), content=theme.eyebrow("Profiles"))
    ]
    for index, name in enumerate(names):
        rows.append(
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=9, bottom=9),
                border=None if index == 0 else ft.Border.only(top=ft.BorderSide(1, t().hair)),
                on_click=lambda _, name=name: set_editing(name),
                content=ft.Row(
                    [
                        ft.Text(name, style=theme.mono(12.5, t().fg, ft.FontWeight.W_500),
                                no_wrap=True),
                        ft.Container(expand=True),
                        ft.Text(f"{identifier}:{name}", style=theme.mono(10, t().fg3),
                                no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Text("›", style=theme.sans(11, t().fg3), no_wrap=True),
                    ],
                    spacing=9,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        )
    if not names:
        rows.append(
            ft.Container(
                padding=ft.Padding(left=2, right=2, top=9, bottom=9),
                content=ft.Text(
                    "None yet — the checkpoint's own defaults answer.",
                    style=theme.sans(12.5, t().fg3),
                ),
            )
        )
    rows.append(
        ft.Container(
            padding=ft.Padding(left=2, right=2, top=10, bottom=6),
            border=ft.Border.only(top=ft.BorderSide(1, t().hair2)),
            on_click=lambda _: create(),
            content=ft.Text(
                "+ New profile", style=theme.sans(12.5, t().accent, ft.FontWeight.W_600)
            ),
        )
    )
    return ft.Column(rows, spacing=0, tight=True)


@ft.component
def _Editor(identifier: str, name: str, back: Callable[[], None]) -> ft.Control:
    profile, set_profile = ft.use_state(None)
    temperature, set_temperature = ft.use_state(0.0)
    top_p, set_top_p = ft.use_state("1.0")
    prompt, set_prompt = ft.use_state("")

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                loaded = await catalog_api.get_profile(identifier, name)
            except Exception:  # noqa: BLE001
                back()
                return
            set_profile(loaded)
            sampling = loaded["sampling"]
            set_temperature(sampling["temperature"] if sampling["temperature"] is not None else 1.0)
            set_top_p("1.0" if sampling["top_p"] is None else str(sampling["top_p"]))
            set_prompt(loaded["system_prompt"] or "")

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [identifier, name])

    def save() -> None:
        async def write() -> None:
            sampling: dict[str, object] = dict(profile["sampling"]) if profile else {}
            sampling["temperature"] = temperature
            with contextlib.suppress(ValueError):
                sampling["top_p"] = float(top_p)
            await catalog_api.save_profile(
                identifier,
                name,
                {"sampling": sampling, "system_prompt": prompt or None},
            )
            back()

        runtime.act(write())

    def drop() -> None:
        async def erase() -> None:
            await catalog_api.remove_profile(identifier, name)
            back()

        runtime.act(erase())

    crumb = ft.Container(
        padding=ft.Padding.only(bottom=4),
        on_click=lambda _: back(),
        content=ft.Row(
            [
                ft.Text("‹", style=theme.sans(12, t().accent, ft.FontWeight.W_600), no_wrap=True),
                theme.eyebrow(f"Profile · {name}"),
            ],
            spacing=6,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    def box(value: str, on_change: Callable[[str], None], width: float = 76) -> ft.Control:
        return ft.Container(
            width=width,
            height=26,
            padding=ft.Padding(left=9, right=9, top=0, bottom=0),
            bgcolor=t().field,
            border=theme.hair(t().hair2),
            border_radius=6,
            alignment=ft.Alignment.CENTER,
            content=ft.TextField(
                value=value,
                text_style=theme.mono(12),
                border=ft.InputBorder.NONE,
                content_padding=ft.Padding.all(0),
                cursor_color=t().accent,
                cursor_width=1,
                dense=True,
                on_change=lambda event: on_change(event.control.value or ""),
            ),
        )

    return ft.Column(
        [
            crumb,
            _setting(
                "Temperature",
                ft.Row(
                    [
                        parts.Slider(
                            120,
                            min(1.0, temperature / 2.0),
                            lambda fraction: set_temperature(round(fraction * 2.0, 2)),
                        ),
                        ft.Text(f"{temperature:.2f}", style=theme.mono(12, t().fg2),
                                no_wrap=True),
                    ],
                    spacing=10,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                first=True,
            ),
            _setting("Top-p", box(top_p, set_top_p)),
            ft.Container(
                padding=ft.Padding(left=0, right=0, top=12, bottom=6),
                content=theme.eyebrow("System prompt"),
            ),
            ft.Container(
                padding=ft.Padding(left=10, right=10, top=8, bottom=8),
                bgcolor=t().field,
                border=theme.hair(t().hair2),
                border_radius=8,
                content=ft.TextField(
                    value=prompt,
                    multiline=True,
                    min_lines=2,
                    max_lines=6,
                    text_style=theme.mono(11),
                    hint_text="No system prompt.",
                    hint_style=theme.mono(11, t().fg3),
                    border=ft.InputBorder.NONE,
                    content_padding=ft.Padding.all(0),
                    cursor_color=t().accent,
                    cursor_width=1,
                    dense=True,
                    on_change=lambda event: set_prompt(event.control.value or ""),
                ),
            ),
            ft.Container(height=12),
            ft.Row(
                [
                    parts.button("Save", save, "primary"),
                    ft.Container(expand=True),
                    parts.button("Delete", drop, "danger"),
                ],
                spacing=8,
            ),
        ],
        spacing=0,
        tight=True,
    )


@ft.component
def _Files(entry: CatalogEntry) -> ft.Control:
    files, set_files = ft.use_state(list[catalog_api.CheckpointFile]())

    def load() -> Callable[[], None]:
        async def fetch() -> None:
            try:
                set_files(await catalog_api.list_files(entry["id"]))
            except Exception:  # noqa: BLE001
                set_files([])

        task = asyncio.get_running_loop().create_task(fetch())
        return task.cancel

    ft.use_effect(load, [entry["id"]])

    def line(left: str, right: str, dim: bool, top: str | None) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=2, right=2, top=8, bottom=8),
            border=None if top is None else ft.Border.only(top=ft.BorderSide(1, top)),
            content=ft.Row(
                [
                    ft.Text(left, style=theme.mono(12, t().fg3 if dim else t().fg), no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                    ft.Text(right, style=theme.mono(11, t().fg2 if dim else t().fg3),
                            no_wrap=True),
                ],
                spacing=10,
            ),
        )

    rows = [
        line(one["name"], size(one["size"]), False, None if index == 0 else t().hair)
        for index, one in enumerate(files)
    ]
    if files:
        total = sum(one["size"] for one in files)
        rows.append(line(f"{len(files)} files", size(total), True, t().hair2))
    else:
        rows.append(line("asking the daemon…", "", True, None))
    return ft.Column(rows, spacing=0, tight=True)


@ft.component
def ModelScreen(
    shell: Shell, engine: Engine, entry: CatalogEntry, back: Callable[[], None]
) -> ft.Control:
    tab, set_tab = ft.use_state(TABS[0])

    crumb = ft.Container(
        padding=ft.Padding(left=2, right=2, top=0, bottom=10),
        on_click=lambda _: back(),
        content=ft.Text(
            "‹ Models", style=theme.sans(12.5, t().accent, ft.FontWeight.W_600), no_wrap=True
        ),
    )

    if tab == "Configure":
        content: ft.Control = ft.Row(
            [
                ft.Container(content=_Features(entry), expand=True),
                ft.Container(width=30),
                ft.Container(content=_Profiles(entry), expand=True),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
    elif tab == "Files":
        content = _Files(entry)
    else:
        content = _overview(entry)

    return ft.Column(
        [
            ft.WindowDragArea(content=ft.Container(height=24), maximizable=False),
            ft.Container(
                expand=True,
                padding=ft.Padding(left=24, right=24, top=0, bottom=24),
                content=ft.Column(
                    [
                        crumb,
                        _header(shell, engine, entry, back),
                        ft.Container(height=14),
                        _facts(engine, entry),
                        ft.Container(height=16),
                        _tabs(tab, set_tab),
                        ft.Container(height=12),
                        content,
                    ],
                    spacing=0,
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                ),
            ),
        ],
        spacing=0,
        expand=True,
    )

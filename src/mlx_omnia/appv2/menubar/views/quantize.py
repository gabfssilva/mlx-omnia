"""Quantize: the window's two columns, stacked.

The rail becomes the block that closes the screen, because on a panel the projected size is
the answer being waited for and not a sidebar to it. The ~600 per-leaf overrides do not come
along: that is a table, and a table is what the window is for.

The rules are the server's, and they are imported rather than restated — an exponent-scaled
mode fixing method, width and group size is a fact about the engine, and a second copy of it
here is a second thing to get wrong.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

import flet as ft

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.engine import GIB, Engine, Job
from mlx_omnia.app.ui.format import display_name, gb
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.menubar.panel import Panel, empty
from mlx_omnia.appv2.theme import t
from mlx_omnia.appv2.views.quantize import (
    BITS,
    DEBOUNCE,
    DTYPES,
    GPTQ_BITS,
    GROUP_SIZES,
    METHODS,
    MODES,
    REPO,
    PricedPlan,
)


def _grain(entry: engine_api.CatalogEntry) -> str:
    dtype = DTYPES.get(entry["dtype"] or "", entry["dtype"] or "?")
    return f"{dtype} · {gb(entry['bytes_on_disk'])} GB"


def _suggest(source: str, mode: str, bits: int) -> str:
    """`local/` because a load by id asks the Hub first, and a name that also exists up
    there resolves to somebody else's weights."""
    if source == "":
        return ""
    tail = source.split("/")[-1]
    return f"local/{tail}-{f'{bits}bit' if mode == 'affine' else mode}"


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _labelled(label: str, control: ft.Control, hint: str | None = None) -> ft.Control:
    rows: list[ft.Control] = [theme.eyebrow(label), control]
    if hint is not None:
        rows.append(ft.Text(hint, style=theme.sans(11, t().fg3, height=1.4), max_lines=3))
    return ft.Column(rows, spacing=6, tight=True)


def _seg(
    options: list[tuple[str, str]],
    chosen: str,
    on_pick: Callable[[str], None],
    dimmed: tuple[str, ...] = (),
) -> ft.Container:
    """The segmented control at the panel's width: 12 pt and no horizontal padding, so
    five methods still fit across 352.

    A key the current selection refuses stays on the list — the rule is worth seeing —
    but takes no click.
    """

    def one(key: str, label: str) -> ft.Control:
        on = key == chosen
        dim = key in dimmed and not on
        return ft.Container(
            expand=True,
            padding=ft.Padding(left=0, right=0, top=4, bottom=4),
            alignment=ft.Alignment.CENTER,
            bgcolor=t().elev if on else None,
            border_radius=7,
            shadow=ft.BoxShadow(
                offset=ft.Offset(0, 1), blur_radius=2, color=theme.alpha("#000000", 0.16)
            )
            if on
            else None,
            content=ft.Text(
                label,
                style=theme.sans(
                    12,
                    t().fg if on else (t().fg3 if dim else t().fg2),
                    ft.FontWeight.W_500 if on else ft.FontWeight.W_400,
                ),
                no_wrap=True,
            ),
            opacity=0.55 if dim else 1.0,
            on_click=None if dim else (lambda _, key=key: on_pick(key)),
        )

    return ft.Container(
        padding=2,
        bgcolor=t().sel,
        border_radius=9,
        content=ft.Row([one(key, label) for key, label in options], spacing=2),
    )


def _field(value: str, on_change: Callable[[str], None], hint: str = "") -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=11, right=11, top=6, bottom=6),
        bgcolor=t().field,
        border=theme.hair(t().hair2),
        border_radius=8,
        content=ft.TextField(
            value=value,
            text_style=theme.mono(12),
            hint_text=hint,
            hint_style=theme.mono(12, t().fg3),
            border=ft.InputBorder.NONE,
            content_padding=ft.Padding.all(0),
            cursor_color=t().accent,
            cursor_width=1,
            dense=True,
            on_change=lambda event: on_change(event.control.value or ""),
        ),
    )


def _kvline(label: str, value: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.symmetric(vertical=6),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                theme.eyebrow(label),
                ft.Container(expand=True),
                ft.Text(value, style=theme.mono(11.5, weight=ft.FontWeight.W_500), no_wrap=True),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def _refusal(message: str) -> ft.Control:
    return ft.Text(message, style=theme.sans(11, t().bad, height=1.5))


@ft.component
def Quantize(panel: Panel, engine: Engine) -> ft.Control:
    source, set_source = ft.use_state("")
    mode, set_mode = ft.use_state("affine")
    method, set_method = ft.use_state("rtn")
    bits, set_bits = ft.use_state(4)
    group_size, set_group_size = ft.use_state(64)
    repo, set_repo = ft.use_state("")
    named, set_named = ft.use_state(False)
    target, set_target = ft.use_state("")
    plan, set_plan = ft.use_state(None)
    refusal, set_refusal = ft.use_state("")
    pricing, set_pricing = ft.use_state(False)
    started, set_started = ft.use_state("")
    job, set_job = ft.use_state(None)
    failure, set_failure = ft.use_state("")

    exponent = next(((label, shape) for key, label, shape in MODES if key == mode and shape), None)
    allocated = method in ("oq", "oqe")

    def request() -> dict[str, object]:
        body: dict[str, object] = {"source": source, "mode": mode, "method": method}
        if exponent is None:
            body["bits"] = bits
            body["group_size"] = group_size
        if allocated:
            asked = _number(target)
            if asked is not None:
                body["target_bpw"] = asked
        body["overrides"] = {}
        return body

    def price() -> Callable[[], None]:
        async def ask() -> None:
            if source == "":
                set_plan(None)
                set_refusal("")
                return
            await asyncio.sleep(DEBOUNCE)
            set_pricing(True)
            try:
                set_plan(
                    cast(
                        PricedPlan,
                        await engine_api.send("POST", "/admin/quantizations/plan", request()),
                    )
                )
                set_refusal("")
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 — the daemon's refusal is the message
                set_plan(None)
                set_refusal(str(error))
            finally:
                set_pricing(False)

        task = asyncio.get_running_loop().create_task(ask())
        return task.cancel

    # The daemon's presence is a dependency too: a price asked while it was still coming up
    # is retried when it answers, instead of standing as a refusal nobody re-asks.
    up = engine.health is not None
    ft.use_effect(price, [source, mode, method, bits, group_size, target, up])

    def follow() -> Callable[[], None]:
        async def watch() -> None:
            if started == "":
                return
            try:
                async for frame in engine_api.job_events(started):
                    set_job(frame)
            except Exception:  # noqa: BLE001 — the jobs stream still shows the progress
                pass

        task = asyncio.get_running_loop().create_task(watch())
        return task.cancel

    ft.use_effect(follow, [started])

    held = cast(Job | None, job)
    running = held is not None and held["state"] in ("pending", "running")
    priced = cast(PricedPlan | None, plan)

    def rename(chosen: str, chosen_mode: str, chosen_bits: int) -> None:
        if not named:
            set_repo(_suggest(chosen, chosen_mode, chosen_bits))

    def pick_source(identifier: str) -> None:
        set_source(identifier)
        rename(identifier, mode, bits)

    def pick_mode(key: str) -> None:
        set_mode(key)
        if next((shape for k, _, shape in MODES if k == key and shape), None) is not None:
            # The exponent grid has no bias to search and no width to pick, so the controls
            # it decides go with it.
            set_method("rtn")
            set_target("")
        rename(source, key, bits)

    def pick_method(key: str) -> None:
        set_method(key)
        if key not in ("oq", "oqe"):
            set_target("")

    def pick_bits(key: str) -> None:
        set_bits(int(key))
        rename(source, mode, int(key))

    def launch() -> None:
        async def create() -> None:
            set_failure("")
            try:
                accepted = cast(
                    Job,
                    await engine_api.send(
                        "POST", "/admin/quantizations", {**request(), "repo": repo}
                    ),
                )
                set_job(accepted)
                set_started(accepted["id"])
            except Exception as error:  # noqa: BLE001 — a refused job is a message, not a crash
                set_failure(str(error))

        runtime.act(create())

    dense = [entry for entry in engine.catalog if entry["quantization"] is None]
    if not dense:
        return ft.Column(
            [empty("Nothing dense on this disk to quantize — fetch a model under Models.")],
            spacing=0,
            expand=True,
        )
    chosen = next((entry for entry in dense if entry["id"] == source), None)

    picker = ft.PopupMenuButton(
        bgcolor=t().elev,
        items=[
            ft.PopupMenuItem(
                content=ft.Row(
                    [
                        ft.Text(
                            display_name(entry["id"]),
                            style=theme.sans(
                                12.5,
                                t().accent if entry["id"] == source else t().fg,
                                ft.FontWeight.W_500,
                            ),
                            no_wrap=True,
                        ),
                        ft.Text(_grain(entry), style=theme.mono(10, t().fg3), no_wrap=True),
                    ],
                    spacing=10,
                ),
                on_click=lambda _, entry=entry: pick_source(entry["id"]),
            )
            for entry in dense
        ],
        content=ft.Container(
            padding=ft.Padding(left=12, right=12, top=8, bottom=8),
            bgcolor=t().elev,
            border=theme.hair(),
            border_radius=10,
            content=ft.Row(
                [
                    ft.Container(
                        width=9,
                        height=9,
                        border_radius=3,
                        bgcolor=t().mat(engine.materials.get(source, 0))
                        if source in runtime.resident_ids(engine)
                        else t().sel,
                    ),
                    ft.Text(
                        display_name(source) if source else "Pick a checkpoint",
                        style=theme.sans(13, t().fg if source else t().fg3, ft.FontWeight.W_500),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                    ft.Text(
                        "" if chosen is None else _grain(chosen),
                        style=theme.mono(10, t().fg3),
                        no_wrap=True,
                    ),
                    ft.Text("▾", style=theme.sans(9, t().fg3), no_wrap=True),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ),
    )

    controls: list[ft.Control] = [
        _labelled("Source", picker),
        _labelled(
            "Format",
            _seg([(key, label) for key, label, _ in MODES], mode, pick_mode),
            None
            if exponent is None
            else f"{exponent[1][1]} bits at group {exponent[1][0]}, packed by mx.quantize. "
            "Method, width and group size are the mode's.",
        ),
    ]
    if exponent is None:
        controls.append(_labelled("Method", _seg(METHODS, method, pick_method)))
        controls.append(
            ft.Row(
                [
                    ft.Container(
                        expand=True,
                        content=_labelled(
                            "Width",
                            _seg(
                                [(str(value), str(value)) for value in BITS],
                                str(bits),
                                pick_bits,
                                dimmed=tuple(
                                    str(value) for value in BITS if value not in GPTQ_BITS
                                )
                                if method == "gptq"
                                else (),
                            ),
                        ),
                    ),
                    ft.Container(
                        width=132,
                        content=_labelled(
                            "Group size",
                            _seg(
                                [(str(value), str(value)) for value in GROUP_SIZES],
                                str(group_size),
                                lambda value: set_group_size(int(value)),
                            ),
                        ),
                    ),
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
    if allocated:
        controls.append(
            _labelled(
                "Budget",
                _field(target, set_target, "RTN + 1.0"),
                "Effective bits per weight. Left empty, the allocator takes the RTN plan "
                "plus one bit of headroom.",
            )
        )

    output: list[ft.Control] = [
        _field(repo, lambda value: (set_named(True), set_repo(value)), "org/name")
    ]
    if repo != "" and not REPO.match(repo):
        output.append(_refusal("Two segments, and nothing that could name a directory: org/name."))
    controls.append(_labelled("Output id", ft.Column(output, spacing=4, tight=True)))

    # ── the projected block ───────────────────────────────────────────────
    rail: list[ft.Control] = [theme.eyebrow("Projected")]
    if priced is None:
        rail.append(
            ft.Container(
                padding=ft.Padding.only(top=4),
                content=ft.Text("—", style=theme.mono(30, spacing=-0.03 * 30)),
            )
        )
        if refusal:
            rail.append(_refusal(refusal))
    else:
        fraction = (
            0.0
            if chosen is None or chosen["bytes_on_disk"] == 0
            else min(1.0, priced["entry_bytes"] / chosen["bytes_on_disk"])
        )
        memory = engine.system["memory_bytes"] if engine.system else None
        resident = (
            None
            if engine.state is None
            else engine.state["resident_bytes"] + engine.state["kv_bytes"]
        )
        rail += [
            ft.Container(
                padding=ft.Padding.only(top=4),
                content=ft.Row(
                    [
                        ft.Text(
                            gb(priced["entry_bytes"]), style=theme.mono(30, spacing=-0.03 * 30)
                        ),
                        ft.Text("GB", style=theme.mono(11, t().fg3)),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.END,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(top=10),
                height=4,
                border_radius=3,
                bgcolor=t().sel,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                content=ft.Row(
                    [
                        ft.Container(expand=max(1, round(fraction * 1000)), bgcolor=t().accent),
                        ft.Container(expand=max(1, round((1 - fraction) * 1000))),
                    ],
                    spacing=0,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(top=5),
                content=ft.Row(
                    [
                        ft.Text(
                            f"{priced['bits_per_weight']:.2f} bits / weight",
                            style=theme.mono(10, t().fg3),
                            no_wrap=True,
                        ),
                        ft.Container(expand=True),
                        ft.Text(
                            "" if chosen is None else f"source {gb(chosen['bytes_on_disk'])} GB",
                            style=theme.mono(10, t().fg3),
                            no_wrap=True,
                        ),
                    ],
                    spacing=8,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(top=7),
                content=_kvline(
                    "If loaded now",
                    "—"
                    if resident is None or memory is None
                    else f"{gb(resident + priced['entry_bytes'])} / {round(memory / GIB)} GB",
                ),
            ),
            _kvline(
                "Leaves packed",
                f"{sum(1 for leaf in priced['leaves'] if leaf['bits'] is not None)}"
                f" / {len(priced['leaves'])}",
            ),
            ft.Container(
                margin=ft.Margin.only(top=9),
                content=ft.Text(
                    "The size and budget hold; which leaves the calibration promotes is "
                    "measured by the job, not projected here."
                    if allocated
                    else "Priced against the checkpoint's own leaves — nothing written yet. "
                    "Parity is measured after the job, never predicted.",
                    style=theme.sans(11, t().fg3, height=1.5),
                ),
            ),
        ]
    if failure:
        rail.append(ft.Container(margin=ft.Margin.only(top=8), content=_refusal(failure)))
    if held is not None and not running:
        rail.append(
            ft.Container(
                margin=ft.Margin.only(top=8),
                content=_refusal(held["error"] or "cancelled — nothing was kept")
                if held["state"] in ("error", "cancelled")
                else ft.Text(
                    f"Done — {held['progress']['message']} is in the catalog.",
                    style=theme.sans(11, t().ok, height=1.5),
                ),
            )
        )
    if running and held is not None:
        total = held["progress"]["total"]
        done = held["progress"]["completed"]
        rail.append(
            ft.Container(
                margin=ft.Margin.only(top=8),
                content=ft.Text(
                    (f"{round(done / total * 100)}% · " if total else "")
                    + held["progress"]["message"],
                    style=theme.mono(10, t().fg2),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
            )
        )

    ready = priced is not None and not refusal and bool(REPO.match(repo)) and not running
    if running:
        action: ft.Control = parts.button(
            "Cancel",
            lambda: runtime.act(engine_api.cancel_job(held["id"])) if held else None,
            "danger",
            expand=True,
        )
    elif ready:
        action = parts.button("Quantize", launch, "primary", expand=True)
    else:
        action = ft.Container(
            expand=True,
            opacity=0.45,
            content=parts.button("Quantize", None, "primary", expand=True),
        )
    rail.append(ft.Container(margin=ft.Margin.only(top=11), content=ft.Row([action])))

    controls.append(
        ft.Container(
            margin=ft.Margin.only(top=2),
            padding=ft.Padding(left=13, right=13, top=12, bottom=12),
            bgcolor=t().elev,
            border=theme.hair(),
            border_radius=12,
            opacity=0.55 if pricing else 1.0,
            content=ft.Column(rail, spacing=2, tight=True),
        )
    )

    return ft.Column(controls, spacing=13, scroll=ft.ScrollMode.AUTO, expand=True)

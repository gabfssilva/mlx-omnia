"""Modelos: the one shelf, and the detail beside it.

A row says one thing — name, the essential line, a state. Everything else about a
checkpoint (README, files, profiles, benchmark, delete) lives in the detail, which is
where quantize and compare will hang when the screen is wired.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from mlx_omnia.appv2 import data, parts, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

STATES = {
    "resident": ("Resident", "resident"),
    "disk": ("On disk", "disk"),
    "hub": ("Hub", "hub"),
}

TABS = ["Overview", "Files", "Profiles", "Benchmark"]

DETAIL = 330


def _material_bar(model: data.Model) -> ft.Control:
    color = t().mat(model.material) if model.material is not None else t().sel
    return ft.Container(width=4, height=52, border_radius=2, bgcolor=color)


def _row(model: data.Model, on: bool, pick: Callable[[], None]) -> ft.Control:
    meta: list[ft.Control] = [
        ft.Text(
            f"{model.kind} · {model.quant} · {model.size_gb:.1f} GB · ctx {model.context}",
            style=theme.sans(12.5, t().fg2),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
    ]
    if model.progress is not None:
        meta.append(
            ft.Container(
                height=4,
                border_radius=2,
                bgcolor=t().sel,
                margin=ft.Margin.only(top=6),
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                content=ft.Row(
                    [
                        ft.Container(expand=round(model.progress * 100), bgcolor=t().accent),
                        ft.Container(expand=round((1 - model.progress) * 100)),
                    ],
                    spacing=0,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            )
        )
    if model.progress is not None:
        state = parts.pill(f"Downloading · {model.progress:.0%}", "disk")
    else:
        label, kind = STATES[model.state]
        state = parts.pill(label, kind)

    def body(tint: str | None) -> ft.Container:
        return ft.Container(
            padding=ft.Padding(left=14, right=18, top=17, bottom=17),
            bgcolor=tint,
            border=theme.hair(t().accent if on else t().hair),
            border_radius=12,
            content=ft.Row(
                [
                    _material_bar(model),
                    ft.Column(
                        [
                            ft.Text(
                                model.name,
                                style=theme.sans(16, weight=ft.FontWeight.W_600),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                model.id,
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
                    state,
                ],
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    return parts.press(body, pick, t().elev, theme.mix(t().fg, 0.03, t().elev), t().sel)


def _overview(shell: Shell, model: data.Model) -> list[ft.Control]:
    content: list[ft.Control] = [
        ft.Text(model.blurb, style=theme.sans(12.5, t().fg2, height=1.5))
    ]
    if model.decode is not None:
        content.append(
            parts.fact("Measured speed", f"{model.decode} · {model.ceiling_share}")
        )
    content.append(
        ft.Row(
            [
                parts.button(
                    "Open in Chat",
                    lambda: setattr(shell, "view", "chat"),
                    "primary",
                )
                if model.state == "resident"
                else parts.button("Load", lambda: None, "primary"),
                parts.button("Quantize", lambda: None),
                parts.button("Delete", lambda: None, "danger"),
            ],
            spacing=8,
            wrap=True,
        )
    )
    return content


def _rows(pairs: list[tuple[str, str]]) -> list[ft.Control]:
    return [
        ft.Container(
            padding=ft.Padding(left=0, right=0, top=7, bottom=7),
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=ft.Row(
                [
                    ft.Text(left, style=theme.mono(12), no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                    ft.Text(right, style=theme.mono(11, t().fg3), no_wrap=True),
                ],
                spacing=10,
            ),
        )
        for left, right in pairs
    ]


def _detail(
    shell: Shell, model: data.Model, tab: str, set_tab: Callable[[str], None]
) -> ft.Control:
    facts = ft.Column(
        [
            ft.Row(
                [
                    ft.Container(content=parts.fact("Architecture", model.arch), expand=True),
                    ft.Container(content=parts.fact("Context", model.context), expand=True),
                ]
            ),
            ft.Row(
                [
                    ft.Container(content=parts.fact("Quantization", model.quant), expand=True),
                    ft.Container(
                        content=parts.fact("Active per token", f"{model.active_gb:.1f} GB"),
                        expand=True,
                    ),
                ]
            ),
        ],
        spacing=12,
        tight=True,
    )

    tabs = ft.Container(
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
                            12.5,
                            t().fg if name == tab else t().fg3,
                            ft.FontWeight.W_600 if name == tab else ft.FontWeight.W_400,
                        ),
                        no_wrap=True,
                    ),
                    on_click=lambda _, name=name: set_tab(name),
                )
                for name in TABS
            ],
            spacing=14,
        ),
    )

    if tab == "Files":
        content = _rows(list(data.FILES))
    elif tab == "Profiles":
        content = _rows(list(data.PROFILES))
    elif tab == "Benchmark":
        content = (
            [
                parts.fact("Decode", model.decode),
                parts.fact("Of physical ceiling", model.ceiling_share or "—"),
            ]
            if model.decode is not None
            else [ft.Text("No measurement for this checkpoint.", style=theme.sans(12.5, t().fg3))]
        )
    else:
        content = _overview(shell, model)

    return ft.Container(
        width=DETAIL,
        padding=20,
        border=ft.Border.only(left=ft.BorderSide(1, t().hair)),
        content=ft.Column(
            [
                ft.Text(model.name, style=theme.display(20), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
                ft.Text(model.id, style=theme.mono(11, t().fg3), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
                ft.Container(height=6),
                facts,
                ft.Container(height=6),
                tabs,
                *content,
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
    )


def _visible(scope_key: str, query: str) -> list[data.Model]:
    models = list(data.MODELS)
    if scope_key == "disk":
        models = [m for m in models if m.state in ("resident", "disk", "downloading")]
    elif scope_key == "resident":
        models = [m for m in models if m.state == "resident"]
    if query:
        needle = query.lower()
        models = [m for m in models if needle in m.name.lower() or needle in m.id.lower()]
    return models


@ft.component
def Models(shell: Shell) -> ft.Control:
    scope_key, set_scope = ft.use_state("all")
    query, set_query = ft.use_state("")
    chosen, set_chosen = ft.use_state(data.MODELS[1].id)
    tab, set_tab = ft.use_state(TABS[0])

    models = _visible(scope_key, query)
    selected = next((m for m in models if m.id == chosen), models[0] if models else None)

    shelf = ft.Container(
        expand=True,
        padding=16,
        content=ft.Column(
            [
                ft.WindowDragArea(
                    content=parts.searchbox("Search disk and the Hub…", set_query),
                    maximizable=False,
                ),
                parts.scope(
                    [("all", "All"), ("disk", "On this Mac"), ("resident", "Resident")],
                    scope_key,
                    set_scope,
                ),
                *[
                    _row(model, selected is not None and model.id == selected.id,
                         lambda model=model: (set_chosen(model.id), set_tab(TABS[0]), None)[-1])
                    for model in models
                ],
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
    )

    row: list[ft.Control] = [shelf]
    if selected is not None:
        row.append(_detail(shell, selected, tab, set_tab))
    return ft.Row(row, spacing=0, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH)

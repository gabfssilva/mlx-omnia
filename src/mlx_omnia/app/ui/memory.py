"""The memory scale, twice: the band that is the subject of Resident, and the 126px meter
that carries it into every other screen's title bar.

Both read the same numbers and use the same materials, so the small one is the big one
compressed and not a second opinion. Percentages are resolved against a measured track,
because the window is fixed and a share of it is a real px width.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from mlx_omnia.app.api.engine import Engine, Slot
from mlx_omnia.app.ui import theme
from mlx_omnia.app.ui.format import gb
from mlx_omnia.app.ui.theme import t

# .band sits 12px inside a 1160px window and adds a 1px rule and 11px of padding on each
# side. Everything the track draws is a share of what is left.
TRACK = 1160 - 2 * 12 - 2 * 1 - 2 * 11
TRACK_HEIGHT = 54
GAP = 2
# A block narrower than this cannot hold two lines of label.
TIGHT = 0.055

METER = 126.0
METER_HEIGHT = 9.0


def _short(model: str) -> str:
    """What fits in a block too narrow for the whole name."""
    leaf = model.split("/")[-1]
    return "-".join(leaf.split("-")[:2])


def _used(models: list[Slot]) -> int:
    return sum(model["weights_bytes"] + model["kv_bytes"] for model in models)


def _slab(model: Slot, material: int, total: int, first: bool) -> ft.Control:
    size = model["weights_bytes"] + model["kv_bytes"]
    tight = size / total < TIGHT
    material_color = t().mat(material)
    width = max(5.0, size / total * TRACK)

    label = ft.Text(
        _short(model["id"]) if tight else model["id"].split("/")[-1],
        style=theme.sans(9.5, weight=ft.FontWeight.W_600, height=1.2),
        no_wrap=True,
        overflow=ft.TextOverflow.ELLIPSIS,
    )
    measure = ft.Text(
        gb(model["weights_bytes"])
        + (f" + {gb(model['kv_bytes'])} KV" if model["kv_bytes"] > 0 else ""),
        style=theme.mono(8.5, t().fg2),
        no_wrap=True,
        overflow=ft.TextOverflow.ELLIPSIS,
    )

    face: list[ft.Control] = []
    if model["kv_bytes"] > 0:
        # What the cache costs, drawn against the right edge of the block it grows in.
        face.append(
            ft.Container(
                right=0,
                top=0,
                bottom=0,
                width=model["kv_bytes"] / size * width,
                bgcolor=theme.mix(material_color, 0.30, theme.TRANSPARENT),
                border=ft.Border.only(
                    left=ft.BorderSide(1, theme.mix(material_color, 0.50, theme.TRANSPARENT))
                ),
            )
        )
    # The 3px rule down the left edge is the block's material, and the fill is the same
    # material at a quarter: identity twice, never state.
    face.append(ft.Container(left=0, top=0, bottom=0, width=3, bgcolor=material_color))
    if tight:
        face.append(ft.Container(left=9, right=5, top=0, bottom=0, content=label,
                                 alignment=ft.Alignment.CENTER_LEFT))
    else:
        face.append(ft.Container(left=9, right=5, top=7, content=label))
        face.append(ft.Container(left=9, right=5, top=21, content=measure))

    return ft.Container(
        content=ft.Stack(face),
        width=width,
        bgcolor=theme.mix(material_color, 0.24, t().surface),
        border=theme.hair(theme.mix(material_color, 0.45, theme.TRANSPARENT)),
        border_radius=ft.BorderRadius(top_left=6, top_right=0, bottom_left=6, bottom_right=0)
        if first
        else 0,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        tooltip=f"{model['id']} · {gb(size)} GB",
    )


def _loading(done: int, total: int) -> ft.Control:
    return ft.Container(
        content=ft.Container(
            left=9,
            right=5,
            top=0,
            bottom=0,
            alignment=ft.Alignment.CENTER_LEFT,
            content=ft.Text(gb(done), style=theme.mono(8.5, t().fg2), no_wrap=True),
        ),
        width=max(5.0, done / total * TRACK),
        gradient=theme.hatch(
            theme.mix(t().mats[5], 0.22, t().surface),
            12,
            TRACK,
            theme.mix(t().mats[5], 0.09, t().surface),
        ),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )


def _dashed(height: float, color: str) -> ft.Control:
    """border-left: 1.5px dashed, which Flutter has no word for."""
    count = int(height // 6)
    return ft.Column(
        [ft.Container(width=1.5, height=3, bgcolor=color) for _ in range(count)],
        spacing=3,
        tight=True,
    )


def band(engine: Engine, on_ceiling: Callable[[int], None] | None = None) -> ft.Control:
    """.band — the machine's memory drawn to scale, with the ceiling as its one control."""
    system = engine.system
    if system is None:
        return _shell(
            [ft.Text("waiting for the engine", style=theme.mono(9.5, t().fg3))],
            ft.Container(height=TRACK_HEIGHT, border_radius=6, bgcolor=_bed()),
        )

    total = system["memory_bytes"]
    ceiling = engine.ceiling or total
    models = engine.models
    materials = engine.materials
    job = next((j for j in engine.jobs if j["kind"] in ("load", "download")), None)
    pending = job["progress"]["completed"] if job is not None else 0
    used = _used(models)
    free = max(0, ceiling - used - pending)
    headroom = total - ceiling
    over = used + pending > ceiling

    row: list[ft.Control] = [
        _slab(model, materials.get(model["id"], 0), total, index == 0)
        for index, model in enumerate(models)
    ]
    if job is not None:
        row.append(_loading(pending, total))
    row.append(
        ft.Container(
            expand=True,
            padding=ft.Padding.only(left=13),
            # justify-content: center on a full-height flex column; a tight Column has no
            # slack for MainAxisAlignment to divide, so the alignment is the box's.
            alignment=ft.Alignment.CENTER_LEFT,
            content=ft.Column(
                [
                    ft.Text(gb(free), style=theme.mono(19, t().fg2, spacing=-0.02 * 19)),
                    ft.Text("GB free below the ceiling", style=theme.sans(9.5, t().fg3)),
                ],
                spacing=3,
                tight=True,
            ),
        )
    )

    over_track: list[ft.Control] = [
        ft.Row(row, spacing=GAP, vertical_alignment=ft.CrossAxisAlignment.STRETCH)
    ]
    if headroom > 0:
        legend: list[ft.Control] = []
        if headroom / total > 0.1:
            legend.append(
                ft.Container(
                    right=9,
                    top=0,
                    bottom=0,
                    alignment=ft.Alignment.CENTER_RIGHT,
                    content=ft.Text(
                        f"headroom\n{gb(headroom, 0)} GB",
                        style=theme.sans(9, t().fg3, height=1.3),
                        text_align=ft.TextAlign.RIGHT,
                    ),
                )
            )
        over_track.append(
            ft.Container(
                right=0,
                top=0,
                bottom=0,
                width=headroom / total * TRACK,
                # transparent 0 4px, --hair 4px 5px: a hairline every five, not a stripe.
                gradient=theme.hatch(t().hair, 5, TRACK, duty=0.2),
                border_radius=ft.BorderRadius(
                    top_left=0, top_right=6, bottom_left=0, bottom_right=6
                ),
                content=ft.Stack(legend) if legend else None,
            )
        )
    over_track.append(_ceiling(ceiling, total, over, on_ceiling))

    return _shell(
        [
            ft.Text(
                f"{gb(total, 0)} GB · {len(models)} resident",
                style=theme.mono(9.5, t().fg3),
            ),
            ft.Container(expand=True),
            ft.Text(
                f"{gb(used + pending)} in use · ceiling {gb(ceiling, 0)} GB",
                style=theme.mono(9.5, t().fg3),
            ),
        ],
        ft.Container(
            content=ft.Stack(over_track, clip_behavior=ft.ClipBehavior.NONE),
            height=TRACK_HEIGHT,
            border_radius=6,
            bgcolor=_bed(),
            border=theme.hair(),
        ),
    )


def _bed() -> str:
    return theme.mix(t().sunken, 0.55, t().surface)


def _shell(headline: list[ft.Control], track: ft.Control) -> ft.Control:
    return ft.Container(
        margin=ft.Margin.only(left=12, right=12, top=12),
        padding=ft.Padding(left=11, right=11, top=9, bottom=11),
        bgcolor=t().surface,
        border=theme.hair(),
        border_radius=10,
        content=ft.Column(
            [
                ft.Container(
                    padding=ft.Padding.only(bottom=7),
                    content=ft.Row(
                        [
                            ft.Text(
                                "Memory",
                                style=theme.sans(11, weight=ft.FontWeight.W_600),
                            ),
                            *headline,
                        ],
                        spacing=10,
                        vertical_alignment=theme.BASELINE,
                    ),
                ),
                track,
            ],
            spacing=0,
            tight=True,
        ),
    )


def _ceiling(
    ceiling: int, total: int, over: bool, on_ceiling: Callable[[int], None] | None
) -> ft.Control:
    color = t().bad if over else t().fg2
    at = ceiling / total * TRACK
    grip = ft.Container(
        width=9,
        height=9,
        bgcolor=color,
        border_radius=2,
        border=theme.hair(t().surface, 2),
    )
    if on_ceiling is not None:
        grip = ft.GestureDetector(
            content=grip,
            mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
            drag_interval=50,
            on_pan_update=lambda e: on_ceiling(
                round(min(max(at + e.local_position.x - 4.5, 0), TRACK) / TRACK * total)
            ),
        )
    return ft.Container(
        left=at,
        top=-3,
        bottom=-3,
        content=ft.Stack(
            [
                _dashed(TRACK_HEIGHT + 6, color),
                ft.Container(left=-5, top=-4, content=grip),
                ft.Container(
                    right=7,
                    bottom=5,
                    content=ft.Text(
                        f"ceiling {gb(ceiling, 0)} GB",
                        style=theme.mono(8.5, color),
                        no_wrap=True,
                    ),
                ),
            ],
            clip_behavior=ft.ClipBehavior.NONE,
        ),
    )


def meter(engine: Engine, ghost: int | None = None) -> list[ft.Control]:
    """The band compressed into the title bar: same scale, same materials, same tick."""
    system = engine.system
    if system is None:
        return []
    total = system["memory_bytes"]
    ceiling = engine.ceiling or total
    models = engine.models
    materials = engine.materials
    used = _used(models)
    over = used + (ghost or 0) > ceiling

    bars: list[ft.Control] = [
        ft.Container(
            width=(model["weights_bytes"] + model["kv_bytes"]) / total * METER,
            bgcolor=t().mat(materials.get(model["id"], 0)),
        )
        for model in models
    ]
    if ghost:
        # What the checkpoint under the cursor would cost, in the same hatch the band
        # draws bytes that are not here yet with.
        bars.append(
            ft.Container(
                width=ghost / total * METER,
                gradient=theme.hatch(theme.mix(t().fg2, 0.62, theme.TRANSPARENT), 6, METER),
            )
        )
    return [
        ft.Container(
            width=METER,
            height=METER_HEIGHT,
            border_radius=3,
            bgcolor=t().sunken,
            border=theme.hair(),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Stack(
                [
                    ft.Row(bars, spacing=1, vertical_alignment=ft.CrossAxisAlignment.STRETCH),
                    ft.Container(
                        left=ceiling / total * METER,
                        top=0,
                        bottom=0,
                        width=1.5,
                        bgcolor=t().bad if over else t().fg2,
                    ),
                ]
            ),
        ),
        ft.Container(
            height=23,
            padding=ft.Padding.symmetric(horizontal=9),
            alignment=ft.Alignment.CENTER,
            bgcolor=t().surface,
            border=theme.hair(),
            border_radius=6,
            content=ft.Text(
                f"{gb(used)} / {gb(ceiling, 0)} GB", style=theme.mono(11, t().fg2)
            ),
        ),
    ]

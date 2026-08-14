"""Quantize: the fifth place. The selection reads top to bottom — source, format,
method, width, group, name — and the right rail answers what it costs, repriced by the
daemon on every touch. The ~600 leaves fold into their groups below, closed until
somebody needs them.

The rules are the server's; this screen only confesses them before the 400: an
exponent-scaled mode fixes method, width and group size, so those controls leave the
screen; GPTQ packs only 2, 4 and 8, so the other widths dim; oQ and oQe take the budget
nobody else may name.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict, cast

import flet as ft

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.engine import CatalogEntry, Engine, Job
from mlx_omnia.app.ui.format import display_name, gb
from mlx_omnia.appv2 import parts, runtime, theme
from mlx_omnia.appv2.shell import Shell
from mlx_omnia.appv2.theme import t

GIB = 1024**3
DEBOUNCE = 0.18
RAIL = 272

# The engine's five. AWQ and GPTQ price as RTN — neither moves a leaf's format. oQ and
# oQe are priced by the same allocator the job runs, handed no scores.
METHODS = [("rtn", "RTN"), ("awq", "AWQ"), ("gptq", "GPTQ"), ("oq", "oQ"), ("oqe", "oQe")]

# `Affine.__post_init__`: anything else raises out of the engine and comes back 400.
BITS = [2, 3, 4, 5, 6, 8]
GROUP_SIZES = [32, 64, 128]

# GPTQ packs its own codes; only these have a layout verified against mx.quantize.
GPTQ_BITS = (2, 4, 8)

# The three exponent-scaled modes carry an exponent per group and no bias, so each fixes
# its own width and group size — and none has the scale-and-bias grid the search methods
# work over, which is why they are RTN's alone.
MODES: list[tuple[str, str, tuple[int, int] | None]] = [
    ("affine", "Affine", None),
    ("mxfp4", "MXFP4", (32, 4)),
    ("mxfp8", "MXFP8", (32, 8)),
    ("nvfp4", "NVFP4", (16, 4)),
]

REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
INDEX = re.compile(r"^\d+$")

DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


class PlanLeaf(TypedDict):
    path: str
    kind: str
    shape: list[int]
    # `null` says the leaf stays dense, which is what the default or an override asked.
    bits: int | None
    group_size: int | None
    bytes: int


class PricedPlan(TypedDict):
    leaves: list[PlanLeaf]
    total_bytes: int
    weights: int
    bits_per_weight: float
    # What the produced `model.safetensors` will weigh: the leaves at their new width plus
    # everything no plan touches.
    entry_bytes: int


@dataclass(frozen=True)
class Override:
    """One group of leaves against the rest of the plan; `bits` None is dense."""

    bits: int | None
    group_size: int | None = None

    def wire(self) -> dict[str, int] | None:
        if self.bits is None:
            return None
        body = {"bits": self.bits}
        if self.group_size is not None:
            body["group_size"] = self.group_size
        return body


@dataclass(frozen=True)
class LeafGroup:
    """The plan is priced leaf by leaf — a 32B has some six hundred — and the override it
    takes back is a `fullmatch` pattern. Both ends meet here: the leaves collapse into the
    groups the numbered segments make, and the key is the regex matching exactly them."""

    pattern: str
    label: str
    leaves: int
    bytes: int
    bits: int | None
    group_size: int | None
    mixed: bool = False


def group_leaves(leaves: list[PlanLeaf]) -> list[LeafGroup]:
    held: dict[str, list[PlanLeaf]] = {}
    for leaf in leaves:
        segments = leaf["path"].split(".")
        pattern = r"\.".join(
            r"\d+" if INDEX.match(part) else re.escape(part) for part in segments
        )
        held.setdefault(pattern, []).append(leaf)
    groups: list[LeafGroup] = []
    for pattern, members in held.items():
        first = members[0]
        segments = first["path"].split(".")
        groups.append(
            LeafGroup(
                pattern=pattern,
                label=".".join("*" if INDEX.match(part) else part for part in segments),
                leaves=len(members),
                bytes=sum(leaf["bytes"] for leaf in members),
                bits=first["bits"],
                group_size=first["group_size"],
                mixed=any(
                    leaf["bits"] != first["bits"] or leaf["group_size"] != first["group_size"]
                    for leaf in members
                ),
            )
        )
    return groups


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


def _grain(entry: CatalogEntry) -> str:
    dtype = DTYPES.get(entry["dtype"] or "", entry["dtype"] or "?")
    return f"{dtype} · {gb(entry['bytes_on_disk'])} GB"


def quantizing(engine: Engine) -> list[Job]:
    """The quantize jobs still moving, off the same stream every screen reads."""
    return [
        job
        for job in engine.jobs
        if job["kind"] == "quantize" and job["state"] in ("pending", "running")
    ]


# ── the pieces ────────────────────────────────────────────────────────────


def _labelled(label: str, control: ft.Control, hint: str | None = None) -> ft.Control:
    rows: list[ft.Control] = [theme.eyebrow(label), control]
    if hint is not None:
        rows.append(
            ft.Text(hint, style=theme.sans(11.5, t().fg3, height=1.4), max_lines=3)
        )
    return ft.Column(rows, spacing=6, tight=True)


def _seg(
    options: list[tuple[str, str]],
    chosen: str,
    on_pick: Callable[[str], None],
    dimmed: tuple[str, ...] = (),
) -> ft.Container:
    """`parts.scope` with a dim state: a key the current selection refuses stays on the
    list — the rule is worth seeing — but takes no click."""

    def one(key: str, label: str) -> ft.Control:
        on = key == chosen
        dim = key in dimmed and not on
        return ft.Container(
            padding=ft.Padding(left=13, right=13, top=4, bottom=4),
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
                    12.5,
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
        content=ft.Row([one(key, label) for key, label in options], spacing=2, tight=True),
    )


def _drop(
    shown: str,
    options: list[tuple[str, str]],
    chosen: str,
    on_pick: Callable[[str], None],
    set_: bool = False,
) -> ft.Control:
    """A field-styled trigger with a real popup: the options float over the screen."""
    return ft.PopupMenuButton(
        bgcolor=t().elev,
        items=[
            ft.PopupMenuItem(
                content=ft.Text(
                    label,
                    style=theme.mono(11.5, t().accent if key == chosen else t().fg2),
                ),
                on_click=lambda _, key=key: on_pick(key),
            )
            for key, label in options
        ],
        content=ft.Container(
            padding=ft.Padding(left=9, right=9, top=3, bottom=3),
            bgcolor=t().field,
            border=theme.hair(t().accent if set_ else t().hair2),
            border_radius=7,
            content=ft.Row(
                [
                    ft.Text(
                        shown,
                        style=theme.mono(
                            11.5, t().accent if set_ else t().fg2,
                            ft.FontWeight.W_600 if set_ else ft.FontWeight.W_400,
                        ),
                        no_wrap=True,
                    ),
                    ft.Text("▾", style=theme.sans(9, t().fg3), no_wrap=True),
                ],
                spacing=6,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ),
    )


def _field(
    value: str,
    on_change: Callable[[str], None],
    hint: str = "",
    width: float | None = None,
) -> ft.Control:
    return ft.Container(
        width=width,
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


def _refusal(message: str) -> ft.Control:
    return ft.Text(message, style=theme.sans(11.5, t().bad, height=1.5))


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


# ── the view ──────────────────────────────────────────────────────────────


@ft.component
def Quantize(shell: Shell, engine: Engine) -> ft.Control:
    source, set_source = ft.use_state("")
    mode, set_mode = ft.use_state("affine")
    method, set_method = ft.use_state("rtn")
    bits, set_bits = ft.use_state(4)
    group_size, set_group_size = ft.use_state(64)
    overrides, set_overrides = ft.use_state(dict[str, Override]())
    target, set_target = ft.use_state("")
    cap, set_cap = ft.use_state("")
    repo, set_repo = ft.use_state("")
    named, set_named = ft.use_state(False)
    plan, set_plan = ft.use_state(None)
    refusal, set_refusal = ft.use_state("")
    pricing, set_pricing = ft.use_state(False)
    opened, set_opened = ft.use_state(False)
    started, set_started = ft.use_state("")
    job, set_job = ft.use_state(None)
    failure, set_failure = ft.use_state("")

    exponent = next(
        ((label, shape) for key, label, shape in MODES if key == mode and shape), None
    )
    allocated = method in ("oq", "oqe")

    def request() -> dict[str, object]:
        """The selection as the wire carries it. Width and group are left out under an
        exponent-scaled mode, and the budget rides only with the allocator: the daemon
        refuses a request that sets what it decided."""
        body: dict[str, object] = {"source": source, "mode": mode, "method": method}
        if exponent is None:
            body["bits"] = bits
            body["group_size"] = group_size
        if allocated:
            asked = _number(target)
            if asked is not None:
                body["target_bpw"] = asked
            ceiling = _number(cap)
            if ceiling is not None:
                body["hard_cap_bpw"] = ceiling
        body["overrides"] = {key: value.wire() for key, value in overrides.items()}
        return body

    def price() -> Callable[[], None]:
        """Every change of the selection is a new price, and the one in flight is dropped:
        the rail never shows the answer to a question no longer on screen."""

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
                        await engine_api.send(
                            "POST", "/admin/quantizations/plan", request()
                        ),
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

    # The daemon's presence is a dependency too: a price asked while it was still coming
    # up is retried when it answers, instead of standing as a refusal nobody re-asks.
    up = engine.health is not None
    ft.use_effect(
        price, [source, mode, method, bits, group_size, overrides, target, cap, up]
    )

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

    # ── the knobs ─────────────────────────────────────────────────────────

    def rename(chosen: str, chosen_mode: str, chosen_bits: int) -> None:
        if not named:
            set_repo(_suggest(chosen, chosen_mode, chosen_bits))

    def pick_source(identifier: str) -> None:
        set_source(identifier)
        # The patterns belong to the tree they were read off: carried over, they match no
        # leaf and the next price fails for the wrong reason.
        set_overrides({})
        rename(identifier, mode, bits)

    def pick_mode(key: str) -> None:
        set_mode(key)
        shape = next((s for k, _, s in MODES if k == key and s), None)
        if shape is not None:
            # The exponent grid has no bias to search and no width to pick, so the two
            # controls it decides go with it.
            set_method("rtn")
            set_target("")
            set_cap("")
            set_overrides(
                {k: v for k, v in overrides.items() if v.bits is None}
            )
        rename(source, key, bits)

    def pick_method(key: str) -> None:
        set_method(key)
        if key not in ("oq", "oqe"):
            set_target("")
            set_cap("")

    def pick_bits(key: str) -> None:
        set_bits(int(key))
        rename(source, mode, int(key))

    def name(value: str) -> None:
        set_named(True)
        set_repo(value)

    def pick_leaf(pattern: str, choice: str) -> None:
        if choice == "":
            set_overrides({k: v for k, v in overrides.items() if k != pattern})
        elif choice == "dense":
            set_overrides({**overrides, pattern: Override(None)})
        else:
            # The group size the row already carried survives a change of width: the two
            # are picked apart, and re-picking one is not a way of forgetting the other.
            held_override = overrides.get(pattern)
            set_overrides(
                {
                    **overrides,
                    pattern: Override(
                        int(choice),
                        None if held_override is None else held_override.group_size,
                    ),
                }
            )

    def regroup(pattern: str, choice: str) -> None:
        held_override = overrides.get(pattern)
        if held_override is None or held_override.bits is None:
            return
        set_overrides(
            {
                **overrides,
                pattern: Override(
                    held_override.bits, None if choice == "" else int(choice)
                ),
            }
        )

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

    def stop() -> None:
        if held is not None:
            runtime.act(engine_api.cancel_job(held["id"]))

    # ── the left column ───────────────────────────────────────────────────

    dense = [entry for entry in engine.catalog if entry["quantization"] is None]
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
                                13,
                                t().accent if entry["id"] == source else t().fg,
                                ft.FontWeight.W_500,
                            ),
                            no_wrap=True,
                        ),
                        ft.Text(
                            _grain(entry), style=theme.mono(10.5, t().fg3), no_wrap=True
                        ),
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
                        style=theme.sans(
                            13.5, t().fg if source else t().fg3, ft.FontWeight.W_500
                        ),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                    ),
                    ft.Text(
                        "" if chosen is None else _grain(chosen),
                        style=theme.mono(10.5, t().fg3),
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
        _labelled(
            "Source",
            picker,
            "Only dense checkpoints already on this disk."
            if dense
            else "Nothing on this disk to quantize — fetch a model under Models.",
        ),
        _labelled(
            "Format",
            parts.scope([(key, label) for key, label, _ in MODES], mode, pick_mode),
            "A scale and a bias per group, at the width and group size below."
            if exponent is None
            else f"An exponent per group and no bias: {exponent[1][1]} bits at group "
            f"{exponent[1][0]}, packed by mx.quantize. Method, width and group size "
            "are the mode's.",
        ),
    ]
    if exponent is None:
        controls.append(
            _labelled(
                "Method",
                parts.scope(METHODS, method, pick_method),
                "AWQ, GPTQ, oQ and oQe read a calibration pass before writing.",
            )
        )
        controls.append(
            _labelled(
                "Width",
                _seg(
                    [(str(v), str(v)) for v in BITS],
                    str(bits),
                    pick_bits,
                    dimmed=tuple(str(v) for v in BITS if v not in GPTQ_BITS)
                    if method == "gptq"
                    else (),
                ),
                "GPTQ packs its own codes: only 2, 4 and 8 have a layout verified "
                "against mx.quantize."
                if method == "gptq"
                else None,
            )
        )
        controls.append(
            _labelled(
                "Group size",
                parts.scope(
                    [(str(v), str(v)) for v in GROUP_SIZES],
                    str(group_size),
                    lambda v: set_group_size(int(v)),
                ),
            )
        )
    if allocated:
        controls.append(
            _labelled(
                "Budget",
                ft.Row(
                    [
                        _field(target, set_target, "RTN + 1.0", width=110),
                        ft.Text("target", style=theme.sans(11.5, t().fg3), no_wrap=True),
                        _field(cap, set_cap, "target", width=110),
                        ft.Text("hard cap", style=theme.sans(11.5, t().fg3), no_wrap=True),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                "Effective bits per weight, scales and biases included. Left empty, the "
                "allocator takes the RTN plan plus one bit of headroom.",
            )
        )

    output: list[ft.Control] = [_field(repo, name)]
    if repo != "" and not REPO.match(repo):
        output.append(
            _refusal(
                "Two segments, and nothing that could name a directory outside the "
                "cache: org/name."
            )
        )
    controls.append(_labelled("Output id", ft.Column(output, spacing=4, tight=True)))

    groups = [] if priced is None else group_leaves(priced["leaves"])
    fold = ft.Container(
        padding=ft.Padding(left=2, right=2, top=2, bottom=0),
        on_click=(lambda _: set_opened(not opened)) if groups else None,
        content=ft.Row(
            [
                ft.Text("▾" if opened else "▸", style=theme.sans(9, t().fg3), no_wrap=True),
                ft.Text(
                    "Per leaf group",
                    style=theme.sans(12.5, t().fg2, ft.FontWeight.W_500),
                    no_wrap=True,
                ),
                ft.Text(
                    f"{len(groups)} groups · {len(priced['leaves'])} leaves"
                    if priced is not None
                    else "the leaves come with the price",
                    style=theme.mono(10.5, t().fg3),
                    no_wrap=True,
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
    controls.append(fold)

    # ── the right rail ────────────────────────────────────────────────────

    rail: list[ft.Control] = [theme.eyebrow("Projected")]
    if priced is None:
        rail.append(
            ft.Container(
                padding=ft.Padding.only(top=6),
                content=ft.Text("—", style=theme.mono(34, spacing=-0.03 * 34)),
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
                padding=ft.Padding.only(top=6),
                content=ft.Row(
                    [
                        ft.Text(
                            gb(priced["entry_bytes"]),
                            style=theme.mono(34, spacing=-0.03 * 34),
                        ),
                        ft.Text("GB", style=theme.mono(11, t().fg3)),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(top=12),
                height=4,
                border_radius=3,
                bgcolor=t().sel,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                content=ft.Row(
                    [
                        ft.Container(
                            expand=max(1, round(fraction * 1000)), bgcolor=t().accent
                        ),
                        ft.Container(expand=max(1, round((1 - fraction) * 1000))),
                    ],
                    spacing=0,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(top=6),
                content=ft.Row(
                    [
                        ft.Text(
                            f"{priced['bits_per_weight']:.2f} bits / weight",
                            style=theme.mono(10.5, t().fg3),
                            no_wrap=True,
                        ),
                        ft.Container(expand=True),
                        ft.Text(
                            "" if chosen is None else f"source {gb(chosen['bytes_on_disk'])} GB",
                            style=theme.mono(10.5, t().fg3),
                            no_wrap=True,
                        ),
                    ],
                    spacing=8,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(top=8),
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
        ]
        rail.append(
            ft.Container(
                margin=ft.Margin.only(top=10),
                content=ft.Text(
                    "The size and budget hold; which leaves the calibration promotes is "
                    "measured by the job, not projected here."
                    if allocated
                    else "Priced against the checkpoint's own leaves — nothing written "
                    "yet. Parity is measured after the job, never predicted.",
                    style=theme.sans(11, t().fg3, height=1.5),
                ),
            )
        )
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
                    style=theme.sans(11.5, t().ok, height=1.5),
                ),
            )
        )
    if running and held is not None:
        part = held["progress"]["total"]
        done = held["progress"]["completed"]
        rail.append(
            ft.Container(
                margin=ft.Margin.only(top=8),
                content=ft.Text(
                    (f"{round(done / part * 100)}% · " if part else "")
                    + held["progress"]["message"],
                    style=theme.mono(10.5, t().fg2),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
            )
        )
    ready = (
        priced is not None and not refusal and bool(REPO.match(repo)) and not running
    )
    if running:
        action: ft.Control = parts.button("Cancel", stop, "danger", expand=True)
    elif ready:
        action = parts.button("Quantize", launch, "primary", expand=True)
    else:
        action = ft.Container(
            expand=True,
            opacity=0.45,
            content=parts.button("Quantize", None, "primary", expand=True),
        )
    rail.append(ft.Container(margin=ft.Margin.only(top=12), content=ft.Row([action])))

    cols = ft.Row(
        [
            ft.Column(controls, spacing=15, tight=True, expand=True),
            ft.Container(
                width=RAIL,
                padding=ft.Padding(left=15, right=15, top=13, bottom=13),
                bgcolor=t().elev,
                border=theme.hair(),
                border_radius=12,
                opacity=0.55 if pricing else 1.0,
                content=ft.Column(rail, spacing=2, tight=True),
            ),
        ],
        spacing=14,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

    rows: list[ft.Control] = [
        ft.WindowDragArea(content=ft.Container(height=24), maximizable=False),
        ft.Container(
            padding=ft.Padding(left=2, right=2, top=0, bottom=0),
            content=ft.Text("Quantize", style=theme.display(20)),
        ),
        ft.Container(
            padding=ft.Padding(left=2, right=2, top=0, bottom=8),
            content=ft.Text(
                "Creates a model — priced by the daemon before a byte is written.",
                style=theme.sans(12.5, t().fg3),
            ),
        ),
        cols,
    ]
    if opened and groups:
        widths = (
            []
            if exponent is not None
            else BITS
            if allocated
            else [value for value in BITS if value != bits]
        )
        placeholder = (
            exponent[0]
            if exponent is not None
            else "auto"
            if allocated
            else f"{bits}-bit"
        )
        rows.append(_leaves(groups, overrides, widths, placeholder, exponent is None,
                            group_size, pick_leaf, regroup))

    return ft.Container(
        expand=True,
        padding=ft.Padding(left=16, right=16, top=0, bottom=16),
        content=ft.Column(rows, spacing=8, scroll=ft.ScrollMode.AUTO, expand=True),
    )


def _leaves(
    groups: list[LeafGroup],
    overrides: dict[str, Override],
    widths: list[int],
    placeholder: str,
    affine: bool,
    plan_group: int,
    pick_leaf: Callable[[str, str], None],
    regroup: Callable[[str, str], None],
) -> ft.Control:
    rows: list[ft.Control] = []
    for index, entry in enumerate(groups):
        held = overrides.get(entry.pattern)
        overridden = entry.pattern in overrides
        selected = (
            ""
            if not overridden
            else "dense"
            if held is None or held.bits is None
            else str(held.bits)
        )
        shown = (
            ("mixed" if entry.mixed else placeholder)
            if not overridden
            else "dense"
            if held is None or held.bits is None
            else f"{held.bits}-bit"
        )
        cells: list[ft.Control] = [
            ft.Text(
                entry.label,
                style=theme.mono(11, t().fg3 if entry.bits is None else t().fg),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
                expand=True,
                tooltip=entry.pattern,
            )
        ]
        if overridden:
            cells.append(
                ft.Container(
                    padding=ft.Padding(left=6, right=6, top=1, bottom=1),
                    border=theme.hair(t().accent),
                    border_radius=5,
                    content=ft.Text(
                        "override",
                        style=theme.sans(9, t().accent, ft.FontWeight.W_600),
                        no_wrap=True,
                    ),
                )
            )
        cells.append(
            ft.Text(
                f"{entry.leaves} {'leaf' if entry.leaves == 1 else 'leaves'} · "
                f"{entry.bytes / GIB:.2f} GB",
                style=theme.mono(10, t().fg3),
                no_wrap=True,
            )
        )
        cells.append(
            _drop(
                shown,
                [("", placeholder), *((str(v), f"{v}-bit") for v in widths), ("dense", "dense")],
                selected,
                lambda choice, pattern=entry.pattern: pick_leaf(pattern, choice),
                set_=overridden,
            )
        )
        if held is not None and held.bits is not None and affine:
            cells.append(
                _drop(
                    f"group {plan_group if held.group_size is None else held.group_size}",
                    [
                        ("", f"group {plan_group}"),
                        *(
                            (str(v), f"group {v}")
                            for v in GROUP_SIZES
                            if v != plan_group
                        ),
                    ],
                    "" if held.group_size is None else str(held.group_size),
                    lambda choice, pattern=entry.pattern: regroup(pattern, choice),
                    set_=held.group_size is not None,
                )
            )
        rows.append(
            ft.Container(
                padding=ft.Padding.symmetric(vertical=8),
                border=ft.Border.only(top=ft.BorderSide(1, t().hair)) if index else None,
                content=ft.Row(
                    cells, spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER
                ),
            )
        )
    return ft.Container(
        padding=ft.Padding(left=16, right=16, top=4, bottom=4),
        bgcolor=t().elev,
        border=theme.hair(),
        border_radius=12,
        content=ft.Column(rows, spacing=0, tight=True),
    )

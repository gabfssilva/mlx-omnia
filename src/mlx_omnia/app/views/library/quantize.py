"""The quantize sheet: pick a dense checkpoint, a format and a width, and the daemon
prices the plan leaf by leaf before a byte is written.

The catalog, residency and the machine arrive on the event stream, so this screen asks
for the two things nothing else knows: the price and the job.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict, cast

import flet as ft

from mlx_omnia.app.api import engine as engine_api
from mlx_omnia.app.api.engine import Engine, Job
from mlx_omnia.app.ui import parts, theme
from mlx_omnia.app.ui.format import gb
from mlx_omnia.app.ui.forms import labelled
from mlx_omnia.app.ui.hooks import act
from mlx_omnia.app.ui.theme import t

GIB = 1024**3
DEBOUNCE = 0.18

# The engine's five. AWQ and GPTQ price as RTN — neither moves a leaf's format. oQ and oQe
# are priced by the same allocator the job runs, handed no scores.
METHODS = [("rtn", "RTN"), ("awq", "AWQ"), ("gptq", "GPTQ"), ("oq", "oQ"), ("oqe", "oQe")]

# `Affine.__post_init__`: anything else raises out of the engine and comes back 400.
BITS = [2, 3, 4, 5, 6, 8]
GROUP_SIZES = [32, 64, 128]

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


@dataclass
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
    groups: dict[str, LeafGroup] = {}
    for leaf in leaves:
        segments = leaf["path"].split(".")
        pattern = r"\.".join(
            r"\d+" if INDEX.match(part) else re.escape(part) for part in segments
        )
        found = groups.get(pattern)
        if found is None:
            groups[pattern] = LeafGroup(
                pattern=pattern,
                label=".".join("*" if INDEX.match(part) else part for part in segments),
                leaves=1,
                bytes=leaf["bytes"],
                bits=leaf["bits"],
                group_size=leaf["group_size"],
            )
            continue
        found.leaves += 1
        found.bytes += leaf["bytes"]
        if found.bits != leaf["bits"] or found.group_size != leaf["group_size"]:
            found.mixed = True
    return list(groups.values())


def _width(dtype: str | None) -> str:
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(dtype or "", dtype or "")


def _suggest(source: str, mode: str, bits: int) -> str:
    """`local/` because a load by id asks the Hub first, and a name that also exists up
    there resolves to somebody else's weights."""
    if source == "":
        return ""
    tail = source.split("/")[-1]
    return f"local/{tail}-{f'{bits}bit' if mode == 'affine' else mode}"


def _finished(job: Job | None) -> bool:
    return job is None or job["state"] in ("ok", "error", "cancelled")


@dataclass
class Override:
    bits: int | None  # None is dense
    group_size: int | None = None

    def wire(self) -> dict[str, int] | None:
        if self.bits is None:
            return None
        body = {"bits": self.bits}
        if self.group_size is not None:
            body["group_size"] = self.group_size
        return body


@dataclass
class Quantize:
    """Kept across rebuilds; opened by Library and closed back to it."""

    redraw: Callable[[], None]
    close: Callable[[], None]
    source: str = ""
    method: str = "rtn"
    mode: str = "affine"
    bits: int = 4
    group_size: int = 64
    overrides: dict[str, Override] = field(default_factory=dict)
    repo: str = ""
    named: bool = False
    plan: PricedPlan | None = None
    refusal: str | None = None
    pricing: bool = False
    job: Job | None = None
    failure: str | None = None
    _price: asyncio.Task[None] | None = None
    _stream: asyncio.Task[None] | None = None

    # ── what the sheet does ───────────────────────────────────────────────

    def open(self, source: str) -> None:
        self.source = source
        self.overrides = {}
        self.plan = None
        self.refusal = None
        self.job = None
        self.failure = None
        self.named = False
        self._rename()
        self._reprice()

    def _rename(self) -> None:
        if not self.named:
            self.repo = _suggest(self.source, self.mode, self.bits)

    def _reprice(self) -> None:
        """Every change of the selection is a new price, and the one in flight is dropped:
        the panel never shows the answer to a question no longer on screen."""
        if self._price is not None:
            self._price.cancel()
        if self.source == "":
            return
        self._price = asyncio.get_running_loop().create_task(self._ask())

    def _selection(self) -> dict[str, object]:
        """The width controls as the request carries them. Left out under an
        exponent-scaled mode: the mode fixes both, and the daemon refuses a request that
        sets what it decided."""
        if self.mode != "affine":
            return {"mode": self.mode}
        return {"mode": self.mode, "bits": self.bits, "group_size": self.group_size}

    def _request(self) -> dict[str, object]:
        return {
            "source": self.source,
            **self._selection(),
            "overrides": {key: value.wire() for key, value in self.overrides.items()},
            "method": self.method,
        }

    async def _ask(self) -> None:
        try:
            await asyncio.sleep(DEBOUNCE)
            self.pricing = True
            self.redraw()
            self.plan = cast(
                PricedPlan,
                await engine_api.send("POST", "/admin/quantizations/plan", self._request()),
            )
            self.refusal = None
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            self.plan = None
            self.refusal = str(error)
        finally:
            self.pricing = False
            self.redraw()

    def _launch(self) -> None:
        act(self._create())

    async def _create(self) -> None:
        self.failure = None
        try:
            started = cast(
                Job,
                await engine_api.send(
                    "POST", "/admin/quantizations", {**self._request(), "repo": self.repo}
                ),
            )
            self.job = started
            self.redraw()
            if self._stream is not None:
                self._stream.cancel()
            self._stream = asyncio.get_running_loop().create_task(self._follow(started["id"]))
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
            self.redraw()

    async def _follow(self, identifier: str) -> None:
        try:
            async for frame in engine_api.job_events(identifier):
                self.job = frame
                self.redraw()
        except Exception:  # noqa: BLE001
            pass

    def _stop(self) -> None:
        job = self.job
        if job is None:
            return
        act(self._cancel(job["id"]))

    async def _cancel(self, identifier: str) -> None:
        try:
            await engine_api.cancel_job(identifier)
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.redraw()

    # ── the knobs ─────────────────────────────────────────────────────────

    def _set_source(self, value: str) -> None:
        self.source = value
        # The patterns belong to the tree they were read off: carried over, they match no
        # leaf and the next price fails for the wrong reason.
        self.overrides = {}
        self._rename()
        self._reprice()
        self.redraw()

    def _set_mode(self, value: str) -> None:
        self.mode = value
        shape = next((s for key, _, s in MODES if key == value), None)
        if shape is not None:
            # The exponent grid has no bias to search and no width to pick, so the two
            # controls it decides go with it.
            self.method = "rtn"
            self.overrides = {
                key: value_ for key, value_ in self.overrides.items() if value_.bits is None
            }
        self._rename()
        self._reprice()
        self.redraw()

    def _set(self, **changes: object) -> None:
        for key, value in changes.items():
            setattr(self, key, value)
        self._rename()
        self._reprice()
        self.redraw()

    def _set_repo(self, value: str) -> None:
        self.named = True
        self.repo = value
        self.redraw()

    def _pick(self, pattern: str, choice: str) -> None:
        if choice == "":
            self.overrides.pop(pattern, None)
        elif choice == "dense":
            self.overrides[pattern] = Override(None)
        else:
            # The group size the row already carried survives a change of width: the two
            # are picked apart, and re-picking one is not a way of forgetting the other.
            held = self.overrides.get(pattern)
            self.overrides[pattern] = Override(
                int(choice), None if held is None else held.group_size
            )
        self._reprice()
        self.redraw()

    def _regroup(self, pattern: str, choice: str) -> None:
        """Only where a width was already picked: the wire has no override naming a group
        size alone, and the plan's own is what a row without one means."""
        held = self.overrides.get(pattern)
        if held is None or held.bits is None:
            return
        held.group_size = None if choice == "" else int(choice)
        self._reprice()
        self.redraw()

    # ── drawing ───────────────────────────────────────────────────────────

    def tree(self, engine: Engine) -> ft.Control:
        entries = [entry for entry in engine.catalog]
        chosen = next((entry for entry in entries if entry["id"] == self.source), None)
        exponent = next(
            ((key, label, s) for key, label, s in MODES if key == self.mode and s), None
        )
        # The two methods whose widths are the allocator's: what a group the caller left
        # alone ends up at is decided by the calibration, so there is no number to name.
        allocated = self.method in ("oq", "oqe")
        # Under the allocator the base width is a width like any other: it is not what a
        # group left alone gets, so naming it is a choice and it stays on the list.
        widths = (
            []
            if exponent is not None
            else BITS
            if allocated
            else [value for value in BITS if value != self.bits]
        )
        groups = [] if self.plan is None else group_leaves(self.plan["leaves"])
        running = self.job is not None and not _finished(self.job)
        ready = (
            self.plan is not None
            and self.refusal is None
            and bool(REPO.match(self.repo))
            and not running
        )

        return parts.sheet(self._header(running, ready), self._body(
            entries, chosen, exponent, allocated, widths, groups, engine
        ), 900)

    def _header(self, running: bool, ready: bool) -> list[ft.Control]:
        row: list[ft.Control] = [
            ft.Text("Quantize", style=theme.sans(13.5, weight=ft.FontWeight.W_600), no_wrap=True),
            ft.Text(
                self.source or "pick a checkpoint",
                style=theme.mono(11, t().fg3),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
                expand=True,
            ),
        ]
        job = self.job
        if job is not None:
            part = job["progress"]["total"]
            done = (
                f"{round(job['progress']['completed'] / part * 100)}% · "
                if part and running
                else ""
            )
            row.append(
                parts.chip(
                    [
                        parts.dot("bad" if job["state"] == "error" else "ok" if running else ""),
                        ft.Container(
                            width=220,
                            content=ft.Text(
                                done + (job["error"] or job["progress"]["message"]),
                                style=theme.sans(11, t().fg2),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ),
                    ],
                    gap=7,
                    left=7,
                )
            )
        row.append(
            parts.btn("Cancel", self._stop, "danger")
            if running
            else parts.btn("Quantize", self._launch, "pri", ready)
        )
        row.append(parts.btn("Close", self.close, "quiet"))
        return row

    def _body(
        self,
        entries: list[engine_api.CatalogEntry],
        chosen: engine_api.CatalogEntry | None,
        exponent: tuple[str, str, tuple[int, int] | None] | None,
        allocated: bool,
        widths: list[int],
        groups: list[LeafGroup],
        engine: Engine,
    ) -> ft.Control:
        stack: list[ft.Control] = []
        if self.failure is not None:
            stack.append(_refusal(self.failure))
        stack.append(
            ft.Row(
                [
                    ft.Container(expand=True, content=self._left(entries, exponent)),
                    ft.Container(width=300, content=self._projected(chosen, engine)),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
        stack.append(self._overrides(groups, exponent, allocated, widths))
        return ft.Container(
            height=560,
            padding=ft.Padding.symmetric(horizontal=15, vertical=13),
            content=ft.Column(stack, spacing=10, scroll=ft.ScrollMode.AUTO, expand=True),
        )

    def _left(
        self,
        entries: list[engine_api.CatalogEntry],
        exponent: tuple[str, str, tuple[int, int] | None] | None,
    ) -> ft.Control:
        def describe(entry: engine_api.CatalogEntry) -> str:
            grain = " ".join(
                part
                for part in ((entry["quantization"] or "dense"), _width(entry["dtype"]))
                if part
            )
            return f"{entry['id']} · {grain} · {gb(entry['bytes_on_disk'])} GB"

        options = (
            [(entry["id"], describe(entry)) for entry in entries]
            if entries
            else [("", "nothing on this disk to quantize")]
        )
        fields: list[ft.Control] = [
            labelled("Source", parts.pick(options, self.source, self._set_source)),
            labelled(
                "Format",
                parts.seg([(key, label) for key, label, _ in MODES], self.mode, self._set_mode),
                parts.note(
                    "A scale and a bias per group, at the width and group size below."
                    if exponent is None
                    else f"An exponent per group and no bias: {exponent[2][1]} bits at group "  # type: ignore[index]
                    f"{exponent[2][0]}, packed by mx.quantize. Method, width and group size "  # type: ignore[index]
                    "are the mode's."
                ),
                stretch=False,
            ),
        ]
        if exponent is None:
            fields += [
                labelled(
                    "Method",
                    parts.seg(METHODS, self.method, lambda v: self._set(method=v)),
                    parts.note("AWQ, GPTQ, oQ and oQe run a calibration pass before writing."),
                    stretch=False,
                ),
                labelled(
                    "Width",
                    parts.seg(
                        [(str(v), str(v)) for v in BITS],
                        str(self.bits),
                        lambda v: self._set(bits=int(v)),
                    ),
                    stretch=False,
                ),
                labelled(
                    "Group size",
                    parts.seg(
                        [(str(v), str(v)) for v in GROUP_SIZES],
                        str(self.group_size),
                        lambda v: self._set(group_size=int(v)),
                    ),
                    stretch=False,
                ),
            ]
        output: list[ft.Control] = [parts.field(self.repo, self._set_repo, mono=True)]
        if self.repo != "" and not REPO.match(self.repo):
            output.append(
                _refusal(
                    "Two segments, and nothing that could name a directory outside the "
                    "cache: org/name."
                )
            )
        fields.append(
            labelled("Output id", ft.Column(output, spacing=4, tight=True))
        )
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=11),
            bgcolor=t().win,
            border=theme.hair(),
            border_radius=9,
            content=ft.Column(fields, spacing=11, tight=True),
        )

    def _projected(
        self, chosen: engine_api.CatalogEntry | None, engine: Engine
    ) -> ft.Control:
        plan = self.plan
        held = (
            None
            if engine.state is None
            else engine.state["resident_bytes"] + engine.state["kv_bytes"]
        )
        stack: list[ft.Control] = [theme.eyebrow("Projected")]
        if plan is None:
            stack.append(
                ft.Container(
                    padding=ft.Padding.only(top=8),
                    content=ft.Text("—", style=theme.mono(30, spacing=-0.03 * 30)),
                )
            )
            if self.refusal is not None:
                stack.append(_refusal(self.refusal))
        else:
            fraction = (
                0.0
                if chosen is None or chosen["bytes_on_disk"] == 0
                else min(1.0, plan["entry_bytes"] / chosen["bytes_on_disk"])
            )
            memory = engine.system["memory_bytes"] if engine.system else None
            stack += [
                ft.Container(
                    padding=ft.Padding.only(top=8),
                    content=ft.Row(
                        [
                            ft.Text(gb(plan["entry_bytes"]),
                                    style=theme.mono(30, spacing=-0.03 * 30)),
                            ft.Text("GB", style=theme.mono(11, t().fg3)),
                        ],
                        spacing=5,
                        vertical_alignment=theme.BASELINE,
                    ),
                ),
                ft.Container(
                    padding=ft.Padding.only(top=16),
                    content=ft.Column(
                        [
                            ft.Container(
                                height=4,
                                border_radius=3,
                                bgcolor=t().sunken,
                                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                                content=ft.Row(
                                    [
                                        ft.Container(
                                            expand=max(1, round(fraction * 1000)),
                                            bgcolor=t().mat(0),
                                        ),
                                        ft.Container(
                                            expand=max(1, round((1 - fraction) * 1000))
                                        ),
                                    ],
                                    spacing=0,
                                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                                ),
                            ),
                            parts.sublab(
                                f"{plan['bits_per_weight']:.2f} bits / weight",
                                ""
                                if chosen is None
                                else f"source {gb(chosen['bytes_on_disk'])} GB",
                            ),
                        ],
                        spacing=5,
                        tight=True,
                    ),
                ),
                ft.Container(
                    padding=ft.Padding.only(top=14),
                    content=_kvline(
                        "If loaded now",
                        "—"
                        if held is None or memory is None
                        else f"{gb(held + plan['entry_bytes'])} / {round(memory / GIB)} GB",
                    ),
                ),
                _kvline(
                    "Leaves packed",
                    f"{sum(1 for leaf in plan['leaves'] if leaf['bits'] is not None)} / "
                    f"{len(plan['leaves'])}",
                ),
            ]
        stack.append(
            ft.Container(
                padding=ft.Padding.only(top=12),
                content=parts.note(
                    "The size and budget hold; which leaves the calibration promotes is "
                    "measured by the job, not projected here."
                    if self.method in ("oq", "oqe") and plan is not None
                    else "Priced by the daemon against the checkpoint's own leaves — nothing "
                    "written yet. Parity is measured after the job, never predicted."
                ),
            )
        )
        return ft.Container(
            padding=ft.Padding(left=11, right=11, top=10, bottom=9),
            bgcolor=t().win,
            border=theme.hair(),
            border_radius=9,
            opacity=0.55 if self.pricing else 1.0,
            content=ft.Column(stack, spacing=8, tight=True),
        )

    def _overrides(
        self,
        groups: list[LeafGroup],
        exponent: tuple[str, str, tuple[int, int] | None] | None,
        allocated: bool,
        widths: list[int],
    ) -> ft.Control:
        stack: list[ft.Control] = [
            ft.Row(
                [
                    ft.Text("Overrides", style=theme.sans(11.5, weight=ft.FontWeight.W_600)),
                    ft.Text(
                        "per leaf group, from the checkpoint itself",
                        style=theme.sans(10.5, t().fg3),
                    ),
                ],
                spacing=7,
                vertical_alignment=theme.BASELINE,
            )
        ]
        if not groups:
            stack.append(
                parts.note("The leaves come with the price: pick a source the daemon can plan.")
            )
        else:
            for index, entry in enumerate(groups):
                stack.append(
                    self._leaf(
                        entry,
                        widths,
                        exponent[1]
                        if exponent is not None
                        else "auto"
                        if allocated
                        else "mixed"
                        if entry.mixed
                        else f"{self.bits}-bit",
                        GROUP_SIZES if exponent is None else [],
                        self.group_size if exponent is None else None,
                        index > 0,
                    )
                )
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=11),
            bgcolor=t().win,
            border=theme.hair(),
            border_radius=9,
            content=ft.Column(stack, spacing=0, tight=True),
        )

    def _leaf(
        self,
        entry: LeafGroup,
        widths: list[int],
        placeholder: str,
        groups: list[int],
        plan_group: int | None,
        ruled: bool,
    ) -> ft.Control:
        held = self.overrides.get(entry.pattern)
        overridden = entry.pattern in self.overrides
        selected = (
            ""
            if not overridden
            else "dense"
            if held is None or held.bits is None
            else str(held.bits)
        )
        frozen = entry.bits is None
        row: list[ft.Control] = [
            ft.Text(
                entry.label,
                style=theme.mono(10.5, t().fg3 if frozen else t().fg),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
                expand=True,
                tooltip=entry.pattern,
            )
        ]
        if overridden:
            row.append(
                ft.Container(
                    padding=ft.Padding(left=6, right=6, top=1, bottom=1),
                    border=theme.hair(theme.mix(t().mat(0), 0.35, theme.TRANSPARENT)),
                    border_radius=5,
                    content=ft.Text(
                        "override", style=theme.sans(9, t().mat(0), ft.FontWeight.W_600)
                    ),
                )
            )
        row += [
            ft.Container(
                width=130,
                content=ft.Text(
                    f"{entry.leaves} {'leaf' if entry.leaves == 1 else 'leaves'} · "
                    f"{entry.bytes / GIB:.2f} GB",
                    style=theme.mono(10, t().fg3),
                    no_wrap=True,
                    text_align=ft.TextAlign.RIGHT,
                ),
            ),
            parts.pick(
                [("", placeholder), *((str(v), f"{v}-bit") for v in widths), ("dense", "dense")],
                selected,
                lambda choice, pattern=entry.pattern: self._pick(pattern, choice),
                width=82,
            ),
        ]
        if held is not None and held.bits is not None and groups:
            row.append(
                parts.pick(
                    [
                        ("", f"group {plan_group}"),
                        *((str(v), f"group {v}") for v in groups if v != plan_group),
                    ],
                    "" if held.group_size is None else str(held.group_size),
                    lambda choice, pattern=entry.pattern: self._regroup(pattern, choice),
                    width=96,
                )
            )
        return ft.Container(
            padding=ft.Padding.symmetric(vertical=7),
            border=ft.Border.only(top=ft.BorderSide(1, t().hair)) if ruled else None,
            content=ft.Row(row, spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )


def _refusal(message: str) -> ft.Control:
    return ft.Text(message, style=theme.sans(11, t().bad, height=1.55))


def _kvline(label: str, value: str) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.symmetric(vertical=5.5),
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(
            [
                theme.eyebrow(label),
                ft.Container(expand=True),
                ft.Text(value, style=theme.mono(11, weight=ft.FontWeight.W_500), no_wrap=True),
            ],
            spacing=10,
            vertical_alignment=theme.BASELINE,
        ),
    )

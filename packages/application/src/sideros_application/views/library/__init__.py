"""Every checkpoint, wherever it is.

What used to be two screens — Models and Download — differed by something the person does
not control: whether the bytes had already come down. Here it is one shelf and a filter,
and the same card answers "load it" and "fetch it, then load it".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

import flet as ft
import flet.canvas as cv

from sideros_application.api import catalog as hub
from sideros_application.api import engine as engine_api
from sideros_application.api.catalog import HubModel, Staged
from sideros_application.api.downloads import Downloads, Pull
from sideros_application.api.engine import CatalogEntry, Engine
from sideros_application.ui import parts, theme
from sideros_application.ui.chrome import Chrome
from sideros_application.ui.format import (
    display_name,
    gb,
    left,
    owner,
    rate,
    size,
    size_pair,
    tokens,
)
from sideros_application.ui.hooks import act, sheets
from sideros_application.ui.theme import t
from sideros_application.views.library.card import Card
from sideros_application.views.library.profiles import Profiles, Speculation
from sideros_application.views.library.quantize import Quantize

FILTERS = [
    ("all", "All"),
    ("resident", "Resident"),
    ("disk", "On disk"),
    ("incomplete", "Incomplete"),
    ("hub", "Hub"),
]

TABS = [
    ("card", "Card"),
    ("features", "Features"),
    ("files", "Files"),
    ("profiles", "Profiles"),
    ("source", "Source"),
]

ON_ENTRY = {"features", "profiles"}
"""Tabs about what the daemon holds for a checkpoint rather than what the checkpoint is.
Nothing to show for one that is not on this machine."""

COLUMNS = 3

PANEL = 529.0
"""What the inspector is here, rather than Resident's `INSPECTOR`: three tiles of the same
width the four-up grid had, and every point the fourth column was taking. The panel is
where a checkpoint is read and configured, and a tile is a name and a size."""

# The Hub is only searched once there is something to search for, and only after the
# typing stops.
SETTLE = 0.35
MIN_TERM = 2


@dataclass
class Chosen:
    id: str
    hub: bool


@ft.component
def Library(
    engine: Engine, chrome: Chrome, downloads: Downloads, jump: Callable[[str], None]
) -> ft.Control:
    shelf, _ = ft.use_state(lambda: Shelf(downloads))

    # Bytes on disk with no model to show for them, and what the selection costs, are read
    # off the same beat as the catalog: a download landing turns staging into a checkpoint.
    ft.use_effect(
        lambda: act(shelf.refresh(engine)),
        [
            tuple(entry["id"] for entry in engine.catalog),
            tuple(sorted(downloads.active)),
            None if shelf.chosen is None else shelf.chosen.id,
        ],
    )
    # Escape closes the sheet over this shelf, and nothing when there is none.
    ft.use_effect(lambda: sheets(lambda: shelf.dismiss), [])
    # The meter in the title bar draws the ghost, and this is the screen that knows it.
    chrome.ghost = shelf.price

    return shelf.tree(engine, jump)


@ft.observable
class Shelf:
    """What this screen holds: the search, the selection, and the four panels that answer
    for it."""

    def __init__(self, downloads: Downloads) -> None:
        self.downloads = downloads
        self.query = ""
        self.filter = "all"
        self.hub: list[HubModel] = []
        self.staged: list[Staged] = []
        self.chosen: Chosen | None = None
        self.tab = "card"
        # What the selection would cost, drawn as a ghost on the title bar's meter.
        self.price: int | None = None
        self.failure: str | None = None
        self.busy = False
        self.confirming = False
        self.cancelling = False
        self._search: asyncio.Task[None] | None = None
        # The inspector's three panels, which own their own fetches: the panel is rebuilt
        # every frame and a fetch started there would ask the Hub sixty times a second.
        # A panel owns its own fetches, and says so here when one lands: the screen is
        # what redraws, and the panel is not the one holding it.
        self.card = Card(self.notify, self._take_price)
        self.profiles = Profiles(self.notify)
        self.speculation = Speculation(self.notify)
        self.quantizing = False
        self.quantize = Quantize(self.notify, self._shut)

    def _open_quantize(self, source: str) -> None:
        self.quantizing = True
        self.quantize.open(source)
        self.notify()

    def _shut(self) -> None:
        self.quantizing = False
        self.notify()

    @property
    def dismiss(self) -> Callable[[], None] | None:
        """What Escape closes here: the quantize sheet, while it is over the shelf."""
        return self._shut if self.quantizing else None

    def _take_price(self, total: int) -> None:
        """The listing the card block fetched is also what a download would cost, so the
        Hub is asked for it once and both readers divide the same number out."""
        chosen = self.chosen
        if chosen is not None and chosen.hub:
            self.price = total
            self.notify()

    # ── drawing ───────────────────────────────────────────────────────────

    def tree(self, engine: Engine, jump: Callable[[str], None]) -> ft.Control:
        self._jump = jump
        work = ft.Container(
            expand=True,
            padding=12,
            content=ft.Row(
                [self._shelf(engine), self._inspector(engine)],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )
        return ft.Container(
            expand=True,
            bgcolor=t().win,
            content=ft.Stack([work, self.quantize.tree(engine)], expand=True)
            if self.quantizing
            else work,
        )

    def _shelf(self, engine: Engine) -> ft.Control:
        disk = engine.catalog
        on_disk = {entry["id"] for entry in disk}
        term = self.query.strip().lower()

        def matches(identifier: str) -> bool:
            return term == "" or term in identifier.lower()

        shown = [
            entry
            for entry in disk
            if (entry["resident"] if self.filter == "resident" else self.filter in ("all", "disk"))
            and matches(entry["id"])
        ]
        hub_shown = [model for model in self.hub if model["id"] not in on_disk]

        # A repository is in exactly one of these: the daemon is fetching it, it has bytes
        # from a download that stopped, or it is neither. A failed pull counts as
        # incomplete — same bytes, and it is the one that can also say why it stopped.
        pulls = [p for p in self.downloads.active.values() if p.repo not in on_disk]
        running = [p for p in pulls if p.state != "error"]
        failed = [p for p in pulls if p.state == "error"]
        abandoned = [
            s
            for s in self.staged
            if s["repo"] not in self.downloads.active and s["repo"] not in on_disk
        ]

        incomplete = self.filter in ("all", "incomplete")
        pulls_shown = [p for p in (running if self.filter == "all" else []) if matches(p.repo)]
        failed_shown = [p for p in (failed if incomplete else []) if matches(p.repo)]
        stopped_shown = [s for s in (abandoned if incomplete else []) if matches(s["repo"])]

        counts = {
            "all": len(disk) + len(running) + len(failed) + len(abandoned) + len(hub_shown),
            "resident": sum(1 for e in disk if e["resident"]),
            "disk": len(disk),
            "incomplete": len(failed) + len(abandoned),
            "hub": len(hub_shown),
        }

        tiles: list[ft.Control] = []
        for item in pulls_shown:
            tiles.append(
                self._tile(
                    item.repo,
                    owner(item.repo),
                    "—" if item.total is None else size(item.completed),
                    "starting" if item.total is None else f"of {size(item.total)}",
                    "Queued" if item.state == "pending" else item.share,
                    None,
                    item.fraction,
                    tag_color=t().fg2,
                    hatched=True,
                )
            )
        for item in failed_shown:
            tiles.append(
                self._tile(
                    item.repo,
                    owner(item.repo),
                    size(item.completed),
                    "on disk" if item.total is None else f"of {size(item.total)}",
                    "Failed",
                    None,
                    item.fraction,
                    tag_color=t().bad,
                    hatched=True,
                )
            )
        for stale in stopped_shown:
            tiles.append(
                self._tile(
                    stale["repo"],
                    owner(stale["repo"]),
                    size(stale["bytes_on_disk"]),
                    "on disk",
                    "Stopped",
                    None,
                    None,
                    tag_color=t().fg3,
                    hatched=True,
                )
            )
        for entry in shown:
            material = engine.materials.get(entry["id"])
            tiles.append(
                self._tile(
                    entry["id"],
                    owner(entry["id"]),
                    f"{gb(entry['bytes_on_disk'])} GB",
                    (entry["quantization"] or "dense")
                    + (f" · {tokens(entry['context'])}" if entry["context"] else ""),
                    "Resident" if entry["resident"] else "",
                    None if material is None else t().mat(material),
                    None,
                    on_disk_entry=True,
                )
            )
        for model in hub_shown:
            downloads = model["downloads"]
            tiles.append(
                self._tile(
                    model["id"],
                    f"{owner(model['id'])} · Hub",
                    "—" if downloads is None else f"{round(downloads / 1000)}k",
                    "on the Hub" if downloads is None else "downloads / month",
                    "Hub",
                    None,
                    None,
                    tag_color=t().fg3,
                )
            )

        blank = self.query.strip() == ""
        shelf: list[ft.Control] = [
            ft.Container(
                padding=ft.Padding.only(bottom=9),
                content=ft.Row(
                    [
                        parts.fchip(
                            f"{label} {counts[key]}",
                            key == self.filter,
                            lambda key=key: self._pick_filter(key),
                        )
                        for key, label in FILTERS
                    ],
                    spacing=5,
                ),
            )
        ]
        if tiles:
            shelf.append(parts.cards(tiles, columns=COLUMNS))
        else:
            shelf.append(
                parts.empty(
                    "Nothing on this machine yet" if blank else "No match",
                    "Search for a repository — anything on the Hub can be pulled straight "
                    "into the catalog."
                    if blank
                    else "Two characters search the Hub as well. Try the organisation, or "
                    "part of the name.",
                )
            )

        return parts.pane(
            [
                ft.Container(
                    height=38,
                    padding=ft.Padding.symmetric(horizontal=13),
                    border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            parts.search(
                                self.query, self._type, "Search this machine and the Hub"
                            ),
                            parts.iconbtn(_refresh(), self.notify, "Refresh"),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
                parts.body(shelf),
            ]
        )

    def _tile(
        self,
        identifier: str,
        org: str,
        measure: str,
        meta: str,
        tag: str,
        material: str | None,
        progress: float | None,
        tag_color: str | None = None,
        hatched: bool = False,
        on_disk_entry: bool = False,
    ) -> ft.Control:
        chosen = self.chosen
        on = chosen is not None and chosen.id == identifier and chosen.hub != on_disk_entry
        return parts.card(
            display_name(identifier),
            org,
            measure,
            meta,
            tag,
            material,
            on,
            lambda: self._choose(identifier, not on_disk_entry),
            progress,
            tag_color,
            hatched,
        )

    # ── the inspector ─────────────────────────────────────────────────────

    def _inspector(self, engine: Engine) -> ft.Control:
        chosen = self.chosen
        if chosen is None:
            return parts.pane(
                [
                    parts.empty(
                        "Nothing selected",
                        "Pick a checkpoint to read its card, price it against the ceiling, "
                        "and load it.",
                    )
                ],
                PANEL,
            )

        entry = None if chosen.hub else engine.entry(chosen.id)
        material = None if entry is None else engine.materials.get(entry["id"])
        flight = self.downloads.active.get(chosen.id)
        halted = next((s for s in self.staged if s["repo"] == chosen.id), None)
        stalled = flight is not None and flight.state == "error"
        fetching = flight is not None and not stalled

        rule = t().fg3
        if stalled:
            rule = t().bad
        elif material is not None:
            rule = t().mat(material)

        ceiling = engine.ceiling
        used = 0 if engine.state is None else (
            engine.state["resident_bytes"] + engine.state["kv_bytes"]
        )
        free = None if ceiling is None else ceiling - used
        too_big = self.price is not None and free is not None and self.price > free

        blocks: list[ft.Control] = []
        if self.failure is not None:
            blocks.append(
                ft.Container(content=parts.err(self.failure), margin=ft.Margin.only(bottom=9))
            )
        if stalled and flight is not None and flight.error is not None:
            blocks.append(
                ft.Container(
                    content=parts.err(
                        f"{flight.error} What was fetched is on disk; resuming continues "
                        "from there."
                    ),
                    margin=ft.Margin.only(bottom=9),
                )
            )
        if too_big:
            blocks.append(
                ft.Container(
                    content=parts.note(
                        f"{gb(self.price or 0)} GB needed, {gb(free or 0)} GB free below the "
                        "ceiling. Raise the ceiling in Settings, or unload something first.",
                        bad=True,
                    ),
                    margin=ft.Margin.only(bottom=9),
                )
            )
        # --m falls back to --mat1: a checkpoint with no material still colours its
        # card's links and its base chip, and teal is what the stylesheet defaults to.
        accent = rule if material is not None else t().mat(0)
        if entry is not None and self.tab == "card":
            blocks.extend(_facts(entry))
        if entry is not None and self.tab == "features":
            self.speculation.look(entry["id"])
            blocks.extend(self.speculation.draw())
        if self.tab == "profiles" and entry is not None:
            self.profiles.look(entry["id"])
            blocks.extend(self.profiles.draw())
        if self.tab not in ON_ENTRY:
            self.card.look(chosen.id, chosen.hub)
            tail = "to download" if entry is None else f"on disk · {entry['store']}"
            blocks.append(
                ft.Container(
                    padding=ft.Padding.only(top=11),
                    content=ft.Column(
                        self.card.draw(self.tab, accent, tail), spacing=0, tight=True
                    ),
                )
            )

        return parts.pane(
            [
                _insph(display_name(chosen.id), self._subtitle(chosen), rule),
                self._itabs(entry is not None),
                ft.Container(
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=13, vertical=11),
                    content=ft.Column(blocks, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
                ),
                self._pullbox(flight)
                if fetching and flight is not None
                else self._acts(chosen, entry, stalled, halted),
            ],
            PANEL,
        )

    def _subtitle(self, chosen: Chosen) -> str:
        return chosen.id + ("" if self.price is None else f" · {gb(self.price)} GB")

    def _itabs(self, has_entry: bool) -> ft.Control:
        shown = [(key, label) for key, label in TABS if key not in ON_ENTRY or has_entry]
        return ft.Container(
            padding=ft.Padding(left=9, right=9, top=7, bottom=0),
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=ft.Row(
                [self._itab(key, label) for key, label in shown], spacing=2, tight=True
            ),
        )

    def _itab(self, key: str, label: str) -> ft.Control:
        on = key == self.tab
        return ft.Container(
            padding=ft.Padding(left=9, right=9, top=4, bottom=7),
            border=ft.Border.only(
                bottom=ft.BorderSide(1.5, t().fg if on else theme.TRANSPARENT)
            ),
            content=ft.Text(
                label,
                style=theme.sans(11, t().fg if on else t().fg3, ft.FontWeight.W_500),
                no_wrap=True,
            ),
            on_click=lambda _, key=key: self._pick_tab(key),
        )

    def _acts(
        self,
        chosen: Chosen,
        entry: CatalogEntry | None,
        stalled: bool,
        halted: Staged | None,
    ) -> ft.Control:
        row: list[ft.Control] = []
        if stalled or halted is not None:
            bytes_here = halted["bytes_on_disk"] if halted is not None else 0
            flight = self.downloads.active.get(chosen.id)
            row += [
                parts.btn("Resume", lambda: self._start(chosen.id), "pri", not self.busy),
                ft.Container(expand=True),
                parts.btn(
                    f"Discard {size(flight.completed if flight else bytes_here)}",
                    lambda: self._discard(chosen.id),
                    "quiet danger",
                    not self.busy,
                ),
            ]
        elif entry is None:
            row.append(
                parts.btn(
                    "Download" + ("" if self.price is None else f" {gb(self.price)} GB"),
                    lambda: self._start(chosen.id),
                    "pri",
                    not self.busy,
                    expand=True,
                )
            )
        elif entry["resident"]:
            row += [
                parts.btn("Chat", lambda: self._jump("chat"), "pri"),
                parts.btn(
                    "Unload",
                    lambda: self._run(hub.unload_model(entry["id"])),
                    enabled=not self.busy,
                ),
            ]
        else:
            row += [
                parts.btn(
                    f"Load {gb(entry['bytes_on_disk'])} GB",
                    lambda: self._run(hub.load_model(entry["id"])),
                    "pri",
                    not self.busy,
                ),
                parts.btn("Quantize", lambda: self._open_quantize(entry["id"])),
            ]
        if entry is not None:
            row += [
                ft.Container(expand=True),
                parts.btn(
                    "Delete from disk" if self.confirming else "Remove",
                    lambda: self._remove(entry["id"]),
                    "danger" if self.confirming else "quiet danger",
                    not self.busy,
                ),
            ]
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=11, vertical=9),
            border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
            content=ft.Row(row, spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _pullbox(self, item: Pull) -> ft.Control:
        """Where the Download button was, doing the job the Download button no longer has.

        Cancelling takes the staging with it, so it asks twice — the same two steps as
        Remove, and for the same reason: the second press is the one that throws bytes
        away.
        """
        part = item.fraction or 0.0
        foot: list[ft.Control] = []
        if item.total is not None:
            foot.append(
                ft.Text(size_pair(item.completed, item.total), style=theme.mono(10, t().fg3))
            )
        if item.rate is not None:
            foot += [_sep(), ft.Text(rate(item.rate), style=theme.mono(10, t().fg3))]
        remaining = item.eta
        if remaining is not None:
            foot += [_sep(), ft.Text(left(remaining), style=theme.mono(10, t().fg3))]
        foot += [
            ft.Container(expand=True),
            parts.btn(
                f"Cancel and discard {size(item.completed)}" if self.cancelling else "Cancel",
                lambda: self._cancel(item),
                "danger" if self.cancelling else "quiet danger",
            ),
        ]
        return ft.Container(
            padding=ft.Padding(left=11, right=11, top=10, bottom=11),
            border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(
                                "Queued" if item.state == "pending" else "Downloading",
                                style=theme.sans(11.5, weight=ft.FontWeight.W_600),
                            ),
                            ft.Text(
                                item.message,
                                style=theme.sans(10.5, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                                expand=True,
                            ),
                            ft.Text(item.share, style=theme.mono(12.5)),
                        ],
                        spacing=9,
                        vertical_alignment=theme.BASELINE,
                    ),
                    ft.Container(
                        height=4,
                        border_radius=3,
                        bgcolor=t().sunken,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        content=ft.Row(
                            [
                                ft.Container(expand=max(1, round(part * 1000)), bgcolor=t().fg2),
                                ft.Container(
                                    expand=max(1, round((1 - part) * 1000)),
                                    gradient=theme.hatch(
                                        theme.mix(t().fg2, 0.42, theme.TRANSPARENT), 6, 300
                                    ),
                                ),
                            ],
                            spacing=0,
                            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                        ),
                    ),
                    ft.Row(foot, spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ],
                spacing=7,
                tight=True,
            ),
        )

    # ── what this screen can do ───────────────────────────────────────────

    def _type(self, value: str) -> None:
        self.query = value
        if self._search is not None:
            self._search.cancel()
        self._search = asyncio.get_running_loop().create_task(self._look(value))
        self.notify()

    async def _look(self, value: str) -> None:
        term = value.strip()
        if len(term) < MIN_TERM or self.filter in ("resident", "disk", "incomplete"):
            self.hub = []
            self.notify()
            return
        try:
            await asyncio.sleep(SETTLE)
            self.hub = await hub.search_hub(term)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self.hub = []
        self.notify()

    def _pick_filter(self, key: str) -> None:
        self.filter = key
        self.notify()
        self._type(self.query)

    def _pick_tab(self, key: str) -> None:
        self.tab = key
        self.notify()

    def _choose(self, identifier: str, is_hub: bool) -> None:
        self.chosen = Chosen(identifier, is_hub)
        self.tab = "files" if is_hub and identifier in self.downloads.active else "card"
        self.confirming = False
        self.cancelling = False
        self.failure = None
        self.price = None
        self.notify()

    async def refresh(self, engine: Engine) -> None:
        """Bytes on disk with no model to show for them, and what the selection costs.

        Both are read off the same beat as the catalog, because that is when they change:
        a download landing turns staging into a checkpoint, and a discard turns it into
        free space.
        """
        try:
            self.staged = await hub.list_staged()
        except Exception:  # noqa: BLE001
            self.staged = []
        await self._price(engine)

    async def _price(self, engine: Engine) -> None:
        chosen = self.chosen
        if chosen is None:
            self.price = None
            return
        entry = None if chosen.hub else engine.entry(chosen.id)
        if entry is not None:
            self.price = None if entry["resident"] else entry["bytes_on_disk"]
        # A repository not on disk is priced by the listing the card block fetches, which
        # is the same call and arrives through `_take_price`.

    def _start(self, repo: str) -> None:
        act(self._pull(repo))

    async def _pull(self, repo: str) -> None:
        try:
            self.downloads.follow(await hub.pull(repo))
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.notify()

    def _cancel(self, item: Pull) -> None:
        if not self.cancelling:
            self.cancelling = True
            self.notify()
            return
        self.cancelling = False
        act(self._run_bare(engine_api.cancel_job(item.job)))

    def _discard(self, repo: str) -> None:
        act(self._drop(repo))

    async def _drop(self, repo: str) -> None:
        try:
            await hub.drop_staged(repo)
            self.downloads.forget(repo)
            self.chosen = None
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.notify()

    def _remove(self, model: str) -> None:
        if not self.confirming:
            self.confirming = True
            self.notify()
            return
        self.confirming = False
        act(self._run_bare(hub.remove_model(model), clear=True))

    def _run(self, work: Coroutine[None, None, object]) -> None:
        act(self._run_bare(work))

    async def _run_bare(
        self, work: Coroutine[None, None, object], clear: bool = False
    ) -> None:
        self.busy = True
        self.failure = None
        self.notify()
        try:
            await work
            if clear:
                self.chosen = None
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        finally:
            self.busy = False
            self.notify()


def _facts(entry: CatalogEntry) -> list[ft.Control]:
    per_token = entry["bytes_per_token"]
    return [
        parts.group("Checkpoint", first=True),
        parts.kvr("Architecture", entry["architecture"]),
        parts.kvr(
            "Quantization",
            entry["quantization"] or "dense",
            None if entry["dtype"] is None else f"· {entry['dtype']}",
        ),
        parts.kvr(
            "Context",
            "—" if entry["context"] is None else f"{tokens(entry['context'])} tokens",
        ),
        parts.kvr(
            "Bytes per token", "—" if per_token is None else f"{per_token / 1e9:.2f} GB"
        ),
        parts.kvr("On disk", f"{gb(entry['bytes_on_disk'])} GB", last=True),
    ]


def _insph(name: str, subtitle: str, rule: str) -> ft.Control:
    return ft.Container(
        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Stack(
            [
                ft.Container(
                    left=0,
                    top=11,
                    width=3,
                    height=26,
                    bgcolor=rule,
                    border_radius=ft.BorderRadius(
                        top_left=0, top_right=3, bottom_left=0, bottom_right=3
                    ),
                ),
                ft.Container(
                    width=PANEL - 2,
                    padding=ft.Padding(left=13, right=13, top=11, bottom=9),
                    content=ft.Column(
                        [
                            ft.Text(
                                name,
                                style=theme.sans(13, weight=ft.FontWeight.W_600, height=1.25),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                subtitle,
                                style=theme.mono(9.5, t().fg3),
                                no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=3,
                        tight=True,
                    ),
                ),
            ]
        ),
    )


def _sep() -> ft.Control:
    return ft.Text("·", style=theme.mono(10, t().hair2))


def _refresh() -> list[cv.Shape]:
    paint = ft.Paint(
        color=t().fg2,
        stroke_width=1.8,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    # The 24-viewport arc and tick of the stylesheet's own icon, at 13.
    s = 13 / 24
    return [
        cv.Arc(1.5 * s, 1.5 * s, 21 * s, 21 * s, 0.6, 5.1, paint=paint),
        cv.Path(
            [
                cv.Path.MoveTo(21 * s, 3 * s),
                cv.Path.LineTo(21 * s, 9 * s),
                cv.Path.LineTo(15 * s, 9 * s),
            ],
            paint=paint,
        ),
    ]

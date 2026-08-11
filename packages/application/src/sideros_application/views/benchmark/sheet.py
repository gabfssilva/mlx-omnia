"""The New benchmark sheet: what a run is asked for."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import flet as ft
import flet.canvas as cv

from sideros_application.api import benchmarks as api
from sideros_application.api import catalog as hub
from sideros_application.api.benchmarks import (
    CONCURRENCIES,
    CONTEXTS,
    GATES,
    GENERATES,
    GREEDY,
    Dataset,
    Kind,
    Plan,
    Sampling,
    human,
)
from sideros_application.api.engine import CatalogEntry, Job
from sideros_application.ui import parts, theme
from sideros_application.ui.format import gb
from sideros_application.ui.forms import labelled
from sideros_application.ui.hooks import act
from sideros_application.ui.theme import t
from sideros_application.views.benchmark.shared import CORPUS, cost, group, keyline


def check(on: bool, on_click: Callable[[], None]) -> ft.Control:
    """A checkbox the house draws: Material's arrives with a ripple and a 48pt hit box."""
    tick = ft.Paint(
        color=theme.ON_ACCENT,
        stroke_width=1.8,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    return ft.Container(
        width=14,
        height=14,
        border_radius=4,
        bgcolor=t().mat(0) if on else t().field,
        border=None if on else theme.hair(t().hair2),
        alignment=ft.Alignment.CENTER,
        content=cv.Canvas(
            [
                cv.Path(
                    [cv.Path.MoveTo(3, 7), cv.Path.LineTo(6, 10), cv.Path.LineTo(11, 4)],
                    paint=tick,
                )
            ],
            width=14,
            height=14,
        )
        if on
        else None,
        on_click=lambda _: on_click(),
    )


def chips(
    values: list[int],
    chosen: list[int],
    label: Callable[[int], str],
    on_toggle: Callable[[int], None],
) -> ft.Control:
    return ft.Row(
        [
            ft.Container(
                # .chips .fchip: 34 minimum, and the pill's own 20pt of padding past
                # the label — which is what clips `128k` when it is counted as four
                # characters of prose.
                width=max(34, parts.advance(label(value)) + 22),
                content=parts.fchip(
                    label(value), value in chosen, lambda value=value: on_toggle(value)
                ),
            )
            for value in values
        ],
        spacing=5,
        wrap=True,
        run_spacing=5,
    )


def toggle(values: list[int], value: int) -> list[int]:
    return sorted(v for v in values if v != value) if value in values else sorted(
        [*values, value]
    )


@dataclass
class Sheet:
    """What to measure, on which checkpoints, and what it will cost.

    Nothing scrolls to decide: everything chosen is on screen when it opens, and the one
    list that scrolls is the model library, because it is the only one that grows without
    bound. The cost line comes from `POST /admin/benchmarks/plan` and is never recomputed
    here — two copies of that arithmetic is two answers to "does this fit".
    """

    kind: Kind
    redraw: Callable[[], None]
    close: Callable[[], None]
    started: Callable[[Job, dict[str, object]], None]
    declare: Callable[[], None]
    view: Kind = "speed"
    models: list[str] = field(default_factory=list)
    contexts: list[int] = field(default_factory=lambda: [4096])
    generates: list[int] = field(default_factory=lambda: [256])
    concurrencies: list[int] = field(default_factory=lambda: [1])
    rounds: int = 3
    sampling: Sampling = GREEDY
    profiles: list[str] = field(default_factory=list)
    profile: str = ""
    page: str = "warm"
    gate: int | None = 45
    skip: bool = True
    pairs: dict[str, str] = field(default_factory=dict)
    chosen_datasets: list[str] = field(default_factory=lambda: ["mmlu"])
    items: int = 1400
    seed: int = 42
    shots: int = 5
    priced: Plan | None = None
    failure: str | None = None
    busy: bool = False
    _price: asyncio.Task[None] | None = None
    _anchor: str | None = None

    def __post_init__(self) -> None:
        self.view = self.kind
        self._reprice()

    # what the request is

    def body(self) -> dict[str, object] | None:
        if self.view == "speed":
            if not self.models:
                return None
            return {
                "kind": "speed",
                "models": self.models,
                "contexts": self.contexts,
                "generates": self.generates,
                "concurrencies": self.concurrencies,
                "rounds": self.rounds,
                "sampling": self.sampling.wire(),
                "page_cache": self.page,
                "thermal_gate_c": self.gate,
                "skip_if_measured": self.skip,
            }
        if self.view == "quality":
            if not self.models or not self.chosen_datasets:
                return None
            return {
                "kind": "quality",
                "models": self.models,
                "datasets": self.chosen_datasets,
                "items": self.items,
                "seed": self.seed,
                "shots": self.shots,
                "skip_if_measured": self.skip,
            }
        declared = [(model, ref) for model, ref in self.pairs.items() if ref != ""]
        if not declared:
            return None
        return {
            "kind": "fidelity",
            "pairs": [{"model": model, "reference": ref} for model, ref in declared],
            "corpus": CORPUS,
            "tokens": 10000,
            "seed": self.seed,
            "skip_if_measured": self.skip,
        }

    def _reprice(self) -> None:
        if self._price is not None:
            self._price.cancel()
        self._price = asyncio.get_running_loop().create_task(self._ask())

    async def _ask(self) -> None:
        request = self.body()
        if request is None:
            self.priced = None
            self.redraw()
            return
        try:
            self.priced = await api.price(request)
            self.failure = None
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            self.priced = None
            self.failure = str(error)
        self.redraw()

    def _set(self, **changes: object) -> None:
        for name, value in changes.items():
            setattr(self, name, value)
        self._reprice()
        self.redraw()

    def _toggle_model(self, identifier: str) -> None:
        self.models = (
            [one for one in self.models if one != identifier]
            if identifier in self.models
            else [*self.models, identifier]
        )
        # The profiles of the first checkpoint chosen. A profile is per model, and the key
        # may not name one — so what is taken from it is the numbers, and those go in the
        # key. Two models under the same numbers stay on the same axis.
        anchor = self.models[0] if self.models else None
        if anchor != self._anchor:
            self._anchor = anchor
            act(self._read_profiles(anchor))
        self._reprice()
        self.redraw()

    async def _read_profiles(self, anchor: str | None) -> None:
        if anchor is None:
            self.profiles = []
        else:
            try:
                self.profiles = await hub.profile_names(anchor)
            except Exception:  # noqa: BLE001
                self.profiles = []
        self.redraw()

    def _fill(self, name: str) -> None:
        self.profile = name
        if name == "" or self._anchor is None:
            self._set(sampling=GREEDY)
            return
        act(self._read_profile(self._anchor, name))

    async def _read_profile(self, model: str, name: str) -> None:
        try:
            found = await hub.get_profile(model, name)
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
            self.redraw()
            return
        knobs = found["sampling"]
        # A knob the profile leaves unset is one it does not opine on, and what stands
        # under it here is this section's own neutral value — not the dialect's, which
        # would make an unset temperature 1.0 and turn "fill from profile" into a drawn
        # run nobody asked for.
        self._set(
            sampling=Sampling(
                temperature=knobs["temperature"] or 0.0,
                top_p=knobs["top_p"] if knobs["top_p"] is not None else 1.0,
                top_k=knobs["top_k"],
                min_p=knobs["min_p"] or 0.0,
                repetition_penalty=knobs["repetition_penalty"] or 1.0,
                seed=knobs["seed"],
            )
        )

    def knob(self, name: str, raw: str) -> None:
        try:
            parsed = None if raw.strip() == "" else float(raw)
        except ValueError:
            return
        neutral = {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}
        value = (
            None
            if name == "top_k" and parsed is None
            else int(parsed)
            if name == "top_k" and parsed is not None
            else (parsed if parsed is not None else neutral[name])
        )
        self.profile = ""
        self._set(sampling=replace(self.sampling, **{name: value}))

    def _run(self) -> None:
        request = self.body()
        if request is None:
            return
        act(self._launch(request))

    async def _launch(self, request: dict[str, object]) -> None:
        self.busy = True
        self.redraw()
        try:
            job = await api.start(request)
            self.started(job, {**request, "_sampling": self.sampling})
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        finally:
            self.busy = False
            self.redraw()

    # drawing

    def _references_for(
        self, entry: CatalogEntry, catalog: list[CatalogEntry]
    ) -> list[CatalogEntry]:
        """Which checkpoints can be a reference for this one. The vocabulary decides — a
        different one has no vector to subtract — and a different architecture is marked
        and allowed."""
        return [
            other
            for other in catalog
            if other["id"] != entry["id"]
            and (
                other["vocab_size"] is None
                or entry["vocab_size"] is None
                or other["vocab_size"] == entry["vocab_size"]
            )
        ]

    def tree(self, catalog: list[CatalogEntry], datasets: list[Dataset]) -> ft.Control:
        shapes = 0 if self.priced is None else len(self.priced["shapes"])
        estimate = (
            "—"
            if self.priced is None or self.priced["estimate_seconds"] is None
            else f"≈ {max(1, round(self.priced['estimate_seconds'] / 60))} min"
        )
        return parts.sheet(
            [
                ft.Text("New benchmark", style=theme.sans(13.5, weight=ft.FontWeight.W_600),
                        no_wrap=True),
                ft.Text(
                    "the key is what lets two checkpoints share an axis",
                    style=theme.mono(11, t().fg3),
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                ),
                parts.btn("Close", self.close, "quiet"),
            ],
            ft.Column(
                [
                    ft.Container(
                        height=540,
                        content=ft.Row(
                            [
                                ft.Container(
                                    width=214,
                                    padding=ft.Padding.symmetric(horizontal=15, vertical=13),
                                    border=ft.Border.only(
                                        right=ft.BorderSide(1, t().hair)
                                    ),
                                    content=ft.Column(
                                        self._left(), spacing=11,
                                        scroll=ft.ScrollMode.AUTO, expand=True,
                                    ),
                                ),
                                ft.Container(
                                    expand=True,
                                    padding=ft.Padding.symmetric(horizontal=15, vertical=13),
                                    content=ft.Column(
                                        self._right(catalog, datasets, shapes, estimate),
                                        spacing=0,
                                        scroll=ft.ScrollMode.AUTO,
                                        expand=True,
                                    ),
                                ),
                            ],
                            spacing=0,
                            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                        ),
                    ),
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=15, vertical=10),
                        border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                        content=ft.Row(
                            [
                                parts.btn(
                                    f"Run {shapes} shape{'' if shapes == 1 else 's'}",
                                    self._run,
                                    "pri",
                                    not self.busy and shapes > 0,
                                ),
                                parts.btn("Cancel", self.close, "quiet"),
                            ],
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                ],
                spacing=0,
                tight=True,
            ),
            1080,
        )

    def _left(self) -> list[ft.Control]:
        drawn: list[ft.Control] = [
            labelled(
                "Measure",
                ft.Column(
                    [
                        stack_option(label, key == self.view, lambda key=key: self._set(view=key))
                        for key, label in api.KINDS
                    ],
                    spacing=2,
                    tight=True,
                ),
            )
        ]
        if self.view == "speed":
            drawn.append(
                labelled(
                    "Sampling",
                    parts.pick(
                        [("", "Greedy — argmax"),
                         *((name, f"from profile {name}") for name in self.profiles)],
                        self.profile,
                        self._fill,
                    ),
                )
            )
            drawn.append(
                two(
                    [
                        labelled("Temperature", parts.field(
                            plain(self.sampling.temperature),
                            lambda v: self._knob("temperature", v), mono=True)),
                        labelled("Top-p", parts.field(
                            plain(self.sampling.top_p),
                            lambda v: self._knob("top_p", v), mono=True)),
                        labelled("Top-k", parts.field(
                            "" if self.sampling.top_k is None else str(self.sampling.top_k),
                            lambda v: self._knob("top_k", v), mono=True, hint="off")),
                        labelled("Repetition", parts.field(
                            plain(self.sampling.repetition_penalty),
                            lambda v: self._knob("repetition_penalty", v), mono=True)),
                    ]
                )
            )
            # The sampler is in the key: a drawn token pays for filters an argmax does not.
            drawn.append(
                parts.note(
                    "Argmax. The four measurements — load, prefill, TTFT, decode — are taken "
                    "on every shape."
                    if self.sampling.temperature == 0
                    else "Drawn. The filters run per step and the key says so, so this does "
                    "not compare with a greedy run."
                )
            )
        if self.failure is not None:
            drawn.append(parts.note(self.failure, bad=True))
        if self.priced is not None and self.priced["skipped"]:
            first = self.priced["skipped"][0]
            count = len(self.priced["skipped"])
            drawn.append(
                parts.note(
                    f"{count} shape{'' if count == 1 else 's'} skipped — "
                    + (
                        f"the largest needed {gb(first['needed_bytes'], 0)} GB"
                        if first["reason"] == "kv_over_budget"
                        and first["needed_bytes"] is not None
                        else first["reason"].replace("_", " ")
                    )
                    + "."
                )
            )
        return drawn

    def _right(
        self, catalog: list[CatalogEntry], datasets: list[Dataset], shapes: int, estimate: str
    ) -> list[ft.Control]:
        if self.view == "speed":
            right = [
                labelled("Context", chips(CONTEXTS, self.contexts, human,
                                           lambda v: self._set(contexts=toggle(self.contexts, v))),
                         stretch=False),
                ft.Container(height=10),
                labelled(
                    "Generate",
                    chips(
                        GENERATES,
                        self.generates,
                        str,
                        lambda v: self._set(generates=toggle(self.generates, v)),
                    ),
                    stretch=False,
                ),
                ft.Container(height=10),
                labelled("Concurrency", chips(CONCURRENCIES, self.concurrencies, str,
                                               lambda v: self._set(
                                                   concurrencies=toggle(self.concurrencies, v))),
                         stretch=False),
                ft.Container(height=12),
                two(
                    [
                        labelled("Rounds", parts.field(
                            str(self.rounds),
                            lambda v: self._set(rounds=int(v) if v.strip().isdigit() else 1),
                            mono=True)),
                        labelled("Page cache on load", parts.pick(
                            [("warm", "Warm — as in use"), ("cold", "Cold — purge first")],
                            self.page, lambda v: self._set(page=v))),
                        labelled("Start below", parts.pick(
                            [("none", "No gate"), *((str(v), f"{v} °C") for v in GATES if v)],
                            "none" if self.gate is None else str(self.gate),
                            lambda v: self._set(gate=None if v == "none" else int(v)))),
                    ]
                ),
                ft.Container(height=13),
                self._skip(),
                ft.Container(height=11),
                keyline(
                    api.speed_key(
                        self.contexts[0] if self.contexts else 4096,
                        self.generates[0] if self.generates else 256,
                        self.concurrencies[0] if self.concurrencies else 1,
                        self.rounds,
                        self.sampling,
                        self.page,
                    )
                ),
                ft.Container(height=8),
                cost("Will run", str(shapes), "shapes", estimate),
                ft.Container(height=9),
                parts.note(
                    "Pick at least one checkpoint."
                    if self.priced is None
                    else f"{len(self.priced['already'])} already answered under this key."
                ),
            ]
            return [two_columns(models_list(catalog, self.models, self._toggle_model), right)]

        if self.view == "quality":
            right = [
                group("Datasets", first=True),
                *[
                    self._dataset_row(entry)
                    for entry in datasets
                    if entry["use"] != "corpus"
                ],
                ft.Container(
                    padding=ft.Padding.only(top=9),
                    content=parts.btn("Add from the Hub", self.declare),
                ),
                ft.Container(height=12),
                two(
                    [
                        labelled("Items", parts.field(
                            str(self.items),
                            lambda v: self._set(items=int(v) if v.strip().isdigit() else 1),
                            mono=True)),
                        labelled("Seed", parts.field(
                            str(self.seed),
                            lambda v: self._set(seed=int(v) if v.strip().isdigit() else 0),
                            mono=True)),
                        labelled("Shots", parts.field(
                            str(self.shots),
                            lambda v: self._set(shots=int(v) if v.strip().isdigit() else 0),
                            mono=True)),
                    ]
                ),
                ft.Container(height=13),
                self._skip(),
                ft.Container(height=11),
                keyline(f"n{self.items} · s{self.seed} · {self.shots}-shot"),
                ft.Container(height=8),
                cost("Will run", str(shapes), "runs", estimate),
            ]
            return [two_columns(models_list(catalog, self.models, self._toggle_model), right)]

        pairable = [
            entry for entry in catalog if self._references_for(entry, catalog)
        ]
        orphans = [entry for entry in catalog if not self._references_for(entry, catalog)]
        left: list[ft.Control] = [
            group("Model and its reference", first=True),
            ft.Container(
                height=420,
                content=ft.Column(
                    [self._pair_row(entry, catalog) for entry in pairable],
                    spacing=0,
                    scroll=ft.ScrollMode.AUTO,
                ),
            ),
        ]
        if orphans:
            left.append(
                ft.Container(
                    padding=ft.Padding.only(top=9),
                    content=parts.note(
                        "No reference shares a vocabulary with "
                        + ", ".join(entry["id"] for entry in orphans)
                        + "."
                    ),
                )
            )
        right = [
            two(
                [
                    labelled("Corpus", parts.pick(
                        [(CORPUS, "wikitext-103 · test")], CORPUS, lambda _: None)),
                    labelled("Tokens", parts.field("10000", lambda _: None, mono=True)),
                    labelled("Seed", parts.field(
                        str(self.seed),
                        lambda v: self._set(seed=int(v) if v.strip().isdigit() else 0),
                        mono=True)),
                    labelled("Reference logits", parts.pick(
                        [("cache", "Cache top-64")], "cache", lambda _: None)),
                ]
            ),
            ft.Container(height=13),
            self._skip(),
            ft.Container(height=11),
            keyline(f"{CORPUS} · n10000 · s{self.seed} · ref per model"),
            ft.Container(height=8),
            cost("Will run", str(shapes), "pairs", estimate),
            ft.Container(height=9),
            parts.note(
                "Each reference is read once and its top-64 cached, so two candidates "
                "sharing a reference pay for it once. A model whose vocabulary nobody else "
                "shares has no pair to make."
            ),
        ]
        return [two_columns(left, right)]

    def _skip(self) -> ft.Control:
        return ft.Row(
            [
                check(self.skip, lambda: self._set(skip=not self.skip)),
                ft.Text("Skip if already run",
                        style=theme.sans(11.5, t().fg, ft.FontWeight.W_500), no_wrap=True),
            ],
            spacing=8,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _dataset_row(self, entry: Dataset) -> ft.Control:
        on = entry["id"] in self.chosen_datasets
        name: list[ft.Control] = [
            ft.Text(entry["id"], style=theme.sans(11, weight=ft.FontWeight.W_500), no_wrap=True)
        ]
        if not entry["builtin"]:
            name.append(theme.eyebrow("custom"))
        return ft.Container(
            padding=ft.Padding(left=4, right=4, top=6, bottom=6),
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=ft.Row(
                [
                    check(
                        on,
                        lambda key=entry["id"]: self._set(
                            chosen_datasets=[c for c in self.chosen_datasets if c != key]
                            if key in self.chosen_datasets
                            else [*self.chosen_datasets, key]
                        ),
                    ),
                    ft.Column(
                        [
                            ft.Row(name, spacing=7, tight=True),
                            ft.Text(entry["repo"], style=theme.mono(9, t().fg3), no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                        ],
                        spacing=1,
                        tight=True,
                        expand=True,
                    ),
                    ft.Container(
                        width=92,
                        content=ft.Text(entry["use"].replace("_", " "),
                                        style=theme.mono(9.5, t().fg3), no_wrap=True),
                    ),
                    ft.Container(
                        width=58,
                        content=ft.Text(
                            str(entry["size"] or "—"),
                            style=theme.mono(10, t().fg2),
                            text_align=ft.TextAlign.RIGHT,
                            no_wrap=True,
                        ),
                    ),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _pair_row(self, entry: CatalogEntry, catalog: list[CatalogEntry]) -> ft.Control:
        offered = self._references_for(entry, catalog)
        chosen = self.pairs.get(entry["id"], "")
        return ft.Container(
            padding=ft.Padding(left=4, right=4, top=5, bottom=5),
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
            content=ft.Row(
                [
                    check(
                        chosen != "",
                        lambda key=entry["id"], first=offered[0]["id"]: self._set(
                            pairs={
                                **self.pairs,
                                key: "" if self.pairs.get(key, "") != "" else first,
                            }
                        ),
                    ),
                    ft.Column(
                        [
                            ft.Text(entry["id"],
                                    style=theme.sans(11, weight=ft.FontWeight.W_500),
                                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                            ft.Text(entry["shape"] or "—", style=theme.mono(9, t().fg3),
                                    no_wrap=True),
                        ],
                        spacing=1,
                        tight=True,
                        expand=True,
                    ),
                    parts.pick(
                        [
                            ("", "no reference"),
                            *(
                                (
                                    other["id"],
                                    other["id"]
                                    + (
                                        " · other shape"
                                        if other["shape"] != entry["shape"]
                                        else ""
                                    ),
                                )
                                for other in offered
                            ),
                        ],
                        chosen,
                        lambda value, key=entry["id"]: self._set(
                            pairs={**self.pairs, key: value}
                        ),
                        width=150,
                    ),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )


def plain(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def stack_option(label: str, on: bool, on_click: Callable[[], None]) -> ft.Control:
    """The Measure switch, which is the segmented control stood on end."""
    return ft.Container(
        padding=ft.Padding.symmetric(horizontal=9, vertical=4),
        bgcolor=t().surface2 if on else t().sunken,
        border_radius=5,
        content=ft.Text(
            label,
            style=theme.sans(11, t().fg if on else t().fg2,
                             ft.FontWeight.W_500 if on else ft.FontWeight.W_400),
            no_wrap=True,
        ),
        on_click=lambda _: on_click(),
    )


def two(cells: list[ft.Control]) -> ft.Control:
    """.cols — two equal columns, filled row by row."""
    rows: list[ft.Control] = []
    for start in range(0, len(cells), 2):
        pair = cells[start : start + 2]
        pair += [ft.Container() for _ in range(2 - len(pair))]
        rows.append(
            ft.Row(
                [ft.Container(content=cell, expand=True) for cell in pair],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
    return ft.Column(rows, spacing=9, tight=True)


def two_columns(left: list[ft.Control], right: list[ft.Control]) -> ft.Control:
    return ft.Row(
        [
            ft.Column(left, spacing=0, tight=True, expand=True,
                      horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            ft.Column(right, spacing=0, tight=True, expand=True,
                      horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
        ],
        spacing=16,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def models_list(
    catalog: list[CatalogEntry], chosen: list[str], on_toggle: Callable[[str], None]
) -> list[ft.Control]:
    """The one list that scrolls, because it is the only one that grows without bound."""
    return [
        group("Models", first=True),
        ft.Container(
            height=420,
            content=ft.Column(
                [
                    ft.Container(
                        padding=ft.Padding(left=4, right=4, top=5, bottom=5),
                        border=ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                        content=ft.Row(
                            [
                                check(
                                    entry["id"] in chosen,
                                    lambda key=entry["id"]: on_toggle(key),
                                ),
                                ft.Column(
                                    [
                                        ft.Text(entry["id"],
                                                style=theme.sans(11, weight=ft.FontWeight.W_500),
                                                no_wrap=True,
                                                overflow=ft.TextOverflow.ELLIPSIS),
                                        ft.Text(entry["quantization"] or "dense",
                                                style=theme.mono(9, t().fg3), no_wrap=True),
                                    ],
                                    spacing=1,
                                    tight=True,
                                    expand=True,
                                ),
                                ft.Container(
                                    width=62,
                                    content=ft.Text(
                                        f"{gb(entry['bytes_on_disk'])} GB",
                                        style=theme.mono(10, t().fg2),
                                        text_align=ft.TextAlign.RIGHT,
                                        no_wrap=True,
                                    ),
                                ),
                            ],
                            spacing=9,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    )
                    for entry in catalog
                ],
                spacing=0,
                scroll=ft.ScrollMode.AUTO,
            ),
        ),
    ]

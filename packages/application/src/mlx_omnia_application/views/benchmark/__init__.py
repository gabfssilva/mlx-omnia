"""The Benchmark tab: three views that never share a chart.

Speed, Fidelity and Quality answer questions that are not traded against each other in a
single run, and concurrency only exists on one of the three — so the band follows the view
switch and the knobs in its header follow with it. What is shared is the comparison: the
set of checkpoints drawn, which is `auto ∪ pinned − dropped` and survives a change of knobs.
"""

from __future__ import annotations

import asyncio
import json
import webbrowser
from collections.abc import Callable

import flet as ft

from mlx_omnia_application.api import benchmarks as api
from mlx_omnia_application.api import engine as engine_api
from mlx_omnia_application.api.benchmarks import (
    CONCURRENCIES,
    CONTEXTS,
    GENERATES,
    Dataset,
    Kind,
    Run,
    human,
)
from mlx_omnia_application.api.engine import Engine, Job
from mlx_omnia_application.ui import parts, theme
from mlx_omnia_application.ui.band import Line, autofill, band
from mlx_omnia_application.ui.format import gb
from mlx_omnia_application.ui.hooks import act, sheets
from mlx_omnia_application.ui.theme import t
from mlx_omnia_application.views.benchmark.axes import (
    TITLES,
    fidelity_axes,
    fidelity_values,
    quality_axes,
    quality_values,
    speed_axes,
    speed_values,
    split,
)
from mlx_omnia_application.views.benchmark.dataset import DatasetForm
from mlx_omnia_application.views.benchmark.rows import (
    INSPECTOR,
    TRACK,
    bars,
    curve,
    fidelity_result,
    finished,
    fix,
    insph,
    itabs,
    kv,
    nb,
    qhead,
    run_row,
    setup,
)
from mlx_omnia_application.views.benchmark.shared import CORPUS, cost, group, keyline, knob
from mlx_omnia_application.views.benchmark.sheet import Sheet

GIB = 1024**3


@ft.component
def Benchmark(engine: Engine) -> ft.Control:
    bench, _ = ft.use_state(Bench)
    # Escape closes whichever of the two sheets is on top.
    ft.use_effect(lambda: sheets(lambda: bench.dismiss), [])
    return bench.tree(engine)


@ft.observable
class Bench:
    """What this screen holds: which key is being read, which runs are on the band, and
    the two sheets that can be over it."""

    def __init__(self) -> None:
        self.view: Kind = "speed"
        self.context = 4096
        self.generate = 256
        self.concurrency = 1
        # The rest of the key — rounds, sampler, page cache, stream source. Held as the
        # whole key rather than as its parts because it is only ever chosen from keys that
        # exist.
        self.variant: str | None = None
        self.reference: str | None = None
        self.all: list[Run] = []
        self.scoped: list[Run] = []
        self.selected: str | None = None
        self.dataset: str | None = None
        self.pinned: list[str] = []
        self.dropped: list[str] = []
        self.query = ""
        self.tab = "result"
        self.definitions: list[Dataset] = []
        self.job: Job | None = None
        self.failure: str | None = None
        self.confirming = False
        self.sheet: Sheet | None = None
        self.form: DatasetForm | None = None
        self._booted = False
        self._loading: str | None = None
        self._stream: asyncio.Task[None] | None = None

    # ── the machine's side ────────────────────────────────────────────────

    def boot(self) -> None:
        if self._booted:
            return
        self._booted = True
        act(self._read_datasets())
        self.reload()

    async def _read_datasets(self) -> None:
        try:
            self.definitions = await api.datasets()
        except Exception:  # noqa: BLE001 — a daemon with no datasets declares none
            self.definitions = []
        self.notify()

    def reload(self) -> None:
        act(self._fetch(self.view))

    async def _fetch(self, kind: Kind) -> None:
        try:
            self.all = await api.runs(kind)
            self.failure = None
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        # The key is read after the whole list lands, because which variants exist is read
        # off it: scoping to a key guessed before the first fetch is scoping to `r3+1` on a
        # machine that only ever measured `r8+2`, and the runs read as missing.
        try:
            self.scoped = await api.runs(kind, self.key) if kind == "speed" else []
        except Exception:  # noqa: BLE001
            self.scoped = []
        self.notify()

    def _follow(self, job: Job) -> None:
        """The batch reports through the SSE every job already has. One subscription per
        job — reopening it on every frame is a connection storm on the daemon that is
        measuring, which is the one process that must be left alone."""
        self.job = job
        if self._stream is not None:
            self._stream.cancel()
        self._stream = asyncio.get_running_loop().create_task(self._watch(job["id"]))

    async def _watch(self, identifier: str) -> None:
        done = -1
        try:
            async for frame in engine_api.job_events(identifier):
                self.job = frame
                # The list follows the batch: one refetch per shape it finishes, and one
                # when it ends. Rows are refetched and never patched.
                if frame["progress"]["completed"] != done:
                    done = frame["progress"]["completed"]
                    self.reload()
                self.notify()
        except Exception:  # noqa: BLE001 — the stream closing with the job is not a failure
            pass
        self.reload()

    # ── what the knobs are worth ──────────────────────────────────────────

    @property
    def prefix(self) -> str:
        return api.speed_prefix(self.context, self.generate, self.concurrency)

    @property
    def variants(self) -> list[str]:
        """Every spelling of the rest of the key that has actually been measured under
        these three knobs, newest first. Assuming one instead — the defaults of the day —
        is what made a benchmark run under `r8+2` and land in a list scoped to `r3+1`,
        which reads from the outside as the run having disappeared."""
        if self.view != "speed":
            return []
        newest: dict[str, float] = {}
        for run in self.all:
            if not run["key"].startswith(self.prefix):
                continue
            newest[run["key"]] = max(newest.get(run["key"], 0.0), run["created_at"])
        return [key for key, _ in sorted(newest.items(), key=lambda pair: -pair[1])]

    @property
    def key(self) -> str | None:
        if self.view != "speed":
            return None
        if self.variant is not None and self.variant.startswith(self.prefix):
            return self.variant
        found = self.variants
        return found[0] if found else api.speed_key(
            self.context, self.generate, self.concurrency, 3
        )

    def _set(self, **changes: object) -> None:
        for name, value in changes.items():
            setattr(self, name, value)
        self.reload()
        self.notify()

    def _pick(self, model: str, dataset: str | None = None) -> None:
        self.selected = model
        if dataset is not None:
            self.dataset = dataset
        if model not in self._drawn:
            self.pinned = [*dict.fromkeys([*self.pinned, model])]
            self.dropped = [one for one in self.dropped if one != model]
        self.notify()

    def _drop(self, model: str) -> None:
        self.dropped = [*dict.fromkeys([*self.dropped, model])]
        self.pinned = [one for one in self.pinned if one != model]
        self.notify()

    def _forget(self) -> None:
        run = self._chosen
        if run is None:
            return
        if not self.confirming:
            self.confirming = True
            self.notify()
            return
        self.confirming = False
        act(self._delete(run["id"]))

    async def _delete(self, identifier: str) -> None:
        try:
            await api.forget(identifier)
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.reload()

    def _open_sheet(self) -> None:
        self.sheet = Sheet(self.view, self.notify, self._shut, self._started,
                           self._declare)
        self.notify()

    def _shut(self) -> None:
        self.sheet = None
        self.notify()

    def _declare(self) -> None:
        self.form = DatasetForm(self.notify, self._close_form, self._read_datasets)
        self.notify()

    def _close_form(self) -> None:
        self.form = None
        self.notify()

    @property
    def dismiss(self) -> Callable[[], None] | None:
        """Innermost first: the dataset form opens over the sheet."""
        if self.form is not None:
            return self._close_form
        return self._shut if self.sheet is not None else None

    def _started(self, job: Job, request: dict[str, object]) -> None:
        self._follow(job)
        self.view = request["kind"]  # type: ignore[assignment]
        # The view moves to what was just asked for. A batch launched at 8k landing in a
        # list scoped to 4k is a run that, from here, did not happen.
        if request["kind"] == "speed":
            contexts = request["contexts"]  # type: ignore[index]
            generates = request["generates"]  # type: ignore[index]
            streams = request["concurrencies"]  # type: ignore[index]
            self.context = contexts[0]  # type: ignore[index]
            self.generate = generates[0]  # type: ignore[index]
            self.concurrency = streams[0]  # type: ignore[index]
            self.variant = api.speed_key(
                self.context,
                self.generate,
                self.concurrency,
                int(request["rounds"]),  # type: ignore[arg-type]
                request["_sampling"],  # type: ignore[arg-type]
                str(request["page_cache"]),
            )
        self.sheet = None
        self.reload()
        self.notify()

    # ── drawing ───────────────────────────────────────────────────────────

    def tree(self, engine: Engine) -> ft.Control:
        self.boot()
        self._catalog = engine.catalog
        if self.reference is None and self._catalog:
            self.reference = self._catalog[0]["id"]
        self._fidelity = split(
            self._catalog,
            self.reference or "",
            [
                run
                for run in self.all
                if run["fidelity"] is not None
                and run["fidelity"]["reference"] == self.reference
            ],
        )
        work = ft.Column(
            [
                ft.Container(
                    margin=ft.Margin.only(left=12, right=12, top=12),
                    padding=ft.Padding(left=11, right=11, top=9, bottom=11),
                    bgcolor=t().surface,
                    border=theme.hair(),
                    border_radius=10,
                    content=ft.Column(
                        [self._header(), self._band(), self._legend()], spacing=0, tight=True
                    ),
                ),
                ft.Container(
                    expand=True,
                    padding=12,
                    content=ft.Row(
                        [self._list(), self._inspector()],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                ),
            ],
            spacing=0,
            expand=True,
        )
        over: list[ft.Control] = [work]
        if self.sheet is not None:
            over.append(self.sheet.tree(self._catalog, self.definitions))
        if self.form is not None:
            over.append(self.form.tree())
        return ft.Container(
            expand=True,
            bgcolor=t().win,
            content=ft.Stack(over, expand=True) if len(over) > 1 else work,
        )

    # what is on the band, and under it

    @property
    def _for_view(self) -> list[Run]:
        return self.scoped if self.view == "speed" else self.all

    @property
    def _models(self) -> list[str]:
        if self.view == "fidelity":
            return [one.entry["id"] for one in self._fidelity[0]]
        return list(dict.fromkeys(run["model"] for run in self._for_view))

    def _score(self, model: str) -> float | None:
        run = next((one for one in self._for_view if one["model"] == model), None)
        if run is None:
            return None
        if self.view == "speed":
            return None if run["speed"] is None else run["speed"]["decode_tps"]
        if self.view == "fidelity":
            found = None if run["fidelity"] is None else run["fidelity"]["kl_mean"]
            return None if found is None else -found
        return None if run["quality"] is None else run["quality"]["accuracy"]

    @property
    def _drawn(self) -> list[str]:
        union = list(dict.fromkeys([*autofill(self._models, self._score), *self.pinned]))
        return [model for model in union if model not in self.dropped][:6]

    @property
    def _materials(self) -> dict[str, int]:
        return {model: index for index, model in enumerate(self._drawn)}

    @property
    def _chosen(self) -> Run | None:
        if self.selected is None:
            return None
        if self.view == "quality":
            return next(
                (
                    run
                    for run in self.all
                    if run["model"] == self.selected
                    and run["quality"] is not None
                    and run["quality"]["dataset"] == self.dataset
                ),
                next((run for run in self.all if run["model"] == self.selected), None),
            )
        return next((run for run in self._for_view if run["model"] == self.selected), None)

    def _header(self) -> ft.Control:
        running = not finished(self.job)
        row: list[ft.Control] = [
            ft.Container(
                # A floor wide enough for the longest of the three keeps the switch at one
                # x instead of sliding 5px on every change of view.
                width=44,
                content=ft.Text(
                    TITLES[self.view],
                    style=theme.sans(11, weight=ft.FontWeight.W_600),
                    no_wrap=True,
                ),
            ),
            ft.Container(
                margin=ft.Margin.only(left=4),
                content=parts.seg(
                    api.KINDS, self.view, lambda value: self._set(view=value, selected=None)
                ),
            ),
        ]
        # The stylesheet hangs the knobs off `margin-left: auto`, which collapses the
        # moment the row is full — and with four speed knobs it always is. What actually
        # takes the slack on both sides is the seed, so it is the only thing that flexes.
        if self.view == "fidelity":
            row += [
                knob(
                    "Reference",
                    parts.pick(
                        [(entry["id"], entry["id"]) for entry in self._catalog],
                        self.reference or "",
                        lambda value: self._set(reference=value),
                        width=210,
                    ),
                ),
                knob(
                    "Corpus",
                    parts.pick([(CORPUS, f"{CORPUS} · test")], CORPUS, lambda _: None, 150),
                ),
            ]
        if self.view == "speed":
            row += [
                knob(
                    "Context",
                    parts.pick(
                        [(str(one), human(one)) for one in CONTEXTS],
                        str(self.context),
                        lambda value: self._set(context=int(value)),
                        width=76,
                    ),
                ),
                knob(
                    "Generate",
                    parts.pick(
                        [(str(one), str(one)) for one in GENERATES],
                        str(self.generate),
                        lambda value: self._set(generate=int(value)),
                        width=76,
                    ),
                ),
                knob(
                    "Concurrency",
                    parts.seg(
                        [(str(one), str(one)) for one in CONCURRENCIES],
                        str(self.concurrency),
                        lambda value: self._set(concurrency=int(value)),
                        mono=True,
                        tight=True,
                    ),
                ),
                knob(
                    "Variant",
                    parts.pick(
                        variants := [
                            (one, one[len(self.prefix) :])
                            for one in (self.variants or ([self.key] if self.key else []))
                        ],
                        self.key or "",
                        lambda value: self._set(variant=value),
                        # The widest key it has to show, and no wider: this is the last
                        # knob on a row that is already full, and the seed after it is the
                        # only thing left to squeeze.
                        width=parts.fits([label for _, label in variants], 210),
                    ),
                ),
            ]
        row.append(
            ft.Text(
                (self.job["error"] or self.job["progress"]["message"])
                if running and self.job is not None
                else f"{len(self._models)} measured",
                style=theme.mono(9.5, t().fg3),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
                expand=True,
                text_align=ft.TextAlign.RIGHT,
            )
        )
        return ft.Container(
            padding=ft.Padding.only(bottom=7),
            content=ft.Row(row, spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _band(self) -> ft.Control:
        drawn = self._drawn
        if self.view == "speed":
            axes = speed_axes(self.context, self.generate, self.concurrency)
        elif self.view == "fidelity":
            axes = fidelity_axes(CORPUS)
        else:
            seen: dict[str, str] = {}
            for run in self.all:
                if run["quality"] is not None:
                    seen[run["quality"]["dataset"]] = run["key"]
            axes = quality_axes(list(seen.items()))
        lines = [
            Line(
                model,
                index,
                speed_values(next((r for r in self._for_view if r["model"] == model), None))
                if self.view == "speed"
                else fidelity_values(
                    next((r for r in self._for_view if r["model"] == model), None)
                )
                if self.view == "fidelity"
                else quality_values([r for r in self.all if r["model"] == model]),
            )
            for index, model in enumerate(drawn)
        ]
        return band(axes, lines, self.selected, self._pick, TRACK)

    def _legend(self) -> ft.Control:
        """.tfoot — the row is reserved whether or not there are chips in it: a chip is
        taller than the note alone, and without the floor the whole band moved by that much
        on every switch between speed, fidelity and quality."""
        rest = len(self._models) - len(self._drawn)
        chips: list[ft.Control] = [
            ft.Container(
                tooltip="Remove from the comparison",
                content=ft.Row(
                    [
                        ft.Container(
                            width=9, height=2.5, border_radius=2, bgcolor=t().mat(index)
                        ),
                        ft.Text(model, style=theme.sans(10, t().fg2, ft.FontWeight.W_500),
                                no_wrap=True),
                        ft.Text("×", style=theme.sans(10, t().fg2)),
                    ],
                    spacing=6,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                on_click=lambda _, model=model: self._drop(model),
            )
            for index, model in enumerate(self._drawn)
        ]
        chips += [
            ft.Container(expand=True),
            ft.Text(
                f"{rest} more measured under this key — click a run to add"
                if rest > 0
                else "every run under this key is here",
                style=theme.sans(10.5, t().fg3),
                no_wrap=True,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
        ]
        return ft.Container(
            height=26,
            padding=ft.Padding.only(top=8),
            content=ft.Row(chips, spacing=7, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    # ── the run list ──────────────────────────────────────────────────────

    def _visible(self, run: Run) -> bool:
        return self.query == "" or self.query.lower() in run["model"].lower()

    def _list(self) -> ft.Control:
        rows: list[ft.Control] = [
            ft.Container(
                margin=ft.Margin.only(bottom=9),
                content=ft.Row(
                    [
                        parts.search(self.query, self._type, "Filter by checkpoint"),
                        parts.fchip("Clear comparison", False, self._clear),
                    ],
                    spacing=9,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        ]
        if self.failure is not None:
            rows.append(parts.note(self.failure, bad=True))
        if not finished(self.job) and self.job is not None:
            total = self.job["progress"]["total"]
            rows.append(
                ft.Container(
                    margin=ft.Margin.only(bottom=10),
                    content=cost(
                        "Running",
                        str(round(self.job["progress"]["completed"])),
                        f"of {'—' if total is None else round(total)}",
                        self.job["progress"]["message"],
                    ),
                )
            )
        if self.view == "speed":
            rows += self._speed_rows()
        elif self.view == "quality":
            rows += self._quality_rows()
        else:
            rows += self._fidelity_rows()

        return parts.pane(
            [
                parts.head(f"{TITLES[self.view]} runs", f"{len(self._for_view)} runs"),
                ft.Container(
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=11, vertical=9),
                    content=ft.Column(rows, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
                ),
                ft.Container(
                    padding=ft.Padding(left=13, right=13, top=8, bottom=8),
                    border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            parts.btn("New benchmark", self._open_sheet, "pri"),
                            parts.btn(
                                "Export CSV",
                                lambda: webbrowser.open(api.export_url(self.view, self.key)),
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
            ]
        )

    def _type(self, value: str) -> None:
        self.query = value
        self.notify()

    def _clear(self) -> None:
        self.pinned = []
        self.dropped = []
        self.notify()

    def _speed_rows(self) -> list[ft.Control]:
        ordered = sorted(
            [run for run in self.scoped if self._visible(run)],
            key=lambda run: -((run["speed"] or {}).get("decode_tps") or -1),
        )
        streams = f"{self.concurrency} stream{'s' if self.concurrency > 1 else ''}"
        drawn: list[ft.Control] = [
            qhead(
                f"{human(self.context)} → {self.generate} tok · {streams}",
                f"{len(ordered)} measured",
            )
        ]
        materials = self._materials
        for run in ordered:
            body = run["speed"]
            refused = api.refusal(body)
            ok = run["state"] == "ok"
            samples = [one["decode_tps"] for one in api.rounds(body)]
            if ok:
                state = (
                    "measured"
                    if body is None or body["ceiling_fraction"] is None
                    else f"{body['ceiling_fraction'] * 100:.1f}% of ceiling"
                )
                under = f"{(body or {}).get('decode_per_stream_tps') or 0:.1f} per stream"
            elif refused is not None and refused.reason == "kv_over_budget" and (
                refused.needed_bytes is not None
            ):
                state = f"needs {gb(refused.needed_bytes, 0)} GB"
                under = "kv over budget"
            else:
                state = (run["reason"] or "not run").replace("_", " ")
                under = "—"
            load = (
                "—"
                if body is None or body["load_s"] is None
                else f"{body['load_s']:.1f} s"
            )
            key = run["key"]
            drawn.append(
                run_row(
                    run["model"],
                    f"{key.removeprefix(self.prefix)} · load {load}",
                    materials.get(run["model"]),
                    run["model"] == self.selected,
                    ok,
                    [
                        (
                            f"{(body or {}).get('decode_tps') or 0:.1f}" if ok else "—",
                            "decode",
                            not ok,
                        ),
                        (nb((body or {}).get("ttft_p50_ms")) if ok else "—", "ms ttft", True),
                        (nb((body or {}).get("prefill_tps")) if ok else "—", "prefill", True),
                    ],
                    samples if ok else [],
                    "ok" if ok else "bad",
                    state,
                    under,
                    lambda model=run["model"]: self._pick(model),
                )
            )
        if not ordered:
            drawn.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=3, vertical=12),
                    content=parts.note(
                        "Nothing measured under this shape. New benchmark runs it — or pick "
                        "another context, generate and concurrency above."
                    ),
                )
            )
        return drawn

    def _quality_rows(self) -> list[ft.Control]:
        groups: dict[str, list[Run]] = {}
        for run in self.all:
            if not self._visible(run):
                continue
            name = (run["quality"] or {}).get("dataset") or "unknown"
            groups.setdefault(name, []).append(run)
        if not groups:
            return [
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=3, vertical=12),
                    content=parts.note(
                        "Nothing scored yet. Declaring a dataset and running it is New "
                        "benchmark → Quality."
                    ),
                )
            ]
        materials = self._materials
        drawn: list[ft.Control] = []
        for name, entries in groups.items():
            ordered = sorted(
                entries, key=lambda run: -((run["quality"] or {}).get("accuracy") or -1)
            )
            best = (ordered[0]["quality"] or {}).get("accuracy")
            # The key on the header is what authorises the comparison: change n or the
            # seed and it becomes a different axis.
            drawn.append(qhead(name, ordered[0]["key"]))
            for run in ordered:
                body = run["quality"]
                accuracy = None if body is None else body["accuracy"]
                if accuracy is None:
                    dot, state = "", (run["reason"] or "not run").replace("_", " ")
                elif accuracy == best:
                    dot, state = "ok", "best here"
                else:
                    dot, state = "ok", f"{(accuracy - (best or 0)) * 100:.1f} pt off best"
                drawn.append(
                    run_row(
                        run["model"],
                        run["key"],
                        materials.get(run["model"]),
                        run["model"] == self.selected,
                        run["state"] == "ok",
                        [
                            (
                                "—" if accuracy is None else f"{accuracy * 100:.1f}%",
                                "accuracy",
                                accuracy is None,
                            ),
                            (str((body or {}).get("items") or "—"), "items", True),
                        ],
                        [],
                        dot,
                        state,
                        (body or {}).get("scoring") or "—",
                        lambda model=run["model"], name=name: self._pick(model, name),
                    )
                )
        return drawn

    def _fidelity_rows(self) -> list[ft.Control]:
        measured, comparable, incompatible = self._fidelity
        materials = self._materials
        drawn: list[ft.Control] = [
            qhead(f"Against {self.reference}", CORPUS)
        ]
        for one in measured:
            body = None if one.run is None else one.run["fidelity"]
            kl = None if body is None else body["kl_mean"]
            ppl = None if body is None else body["ppl_delta"]
            top5 = None if body is None else body["top5"]
            drawn.append(
                run_row(
                    one.entry["id"],
                    (one.run or {}).get("key") or CORPUS,
                    materials.get(one.entry["id"]),
                    one.entry["id"] == self.selected,
                    True,
                    [
                        (fix(kl, 3), "kl nats", False),
                        (
                            "—" if body is None or body["top1"] is None
                            else f"{body['top1'] * 100:.1f}",
                            "% top-1",
                            True,
                        ),
                    ],
                    [],
                    "" if kl is None else "ok" if kl < 0.02 else "warn" if kl < 0.08 else "bad",
                    "—" if ppl is None else f"ppl {'+' if ppl > 0 else ''}{ppl:.1f}%",
                    "" if top5 is None else f"top-5 {top5 * 100:.1f}%",
                    lambda model=one.entry["id"]: self._pick(model),
                )
            )
        if comparable:
            drawn.append(
                qhead("Same vocabulary, not measured against this reference", "can run")
            )
            for one in comparable:
                drawn.append(
                    ft.Container(
                        opacity=0.6,
                        content=run_row(
                            one.entry["id"],
                            f"shape {one.entry['shape'] or '—'}",
                            None,
                            False,
                            not one.other_shape,
                            [("—", "kl nats", True), ("—", "% top-1", True)],
                            [],
                            "",
                            "other architecture" if one.other_shape else "not run",
                            "kl defined, not meaningful" if one.other_shape else "same shape",
                            lambda model=one.entry["id"]: self._pick(model),
                        ),
                    )
                )
        # No number and no click: logits of two vocabularies do not subtract.
        if incompatible:
            drawn.append(qhead("Different vocabulary", f"{len(incompatible)} checkpoints"))
            for entry in incompatible:
                drawn.append(
                    ft.Container(
                        opacity=0.45,
                        content=run_row(
                            entry["id"],
                            f"vocab {entry['vocab_size'] or '—'}",
                            None,
                            False,
                            False,
                            [("—", "kl nats", True), ("—", "% top-1", True)],
                            [],
                            "",
                            "not comparable",
                            "logits don't subtract",
                            None,
                        ),
                    )
                )
        return drawn

    # ── the inspector ─────────────────────────────────────────────────────

    def _inspector(self) -> ft.Control:
        run = self._chosen
        body: list[ft.Control]
        if run is None:
            body = [
                parts.note("The band draws the comparison; the inspector explains one line of it.")
            ]
        elif self.tab == "raw":
            body = [
                ft.Text(
                    json.dumps(run, indent=2),
                    style=theme.mono(10, t().fg2),
                    selectable=True,
                )
            ]
        elif self.tab == "setup":
            body = setup(run)
        elif self.view == "speed":
            body = self._speed_result(run)
        elif self.view == "quality":
            body = self._quality_result(run)
        else:
            body = fidelity_result(run, self.reference or "")

        material = self._materials.get(self.selected or "")
        return parts.pane(
            [
                insph(
                    self.selected or "Nothing selected",
                    "pick a run" if run is None else f"run {run['id'][:6]} · {run['key']}",
                    t().mat(material) if material is not None else t().mat(0),
                ),
                itabs(self.tab, self._pick_tab),
                ft.Container(
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=13, vertical=11),
                    content=ft.Column(body, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True),
                ),
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=11, vertical=9),
                    border=ft.Border.only(top=ft.BorderSide(1, t().hair)),
                    content=ft.Row(
                        [
                            parts.btn(
                                "Run again", self._open_sheet, "", run is not None, expand=True
                            ),
                            parts.btn(
                                "Delete this run" if self.confirming else "Delete",
                                self._forget,
                                "danger",
                                run is not None,
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
            ],
            INSPECTOR,
        )

    def _pick_tab(self, key: str) -> None:
        self.tab = key
        self.notify()

    def _speed_result(self, run: Run) -> list[ft.Control]:
        body = run["speed"]
        refused = api.refusal(body)
        samples = api.rounds(body)
        decode = [one["decode_tps"] for one in samples]
        spread = (
            None
            if len(decode) < 2
            else (max(decode) - min(decode)) / ((body or {}).get("decode_tps") or 1) * 100
        )
        # The same checkpoint under every stream count, which is what shows the number
        # falling off with context long before it falls off with streams.
        series = [
            (
                value,
                next(
                    (
                        one["speed"]["decode_tps"]
                        for one in self.all
                        if one["model"] == self.selected
                        and one["speed"] is not None
                        and one["speed"]["concurrency"] == value
                    ),
                    None,
                ),
            )
            for value in CONCURRENCIES
        ]
        high = max([1.0, *[value or 0 for _, value in series]])
        streams = f"{self.concurrency} stream{'s' if self.concurrency > 1 else ''}"

        drawn: list[ft.Control] = [
            group("Load", "· no context, no concurrency", first=True),
            kv("time to ready",
                "—" if body is None or body["load_s"] is None else f"{body['load_s']:.2f} s"),
            kv("page cache", (body or {}).get("page_cache") or "—", last=True),
        ]
        if refused is not None:
            drawn.append(
                ft.Container(
                    padding=ft.Padding.only(top=9),
                    content=parts.note(
                        refused.detail
                        or (
                            f"{refused.reason.replace('_', ' ')}: needed "
                            f"{gb(refused.needed_bytes)} GB against a budget of "
                            f"{gb(refused.budget_bytes)} GB"
                            if refused.needed_bytes is not None
                            and refused.budget_bytes is not None
                            else refused.reason.replace("_", " ")
                        ),
                        bad=True,
                    ),
                )
            )
        drawn += [
            group(f"At {streams} · {human(self.context)} context"),
            kv("prefill", f"{nb((body or {}).get('prefill_tps'))} tok/s"),
            kv("ttft p50", f"{nb((body or {}).get('ttft_p50_ms'))} ms"),
            kv("ttft p95", f"{nb((body or {}).get('ttft_p95_ms'))} ms"),
            kv("decode total", f"{(body or {}).get('decode_tps') or 0:.1f} tok/s"),
            kv(
                "decode per stream",
                f"{(body or {}).get('decode_per_stream_tps') or 0:.1f} tok/s",
            ),
            kv(
                "spread across rounds",
                "—" if spread is None else f"±{spread:.1f}%",
                last=True,
            ),
            group("Scaling with concurrency", f"· at {human(self.context)}"),
            curve(series, self.concurrency, high, lambda v: self._set(concurrency=v)),
            parts.sublab("decode total", f"{high:.0f} tok/s"),
            # The ceiling opened: weights amortise between streams and the cache does not.
            group("Bandwidth ceiling"),
            kv(
                "weights read / step",
                "—" if body is None or body["step_weight_bytes"] is None
                else f"{gb(body['step_weight_bytes'], 2)} GB",
            ),
            kv(
                "kv read / step",
                "—" if body is None or body["step_kv_bytes"] is None
                else f"{gb(body['step_kv_bytes'], 2)} GB",
            ),
            kv("ceiling here", f"{nb((body or {}).get('ceiling_tps'))} tok/s"),
            kv(
                "% of ceiling",
                "—" if body is None or body["ceiling_fraction"] is None
                else f"{body['ceiling_fraction'] * 100:.1f}%",
                last=True,
            ),
            ft.Container(
                padding=ft.Padding.only(top=8),
                content=parts.note(
                    "The ceiling is recomputed for this shape. Change the weight format and "
                    "it moves — a percentage that rose because its denominator shrank is not "
                    "a gain."
                ),
            ),
        ]
        if samples:
            drawn += [
                group("Rounds"),
                bars(decode),
                ft.Container(
                    padding=ft.Padding.only(top=8),
                    content=parts.note(
                        "One warm-up round ran before the first shape of this checkpoint and "
                        "stayed out of every median."
                    ),
                ),
            ]
        return drawn

    def _quality_result(self, run: Run) -> list[ft.Control]:
        body = run["quality"]
        areas = api.pairs((body or {}).get("breakdown") or "{}")
        others = [
            one
            for one in self.all
            if one["model"] == run["model"] and one["id"] != run["id"]
        ]
        drawn: list[ft.Control] = [
            group("Score", first=True),
            kv(
                "Accuracy",
                "—" if body is None or body["accuracy"] is None
                else f"{body['accuracy'] * 100:.1f}%",
            ),
            kv(
                "Correct",
                f"{(body or {}).get('correct') or '—'} / {(body or {}).get('items') or '—'}",
            ),
            kv("Scoring", (body or {}).get("scoring") or "—"),
            kv(
                "Wall",
                "—" if body is None or body["wall_s"] is None else f"{body['wall_s']:.0f} s",
                last=True,
            ),
            group("Comparison key"),
            keyline(run["key"]),
            ft.Container(
                padding=ft.Padding.only(top=6),
                content=parts.note(
                    "Only comparable to runs carrying exactly this key. Change n or the seed "
                    "and it becomes a different axis."
                ),
            ),
        ]
        if run["state"] != "ok":
            drawn += [
                group("Did not run"),
                parts.err(
                    "Quality scoring is not implemented yet: the log-likelihood pass over a "
                    "dataset's continuations is task 59.8. The shape is recorded so nothing "
                    "about it has to be typed twice."
                    if run["reason"] == "not_implemented"
                    else (run["reason"] or "unknown")
                ),
            ]
        if areas:
            drawn.append(group("By area"))
            for index, (area, value) in enumerate(areas):
                drawn.append(kv(area, f"{value * 100:.1f}%", last=index == len(areas) - 1))
        if others:
            drawn.append(group("This checkpoint elsewhere"))
            for index, other in enumerate(others):
                name = (other["quality"] or {}).get("dataset") or ""
                accuracy = (other["quality"] or {}).get("accuracy")
                drawn.append(
                    ft.Container(
                        padding=ft.Padding.symmetric(vertical=5),
                        border=None
                        if index == len(others) - 1
                        else ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
                        content=ft.Row(
                            [
                                ft.Text(name, style=theme.sans(11), no_wrap=True,
                                        overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                                ft.Text(
                                    "—" if accuracy is None else f"{accuracy * 100:.1f}%",
                                    style=theme.mono(9.5, t().fg3),
                                    no_wrap=True,
                                ),
                            ],
                            spacing=8,
                        ),
                        on_click=lambda _, name=name: self._set_dataset(name),
                    )
                )
        return drawn

    def _set_dataset(self, name: str) -> None:
        self.dataset = name
        self.notify()

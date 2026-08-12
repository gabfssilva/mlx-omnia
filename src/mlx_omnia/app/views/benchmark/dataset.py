"""The dataset definition form, and its preview."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field

import flet as ft

from mlx_omnia.app.api import benchmarks as api
from mlx_omnia.app.api.benchmarks import (
    Preview,
)
from mlx_omnia.app.ui import parts, theme
from mlx_omnia.app.ui.forms import labelled
from mlx_omnia.app.ui.hooks import act
from mlx_omnia.app.ui.theme import t
from mlx_omnia.app.views.benchmark.shared import group

USES = [("multiple_choice", "Choice"), ("generation", "Generate"), ("corpus", "Corpus")]

EMPTY_DATASET: dict[str, object] = {
    "id": "",
    "use": "multiple_choice",
    "repo": "",
    "config": None,
    "split": "test",
    "template": "{question}\nAnswer:",
    "columns": {"question": "question", "choices": "choices", "answer": "answer"},
    "size": None,
}


@dataclass
class DatasetForm:
    """Declaring a dataset, with the first item rendered beside the form.

    The preview is the point of the form. A column mapped to the wrong name is a silent
    error: the run completes, twenty minutes later, at chance accuracy, and nothing says
    why. Here it fails in a second, by name, with the columns the repository ships.
    """

    redraw: Callable[[], None]
    close: Callable[[], None]
    saved: Callable[[], Coroutine[None, None, None]]
    draft: dict[str, object] = field(default_factory=lambda: dict(EMPTY_DATASET))
    shown: Preview | None = None
    failure: str | None = None
    busy: bool = False

    def _field(self, key: str, value: str) -> None:
        self.draft[key] = None if value == "" and key == "config" else value
        self.redraw()

    def _column(self, role: str, value: str) -> None:
        columns = dict(self.draft["columns"])  # type: ignore[arg-type]
        columns[role] = value
        self.draft["columns"] = columns
        self.redraw()

    def _look(self) -> None:
        act(self._preview())

    async def _preview(self) -> None:
        self.busy = True
        self.failure = None
        self.redraw()
        try:
            self.shown = await api.preview(self.draft)
        except Exception as error:  # noqa: BLE001
            self.shown = None
            self.failure = str(error)
        finally:
            self.busy = False
            self.redraw()

    def _keep(self) -> None:
        act(self._save())

    async def _save(self) -> None:
        self.busy = True
        self.failure = None
        self.redraw()
        try:
            await api.save_dataset(self.draft)
            await self.saved()
            self.close()
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        finally:
            self.busy = False
            self.redraw()

    def tree(self) -> ft.Control:
        left: list[ft.Control] = [
            labelled("Id", parts.field(str(self.draft["id"]),
                                       lambda v: self._field("id", v), mono=True)),
            labelled("Repository", parts.field(str(self.draft["repo"]),
                                               lambda v: self._field("repo", v), mono=True)),
            labelled("Config", parts.field(str(self.draft["config"] or ""),
                                           lambda v: self._field("config", v), mono=True)),
            labelled("Split", parts.field(str(self.draft["split"]),
                                          lambda v: self._field("split", v), mono=True)),
            parts.seg(USES, str(self.draft["use"]), lambda v: self._field("use", v)),
        ]
        columns = self.draft["columns"]
        right: list[ft.Control] = [group("Columns", first=True)]
        for role in ("question", "choices", "answer", "group"):
            right.append(
                ft.Container(
                    padding=ft.Padding.only(bottom=8),
                    content=labelled(
                        role,
                        parts.field(
                            str(columns.get(role, "")) if isinstance(columns, dict) else "",
                            lambda v, role=role: self._column(role, v),
                            mono=True,
                            hint="a column, or a.b to reach into a struct",
                        ),
                    ),
                )
            )
        right.append(
            labelled(
                "Template",
                parts.field(str(self.draft["template"]),
                            lambda v: self._field("template", v), mono=True),
            )
        )
        if self.failure is not None:
            right.append(
                ft.Container(padding=ft.Padding.only(top=9), content=parts.err(self.failure))
            )
        if self.shown is not None:
            right.append(self._preview_box(self.shown))

        return parts.sheet(
            [
                ft.Text("Add a dataset from the Hub",
                        style=theme.sans(13.5, weight=ft.FontWeight.W_600), no_wrap=True),
                ft.Text("the preview is what catches a wrong mapping",
                        style=theme.mono(11, t().fg3), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                parts.btn("Close", self.close, "quiet"),
            ],
            ft.Column(
                [
                    ft.Container(
                        height=460,
                        content=ft.Row(
                            [
                                ft.Container(
                                    width=246,
                                    padding=ft.Padding.symmetric(horizontal=15, vertical=13),
                                    border=ft.Border.only(right=ft.BorderSide(1, t().hair)),
                                    content=ft.Column(left, spacing=13,
                                                      scroll=ft.ScrollMode.AUTO, expand=True),
                                ),
                                ft.Container(
                                    expand=True,
                                    padding=ft.Padding.symmetric(horizontal=15, vertical=13),
                                    content=ft.Column(right, spacing=0,
                                                      scroll=ft.ScrollMode.AUTO, expand=True),
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
                                parts.btn("Preview", self._look, "", not self.busy),
                                parts.btn(
                                    "Save definition",
                                    self._keep,
                                    "pri",
                                    not self.busy
                                    and bool(self.draft["id"])
                                    and bool(self.draft["repo"]),
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
        )

    def _preview_box(self, shown: Preview) -> ft.Control:
        drawn: list[ft.Control] = [
            group("First item, as the model reads it", first=True),
            ft.Text(shown["prompt"], style=theme.mono(9.5, t().fg2), selectable=True),
            group("Continuations scored"),
        ]
        for index, option in enumerate(shown["continuations"]):
            right = str(index) == shown["answer"]
            drawn.append(
                ft.Text(
                    f"· {option}",
                    style=theme.mono(
                        9.5, t().fg if right else t().fg2,
                        ft.FontWeight.W_500 if right else None,
                    ),
                )
            )
        drawn.append(
            ft.Container(
                padding=ft.Padding.only(top=6),
                content=parts.note(
                    f"{shown['rows']} rows · columns: {', '.join(shown['columns'])}"
                ),
            )
        )
        return ft.Container(
            margin=ft.Margin.only(top=11),
            padding=ft.Padding.symmetric(horizontal=10, vertical=9),
            bgcolor=t().sunken,
            border_radius=7,
            content=ft.Column(drawn, spacing=0, tight=True),
        )

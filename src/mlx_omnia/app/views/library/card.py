"""The checkpoint's own README, file listing and raw source.

There is no webview here, and so no arbitrary HTML to sanitise for one: flutter_markdown
renders markdown and nothing else, so a card that leans on raw `<img>`/`<div>` shows its
markdown and leaves the tags out. The Source tab is where
that card is still readable in full.
"""

from __future__ import annotations

import asyncio
import re
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field

import flet as ft
import flet.canvas as cv

from mlx_omnia.app.api import catalog as hub
from mlx_omnia.app.api.catalog import CheckpointFile
from mlx_omnia.app.ui import theme
from mlx_omnia.app.ui.theme import t

SHARD = re.compile(r"^model-\d+-of-\d+\.safetensors$")

# ![alt](url) and [text](url), captured so a relative one can be resolved against the
# checkpoint's asset route before Flutter tries to fetch it.
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)")
ABSOLUTE = re.compile(r"^[a-z][a-z0-9+.-]*:|^//|^#", re.I)
# A frontmatter key, which is what the Source tab marks.
KEY = re.compile(r"^([A-Za-z0-9_-]+:)(.*)$")


def bytes_(size: float) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{round(size)} B"


@dataclass
class Frontmatter:
    """A YAML subset: cards use scalars and string lists, and a card that uses more keeps
    its extra shapes in the Source tab rather than breaking the chips."""

    scalars: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def split(raw: str) -> tuple[Frontmatter | None, str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, raw
    close = next((i for i, line in enumerate(lines) if i > 0 and line.strip() == "---"), -1)
    if close < 0:
        return None, raw
    front = Frontmatter()
    listing: str | None = None
    for line in lines[1:close]:
        key = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if key:
            listing = key[1] if key[2] == "" else None
            if key[2] != "":
                front.scalars[key[1]] = key[2]
            continue
        item = re.match(r"^\s*-\s+(.+)$", line)
        if item and listing == "tags":
            front.tags.append(item[1])
    return front, "\n".join(lines[close + 1 :])


def _open(url: str) -> None:
    webbrowser.open(url)


# ── the frontmatter, worn as chips ────────────────────────────────────────


def _chip(label: str | None, value: str, color: str | None = None) -> ft.Control:
    face: list[ft.Control] = []
    if label is not None:
        face.append(ft.Text(label, style=theme.mono(9.5, t().fg3), no_wrap=True))
    face.append(ft.Text(value, style=theme.mono(9.5, color or t().fg2), no_wrap=True))
    return ft.Container(
        padding=ft.Padding(left=7, right=7, top=2, bottom=2),
        border=theme.hair(t().hair2),
        border_radius=11,
        content=ft.Row(face, spacing=5, tight=True),
    )


def chips(front: Frontmatter, accent: str) -> ft.Control:
    scalars = front.scalars
    row: list[ft.Control] = []
    if "license" in scalars:
        row.append(_chip("license", scalars["license"], t().ok))
    base = scalars.get("base_model")
    if base is not None:
        relation = scalars.get("base_model_relation")
        row.append(_chip("base", f"{base} · {relation}" if relation else base, accent))
    if "library_name" in scalars:
        row.append(_chip("library", scalars["library_name"]))
    if "pipeline_tag" in scalars:
        row.append(_chip(None, scalars["pipeline_tag"]))
    row += [_chip(None, tag) for tag in front.tags]
    return ft.Container(
        padding=ft.Padding.only(bottom=11),
        content=ft.Row(row, spacing=4, wrap=True, run_spacing=4),
    )


# ── the card itself ───────────────────────────────────────────────────────


def _sheet(accent: str) -> ft.MarkdownStyleSheet:
    """.mdcard, as far as one stylesheet object reaches. The h2 rule above each section is
    a per-heading border flutter_markdown has no slot for; the extra air stands in."""
    return ft.MarkdownStyleSheet(
        p_text_style=theme.sans(11.5, t().fg2, height=1.62),
        a_text_style=theme.sans(11.5, accent, height=1.62),
        em_text_style=theme.sans(11.5, t().fg2, height=1.62),
        strong_text_style=theme.sans(11.5, t().fg, ft.FontWeight.W_600, height=1.62),
        h1_text_style=theme.sans(13.5, t().fg, ft.FontWeight.W_600),
        h2_text_style=theme.sans(12.5, t().fg, ft.FontWeight.W_600),
        h3_text_style=theme.sans(12, t().fg, ft.FontWeight.W_600),
        h4_text_style=theme.sans(12, t().fg, ft.FontWeight.W_600),
        h5_text_style=theme.sans(12, t().fg, ft.FontWeight.W_600),
        h6_text_style=theme.sans(12, t().fg, ft.FontWeight.W_600),
        h1_padding=ft.Padding.only(top=16, bottom=6),
        h2_padding=ft.Padding.only(top=18, bottom=6),
        h3_padding=ft.Padding.only(top=14, bottom=5),
        h4_padding=ft.Padding.only(top=14, bottom=5),
        code_text_style=theme.mono(10.5, t().fg),
        blockquote_text_style=theme.sans(11.5, t().fg2, height=1.62),
        blockquote_padding=ft.Padding(left=11, right=0, top=2, bottom=2),
        blockquote_decoration=ft.BoxDecoration(
            border=ft.Border.only(left=ft.BorderSide(3, t().hair2))
        ),
        codeblock_padding=ft.Padding(left=11, right=11, top=9, bottom=9),
        codeblock_decoration=ft.BoxDecoration(bgcolor=t().sunken, border_radius=7),
        horizontal_rule_decoration=ft.BoxDecoration(
            border=ft.Border.only(top=ft.BorderSide(1, t().hair))
        ),
        list_bullet_text_style=theme.sans(11.5, t().fg3, height=1.62),
        list_indent=18,
        block_spacing=9,
        table_head_text_style=theme.mono(9, t().fg3, ft.FontWeight.W_500, spacing=0.05 * 9),
        table_body_text_style=theme.mono(10.5, t().fg2),
        table_cells_padding=ft.Padding(left=0, right=12, top=5, bottom=5),
        table_cells_decoration=ft.BoxDecoration(
            border=ft.Border.only(bottom=ft.BorderSide(1, t().hair))
        ),
    )


def _absolute(body: str, assets: str) -> str:
    def one(match: re.Match[str]) -> str:
        url = match[2]
        if ABSOLUTE.match(url):
            return match[0]
        return match[1] + f"{assets}/{url.removeprefix('./').removeprefix('/')}"

    return LINK.sub(one, body)


def rendered(body: str, assets: str, accent: str) -> ft.Control:
    return ft.Markdown(
        _absolute(body, assets),
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        md_style_sheet=_sheet(accent),
        code_style_sheet=_sheet(accent),
        code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK
        if t() is theme.DARK
        else ft.MarkdownCodeTheme.ATOM_ONE_LIGHT,
        on_tap_link=lambda e: _open(str(e.data)),
    )


# ── the raw source, with the two things a reader checks marked ────────────


def source(raw: str, accent: str) -> ft.Control:
    lines = raw.splitlines()
    opens = bool(lines) and lines[0].strip() == "---"
    close = (
        next((i for i, line in enumerate(lines) if i > 0 and line.strip() == "---"), -1)
        if opens
        else -1
    )
    drawn: list[ft.Control] = []
    for index, line in enumerate(lines):
        if opens and index in (0, close):
            drawn.append(ft.Text(line, style=theme.mono(10, t().fg3)))
        elif close > 0 and 0 < index < close and (key := KEY.match(line)):
            drawn.append(
                ft.Text(
                    spans=[
                        ft.TextSpan(key[1], style=theme.mono(10, accent)),
                        ft.TextSpan(key[2], style=theme.mono(10, t().fg2)),
                    ]
                )
            )
        elif re.match(r"^#{1,6}\s", line):
            drawn.append(ft.Text(line, style=theme.mono(10, t().fg, ft.FontWeight.W_500)))
        else:
            drawn.append(ft.Text(line, style=theme.mono(10, t().fg2), selectable=True))
    return ft.Column(drawn, spacing=3, tight=True)


# ── the listing ───────────────────────────────────────────────────────────


def _chev(open_: bool) -> ft.Control:
    paint = ft.Paint(
        color=t().fg3,
        stroke_width=2.2 * 11 / 24,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
    )
    s = 11 / 24
    points = (
        [(6, 9), (12, 15), (18, 9)] if open_ else [(9, 6), (15, 12), (9, 18)]
    )
    (x0, y0), (x1, y1), (x2, y2) = points
    return cv.Canvas(
        [
            cv.Path(
                [
                    cv.Path.MoveTo(x0 * s, y0 * s),
                    cv.Path.LineTo(x1 * s, y1 * s),
                    cv.Path.LineTo(x2 * s, y2 * s),
                ],
                paint=paint,
            )
        ],
        width=11,
        height=11,
    )


def _frow(cells: list[ft.Control], last: bool, indent: float = 0.0) -> ft.Control:
    return ft.Container(
        padding=ft.Padding(left=indent, right=0, top=5, bottom=5),
        border=None if last else ft.Border.only(bottom=ft.BorderSide(1, t().hair)),
        content=ft.Row(cells, spacing=9, vertical_alignment=ft.CrossAxisAlignment.CENTER),
    )


def _tag(text: str) -> ft.Control:
    return ft.Text(
        text.upper(),
        style=theme.sans(8.5, t().fg3, spacing=0.08 * 8.5),
        no_wrap=True,
    )


def _name(text: str, dim: bool = False, grow: bool = False) -> ft.Control:
    return ft.Text(
        text,
        style=theme.mono(10.5, t().fg2 if dim else t().fg),
        no_wrap=True,
        overflow=ft.TextOverflow.ELLIPSIS,
        expand=grow,
    )


def _size(text: str) -> ft.Control:
    return ft.Text(text, style=theme.mono(10.5, t().fg3), no_wrap=True)


def listing(
    files: list[CheckpointFile], tail: str, open_: bool, toggle: Callable[[], None]
) -> ft.Control:
    shards = [item for item in files if SHARD.match(item["name"])]
    total = sum(item["size"] for item in files)
    rows: list[ft.Control] = []

    def plain(item: CheckpointFile, shard: bool) -> ft.Control:
        cells: list[ft.Control] = [_name(item["name"], shard, grow=item["name"] != "README.md")]
        if item["name"] == "README.md":
            cells += [_tag("card"), ft.Container(expand=True)]
        cells.append(_size(bytes_(item["size"])))
        return _frow(cells, False, 20 if shard else 0)

    for item in files:
        if not SHARD.match(item["name"]):
            rows.append(plain(item, False))
            continue
        # The shard run collapses into one row at its first member; the rest ride it.
        if item["name"] != shards[0]["name"]:
            continue
        rows.append(
            ft.Container(
                content=_frow(
                    [
                        _chev(open_),
                        _name(f"model-*-of-{shards[0]['name'].split('-of-')[1]}", grow=True),
                        _tag(f"{len(shards)} shards"),
                        _size(bytes_(sum(s["size"] for s in shards))),
                    ],
                    False,
                ),
                on_click=lambda _: toggle(),
            )
        )
        if open_:
            rows += [plain(shard, True) for shard in shards]

    rows.append(
        ft.Container(
            padding=ft.Padding.only(top=9),
            content=ft.Row(
                [
                    ft.Text(f"{len(files)} files", style=theme.sans(10, t().fg3), no_wrap=True),
                    ft.Text(
                        f"{bytes_(total)} {tail}",
                        style=theme.sans(10, t().fg3),
                        no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True,
                        text_align=ft.TextAlign.RIGHT,
                    ),
                ],
                spacing=12,
            ),
        )
    )
    return ft.Column(rows, spacing=0, tight=True)


# ── the block, which owns one subject's two fetches ───────────────────────


class Card:
    """One checkpoint's card and listing, kept across the rebuilds of the frame that
    draws them. `look` is idempotent per subject: the panel is rebuilt every frame and
    refetching there would ask the Hub for the same listing sixty times a second."""

    def __init__(self, redraw: Callable[[], None], priced: Callable[[int], None]) -> None:
        self.redraw = redraw
        self.priced = priced
        self.subject: tuple[str, bool] | None = None
        self.raw: str | None = None
        self.files: list[CheckpointFile] = []
        self.settled = False
        self.shards_open = False
        self._task: asyncio.Task[None] | None = None

    def look(self, identifier: str, from_hub: bool) -> None:
        if self.subject == (identifier, from_hub):
            return
        self.subject = (identifier, from_hub)
        self.raw = None
        self.files = []
        self.settled = False
        self.shards_open = False
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._read(identifier, from_hub))

    async def _read(self, identifier: str, from_hub: bool) -> None:
        async def card() -> None:
            try:
                self.raw = await hub.get_card(identifier, from_hub)
            except Exception:  # noqa: BLE001 — a card that will not load is a card-less model
                self.raw = None

        async def files() -> None:
            try:
                found = await (
                    hub.hub_files(identifier) if from_hub else hub.list_files(identifier)
                )
            except Exception:  # noqa: BLE001
                found = []
            self.files = found
            if found:
                self.priced(sum(item["size"] for item in found))

        await asyncio.gather(card(), files())
        if self.subject == (identifier, from_hub):
            self.settled = True
            self.redraw()

    def toggle(self) -> None:
        self.shards_open = not self.shards_open
        self.redraw()

    # What the inspector's tab bar may offer for this subject.
    @property
    def has_card(self) -> bool:
        return self.raw is not None

    def draw(self, tab: str, accent: str, tail: str) -> list[ft.Control]:
        if not self.settled:
            return [ft.Text("Reading the checkpoint…", style=theme.sans(11, t().fg3))]
        if tab == "files":
            return [listing(self.files, tail, self.shards_open, self.toggle)]
        if self.raw is None:
            return [
                ft.Text("This checkpoint ships no card.", style=theme.sans(11, t().fg3))
            ]
        if tab == "source":
            return [source(self.raw, accent)]
        front, body = split(self.raw)
        drawn: list[ft.Control] = []
        if front is not None:
            drawn.append(chips(front, accent))
        drawn.append(rendered(body, hub.asset_base(*self.subject or ("", False)), accent))
        return drawn

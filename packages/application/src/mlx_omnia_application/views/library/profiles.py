"""A model's saved profiles and its own speculation setting.

A profile is sampling plus a system prompt, saved on the model and served as `model:name`;
a knob left blank is not part of the profile, which is what `—` means on screen and `null`
on the wire.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import flet as ft

from mlx_omnia_application.api import catalog as hub
from mlx_omnia_application.api.catalog import (
    EFFORTS,
    ProfileView,
    SettingsView,
)
from mlx_omnia_application.api.catalog import Speculation as Speculating
from mlx_omnia_application.ui import forms, parts, theme
from mlx_omnia_application.ui.hooks import act
from mlx_omnia_application.ui.theme import t

# The two values the picker uses that are not ids. Neither is a valid model id — a `/` is
# mandatory in one — so they can never collide with something selectable.
OTHER = "?other"
MTP = "?mtp"
"""This model's own head, which has no id to name: it is `mtp.*` inside the checkpoint the
model already loads from."""

KNOBS = [
    ("temperature", "Temperature"),
    ("top_p", "Top-p"),
    ("top_k", "Top-k"),
    ("min_p", "Min-p"),
    ("repetition_penalty", "Repetition penalty"),
    ("seed", "Seed"),
]


def _num(raw: str) -> float | None:
    trimmed = raw.strip()
    if trimmed == "":
        return None
    try:
        parsed = float(trimmed)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _text(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else str(value)


@dataclass
class Draft:
    name: str = ""
    knobs: dict[str, str] = field(default_factory=lambda: {key: "" for key, _ in KNOBS})
    # `auto` is the profile not setting it, which is the same `null` every other blank
    # means: the checkpoint's own template decides.
    reasoning_effort: str = "auto"
    reasoning_budget: str = ""
    system_prompt: str = ""


def _of(view: ProfileView) -> Draft:
    sampling = view["sampling"]
    return Draft(
        name=view["name"],
        knobs={key: _text(sampling[key]) for key, _ in KNOBS},  # type: ignore[literal-required]
        reasoning_effort=sampling["reasoning_effort"] or "auto",
        reasoning_budget=_text(sampling["reasoning_budget"]),
        system_prompt=view["system_prompt"] or "",
    )


# ── the profile form ──────────────────────────────────────────────────────


class Profiles:
    """Kept across rebuilds: the panel is redrawn every frame, and a form that reset
    itself there would lose a character per keystroke."""

    def __init__(self, redraw: Callable[[], None]) -> None:
        self.redraw = redraw
        self.model: str | None = None
        self.names: list[str] = []
        self.selected: str | None = None
        self.profile: ProfileView | None = None
        self.draft: Draft | None = None
        self.failure: str | None = None
        self._task: asyncio.Task[None] | None = None

    def look(self, model: str) -> None:
        if self.model == model:
            return
        self.model = model
        self.names = []
        self.selected = None
        self.profile = None
        self.draft = None
        self.failure = None
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._read(model))

    async def _list(self, model: str) -> list[str]:
        self.names = await hub.profile_names(model)
        return self.names

    async def _read(self, model: str) -> None:
        try:
            found = await self._list(model)
            self.selected = found[0] if found else None
            if self.selected is not None:
                self.profile = await hub.get_profile(model, self.selected)
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.redraw()

    def _choose(self, name: str) -> None:
        self.selected = name
        act(self._fetch(name))

    async def _fetch(self, name: str) -> None:
        model = self.model
        if model is None:
            return
        try:
            self.profile = await hub.get_profile(model, name)
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.redraw()

    def _edit(self, **changes: object) -> None:
        current = self.draft or Draft()
        for key, value in changes.items():
            setattr(current, key, value)
        self.draft = current
        self.redraw()

    def _knob(self, key: str, value: str) -> None:
        current = self.draft or Draft()
        current.knobs[key] = value
        self.draft = current
        self.redraw()

    def _save(self) -> None:
        draft = self.draft
        if draft is None:
            return
        name = draft.name.strip()
        if name == "":
            self.failure = "a profile needs a name — it is what a request selects it by"
            self.redraw()
            return
        if ":" in name:
            self.failure = (
                "profile name may not contain ':' — resolution splits the model at the last one"
            )
            self.redraw()
            return
        self.failure = None
        act(self._write(name, draft))

    async def _write(self, name: str, draft: Draft) -> None:
        model = self.model
        if model is None:
            return
        try:
            sampling: dict[str, object] = {
                key: _num(draft.knobs[key]) for key, _ in KNOBS
            }
            sampling["reasoning_effort"] = (
                None if draft.reasoning_effort == "auto" else draft.reasoning_effort
            )
            sampling["reasoning_budget"] = _num(draft.reasoning_budget)
            saved = await hub.save_profile(
                model,
                name,
                {
                    "sampling": sampling,
                    "system_prompt": draft.system_prompt.strip() or None,
                },
            )
            await self._list(model)
            self.profile = saved
            self.selected = saved["name"]
            self.draft = None
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.redraw()

    def _remove(self, name: str) -> None:
        self.failure = None
        act(self._drop(name))

    async def _drop(self, name: str) -> None:
        model = self.model
        if model is None:
            return
        try:
            await hub.remove_profile(model, name)
            left = await self._list(model)
            self.selected = left[0] if left else None
            self.profile = (
                await hub.get_profile(model, self.selected) if self.selected else None
            )
            self.draft = None
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.redraw()

    # ── drawing ───────────────────────────────────────────────────────────

    def draw(self) -> list[ft.Control]:
        model = self.model or ""
        drawn: list[ft.Control] = [
            forms.h3("Profiles", "none saved" if not self.names else f"{len(self.names)} saved")
        ]
        if self.names and self.draft is None:
            drawn.append(
                ft.Container(
                    padding=ft.Padding.only(bottom=11),
                    content=parts.seg(
                        [(name, name) for name in self.names],
                        self.selected or "",
                        self._choose,
                        mono=True,
                    ),
                )
            )
        if self.failure is not None:
            drawn.append(forms.notice(self.failure, lambda: self._edit()))
        if self.draft is not None:
            drawn += self._form(model, self.draft)
        elif self.profile is None:
            drawn += [
                ft.Text(
                    "No profile saved. A profile fills the knobs a request leaves out, and is "
                    f"selected by asking for {model}:name.",
                    style=theme.sans(10.5, t().fg3, height=1.5),
                ),
                self._footer([parts.btn("New profile", lambda: self._edit(name=""), "pri")]),
            ]
        else:
            drawn += self._read_only(self.profile)
        return drawn

    def _form(self, model: str, draft: Draft) -> list[ft.Control]:
        served = f"{model}:{draft.name.strip() or 'name'}"
        cells = [
            forms.labelled(
                label,
                parts.field(
                    draft.knobs[key],
                    lambda value, key=key: self._knob(key, value),
                    mono=False,
                    hint="—",
                ),
            )
            for key, label in KNOBS
        ]
        cells += [
            forms.labelled(
                "Reasoning effort",
                parts.pick(
                    [(rung, rung) for rung in EFFORTS],
                    draft.reasoning_effort,
                    lambda value: self._edit(reasoning_effort=value),
                ),
            ),
            forms.labelled(
                "Reasoning budget",
                parts.field(
                    draft.reasoning_budget,
                    lambda value: self._edit(reasoning_budget=value),
                    hint="—",
                ),
            ),
        ]
        return [
            forms.labelled(
                "Name",
                parts.field(draft.name, lambda value: self._edit(name=value), mono=True,
                            hint="code"),
                ft.Text(f"Served as {served}", style=theme.mono(10, t().fg2)),
            ),
            ft.Container(padding=ft.Padding.only(top=11), content=forms.grid(cells)),
            ft.Container(
                padding=ft.Padding.only(top=11),
                content=forms.labelled(
                    "System prompt",
                    forms.area(
                        draft.system_prompt,
                        lambda value: self._edit(system_prompt=value),
                        "None — the checkpoint's template stands.",
                    ),
                ),
            ),
            self._footer(
                [
                    parts.btn("Save", self._save, "pri"),
                    parts.btn("Cancel", lambda: self._cancel()),
                ]
            ),
        ]

    def _cancel(self) -> None:
        self.draft = None
        self.failure = None
        self.redraw()

    def _read_only(self, profile: ProfileView) -> list[ft.Control]:
        sampling = profile["sampling"]
        drawn: list[ft.Control] = [
            forms.line("Temperature", _text(sampling["temperature"]) or "—",
                 sampling["temperature"] is None),
            forms.line("Top-p", _text(sampling["top_p"]) or "—", sampling["top_p"] is None),
            forms.line("Top-k", _text(sampling["top_k"]) or "—", sampling["top_k"] is None),
        ]
        for key, label in (
            ("min_p", "Min-p"),
            ("repetition_penalty", "Repetition penalty"),
            ("seed", "Seed"),
        ):
            value = sampling[key]  # type: ignore[literal-required]
            if value is not None:
                drawn.append(forms.line(label, _text(value)))
        if sampling["reasoning_effort"] is not None:
            drawn.append(forms.line("Reasoning effort", sampling["reasoning_effort"]))
        if sampling["reasoning_budget"] is not None:
            drawn.append(forms.line("Reasoning budget", _text(sampling["reasoning_budget"])))
        drawn.append(
            forms.line("System prompt", profile["system_prompt"] or "none",
                 profile["system_prompt"] is None)
        )
        drawn.append(
            self._footer(
                [
                    parts.btn("Edit", lambda: self._start(profile), "pri"),
                    parts.btn("New", lambda: self._edit(name="")),
                    parts.btn(
                        "Delete", lambda: self._remove(profile["name"]), "danger"
                    ),
                ]
            )
        )
        return drawn

    def _start(self, profile: ProfileView) -> None:
        self.draft = _of(profile)
        self.redraw()

    def _footer(self, buttons: list[ft.Control]) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.only(top=13),
            content=ft.Row(buttons, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )


# ── the model's own drafter ───────────────────────────────────────────────


class Speculation:
    """There is no on/off here: what turns speculation on is naming the *technique*, since
    naming it is naming the thing the daemon loads beside the model. It never changes what
    the model writes — verification accepts only the target's own argmax — so the setting
    buys speed or it buys nothing.

    Two techniques, drawn by one picker because on screen it is one choice:

    - a **DFlash drafter**, a second checkpoint, picked by id out of what is installed;
    - this model's **own MTP head**, which is inside its checkpoint and has no id to pick.

    Which of the two a model can do is the daemon's answer, read on every load: `available`
    lists the drafters, `mtp_available` says whether the head is in the shards.
    """

    def __init__(self, redraw: Callable[[], None]) -> None:
        self.redraw = redraw
        self.model: str | None = None
        self.view: SettingsView | None = None
        self.failure: str | None = None
        self.saving = False
        self.choice = ""
        """What the picker holds: `""` off, `MTP` this model's own head, anything else the
        id of a drafter checkpoint."""
        self.typing = False
        self.block = ""
        self._task: asyncio.Task[None] | None = None

    def look(self, model: str) -> None:
        if self.model == model:
            return
        self.model = model
        self.view = None
        self.failure = None
        self.typing = False
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._read(model))

    @staticmethod
    def _chosen(stored: Speculating | None) -> str:
        """The stored row as the picker's one value. An MTP head names no drafter, so the
        kind is the only thing that tells it from off."""
        if stored is None or stored.get("kind") is None:
            return ""
        return MTP if stored["kind"] == "mtp" else (stored.get("drafter") or "")

    def _take(self, view: SettingsView) -> None:
        self.view = view
        stored = view["features"]["speculation"]
        self.choice = self._chosen(stored)
        self.typing = False
        size = (stored or {}).get("block_size")
        self.block = "" if size is None else str(size)

    async def _read(self, model: str) -> None:
        try:
            self._take(await hub.get_settings(model))
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
        self.redraw()

    def _commit(self, choice: str) -> None:
        view = self.view
        if view is None:
            return
        stored = view["features"]["speculation"]
        length = None if self.block.strip() == "" else int(float(self.block))
        if choice == self._chosen(stored) and length == (stored or {}).get("block_size"):
            return
        if choice == "":
            asked: Speculating = {"kind": None, "drafter": None, "block_size": None}
        elif choice == MTP:
            asked = {"kind": "mtp", "drafter": None, "block_size": length}
        else:
            asked = {"kind": "dflash", "drafter": choice, "block_size": length}
        act(self._write(asked))

    async def _write(self, next_: Speculating) -> None:
        model = self.model
        if model is None:
            return
        self.saving = True
        self.failure = None
        self.redraw()
        try:
            self._take(await hub.save_settings(model, {"speculation": next_}))
        except Exception as error:  # noqa: BLE001
            self.failure = str(error)
            # The daemon refused, so what is on screen is not what is stored.
            await self._read(model)
        finally:
            self.saving = False
            self.redraw()

    def _pick(self, value: str) -> None:
        if value == OTHER:
            self.choice = ""
            self.typing = True
            self.redraw()
            return
        self.choice = value
        self._commit(value)

    def draw(self) -> list[ft.Control]:
        view = self.view
        if view is None:
            return []
        # What the picker offers: off, this model's own head when it has one, everything
        # installed, and — when the stored id is not among them — the stored one, so opening
        # the list never proposes to forget it.
        installed = view["available"]
        named = "" if self.choice == MTP else self.choice
        offered = installed if named in installed or named == "" else [named, *installed]
        options = [("", "off")]
        if view["mtp_available"]:
            options.append((MTP, "This model's own MTP head"))
        options += [(one, one) for one in offered]
        options.append((OTHER, "Another checkpoint…"))

        if self.typing or (not offered and not view["mtp_available"]):
            chooser: ft.Control = parts.field(
                named,
                lambda value: self._commit(value.strip()),
                mono=True,
                hint="off",
            )
        else:
            chooser = parts.pick(options, self.choice, self._pick)
        drawn: list[ft.Control] = [
            # First on the Features tab, which it has to itself until there is a second one.
            parts.group("Speculation", first=True),
            forms.labelled("Draft with", chooser),
        ]
        stored = view["features"]["speculation"]
        if stored is not None and stored["kind"] is not None:
            drawn.append(
                ft.Container(
                    padding=ft.Padding.only(top=11),
                    content=forms.labelled(
                        "Ids proposed a round",
                        parts.field(
                            self.block,
                            lambda value: self._set_block(value),
                            hint="the engine's default",
                        ),
                    ),
                )
            )
            # Said before the click and not as a 409 after it. The engine refuses to compose
            # speculation with prefix reuse (`generate.stream_ids` says why), and prefix
            # reuse is on by default — on a 21k conversation it is the difference between
            # 0.14 s and 15.78 s to the first token of the third turn.
            drawn.append(
                ft.Container(
                    padding=ft.Padding.only(top=9),
                    content=parts.note(
                        "While this is on, requests that speculate do not reuse the "
                        "conversation prefix, and a sampled, penalized or grammar-constrained "
                        "request decodes without the draft."
                    ),
                )
            )
        if view["unavailable_reason"] is not None:
            drawn.append(
                ft.Container(
                    padding=ft.Padding.only(top=9),
                    content=parts.note(f"{view['unavailable_reason']}."),
                )
            )
        if self.failure is not None:
            drawn.append(
                ft.Container(padding=ft.Padding.only(top=9), content=parts.err(self.failure))
            )
        return drawn

    def _set_block(self, value: str) -> None:
        self.block = value
        self._commit(self.choice)

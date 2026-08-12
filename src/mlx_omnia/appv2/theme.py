"""The Quieta tokens: the v1 family of green-cast neutrals, one step lighter and
quieter, with the system blue replaced by an accent of the app's own — the verdete,
taken from the first material of the band.

The colour arithmetic (oklab mix, alpha, hatch) is pure and already written in the v1
theme; it is imported rather than copied. The tokens and the scale are this package's:
body goes from 12 to 13 pt, and screen titles get the Display face at 20+.
"""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from mlx_omnia.app.ui.theme import TRANSPARENT, alpha, hatch, mix

__all__ = ["TRANSPARENT", "alpha", "hatch", "mix"]


@dataclass(frozen=True)
class Tokens:
    win: str
    side: str
    surface: str
    elev: str
    field: str
    fg: str
    fg2: str
    fg3: str
    hair: str
    hair2: str
    accent: str
    accent_soft: str
    on_accent: str
    ok: str
    warn: str
    bad: str
    sel: str
    shade: str
    mats: tuple[str, str, str, str, str, str]

    def mat(self, material: int) -> str:
        """The material a model wears, handed out by load order — the v1 rule."""
        return self.mats[material % len(self.mats)]


LIGHT = Tokens(
    win="#EDEFEE",
    side="#E6E9E7",
    surface="#FAFBFA",
    elev="#FFFFFF",
    field="#FFFFFF",
    fg="#1A1D1C",
    fg2="#5B635F",
    fg3="#8B938F",
    hair="#1F141C19",
    hair2="#38141C19",
    accent="#2F8C7E",
    accent_soft="#212F8C7E",
    on_accent="#FFFFFF",
    ok="#2E7D45",
    warn="#8A6B0C",
    bad="#B23A2C",
    sel="#0E000000",
    shade="#33000000",
    mats=("#1F7A70", "#4E58C0", "#B2542C", "#4F6B1F", "#8B4694", "#7A5D08"),
)

DARK = Tokens(
    win="#16191A",
    side="#1B1F20",
    surface="#1E2224",
    elev="#262B2D",
    field="#171B1C",
    fg="#ECEEED",
    fg2="#9BA29E",
    fg3="#6B736F",
    hair="#1AEBF5F0",
    hair2="#33EBF5F0",
    accent="#66C2B3",
    accent_soft="#2666C2B3",
    on_accent="#10312B",
    ok="#63B274",
    warn="#D2A03B",
    bad="#DE6D5E",
    sel="#0EFFFFFF",
    shade="#6B000000",
    mats=("#57B0A6", "#8A93E0", "#D8825C", "#9DBB5E", "#BB7FC0", "#D0A93A"),
)

_current = LIGHT


def use(dark: bool) -> None:
    global _current
    _current = DARK if dark else LIGHT


def t() -> Tokens:
    return _current


# ── faces ─────────────────────────────────────────────────────────────────
# The system's own, as in v1. The Display face carries titles from 17 pt up, which is
# where macOS itself switches optical sizes.

SANS = "SF Pro Text"
SANS_FALLBACK = [".AppleSystemUIFont", "Helvetica Neue"]
DISPLAY = "SF Pro Display"
DISPLAY_FALLBACK = ["SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue"]
MONO = "SF Mono"
MONO_FALLBACK = ["Menlo", "Monaco"]

BODY = 13.0


def sans(
    size: float = BODY,
    color: str | None = None,
    weight: ft.FontWeight = ft.FontWeight.W_400,
    height: float | None = None,
    spacing: float | None = None,
) -> ft.TextStyle:
    return ft.TextStyle(
        font_family=SANS,
        font_family_fallback=SANS_FALLBACK,
        size=size,
        weight=weight,
        color=color if color is not None else t().fg,
        height=height,
        letter_spacing=spacing if spacing is not None else -0.004 * size,
    )


def display(
    size: float = 26.0,
    color: str | None = None,
    weight: ft.FontWeight = ft.FontWeight.W_600,
) -> ft.TextStyle:
    return ft.TextStyle(
        font_family=DISPLAY,
        font_family_fallback=DISPLAY_FALLBACK,
        size=size,
        weight=weight,
        color=color if color is not None else t().fg,
        letter_spacing=-0.015 * size,
    )


def mono(
    size: float = 12.0,
    color: str | None = None,
    weight: ft.FontWeight = ft.FontWeight.W_400,
    spacing: float | None = None,
) -> ft.TextStyle:
    return ft.TextStyle(
        font_family=MONO,
        font_family_fallback=MONO_FALLBACK,
        size=size,
        weight=weight,
        color=color if color is not None else t().fg,
        letter_spacing=spacing,
    )


def eyebrow(text: str) -> ft.Text:
    return ft.Text(
        text.upper(),
        style=mono(10.0, t().fg3, spacing=0.11 * 10.0),
        no_wrap=True,
    )


def hair(color: str | None = None, width: float = 1.0) -> ft.Border:
    return ft.Border.all(width, color if color is not None else t().hair)

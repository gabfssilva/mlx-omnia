"""The house style, which is theme.css resolved ahead of time.

Flutter, not WebKit: a colour is #AARRGGBB and never a var(), so what the stylesheet
resolves at paint time is resolved here at build time — including color-mix(), which is
the function below rather than the browser's. The window is the same fixed 1160x760, so
every measure that follows is the same real px measure the stylesheet counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flet as ft

# ── colour arithmetic ─────────────────────────────────────────────────────
# CSS mixes in oklab, and sRGB mixing is not the same picture: the band's slabs are
# 24 % of a saturated material into a light grey, where the two spaces differ in
# lightness by more than the tint itself.

TRANSPARENT = "transparent"


def _split(color: str) -> tuple[float, float, float, float]:
    body = color.removeprefix("#")
    if len(body) == 6:
        body = "FF" + body
    a, r, g, b = (int(body[i : i + 2], 16) / 255 for i in range(0, 8, 2))
    return a, r, g, b


def _join(a: float, r: float, g: float, b: float) -> str:
    return "#" + "".join(f"{round(min(max(v, 0.0), 1.0) * 255):02X}" for v in (a, r, g, b))


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * abs(c) ** (1 / 2.4) - 0.055


def _oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    lr, lg, lb = _linear(r), _linear(g), _linear(b)
    long = _cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb)
    med = _cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb)
    short = _cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb)
    return (
        0.2104542553 * long + 0.7936177850 * med - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * med + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * med - 0.8086757660 * short,
    )


def _srgb(lightness: float, a: float, b: float) -> tuple[float, float, float]:
    long = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    med = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    short = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        _gamma(4.0767416621 * long - 3.3077115913 * med + 0.2309699292 * short),
        _gamma(-1.2684380046 * long + 2.6097574011 * med - 0.3413193965 * short),
        _gamma(-0.0041960863 * long - 0.7034186147 * med + 1.7076147010 * short),
    )


def mix(color: str, share: float, into: str) -> str:
    """color-mix(in oklab, `color` `share`%, `into`), share in 0..1.

    Mixing into `transparent` is premultiplied, which for a fully opaque colour is
    exactly a scaled alpha — the same shortcut the browser takes.
    """
    alpha, r, g, b = _split(color)
    if into is TRANSPARENT or into == TRANSPARENT:
        return _join(alpha * share, r, g, b)
    beta, r2, g2, b2 = _split(into)
    la, aa, ba = _oklab(r, g, b)
    lb, ab, bb = _oklab(r2, g2, b2)
    rest = 1 - share
    red, green, blue = _srgb(
        la * share + lb * rest, aa * share + ab * rest, ba * share + bb * rest
    )
    return _join(alpha * share + beta * rest, red, green, blue)


def alpha(color: str, opacity: float) -> str:
    """rgba() over a hex colour, which the stylesheet writes as a literal."""
    _, r, g, b = _split(color)
    return _join(opacity, r, g, b)


# ── palette ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tokens:
    win: str
    surface: str
    surface2: str
    sunken: str
    field: str
    fg: str
    fg2: str
    fg3: str
    hair: str
    hair2: str
    mats: tuple[str, str, str, str, str, str]
    ok: str
    warn: str
    bad: str
    sel: str
    shade: str
    lift: str

    def mat(self, material: int) -> str:
        """The material a resident model wears, handed out by load order."""
        return self.mats[material % len(self.mats)]


LIGHT = Tokens(
    win="#E4E7E5",
    surface="#F8F9F8",
    surface2="#FFFFFF",
    sunken="#D5D9D7",
    field="#FFFFFF",
    fg="#191C1E",
    fg2="#565D60",
    fg3="#848B8E",
    hair="#1A000000",
    hair2="#2E000000",
    mats=("#1F7A70", "#4E58C0", "#B2542C", "#4F6B1F", "#8B4694", "#7A5D08"),
    ok="#2E7D45",
    warn="#8A6B0C",
    bad="#B23A2C",
    sel="#0E000000",
    shade="#33000000",
    lift="#21000000",
)

DARK = Tokens(
    win="#1B1E20",
    surface="#24272A",
    surface2="#2E3236",
    sunken="#141618",
    field="#171A1C",
    fg="#E8EAE9",
    fg2="#9AA1A4",
    fg3="#697073",
    hair="#17FFFFFF",
    hair2="#29FFFFFF",
    mats=("#57B0A6", "#8A93E0", "#D8825C", "#9DBB5E", "#BB7FC0", "#D0A93A"),
    ok="#63B274",
    warn="#D2A03B",
    bad="#DE6D5E",
    sel="#0EFFFFFF",
    shade="#6B000000",
    lift="#4D000000",
)

# macOS reports its accent through AppKit, which nothing here can read: this is the
# system default, which is what it is until somebody changes it in System Settings.
ACCENT = "#0A84FF"
ON_ACCENT = "#FFFFFF"

_current = LIGHT


def use(dark: bool) -> None:
    global _current
    _current = DARK if dark else LIGHT


def t() -> Tokens:
    return _current


# ── faces ─────────────────────────────────────────────────────────────────
# The system's own, which is what a native window is set in. Flutter bundles Roboto and
# will use it for anything unnamed, so every style below names a family.

SANS = "SF Pro Text"
SANS_FALLBACK = [".AppleSystemUIFont", "Helvetica Neue"]
MONO = "SF Mono"
MONO_FALLBACK = ["Menlo", "Monaco"]

BODY = 12.0
LINE = 1.45


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
        # -.004em on the body face, the only tracking the stylesheet sets globally.
        letter_spacing=spacing if spacing is not None else -0.004 * size,
    )


def mono(
    size: float = 11.0,
    color: str | None = None,
    weight: ft.FontWeight = ft.FontWeight.W_400,
    spacing: float | None = None,
) -> ft.TextStyle:
    # SF Mono is monospaced, so its figures are already tabular: Flutter's TextStyle has
    # no font-feature knob, and every measure in this app is set in the mono face.
    return ft.TextStyle(
        font_family=MONO,
        font_family_fallback=MONO_FALLBACK,
        size=size,
        weight=weight,
        color=color if color is not None else t().fg,
        letter_spacing=spacing,
    )


def eyebrow(text: str) -> ft.Text:
    """.eyebrow / .grouplab: 9px mono, uppercase, .11em."""
    return ft.Text(
        text.upper(),
        style=mono(9.0, t().fg3, spacing=0.11 * 9.0),
        no_wrap=True,
    )


# ── shared shapes ─────────────────────────────────────────────────────────


# align-items: baseline. Flutter's CrossAxisAlignment.baseline needs a textBaseline that
# Flet does not expose, and a Row that asks for it fails to lay out and takes the whole
# subtree down with it — silently, since a layout assertion never reaches Python. Bottom
# alignment is what stands in; it differs from a real baseline only by the descender.
BASELINE = ft.CrossAxisAlignment.END


def hair(color: str | None = None, width: float = 1.0) -> ft.Border:
    return ft.Border.all(width, color if color is not None else t().hair)


def lift() -> ft.BoxShadow:
    """--lift: 0 1px 3px."""
    return ft.BoxShadow(offset=ft.Offset(0, 1), blur_radius=3, color=t().lift)


def hatch(
    color: str,
    period: float = 6.0,
    width: float = 100.0,
    base: str = TRANSPARENT,
    duty: float = 0.5,
) -> ft.LinearGradient:
    """repeating-linear-gradient(135deg, C 0 Npx, base Npx periodpx).

    Flutter repeats a gradient over its own begin/end span rather than a px period, so
    the span is expressed as a fraction of the box it paints. A Flutter decoration may
    not carry a colour and a gradient at once, which is why the second colour is the
    gradient's and never the container's `bgcolor`.
    """
    step = period / max(width, 1.0)
    return ft.LinearGradient(
        begin=ft.Alignment(-1, -1),
        end=ft.Alignment(-1 + 2 * step, -1 + 2 * step),
        colors=[color, color, base, base],
        stops=[0.0, duty, duty, 1.0],
        tile_mode=ft.GradientTileMode.REPEATED,
    )

// The Quieta tokens, transcribed from `mlx_omnia/appv2/theme.py` — same strings, same
// order, so a change there is a diff here and not a redesign. The colour arithmetic is
// the same too: mixing happens in oklab, and mixing into transparent is a scaled alpha,
// which is the shortcut the browser takes and the one the window takes.

import SwiftUI

struct Tokens {
    /// Which of the two sets this is. Every colour below already answers it, but a palette
    /// handed to something outside this file — a syntax theme, say — needs to ask.
    let dark: Bool
    let win: Color
    let side: Color
    let surface: Color
    let elev: Color
    let field: Color
    let fg: Color
    let fg2: Color
    let fg3: Color
    let hair: Color
    let hair2: Color
    let accent: Color
    let accentSoft: Color
    let onAccent: Color
    let ok: Color
    let warn: Color
    let bad: Color
    let sel: Color
    let shade: Color
    let mats: [Color]

    /// The material a model wears, handed out by load order — the v1 rule.
    func mat(_ material: Int) -> Color {
        mats[((material % mats.count) + mats.count) % mats.count]
    }

    static let light = Tokens(
        dark: false,
        win: Color(hex: "#EDEFEE"),
        side: Color(hex: "#E6E9E7"),
        surface: Color(hex: "#FAFBFA"),
        elev: Color(hex: "#FFFFFF"),
        field: Color(hex: "#FFFFFF"),
        fg: Color(hex: "#1A1D1C"),
        fg2: Color(hex: "#5B635F"),
        fg3: Color(hex: "#8B938F"),
        hair: Color(hex: "#1F141C19"),
        hair2: Color(hex: "#38141C19"),
        accent: Color(hex: "#2F8C7E"),
        accentSoft: Color(hex: "#212F8C7E"),
        onAccent: Color(hex: "#FFFFFF"),
        ok: Color(hex: "#2E7D45"),
        warn: Color(hex: "#8A6B0C"),
        bad: Color(hex: "#B23A2C"),
        sel: Color(hex: "#0E000000"),
        shade: Color(hex: "#33000000"),
        mats: ["#1F7A70", "#4E58C0", "#B2542C", "#4F6B1F", "#8B4694", "#7A5D08"].map(Color.init(hex:))
    )

    static let dark = Tokens(
        dark: true,
        win: Color(hex: "#16191A"),
        side: Color(hex: "#1B1F20"),
        surface: Color(hex: "#1E2224"),
        elev: Color(hex: "#262B2D"),
        field: Color(hex: "#171B1C"),
        fg: Color(hex: "#ECEEED"),
        fg2: Color(hex: "#9BA29E"),
        fg3: Color(hex: "#6B736F"),
        hair: Color(hex: "#1AEBF5F0"),
        hair2: Color(hex: "#33EBF5F0"),
        accent: Color(hex: "#66C2B3"),
        accentSoft: Color(hex: "#2666C2B3"),
        onAccent: Color(hex: "#10312B"),
        ok: Color(hex: "#63B274"),
        warn: Color(hex: "#D2A03B"),
        bad: Color(hex: "#DE6D5E"),
        sel: Color(hex: "#0EFFFFFF"),
        shade: Color(hex: "#6B000000"),
        mats: ["#57B0A6", "#8A93E0", "#D8825C", "#9DBB5E", "#BB7FC0", "#D0A93A"].map(Color.init(hex:))
    )
}

// ── the colour arithmetic ────────────────────────────────────────────────

struct RGBA {
    var alpha: Double
    var red: Double
    var green: Double
    var blue: Double

    /// `#RRGGBB` or Flet's `#AARRGGBB`, which is what the tokens are written in.
    init(hex: String) {
        var text = hex
        if text.hasPrefix("#") { text.removeFirst() }
        let value = UInt64(text, radix: 16) ?? 0
        if text.count == 8 {
            alpha = Double((value >> 24) & 0xFF) / 255
            red = Double((value >> 16) & 0xFF) / 255
            green = Double((value >> 8) & 0xFF) / 255
            blue = Double(value & 0xFF) / 255
        } else {
            alpha = 1
            red = Double((value >> 16) & 0xFF) / 255
            green = Double((value >> 8) & 0xFF) / 255
            blue = Double(value & 0xFF) / 255
        }
    }

    init(alpha: Double, red: Double, green: Double, blue: Double) {
        self.alpha = alpha
        self.red = red
        self.green = green
        self.blue = blue
    }

    var color: Color { Color(.sRGB, red: red, green: green, blue: blue, opacity: alpha) }

    /// `color-mix(in oklab, self share%, into)`, share in 0...1.
    func mixed(_ share: Double, into: RGBA) -> RGBA {
        let (l1, a1, b1) = RGBA.oklab(red, green, blue)
        let (l2, a2, b2) = RGBA.oklab(into.red, into.green, into.blue)
        let rest = 1 - share
        let (r, g, b) = RGBA.srgb(
            l1 * share + l2 * rest, a1 * share + a2 * rest, b1 * share + b2 * rest
        )
        return RGBA(alpha: alpha * share + into.alpha * rest, red: r, green: g, blue: b)
    }

    private static func linear(_ value: Double) -> Double {
        value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
    }

    private static func gamma(_ value: Double) -> Double {
        let clamped = min(max(value, 0), 1)
        return clamped <= 0.0031308
            ? clamped * 12.92 : 1.055 * pow(clamped, 1 / 2.4) - 0.055
    }

    static func oklab(_ red: Double, _ green: Double, _ blue: Double) -> (Double, Double, Double) {
        let r = linear(red), g = linear(green), b = linear(blue)
        let l = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
        let m = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
        let s = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
        return (
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
        )
    }

    static func srgb(_ lightness: Double, _ a: Double, _ b: Double) -> (Double, Double, Double) {
        let l = pow(lightness + 0.3963377774 * a + 0.2158037573 * b, 3)
        let m = pow(lightness - 0.1055613458 * a - 0.0638541728 * b, 3)
        let s = pow(lightness - 0.0894841775 * a - 1.2914855480 * b, 3)
        return (
            gamma(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
            gamma(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
            gamma(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
        )
    }
}

extension Color {
    init(hex: String) { self = RGBA(hex: hex).color }
}

/// `theme.mix(color, share, into)` for the two tints the panel uses — a pressed face and
/// a hovered one — with the hex strings the tokens are written in.
func mix(_ color: String, _ share: Double, into: String) -> Color {
    RGBA(hex: color).mixed(share, into: RGBA(hex: into)).color
}

// ── the faces ────────────────────────────────────────────────────────────
// The system's own, as in the window. Tracking is the window's rule: -0.4% of the size
// for text, -1.5% for the display face, and the eyebrow opens up by 11%.

extension Text {
    func sans(_ size: CGFloat, _ color: Color, weight: Font.Weight = .regular) -> Text {
        font(.system(size: size, weight: weight))
            .tracking(-0.004 * size)
            .foregroundColor(color)
    }

    func display(_ size: CGFloat, _ color: Color, weight: Font.Weight = .semibold) -> Text {
        font(.system(size: size, weight: weight))
            .tracking(-0.015 * size)
            .foregroundColor(color)
    }

    func mono(_ size: CGFloat, _ color: Color, weight: Font.Weight = .regular) -> Text {
        font(.system(size: size, weight: weight, design: .monospaced))
            .foregroundColor(color)
    }
}

private struct TokensKey: EnvironmentKey {
    static let defaultValue = Tokens.light
}

extension EnvironmentValues {
    var tokens: Tokens {
        get { self[TokensKey.self] }
        set { self[TokensKey.self] = newValue }
    }
}

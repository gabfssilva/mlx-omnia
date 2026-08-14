// The Quieta controls at the panel's scale: cards, pills, chips, push buttons, the
// segmented control, tracks, the search box, facts, the dot and the pulse.
//
// Nothing here is a stock AppKit control with its own idea of a face — the window's parts
// are Containers with the proposal's own measures, and so are these.

import SwiftUI

// ── the card, and the pressable face ─────────────────────────────────────

struct Card<Content: View>: View {
    @Environment(\.tokens) private var t
    var tint: Color?
    var stroke: Color?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) { content }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(tint ?? t.elev)
            .overlay(
                RoundedRectangle(cornerRadius: 12).strokeBorder(stroke ?? t.hair, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

/// The window's `parts.press`: which of the three phases the pointer is in, spent on the
/// face's tint. A row with nothing to open takes no tint and no cursor.
struct Pressable<Content: View>: View {
    @Environment(\.tokens) private var t
    var stroke: Color?
    var action: (() -> Void)?
    @ViewBuilder var content: Content

    @State private var hovering = false
    @State private var pressing = false

    var body: some View {
        let tint: Color? = action == nil
            ? t.elev
            : (pressing ? t.sel : (hovering ? mix("#1A1D1C", 0.03, into: "#FFFFFF") : t.elev))
        Card(tint: tint, stroke: stroke) { content }
            .contentShape(Rectangle())
            .onHover { inside in if action != nil { hovering = inside } }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in if action != nil { pressing = true } }
                    .onEnded { _ in
                        pressing = false
                        action?()
                    }
            )
    }
}

// ── state worn small ─────────────────────────────────────────────────────

struct PillView: View {
    @Environment(\.tokens) private var t
    let label: String
    var kind: String = "disk"

    var body: some View {
        let face: Color = kind == "resident" ? t.accent : (kind == "hub" ? t.fg3 : t.fg2)
        let ground: Color? = kind == "resident" ? t.accentSoft : (kind == "hub" ? nil : t.sel)
        Text(label).sans(11.5, face, weight: .medium)
            .lineLimit(1)
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .background(ground ?? .clear)
            .overlay(
                Capsule().strokeBorder(kind == "hub" ? t.hair : .clear, lineWidth: 1)
            )
            .clipShape(Capsule())
    }
}

/// A quiet measure worn inline — the decode chip on a model's row.
struct ChipView: View {
    @Environment(\.tokens) private var t
    let text: String

    var body: some View {
        Text(text).mono(11, t.fg2)
            .lineLimit(1)
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(t.sel)
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

/// The verdict pill of the Bench rows: the accent when there is a number, a dash when the
/// axis was never measured.
struct VerdictView: View {
    @Environment(\.tokens) private var t
    let text: String?

    var body: some View {
        Text(text ?? "—").sans(10.5, text == nil ? t.fg3 : t.accent, weight: .medium)
            .lineLimit(1)
            .padding(.horizontal, 9)
            .padding(.vertical, 2)
            .background(text == nil ? t.sel : t.accentSoft)
            .clipShape(Capsule())
    }
}

// ── push button, at the new scale: height 30, radius 8, 13 pt ────────────

struct PushButton: View {
    @Environment(\.tokens) private var t
    let label: String
    var kind: String = ""
    var wide = false
    var action: (() -> Void)?

    @State private var pressing = false

    var body: some View {
        let primary = kind.contains("primary")
        let danger = kind.contains("danger")
        let face: Color = primary ? t.onAccent : (danger ? t.bad : t.fg)
        let resting: Color = primary ? t.accent : t.elev
        let pressed: Color = primary
            ? mix("#000000", 0.12, into: "#2F8C7E")
            : mix("#1A1D1C", 0.09, into: "#FFFFFF")
        Text(label).sans(13, face, weight: primary ? .semibold : .medium)
            .lineLimit(1)
            .frame(maxWidth: wide ? .infinity : nil)
            .frame(height: 30)
            .padding(.horizontal, 14)
            .background(pressing ? pressed : resting)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(primary ? .clear : t.hair2, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .opacity(action == nil ? 0.45 : 1)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in if action != nil { pressing = true } }
                    .onEnded { _ in
                        pressing = false
                        action?()
                    }
            )
    }
}

// ── the segmented control ────────────────────────────────────────────────

/// 12 pt and no horizontal padding, so five methods still fit across 352. A key the
/// current selection refuses stays on the list — the rule is worth seeing — but takes no
/// click.
struct Segmented<Key: Hashable>: View {
    @Environment(\.tokens) private var t
    let options: [(Key, String)]
    let chosen: Key
    var dimmed: Set<Key> = []
    var size: CGFloat = 12
    var status: ((Key) -> (state: String, text: String)?)?
    let pick: (Key) -> Void

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options, id: \.0) { key, label in
                let on = key == chosen
                let dim = dimmed.contains(key) && !on
                HStack(spacing: 5) {
                    if let status = status?(key) {
                        DotView(state: status.state)
                    }
                    Text(label)
                        .sans(
                            size, on ? t.fg : (dim ? t.fg3 : t.fg2),
                            weight: on ? .medium : .regular
                        )
                    if let status = status?(key) {
                        Text(status.text)
                            .mono(size - 1, on ? t.fg2 : t.fg3)
                            .padding(.leading, 3)
                    }
                }
                    .lineLimit(1)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                    // Room around the word, which a control sized to its own text has to
                    // carry itself: the tabs get theirs from the width they share.
                    .padding(.horizontal, 9)
                    .background(
                        RoundedRectangle(cornerRadius: 7)
                            .fill(on ? t.elev : .clear)
                            .shadow(color: on ? .black.opacity(0.16) : .clear, radius: 1, y: 1)
                    )
                    .opacity(dim ? 0.55 : 1)
                    .contentShape(Rectangle())
                    .onTapGesture { if !dim { pick(key) } }
            }
        }
        .padding(2)
        .background(t.sel)
        .clipShape(RoundedRectangle(cornerRadius: 9))
    }
}

// ── tracks ───────────────────────────────────────────────────────────────

struct TrackBar: View {
    @Environment(\.tokens) private var t
    let fraction: Double
    var height: CGFloat = 3

    var body: some View {
        GeometryReader { space in
            ZStack(alignment: .leading) {
                Rectangle().fill(t.sel)
                Rectangle().fill(t.accent)
                    .frame(width: max(0, min(1, fraction)) * space.size.width)
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: height / 2 + 0.5))
    }
}

/// A value dragged along a line: the tuning knobs, and the memory ceiling. `settle` is the
/// gesture ending, which is where a setting that costs a round trip is written — dragging
/// one that PATCHed per pixel would send a hundred requests to reach one number.
struct Rail: View {
    @Environment(\.tokens) private var t
    let fraction: Double?
    var settle: ((Double) -> Void)?
    let slide: (Double) -> Void

    var body: some View {
        GeometryReader { space in
            let width = space.size.width
            let part = min(max(fraction ?? 0, 0), 1)
            ZStack(alignment: .leading) {
                Capsule().fill(t.sel).frame(height: 3)
                Capsule().fill(t.accent).frame(width: part * width, height: 3)
                Circle()
                    .fill(t.elev)
                    .overlay(Circle().strokeBorder(t.hair2, lineWidth: 1))
                    .shadow(color: .black.opacity(0.18), radius: 1, y: 1)
                    .frame(width: 12, height: 12)
                    .offset(x: part * width - 6)
                    .opacity(fraction == nil ? 0.4 : 1)
            }
            .frame(height: 12)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { touch in slide(min(max(touch.location.x / width, 0), 1)) }
                    .onEnded { touch in settle?(min(max(touch.location.x / width, 0), 1)) }
            )
        }
        .frame(height: 12)
    }
}

/// Indeterminate on purpose: the daemon reports nothing until it answers, and a fraction
/// invented here would be a number about nothing.
struct Waiting: View {
    @Environment(\.tokens) private var t
    @State private var travel = false

    var body: some View {
        GeometryReader { space in
            let width = space.size.width
            ZStack(alignment: .leading) {
                Rectangle().fill(t.sel)
                Rectangle().fill(t.accent)
                    .frame(width: width * 0.32)
                    .offset(x: travel ? width * 0.68 : 0)
                    .animation(
                        .easeInOut(duration: 1.1).repeatForever(autoreverses: true), value: travel
                    )
            }
            .onAppear { travel = true }
        }
        .frame(height: 3)
        .clipShape(RoundedRectangle(cornerRadius: 2))
    }
}

// ── the field and the search box ─────────────────────────────────────────

struct MonoField: View {
    @Environment(\.tokens) private var t
    let hint: String
    @Binding var text: String

    var body: some View {
        TextField(hint, text: $text)
            .textFieldStyle(.plain)
            .font(.system(size: 12, design: .monospaced))
            .foregroundColor(t.fg)
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(t.field)
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(t.hair2, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// The v1 search box, one size up: a field with a magnifier, 32 pt tall.
struct SearchBox: View {
    @Environment(\.tokens) private var t
    let placeholder: String
    @Binding var text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(t.fg3)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundColor(t.fg)
        }
        .padding(.horizontal, 11)
        .frame(height: 32)
        .background(t.field)
        .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(t.hair2, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 9))
    }
}

// ── the small marks ──────────────────────────────────────────────────────

struct Eyebrow: View {
    @Environment(\.tokens) private var t
    let text: String
    var color: Color?

    var body: some View {
        Text(text.uppercased())
            .mono(10, color ?? t.fg3)
            .tracking(1.1)
            .lineLimit(1)
    }
}

/// An eyebrow over a measure — the key/value of the proposal.
struct FactView: View {
    @Environment(\.tokens) private var t
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Eyebrow(text: label)
            Text(value).mono(13, t.fg).lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct DotView: View {
    @Environment(\.tokens) private var t
    var state: String = "ok"

    var body: some View {
        Circle()
            .fill(state == "bad" ? t.bad : (state == "warn" ? t.warn : t.ok))
            .frame(width: 7, height: 7)
    }
}

/// This model's share of its ceiling over the last minute; the dot is now. The Server
/// card's pulse at the panel's scale.
struct Sparkline: View {
    @Environment(\.tokens) private var t
    let values: [Double]
    var size = CGSize(width: 62, height: 18)

    var body: some View {
        Canvas { context, area in
            guard values.count >= 2 else { return }
            let step = area.width / CGFloat(EngineStore.traceLength - 1)
            let start = EngineStore.traceLength - values.count
            let points = values.enumerated().map { index, value in
                CGPoint(
                    x: CGFloat(start + index) * step,
                    y: 3 + (area.height - 8) * (1 - min(value, 100) / 100)
                )
            }
            var line = Path()
            line.move(to: points[0])
            for point in points.dropFirst() { line.addLine(to: point) }
            context.stroke(
                line,
                with: .color(t.accent),
                style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round)
            )
            if let last = points.last {
                context.fill(
                    Path(ellipseIn: CGRect(x: last.x - 2.4, y: last.y - 2.4, width: 4.8, height: 4.8)),
                    with: .color(t.accent)
                )
            }
        }
        .frame(width: size.width, height: size.height)
    }
}

/// The material a resident model wears, as the stripe down the left of its row.
struct Material: View {
    @Environment(\.tokens) private var t
    let color: Color
    var height: CGFloat = 32
    var width: CGFloat = 4

    var body: some View {
        RoundedRectangle(cornerRadius: 2).fill(color).frame(width: width, height: height)
    }
}

/// A section eyebrow with the space the shelf gives it.
struct SectionHead: View {
    let title: String

    var body: some View {
        Eyebrow(text: title)
            .padding(.horizontal, 2)
            .padding(.top, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct EmptyNote: View {
    @Environment(\.tokens) private var t
    let message: String

    var body: some View {
        Text(message).sans(12.5, t.fg3)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.top, 26)
            .padding(.horizontal, 2)
    }
}

/// A row of the settings kind: a label, and what it is set to on the right.
struct SettingRow<Value: View>: View {
    @Environment(\.tokens) private var t
    let label: String
    var dim = false
    var first = false
    @ViewBuilder var value: Value

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Text(label).sans(13, dim ? t.fg3 : t.fg).lineLimit(1)
            Spacer(minLength: 0)
            value
        }
        .padding(.horizontal, 2)
        .padding(.vertical, 9)
        .overlay(alignment: .top) {
            if !first { Rectangle().fill(t.hair).frame(height: 1) }
        }
    }
}

/// The card the settings rows sit in. Its own container rather than a `Card`, because the
/// rows carry the vertical rhythm and the hairlines between them.
struct SettingsGroup<Content: View>: View {
    @Environment(\.tokens) private var t
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) { content }
            .padding(.horizontal, 12)
            .padding(.top, 2)
            .padding(.bottom, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(t.elev)
            .overlay(RoundedRectangle(cornerRadius: 11).strokeBorder(t.hair, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 11))
    }
}

/// What a setting costs or is decided by, under the row it belongs to. The rule for writing
/// one: it says what happens, never how it works.
struct SettingNote: View {
    @Environment(\.tokens) private var t
    let text: String

    var body: some View {
        Text(text).sans(10.5, t.fg3)
            .lineSpacing(2)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 2)
            .padding(.bottom, 8)
            .padding(.top, -2)
    }
}

/// − and +, for a number that moves by a known step.
struct Nudge: View {
    @Environment(\.tokens) private var t
    let down: (() -> Void)?
    let up: (() -> Void)?

    var body: some View {
        HStack(spacing: 1) {
            half("−", down, corners: [.topLeft, .bottomLeft])
            half("+", up, corners: [.topRight, .bottomRight])
        }
    }

    private func half(
        _ glyph: String, _ action: (() -> Void)?, corners: RectCorner
    ) -> some View {
        Text(glyph).sans(11, action == nil ? t.fg3 : t.fg2)
            .frame(width: 21, height: 20)
            .background(t.sel)
            .clipShape(Corners(radius: 5, corners: corners))
            .opacity(action == nil ? 0.5 : 1)
            .contentShape(Rectangle())
            .onTapGesture { action?() }
    }
}

struct RectCorner: OptionSet {
    let rawValue: Int
    static let topLeft = RectCorner(rawValue: 1)
    static let topRight = RectCorner(rawValue: 2)
    static let bottomRight = RectCorner(rawValue: 4)
    static let bottomLeft = RectCorner(rawValue: 8)
}

/// A rounded rectangle that rounds only the corners it was given — what the two halves of
/// `Nudge` need to read as one control.
struct Corners: Shape {
    let radius: CGFloat
    let corners: RectCorner

    func path(in rect: CGRect) -> Path {
        var made = Path()
        let topLeft = corners.contains(.topLeft) ? radius : 0
        let topRight = corners.contains(.topRight) ? radius : 0
        let bottomRight = corners.contains(.bottomRight) ? radius : 0
        let bottomLeft = corners.contains(.bottomLeft) ? radius : 0
        made.move(to: CGPoint(x: rect.minX + topLeft, y: rect.minY))
        made.addLine(to: CGPoint(x: rect.maxX - topRight, y: rect.minY))
        made.addArc(
            center: CGPoint(x: rect.maxX - topRight, y: rect.minY + topRight),
            radius: topRight, startAngle: .degrees(-90), endAngle: .degrees(0), clockwise: false
        )
        made.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - bottomRight))
        made.addArc(
            center: CGPoint(x: rect.maxX - bottomRight, y: rect.maxY - bottomRight),
            radius: bottomRight, startAngle: .degrees(0), endAngle: .degrees(90), clockwise: false
        )
        made.addLine(to: CGPoint(x: rect.minX + bottomLeft, y: rect.maxY))
        made.addArc(
            center: CGPoint(x: rect.minX + bottomLeft, y: rect.maxY - bottomLeft),
            radius: bottomLeft, startAngle: .degrees(90), endAngle: .degrees(180), clockwise: false
        )
        made.addLine(to: CGPoint(x: rect.minX, y: rect.minY + topLeft))
        made.addArc(
            center: CGPoint(x: rect.minX + topLeft, y: rect.minY + topLeft),
            radius: topLeft, startAngle: .degrees(180), endAngle: .degrees(270), clockwise: false
        )
        made.closeSubpath()
        return made
    }
}

/// On or off, where off is a value the daemon means — never, rather than zero.
struct Latch: View {
    @Environment(\.tokens) private var t
    let on: Bool
    let flip: () -> Void

    var body: some View {
        ZStack(alignment: on ? .trailing : .leading) {
            Capsule().fill(on ? t.accent : t.sel)
            Circle().fill(.white)
                .shadow(color: .black.opacity(0.25), radius: 1, y: 1)
                .frame(width: 15, height: 15)
                .padding(2)
        }
        .frame(width: 32, height: 19)
        .contentShape(Rectangle())
        .onTapGesture(perform: flip)
    }
}

/// A value the daemon reports and this panel does not set. Dashed, so the field says so
/// before it is clicked.
struct InertField: View {
    @Environment(\.tokens) private var t
    let text: String

    var body: some View {
        Text(text).mono(11.5, t.fg2)
            .lineLimit(1).truncationMode(.middle)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(t.field)
            .overlay(
                RoundedRectangle(cornerRadius: 8).strokeBorder(
                    t.hair2, style: StrokeStyle(lineWidth: 1, dash: [3, 2])
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// A verb worn as text — the settings' own small action, where a push button would weigh
/// more than what it does.
struct ActionText: View {
    @Environment(\.tokens) private var t
    let label: String
    var tint: Color?
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Text(label).sans(11.5, tint ?? t.accent, weight: .semibold)
            .lineLimit(1)
            .opacity(hovering ? 0.72 : 1)
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .onTapGesture(perform: action)
    }
}

// The checkpoint's own card: what the repository says about itself, on either side of a
// download.
//
// A card opens with YAML, and that half is not prose — it is the licence, what this was
// made from, and what it is for, which is most of what anybody wants before deciding to
// fetch 7 GB. It is read as chips and the rest is left to the renderer.
//
// The prose half has no predictable size: 125 bytes for a conversion that is all front
// matter, 79 kB for a card with every benchmark table in it. So it is folded at a ceiling
// and opens on a word.

import SwiftUI

/// A YAML subset: scalars and lists of scalars, which is what a card's front matter is
/// made of. A key with a nested shape under it (`widget`) is read and never shown — the
/// chips are an allowlist, so what this parser cannot shape has nowhere to leak into.
struct Frontmatter {
    var scalars: [String: String] = [:]
    var lists: [String: [String]] = [:]

    func first(_ key: String) -> String? {
        scalars[key] ?? lists[key]?.first
    }
}

enum Front {
    static func split(_ raw: String) -> (Frontmatter?, String) {
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false)
        guard let opening = lines.first, opening.trimmingCharacters(in: .whitespaces) == "---"
        else { return (nil, raw) }
        guard let close = lines.dropFirst().firstIndex(where: {
            $0.trimmingCharacters(in: .whitespaces) == "---"
        }) else { return (nil, raw) }

        var front = Frontmatter()
        var listing: String?
        for line in lines[1..<close] {
            if let (key, value) = keyed(String(line)) {
                if value.isEmpty {
                    listing = key
                } else {
                    listing = nil
                    front.scalars[key] = value
                }
                continue
            }
            if let listing, let item = itemed(String(line)) {
                front.lists[listing, default: []].append(item)
            }
        }
        return (front, lines[(close + 1)...].joined(separator: "\n"))
    }

    private static func keyed(_ line: String) -> (String, String)? {
        guard let colon = line.firstIndex(of: ":"), line.first?.isWhitespace == false else {
            return nil
        }
        let key = String(line[line.startIndex..<colon])
        guard !key.isEmpty, key.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" })
        else { return nil }
        return (key, unquoted(String(line[line.index(after: colon)...])))
    }

    private static func itemed(_ line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("- ") else { return nil }
        let value = unquoted(String(trimmed.dropFirst(2)))
        // A nested mapping under a list item is a shape the chips have no room for.
        return value.contains(":") ? nil : value
    }

    private static func unquoted(_ text: String) -> String {
        var value = text.trimmingCharacters(in: .whitespaces)
        for quote in ["\"", "'"] where value.hasPrefix(quote) && value.hasSuffix(quote) {
            value = String(value.dropFirst().dropLast())
        }
        return value
    }
}

/// The front matter, worn small. The order is what a reader checks first: what it costs to
/// use, what it was made from, and what it is for.
struct FrontChips: View {
    @Environment(\.tokens) private var t
    let front: Frontmatter

    /// Enough tags to say what kind of thing this is. The rest is a count — the Phi-4 card
    /// lists twenty-four languages, and twenty-four chips is a screen.
    private static let shown = 5

    var body: some View {
        FlowRow(spacing: 4, trailing: false) {
            if let license = front.first("license") {
                chip("license", license, tint: t.ok)
            }
            if let base = front.first("base_model") {
                let relation = front.first("base_model_relation")
                chip("base", relation.map { "\(base) · \($0)" } ?? base, tint: t.accent)
            }
            if let library = front.first("library_name") {
                chip("library", library)
            }
            if let pipeline = front.first("pipeline_tag") {
                chip(nil, pipeline)
            }
            if let languages = front.lists["language"], let first = languages.first {
                chip("language", languages.count > 1 ? "\(first) +\(languages.count - 1)" : first)
            }
            let tags = front.lists["tags"] ?? []
            ForEach(Array(tags.prefix(Self.shown).enumerated()), id: \.offset) { _, tag in
                chip(nil, tag)
            }
            if tags.count > Self.shown {
                chip(nil, "+\(tags.count - Self.shown)")
            }
        }
    }

    private func chip(_ label: String?, _ value: String, tint: Color? = nil) -> some View {
        HStack(spacing: 5) {
            if let label { Text(label).mono(9.5, t.fg3) }
            Text(value).mono(9.5, tint ?? t.fg2)
        }
        .lineLimit(1)
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .overlay(Capsule().strokeBorder(t.hair2, lineWidth: 1))
        .clipShape(Capsule())
    }
}

/// Content with a ceiling, and the word that lifts it. Nothing is folded that fits: the
/// fade and the verb appear only when something is actually under them.
struct Clamped<Content: View>: View {
    @Environment(\.tokens) private var t
    var ceiling: CGFloat = 430
    @ViewBuilder var content: Content

    @State private var full: CGFloat = 0
    @State private var open = false

    var body: some View {
        let over = full > ceiling + 1
        let folded = over && !open
        VStack(alignment: .leading, spacing: 8) {
            content
                .fixedSize(horizontal: false, vertical: true)
                .background(
                    GeometryReader { space in
                        Color.clear.preference(key: FullHeight.self, value: space.size.height)
                    }
                )
                .frame(maxHeight: folded ? ceiling : nil, alignment: .top)
                .clipped()
                .overlay(alignment: .bottom) {
                    if folded {
                        LinearGradient(
                            colors: [t.surface.opacity(0), t.surface],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                        .frame(height: 54)
                        .allowsHitTesting(false)
                    }
                }
            if over {
                ActionText(label: open ? "Show less" : "Show all") { open.toggle() }
            }
        }
        .onPreferenceChange(FullHeight.self) { full = $0 }
    }
}

private struct FullHeight: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

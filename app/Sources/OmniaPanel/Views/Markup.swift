// What a model writes and what a checkpoint ships with, drawn at 392 pt.
//
// One renderer for both, because they are the same markdown: a turn arriving from the engine
// and a README read off the disk differ in where they came from, not in what they are.
//
// Textual draws it — SwiftUI's own `Text` over Foundation's `AttributedString` parser, which
// is the parser this file already called for the inline half before it had a block splitter
// of its own to feed it. What that buys over the splitter is the part the splitter was never
// going to have: syntax highlighting in a fence, nested lists, and tables it does not have to
// recognise by their delimiter row.
//
// What stays here is the half no markdown renderer does — taking out what this panel will not
// draw. HTML and images are dropped: there is no webview here, so there is nothing to sanitise
// raw HTML with, and the images in a model card are almost all shields.io badges, a fetch each
// at a width where they would wrap to their own line. An HTML table is rewritten as the
// markdown one it means rather than dropped, because a card's benchmarks are usually in one.

import SwiftUI
import Textual

enum Markup {
    /// The document as this panel will draw it. The strippers run over everything outside a
    /// fence and nothing inside one: a tag in a code block is the subject being written about,
    /// not markup to take out.
    ///
    /// `breaking` keeps the newlines that markdown would fold. It is for what somebody typed
    /// and not for what a model wrote: a line opened with shift-return is a line that was
    /// meant, where a model's single newline is usually a soft wrap.
    static func readable(_ text: String, breaking: Bool = false) -> String {
        var drawn: [String] = []
        var held: [Substring] = []
        var fenced = false

        func flush() {
            guard !held.isEmpty else { return }
            drawn.append(stripped(tabled(held.joined(separator: "\n"))))
            held = []
        }

        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("```") {
                if !fenced { flush() }
                fenced.toggle()
                drawn.append(String(line))
                continue
            }
            if fenced {
                drawn.append(String(line))
            } else {
                held.append(line)
            }
        }
        flush()
        let made = drawn.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return breaking ? hardened(made) : made
    }

    /// Two spaces before a newline, which is how markdown is told the break was meant. Not
    /// inside a fence, where the characters are the subject, and not on the line before a
    /// blank one, which is already a paragraph break.
    private static func hardened(_ text: String) -> String {
        var drawn: [String] = []
        var fenced = false
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        for (offset, line) in lines.enumerated() {
            if line.hasPrefix("```") {
                fenced.toggle()
                drawn.append(String(line))
                continue
            }
            let next = offset + 1 < lines.count ? lines[offset + 1] : ""
            let soft = !fenced && !line.isEmpty && !next.isEmpty
                && !line.hasSuffix("  ") && !line.hasSuffix("\\")
            drawn.append(soft ? line + "  " : String(line))
        }
        return drawn.joined(separator: "\n")
    }

    // ── what the panel does not draw ─────────────────────────────────────

    private static let strippers: [NSRegularExpression] = [
        #"<!--[\s\S]*?-->"#,
        #"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>"#,
        #"!\[[^\]]*\]\([^)]*\)"#,
        // What a badge leaves behind. A card's top row is `[![alt](shield)](link)`, and the
        // link with its image taken out is a paragraph with nothing in it but height.
        #"\[\s*\]\([^)]*\)"#,
    ].compactMap { try? NSRegularExpression(pattern: $0) }

    private static let tables = regex(#"<table[\s\S]*?</table>"#)
    private static let rows = regex(#"<tr[\s\S]*?</tr>"#)
    private static let cells = regex(#"<t[hd][^>]*>([\s\S]*?)</t[hd]>"#)

    private static func regex(_ pattern: String) -> NSRegularExpression? {
        try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }

    private static func stripped(_ text: String) -> String {
        var made = text
        for stripper in strippers {
            made = stripper.stringByReplacingMatches(
                in: made, range: NSRange(made.startIndex..., in: made), withTemplate: ""
            )
        }
        return made
    }

    private static func clean(_ text: String) -> String {
        stripped(text).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A card that lays its benchmarks out in an HTML table, rewritten as the markdown one it
    /// means. Stripping the tags and keeping the text would leave every cell alone on its own
    /// line — a column of numbers with nothing to divide them by, which is worse than either
    /// drawing the table or dropping it.
    private static func tabled(_ text: String) -> String {
        guard let tables else { return text }
        var made = text
        let found = tables.matches(in: text, range: NSRange(text.startIndex..., in: text))
        // Last first, so every range still counted from the start of an untouched prefix.
        for match in found.reversed() {
            guard let read = Range(match.range, in: text),
                  let write = Range(match.range, in: made) else { continue }
            made.replaceSubrange(write, with: pipes(String(text[read])))
        }
        return made
    }

    private static func pipes(_ table: String) -> String {
        guard let rows, let cells else { return "" }
        var drawn: [String] = []
        for row in rows.matches(in: table, range: NSRange(table.startIndex..., in: table)) {
            guard let range = Range(row.range, in: table) else { continue }
            let line = String(table[range])
            let found = cells.matches(in: line, range: NSRange(line.startIndex..., in: line))
                .compactMap { Range($0.range(at: 1), in: line).map { flattened(String(line[$0])) } }
            guard !found.isEmpty else { continue }
            drawn.append("| " + found.joined(separator: " | ") + " |")
            if drawn.count == 1 {
                drawn.append("|" + String(repeating: "---|", count: found.count))
            }
        }
        return drawn.isEmpty ? "" : "\n\n" + drawn.joined(separator: "\n") + "\n\n"
    }

    /// One cell, as one line: its own markup taken out, its wrapping folded, and its pipes
    /// spent — a cell that carried one would be read as two.
    private static func flattened(_ cell: String) -> String {
        let text = clean(cell.replacingOccurrences(of: "|", with: "/"))
            .replacingOccurrences(of: "\n", with: " ")
        return text.split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ")
    }
}

// ── the source, marked ───────────────────────────────────────────────────
//
// What the composer draws, which is not what the bubble draws. A field that rendered the
// markdown would take the markup off the screen while it is still being written, and the
// character somebody is about to delete has to be visible to be deleted. So the delimiters
// stay where they were typed and go grey, and what they enclose takes the shape it will have
// once it is sent.

extension Markup {
    /// Every attribute on `text` replaced by what its own characters say it should be. The
    /// characters are never touched — this runs on a `transform(updating:)`, and an edit to
    /// the text there would be an edit under the reader's cursor.
    static func mark(_ text: inout AttributedString, _ t: Tokens, _ size: CGFloat) {
        var ground = AttributeContainer()
        ground.font = .system(size: size)
        ground.foregroundColor = t.fg
        text.setAttributes(ground)

        let chars = Array(text.characters)
        guard !chars.isEmpty else { return }
        // The offsets a scan works in, resolved to indices once: `index(afterCharacter:)` per
        // mark would make a linear scan quadratic.
        var spots: [AttributedString.Index] = []
        spots.reserveCapacity(chars.count + 1)
        var walk = text.startIndex
        while walk < text.endIndex {
            spots.append(walk)
            walk = text.index(afterCharacter: walk)
        }
        spots.append(text.endIndex)

        func paint(_ span: Range<Int>, _ apply: (inout AttributeContainer) -> Void) {
            guard span.lowerBound >= 0, span.upperBound < spots.count,
                  span.lowerBound < span.upperBound else { return }
            var held = AttributeContainer()
            apply(&held)
            text[spots[span.lowerBound]..<spots[span.upperBound]].mergeAttributes(held)
        }

        let dim: (inout AttributeContainer) -> Void = { $0.foregroundColor = t.fg3 }
        let quiet: (inout AttributeContainer) -> Void = { $0.foregroundColor = t.fg2 }
        let code: (inout AttributeContainer) -> Void = {
            $0.font = .system(size: size * 0.94, design: .monospaced)
            $0.foregroundColor = t.fg2
        }
        let strong: (inout AttributeContainer) -> Void = {
            $0.font = .system(size: size, weight: .semibold)
        }
        let slanted: (inout AttributeContainer) -> Void = {
            $0.font = .system(size: size).italic()
        }
        let linked: (inout AttributeContainer) -> Void = { $0.foregroundColor = t.accent }

        // ── inside one line ──────────────────────────────────────────────

        /// The next unescaped `mark`, doubled or single, before the end of the line.
        func closing(_ span: Range<Int>, _ mark: Character, doubled: Bool) -> Int? {
            var index = span.lowerBound
            while index < span.upperBound {
                if chars[index] == mark {
                    guard doubled else { return index }
                    if index + 1 < span.upperBound, chars[index + 1] == mark { return index }
                    index += 1
                    continue
                }
                index += 1
            }
            return nil
        }

        func inline(_ span: Range<Int>) {
            var index = span.lowerBound
            while index < span.upperBound {
                let here = chars[index]
                // A code span outranks the rest: `**` inside one is two asterisks.
                if here == "`", let close = closing((index + 1)..<span.upperBound, "`", doubled: false) {
                    paint(index..<(index + 1), dim)
                    paint((index + 1)..<close, code)
                    paint(close..<(close + 1), dim)
                    index = close + 1
                    continue
                }
                if here == "*" || here == "_" {
                    let twofold = index + 1 < span.upperBound && chars[index + 1] == here
                    let width = twofold ? 2 : 1
                    let from = index + width
                    if from < span.upperBound, chars[from] != " ",
                       let close = closing(from..<span.upperBound, here, doubled: twofold) {
                        paint(index..<from, dim)
                        paint(from..<close, twofold ? strong : slanted)
                        paint(close..<(close + width), dim)
                        index = close + width
                        continue
                    }
                }
                if here == "[", let shut = closing((index + 1)..<span.upperBound, "]", doubled: false),
                   shut + 1 < span.upperBound, chars[shut + 1] == "(",
                   let end = closing((shut + 2)..<span.upperBound, ")", doubled: false) {
                    paint(index..<(index + 1), dim)
                    paint((index + 1)..<shut, linked)
                    paint(shut..<(end + 1), dim)
                    index = end + 1
                    continue
                }
                index += 1
            }
        }

        // ── line by line ─────────────────────────────────────────────────

        var lines: [Range<Int>] = []
        var opened = 0
        for (offset, character) in chars.enumerated() where character == "\n" {
            lines.append(opened..<offset)
            opened = offset + 1
        }
        lines.append(opened..<chars.count)

        var fenced = false
        for span in lines {
            let lead = span.lowerBound + chars[span].prefix { $0 == " " }.count
            if span.upperBound - lead >= 3,
               chars[lead] == "`", chars[lead + 1] == "`", chars[lead + 2] == "`" {
                paint(span, dim)
                fenced.toggle()
                continue
            }
            if fenced {
                paint(span, code)
                continue
            }
            var body = span
            if let after = headed(chars, span, lead) {
                paint(span.lowerBound..<after, dim)
                paint(after..<span.upperBound, strong)
                body = after..<span.upperBound
            } else if let after = quoted(chars, span, lead) {
                paint(span.lowerBound..<after, dim)
                paint(after..<span.upperBound, quiet)
                body = after..<span.upperBound
            } else if let after = listed(chars, span, lead) {
                paint(span.lowerBound..<after, dim)
                body = after..<span.upperBound
            }
            inline(body)
        }
    }

    /// Where `### ` ends, when the line opens with one.
    private static func headed(_ chars: [Character], _ span: Range<Int>, _ lead: Int) -> Int? {
        var index = lead
        while index < span.upperBound, chars[index] == "#", index - lead < 6 { index += 1 }
        guard index > lead, index < span.upperBound, chars[index] == " " else { return nil }
        return index + 1
    }

    private static func quoted(_ chars: [Character], _ span: Range<Int>, _ lead: Int) -> Int? {
        guard lead < span.upperBound, chars[lead] == ">" else { return nil }
        return lead + 1 < span.upperBound && chars[lead + 1] == " " ? lead + 2 : lead + 1
    }

    /// Where `- ` or `12. ` ends, indent included.
    private static func listed(_ chars: [Character], _ span: Range<Int>, _ lead: Int) -> Int? {
        guard lead < span.upperBound else { return nil }
        if "-*+".contains(chars[lead]) {
            return lead + 1 < span.upperBound && chars[lead + 1] == " " ? lead + 2 : nil
        }
        var index = lead
        while index < span.upperBound, chars[index].isNumber { index += 1 }
        guard index > lead, index + 1 < span.upperBound,
              chars[index] == ".", chars[index + 1] == " " else { return nil }
        return index + 2
    }
}

// ── a document that is still arriving ────────────────────────────────────

extension Markup {
    /// The spacing the block styles below declare, font-relative. Named because a document cut
    /// in pieces has to put back the air between them, and a number spelled in two places is a
    /// gap that changes in one of them.
    ///
    /// Textual collapses the two sides the way CSS margins do — the gap between blocks is the
    /// larger of what the one above asks below itself and the one below asks above — so what a
    /// block asks below itself only ever matters where it is the larger of the two, which in
    /// this set is the rule alone.
    enum Gap {
        static let block: CGFloat = 0.64
        static let heading: CGFloat = 1.1
        /// Tables and rules, which stand further off what precedes them.
        static let apart: CGFloat = 0.8
    }

    /// Every place a document still being written can be cut, so that what is above a cut
    /// cannot change when the rest of it arrives.
    ///
    /// A markdown block ends at a blank line, so a block that starts after one is settled —
    /// with the three exceptions this takes out. A blank line inside a fence is content and
    /// not a boundary. A block that *continues* what is above it — a list item, a quote line,
    /// a table row — would be cut out of the construct it belongs to and drawn as its own: a
    /// list cut in two is two lists, and an ordered one starts again at 1. And the line the
    /// answer stops in the middle of is not read at all, because what it is is not settled
    /// either: `1` is a paragraph and `1.` is a list, and a boundary that was there for one
    /// token and gone the next would reopen a piece that had closed.
    static func boundaries(_ text: String) -> [String.Index] {
        var found: [String.Index] = []
        var fenced = false
        var blank = false
        var index = text.startIndex
        while index < text.endIndex {
            let end = text[index...].firstIndex(of: "\n") ?? text.endIndex
            guard end < text.endIndex else { break }
            let line = text[index..<end]
            if line.hasPrefix("```") {
                if fenced { blank = false } else if blank { found.append(index) }
                fenced.toggle()
            } else if fenced {
                blank = false
            } else if line.allSatisfy(\.isWhitespace) {
                blank = true
            } else {
                if blank, opens(line) { found.append(index) }
                blank = false
            }
            index = text.index(after: end)
        }
        return found
    }

    /// Whether a line starts a block rather than carrying one on. Four spaces or a tab is an
    /// indented code block or a nested item, and either way it belongs to what is above it.
    private static func opens(_ line: Substring) -> Bool {
        if line.hasPrefix("    ") || line.hasPrefix("\t") { return false }
        guard let first = line.first(where: { !$0.isWhitespace }) else { return false }
        if ruled(line) { return true }
        return !"-*+>|".contains(first) && !ordered(line)
    }

    /// A line of nothing but the same rule character, three or more: a thematic break, which
    /// is a block of its own and not a list item however much it looks like three bullets.
    private static func ruled(_ line: Substring) -> Bool {
        let marks = line.filter { !$0.isWhitespace }
        guard let mark = marks.first, marks.count >= 3 else { return false }
        return "-*_".contains(mark) && marks.allSatisfy { $0 == mark }
    }

    private static func ordered(_ line: Substring) -> Bool {
        let body = line.drop(while: \.isWhitespace)
        let digits = body.prefix(while: \.isNumber)
        guard !digits.isEmpty else { return false }
        let after = body[digits.endIndex...]
        return after.hasPrefix(".") || after.hasPrefix(")")
    }

    /// The air a document would have left between these two pieces, font-relative.
    static func gap(after previous: Substring?, before next: Substring) -> CGFloat {
        guard let previous else { return 0 }
        return max(under(previous), over(next))
    }

    /// What the block a piece ends with asks below itself. Only a rule asks for more than the
    /// block after it does, so only a rule is read here.
    private static func under(_ piece: Substring) -> CGFloat {
        let last = piece.split(separator: "\n", omittingEmptySubsequences: true).last ?? ""
        return ruled(last) ? Gap.apart : 0
    }

    /// What the block a piece starts with asks above itself.
    private static func over(_ piece: Substring) -> CGFloat {
        let lines = piece.split(separator: "\n", omittingEmptySubsequences: false)
        guard let first = lines.first else { return Gap.block }
        if first.hasPrefix("#") { return Gap.heading }
        if ruled(first) { return Gap.apart }
        // A table whose header row carries no leading pipe is still a table, and the delimiter
        // row under it is what says so.
        guard let beneath = lines.dropFirst().first, beneath.contains("-") else { return Gap.block }
        return beneath.allSatisfy { "-:| \t".contains($0) } ? Gap.apart : Gap.block
    }

    /// One piece of a document that is still arriving: where it starts — which is its identity,
    /// because a boundary chosen from the text before it does not move when more lands after —
    /// and the air that goes above it.
    struct Piece: Identifiable {
        let id: Int
        let text: String
        let gap: CGFloat
    }

    /// How long a settled piece grows before the next boundary closes it, in characters.
    ///
    /// The two costs it stands between: the open piece is redrawn whenever a block joins it,
    /// so short pieces are redrawn cheaply — and every closed piece is a view of its own, so
    /// short pieces mean many. Around a thousand characters that redraw is a few milliseconds,
    /// which is where making them shorter stops buying anything a reader could see.
    static let segment = 1200

    /// A document being written, as the pieces it should be drawn in: the settled ones, then
    /// the block still being written. `nil` when there is no safe boundary yet, which is a
    /// document drawn whole.
    ///
    /// Every piece but the last is closed and will not change again, so drawing the answer
    /// costs the block arriving rather than the answer so far.
    static func written(_ text: String) -> (settled: [Piece], writing: Piece)? {
        let found = boundaries(text)
        guard let last = found.last else { return nil }
        var settled: [Piece] = []
        var start = text.startIndex
        var at = 0
        for boundary in found where boundary < last {
            let length = text.distance(from: start, to: boundary)
            guard length >= segment else { continue }
            settled.append(piece(text, start..<boundary, at: at))
            start = boundary
            at += length
        }
        if start < last {
            settled.append(piece(text, start..<last, at: at))
            at += text.distance(from: start, to: last)
        }
        return (settled, piece(text, last..<text.endIndex, at: at))
    }

    private static func piece(_ text: String, _ range: Range<String.Index>, at: Int) -> Piece {
        Piece(
            id: at,
            text: String(text[range]),
            gap: gap(after: at == 0 ? nil : text[text.startIndex..<range.lowerBound], before: text[range])
        )
    }
}

// ── the drawing ──────────────────────────────────────────────────────────

struct Prose: View {
    @Environment(\.tokens) private var t
    let text: String
    var size: CGFloat = 12.5
    /// The colour the running text is set in. The reasoning is drawn a step back from the
    /// answer, and it is the same markdown either way.
    var tone: Color?
    /// Whether a newline in it was meant — see `Markup.readable(_:breaking:)`.
    var breaks = false
    /// Whether the rest of this text is still arriving.
    ///
    /// A turn being written is redrawn as often as the eye reads, and drawing a document costs
    /// what the document is long: 7 ms at forty tokens and 190 at fourteen hundred, measured
    /// through `ImageRenderer` at this width. Held as one view that is the answer so far, the
    /// panel therefore lays out the whole answer twenty-five times a second, which saturates
    /// the main thread from a few hundred tokens on — and the daemon decoding beside it pays
    /// for the contention, a token at a time, in the number it reports back.
    ///
    /// Drawn in pieces, what is settled is a view whose input did not change, so it is neither
    /// parsed nor laid out again; what is redrawn per frame is the block being written. The
    /// same answer measured both ways, over the whole of its stream: 6.1 seconds of drawing
    /// held as one document, 0.8 in pieces.
    var writing = false

    var body: some View {
        if writing, let (settled, open) = Markup.written(text) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(settled) { piece in document(piece.text).padding(.top, size * piece.gap) }
                document(open.text).padding(.top, size * open.gap)
            }
        } else {
            document(text)
        }
    }

    private func document(_ part: String) -> some View {
        StructuredText(markdown: Markup.readable(part, breaking: breaks))
            .font(.system(size: size))
            .foregroundStyle(tone ?? t.fg)
            .textual.structuredTextStyle(PanelStyle(t))
            .textual.highlighterTheme(PanelStyle.highlighting(t))
            .textual.textSelection(.enabled)
    }
}

/// The panel's own set. Every measurement here is font-relative, which is what lets the
/// reasoning be drawn at 11.5 and the answer at 12.5 off one style — the two differ by the
/// `.font()` over them and by nothing else.
struct PanelStyle: StructuredText.Style {
    let inlineStyle: InlineStyle
    let headingStyle = PanelHeadingStyle()
    let paragraphStyle = PanelParagraphStyle()
    let blockQuoteStyle: PanelBlockQuoteStyle
    let codeBlockStyle: PanelCodeBlockStyle
    let listItemStyle: StructuredText.DefaultListItemStyle = .default(markerSpacing: .fontScaled(0.56))
    let unorderedListMarker: StructuredText.HierarchicalSymbolListMarker = .hierarchical(
        .disc, .circle, .square
    )
    let orderedListMarker: StructuredText.DecimalListMarker = .decimal
    let tableStyle: PanelTableStyle
    let tableCellStyle: PanelTableCellStyle
    let thematicBreakStyle: PanelThematicBreakStyle

    init(_ t: Tokens) {
        inlineStyle = InlineStyle()
            .code(.monospaced, .fontScale(0.9))
            .strong(.fontWeight(.semibold))
            .link(.foregroundColor(t.accent))
        blockQuoteStyle = PanelBlockQuoteStyle(t)
        codeBlockStyle = PanelCodeBlockStyle(t)
        tableStyle = PanelTableStyle(t)
        tableCellStyle = PanelTableCellStyle(t)
        thematicBreakStyle = PanelThematicBreakStyle(t)
    }

    /// The highlighting palette, transcribed the way the tokens are — six slots, which is
    /// what the hljs classes the transcript used to be drawn with came down to. Prism's
    /// token names map onto the same six.
    static func highlighting(_ t: Tokens) -> StructuredText.HighlighterTheme {
        let key = Color(hex: t.dark ? "#C792EA" : "#7B2FA0")
        let text = Color(hex: t.dark ? "#7EE787" : "#1B6B3A")
        let number = Color(hex: t.dark ? "#FFC53D" : "#9A4E14")
        let name = Color(hex: t.dark ? "#79B8FF" : "#22599F")
        let comment = Color(hex: t.dark ? "#6B6E76" : "#848B8E")
        let attribute = Color(hex: t.dark ? "#FF8A5C" : "#B2542C")
        return StructuredText.HighlighterTheme(
            foregroundColor: DynamicColor(t.fg2),
            // Painted by the code block itself, which needs the border in the same breath.
            backgroundColor: DynamicColor(.clear),
            tokenProperties: [
                .keyword: AnyTextProperty(.foregroundColor(key), .fontWeight(.semibold)),
                .builtin: AnyTextProperty(.foregroundColor(key)),
                .literal: AnyTextProperty(.foregroundColor(key)),
                .boolean: AnyTextProperty(.foregroundColor(key)),
                .operator: AnyTextProperty(.foregroundColor(key)),
                .string: AnyTextProperty(.foregroundColor(text)),
                .char: AnyTextProperty(.foregroundColor(text)),
                .regex: AnyTextProperty(.foregroundColor(text)),
                .url: AnyTextProperty(.foregroundColor(text)),
                .number: AnyTextProperty(.foregroundColor(number)),
                .constant: AnyTextProperty(.foregroundColor(number)),
                .function: AnyTextProperty(.foregroundColor(name)),
                .functionName: AnyTextProperty(.foregroundColor(name)),
                .className: AnyTextProperty(.foregroundColor(name)),
                .tag: AnyTextProperty(.foregroundColor(name)),
                .comment: AnyTextProperty(.foregroundColor(comment)),
                .blockComment: AnyTextProperty(.foregroundColor(comment)),
                .docComment: AnyTextProperty(.foregroundColor(comment)),
                .variable: AnyTextProperty(.foregroundColor(attribute)),
                .property: AnyTextProperty(.foregroundColor(attribute)),
                .attribute: AnyTextProperty(.foregroundColor(attribute)),
                .attributeName: AnyTextProperty(.foregroundColor(attribute)),
            ]
        )
    }
}

/// Semibold, and barely larger than the paragraph under it. A 392 pt column has no room for a
/// heading scale — what separates a section here is the air above it, not the type size.
struct PanelHeadingStyle: StructuredText.HeadingStyle {
    private static let scales: [CGFloat] = [1.08, 1, 0.96, 0.96, 0.96, 0.96]

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .textual.fontScale(Self.scales[min(configuration.headingLevel, 6) - 1])
            .fontWeight(.semibold)
            .textual.lineSpacing(.fontScaled(0.16))
            .textual.blockSpacing(.fontScaled(top: Markup.Gap.heading, bottom: 0.2))
    }
}

struct PanelParagraphStyle: StructuredText.ParagraphStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .textual.lineSpacing(.fontScaled(0.28))
            .textual.blockSpacing(.fontScaled(top: Markup.Gap.block))
    }
}

/// A rule and the text beside it, with no well behind it: the transcript is already a card,
/// and a second fill inside it reads as a third surface.
struct PanelBlockQuoteStyle: StructuredText.BlockQuoteStyle {
    private let t: Tokens

    init(_ t: Tokens) { self.t = t }

    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Rectangle().fill(t.hair2).frame(width: 3)
            configuration.label
                .textual.lineSpacing(.fontScaled(0.28))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .fixedSize(horizontal: false, vertical: true)
        .textual.blockSpacing(.fontScaled(top: Markup.Gap.block, bottom: 0.16))
    }
}

/// The language over the code, in the eyebrow the panel wears everywhere else. A long line is
/// the model's and scrolls rather than wraps — wrapping it would be this panel editing the
/// answer to fit.
struct PanelCodeBlockStyle: StructuredText.CodeBlockStyle {
    private let t: Tokens

    init(_ t: Tokens) { self.t = t }

    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if let language = configuration.languageHint, !language.isEmpty {
                Text(language.uppercased()).mono(10, t.fg3)
                    .tracking(1.1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }
            }
            // Qualified: the panel's own `Overflow` is the ⋯ menu in the title row.
            Textual.Overflow {
                configuration.label
                    .textual.fontScale(0.88)
                    .textual.lineSpacing(.fontScaled(0.23))
                    .monospaced()
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
            }
        }
        .background(t.elev)
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.hair, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .textual.blockSpacing(.fontScaled(top: Markup.Gap.block, bottom: 0.16))
    }
}

struct PanelTableStyle: StructuredText.TableStyle {
    private let t: Tokens

    init(_ t: Tokens) { self.t = t }

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .textual.tableCellSpacing(horizontal: 1, vertical: 1)
            .textual.blockSpacing(.fontScaled(top: Markup.Gap.apart, bottom: 0.4))
            .textual.tableOverlay { layout in
                Canvas { context, _ in
                    for divider in layout.dividers() {
                        context.fill(Path(divider), with: .color(t.hair))
                    }
                }
            }
            .padding(1)
    }
}

struct PanelTableCellStyle: StructuredText.TableCellStyle {
    private let t: Tokens

    init(_ t: Tokens) { self.t = t }

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .textual.fontScale(configuration.row == 0 ? 0.8 : 0.86)
            .monospaced()
            .fontWeight(configuration.row == 0 ? .semibold : .regular)
            .foregroundStyle(configuration.row == 0 ? t.fg3 : t.fg2)
            .textual.padding(.fontScaled(0.4))
    }
}

struct PanelThematicBreakStyle: StructuredText.ThematicBreakStyle {
    private let t: Tokens

    init(_ t: Tokens) { self.t = t }

    func makeBody(configuration _: Configuration) -> some View {
        Rectangle().fill(t.hair).frame(height: 1)
            .textual.blockSpacing(.fontScaled(top: Markup.Gap.apart, bottom: 0.8))
    }
}

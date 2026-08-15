// Chat: the probe.
//
// The other four tabs are about what the daemon holds; this one is about whether it
// answers. It is laid out for that — the model in force never leaves the top, every turn
// carries its numbers, and the composer states what it is about to send.

import AppKit
import SwiftUI

struct ChatView: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    @State private var sheet: Sheet?

    enum Sheet: String, Identifiable {
        case models, tuning, history
        var id: String { rawValue }
    }

    var body: some View {
        let chat = app.chat
        VStack(spacing: 0) {
            ModelLine(app: app, sheet: $sheet)
            ZStack {
                Transcript(app: app, openHistory: { sheet = .history })
                switch sheet {
                case .models: ModelSheet(app: app, close: { sheet = nil })
                case .tuning: TuningSheet(app: app, close: { sheet = nil })
                case .history: HistorySheet(app: app, close: { sheet = nil })
                case nil: EmptyView()
                }
            }
            Composer(app: app, openModels: { sheet = .models })
        }
        .task { await chat.sessions.load() }
        .onAppear { chat.adopt(app.store) }
        .onChange(of: app.store.residentIds) { _, _ in chat.adopt(app.store) }
        // The catalog can arrive after a model was picked — at boot it does. Whether that
        // model has eyes is read again when it does, otherwise the composer would keep the
        // answer it had before there was one.
        .onChange(of: app.store.catalog) { _, _ in
            chat.sees = app.store.entry(chat.model)?.sees == true
        }
    }
}

// ── the model in force ───────────────────────────────────────────────────

struct ModelLine: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    @Binding var sheet: ChatView.Sheet?

    var body: some View {
        let chat = app.chat
        let entry = app.store.entry(chat.model)
        let material = t.mat(app.store.materials[chat.model] ?? 0)

        HStack(spacing: 8) {
            Pressable(action: { sheet = sheet == .models ? nil : .models }) {
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(app.store.residentIds.contains(chat.model) ? material : t.fg3)
                        .frame(width: 8, height: 8)
                    Text(chat.model.isEmpty ? "No model" : Fmt.displayName(chat.model))
                        .sans(12.5, chat.model.isEmpty ? t.fg3 : t.fg, weight: .medium)
                        .lineLimit(1).truncationMode(.tail)
                        // A long checkpoint name would otherwise take the row and leave the
                        // context reading beside it cut down to an ellipsis.
                        .frame(maxWidth: 190, alignment: .leading)
                    Text("▾").sans(9, t.fg3)
                }
            }
            .fixedSize()

            Spacer(minLength: 0)
            // The reading keeps its width and the name gives way: a name that is cut still
            // says which model, and a percentage that is cut says nothing.
            Text(reading(entry)).mono(10, t.fg3).lineLimit(1).layoutPriority(1)
            IconButton(symbol: "gearshape", on: sheet == .tuning) {
                sheet = sheet == .tuning ? nil : .tuning
            }
            IconButton(symbol: "clock.arrow.circlepath", on: sheet == .history) {
                sheet = sheet == .history ? nil : .history
            }
            IconButton(symbol: "square.and.pencil") { app.chat.clear() }
        }
        .padding(.leading, 12)
        .padding(.trailing, 14)
        .padding(.vertical, 8)
        .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }
    }

    /// What the conversation has spent, against what the checkpoint declares it can hold —
    /// the window's own top-right reading, moved here and made to count the pictures.
    private func reading(_ entry: CatalogEntry?) -> String {
        guard let window = entry?.context else { return "" }
        let used = app.chat.used(true)
        guard used > 0 else { return "0 · \(Fmt.tokens(window)) ctx" }
        let share = Double(used) / Double(window)
        return "\(Fmt.tokens(used)) · \(Fmt.percent(share, share < 0.01 ? 2 : 1)) of \(Fmt.tokens(window))"
    }
}

/// One verb, drawn. The symbol and not a glyph: `✎` and `↺` land in the emoji face at this
/// size, where a pencil reads as a paperclip.
struct IconButton: View {
    @Environment(\.tokens) private var t
    let symbol: String
    var on = false
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 12, weight: .medium))
            .foregroundColor(on ? t.accent : t.fg3)
            .frame(width: 22, height: 22)
            .background(on ? t.accentSoft : (hovering ? t.hair : t.sel))
            .clipShape(RoundedRectangle(cornerRadius: 7))
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .onTapGesture(perform: action)
    }
}

// ── the transcript ───────────────────────────────────────────────────────

struct Transcript: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let openHistory: () -> Void

    @State private var following = true

    var body: some View {
        let chat = app.chat
        // The daemon republishes the request it is serving twice a second, and that is where
        // a turn's numbers come from while it is still being written — the frame that closes
        // the stream carries them once, at the end. Only read while something is writing, so
        // an idle transcript is not redrawn on the register's beat.
        let live = chat.streaming ? app.store.live(chat.model) : nil
        ScrollViewReader { rail in
            ScrollView {
                // Not lazy, though a conversation is taller than the panel and this measures
                // every turn on every token of the one being written. A lazy stack has no
                // height for a row it has not built, so what it reports as the content is
                // whatever it happens to have realised — 2558 pt of a transcript 8346 pt tall,
                // measured at forty turns — and it revises that while the last turn grows, by
                // more than the turn grew. The reader scrolls against that number: past the
                // end of it is a viewport with no row in it, which is the panel going blank
                // mid-answer and coming back when you scroll up.
                //
                // The price is the whole transcript laid out per paint: 12 ms at eight turns,
                // 24 at forty. Grouping the settled turns behind one `Equatable` view does not
                // avoid it — measured, no difference — and it is still a fraction of what one
                // turn cost before `Prose` was cut into pieces.
                VStack(alignment: .leading, spacing: 14) {
                    Color.clear.frame(height: 1).id(Self.head)
                    if !chat.session.isEmpty { ThreadHead(app: app) }
                    if chat.turns.isEmpty {
                        if chat.session.isEmpty {
                            Probes(app: app, openHistory: openHistory)
                        } else {
                            Text("No messages. Another client made this one.").sans(12.5, t.fg3)
                        }
                    } else {
                        ForEach(Array(chat.turns.enumerated()), id: \.element.id) { index, turn in
                            VStack(alignment: .leading, spacing: 8) {
                                TurnView(turn: turn, live: turn.writing ? live : nil)
                                if let load = chat.loadReceipt(
                                    after: index,
                                    liveLoad: live?.loadSeconds
                                ) {
                                    ModelLoadReceipt(seconds: load)
                                }
                            }
                            .id(turn.id)
                        }
                    }
                    if !chat.refusal.isEmpty { Refusal(message: chat.refusal) }
                    if !chat.sessions.trouble.isEmpty { Refusal(message: chat.sessions.trouble) }
                    Color.clear.frame(height: 1).id(Self.foot)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, Panel.pad)
                .padding(.top, 13)
                .padding(.bottom, 10)
            }
            .scrollIndicators(.never)
            // Only while one is being written and the reader is at its end: a conversation
            // read back out of the store is opened at its head, and the same handler fires
            // when its turns land. Unanimated — the transcript is painted at a fixed rate,
            // and an ease that takes longer than the gap between two paints is restarted
            // before it has arrived anywhere.
            .onChange(of: chat.turns.last) { _, _ in
                guard chat.streaming, following else { return }
                rail.scrollTo(Self.foot, anchor: .bottom)
            }
            .onChange(of: following) {
                if following { rail.scrollTo(Self.foot, anchor: .bottom) }
            }
            .onChange(of: chat.session) { _, _ in rail.scrollTo(Self.head, anchor: .top) }
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.contentOffset.y + geometry.containerSize.height
                    >= geometry.contentSize.height - 8
            } action: { _, atEnd in
                following = atEnd
            }
        }
        .background(t.surface)
    }

    private static let head = "head"
    private static let foot = "foot"
}

private struct ModelLoadReceipt: View {
    @Environment(\.tokens) private var t
    let seconds: Double

    var body: some View {
        HStack(spacing: 7) {
            Spacer(minLength: 40)
            Circle().fill(t.ok).frame(width: 5, height: 5)
            Text("Model loaded").sans(10.5, t.fg3, weight: .medium)
            Text(Fmt.duration(seconds)).mono(10, t.fg2, weight: .medium)
        }
        .accessibilityElement(children: .combine)
    }
}

struct TurnView: View {
    @Environment(\.tokens) private var t
    let turn: Turn
    /// The register as it stands, handed in only while this turn is the one being written.
    /// It is what puts the numbers under an answer as it arrives; once the turn is over they
    /// come off `turn.timings`, which is the same arithmetic on the closing frame.
    var live: Sample?

    var body: some View {
        if turn.role == "user" {
            HStack {
                Spacer(minLength: 40)
                VStack(alignment: .leading, spacing: 7) {
                    ForEach(turn.clips) { clip in ClipImage(clip: clip) }
                    // The same renderer the answer gets. A prompt is written in markdown as
                    // often as an answer is — a fence around the code being asked about, a
                    // list of the cases to cover — and a bubble that shows the backticks is
                    // the one place in this window where markup is spelled out rather than
                    // drawn.
                    if !turn.text.isEmpty { Prose(text: turn.text, breaks: true) }
                }
                .padding(.horizontal, 11)
                .padding(.vertical, 8)
                .background(t.elev)
                .overlay(RoundedRectangle(cornerRadius: 13).strokeBorder(t.hair, lineWidth: 1))
                .clipShape(RoundedRectangle(cornerRadius: 13))
            }
        } else {
            VStack(alignment: .leading, spacing: 0) {
                if !turn.reasoning.isEmpty { Thinking(turn: turn) }
                if !turn.text.isEmpty { Prose(text: turn.text, writing: turn.writing) }
                if turn.writing && turn.text.isEmpty && turn.reasoning.isEmpty {
                    Text("…").sans(12.5, t.fg3)
                }
                ForEach(turn.calls) { call in
                    CallCard(call: call, writing: turn.writing).padding(.top, 8)
                }
                if let failure = turn.failure {
                    Refusal(message: failure).padding(.top, 6)
                }
                if let line = turn.metricsLine(live: live) {
                    Text(line).mono(10, t.fg3).padding(.top, 7)
                }
                if turn.finish == "length" {
                    Text("Stopped at the token budget, not at an end of turn.")
                        .sans(10.5, t.warn).padding(.top, 4)
                }
            }
        }
    }

}

/// The reasoning: one line until the reader opens it, streaming or not. The thinking is where
/// the answer came from and not the answer. What the line says when closed is how long the
/// wait was, because that is what the reader was watching.
struct Thinking: View {
    @Environment(\.tokens) private var t
    let turn: Turn

    @State private var opened = false
    @State private var following = true

    private var thinking: Bool { turn.writing && turn.text.isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text(opened ? "▾" : "▸").sans(9, t.fg3)
                Text(label).sans(11.5, t.fg3)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
            .onTapGesture {
                if !opened { following = true }
                opened.toggle()
            }

            if opened {
                ScrollViewReader { rail in
                    ScrollView {
                        // The same renderer the answer gets: a model that numbers its steps and
                        // quotes code while it thinks wrote markdown, whichever channel it went out on.
                        Prose(text: turn.reasoning, size: 11.5, tone: t.fg2, writing: turn.writing)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                        Color.clear.frame(height: 1).id(Self.foot)
                    }
                    .onAppear { rail.scrollTo(Self.foot, anchor: .bottom) }
                    .onChange(of: turn.reasoning) {
                        if following { rail.scrollTo(Self.foot, anchor: .bottom) }
                    }
                    .onChange(of: following) {
                        if following { rail.scrollTo(Self.foot, anchor: .bottom) }
                    }
                    .onScrollGeometryChange(for: Bool.self) { geometry in
                        geometry.contentOffset.y + geometry.containerSize.height
                            >= geometry.contentSize.height - 8
                    } action: { _, atEnd in
                        following = atEnd
                    }
                }
                .frame(maxHeight: 220)
                .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
            }
        }
        .background(t.elev)
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.hair, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.bottom, 8)
    }

    private var label: String {
        if thinking { return "Thinking…" }
        guard let spent = turn.thoughtFor else { return "Thought" }
        return String(format: "Thought for %.1fs", spent)
    }

    private static let foot = "thinking-end"
}

/// A call, as the model wrote it. Nothing here runs it: the panel is not an agent, and what
/// is being checked is the shape — whether this checkpoint's dialect produced the call the
/// schema asked for.
struct CallCard: View {
    @Environment(\.tokens) private var t
    let call: ToolCall
    let writing: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 7) {
                Image(systemName: "gearshape")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(t.fg3)
                Text("TOOL CALL · \(call.name.isEmpty ? "…" : call.name)")
                    .mono(10, t.fg3).tracking(1.1).lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }

            ScrollView(.horizontal) {
                Text(pretty).mono(11, t.fg2)
                    .lineSpacing(2.5)
                    .textSelection(.enabled)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
            }
            .scrollIndicators(.never)

            if !writing {
                Text("Nothing runs this here.").sans(11, t.fg3)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
            }
        }
        .background(t.elev)
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.hair, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    /// Indented once it parses, raw until then. A half-written argument list is not JSON
    /// yet, and showing it as it arrives is the point.
    private var pretty: String {
        guard let data = call.arguments.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data),
              let out = try? JSONSerialization.data(
                withJSONObject: parsed, options: [.prettyPrinted, .sortedKeys]
              ),
              let text = String(data: out, encoding: .utf8)
        else { return call.arguments.isEmpty ? "…" : call.arguments }
        return text
    }
}

struct ClipImage: View {
    @Environment(\.tokens) private var t
    let clip: Clip

    var body: some View {
        if let image = clip.thumbnail {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 240, maxHeight: 150)
                .clipShape(RoundedRectangle(cornerRadius: 7))
                .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(t.hair2, lineWidth: 1))
        }
    }
}

// ── the empty state ──────────────────────────────────────────────────────

/// Four probes, not four subjects. Each one exercises a path of the engine: the tower, the
/// grammar, the tool dialect, the context. Under them, the three conversations most recently
/// touched — the panel opens new, and what is wanted is usually yesterday's answer.
struct Probes: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    var openHistory: () -> Void = {}

    private var sees: Bool { app.store.entry(app.chat.model)?.sees == true }

    var body: some View {
        let chat = app.chat
        let recent = chat.sessions.recent(3)
        VStack(alignment: .leading, spacing: 0) {
            Text(headline).display(17, t.fg)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            if let sub { Text(sub).sans(12.5, t.fg3).padding(.top, 2) }

            if !chat.model.isEmpty {
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible())], spacing: 8) {
                    probe("Look at the screen", "Grab a region and ask what it says.") {
                        app.grab()
                        chat.draft = "What is this showing?"
                    }
                    probe("Answer as JSON", "Reply only with {\"ok\": true}.") {
                        chat.draft = "Answer with exactly this and nothing else: {\"ok\": true}"
                    }
                    probe("Call a tool", "Write one call and stop.") {
                        chat.draft = "Call the tool you were given for Lisbon, then stop."
                    }
                    probe("Read it back", "Repeat a line verbatim.") {
                        chat.draft = "Repeat the following back, character for character: mlx-omnia"
                    }
                }
                .padding(.top, 13)
            }

            if !recent.isEmpty {
                SectionHead(title: "Pick up where you left off").padding(.top, 12)
                ForEach(recent) { row in
                    ConvoRow(
                        summary: row,
                        open: { chat.act { await chat.reopen(row.id, app.store) } },
                        forget: { chat.act { await chat.sessions.forget(row.id) } }
                    )
                }
                if chat.sessions.count > recent.count {
                    HStack(spacing: 6) {
                        ActionText(label: "All \(chat.sessions.count)", action: openHistory)
                        Text("→").sans(10, t.accent)
                    }
                    .padding(.top, 9)
                    .padding(.horizontal, 9)
                }
            }
        }
    }

    private var headline: String {
        if app.store.down { return "The engine is not answering." }
        if app.chat.model.isEmpty { return "Nothing resident." }
        let state = app.store.residentIds.contains(app.chat.model) ? "is resident" : "is on disk"
        return "\(Fmt.displayName(app.chat.model)) \(state)."
    }

    /// What to do about the headline, when there is something to do. A model that answers
    /// and reads pictures leaves nothing to say, and the probes below say it themselves.
    private var sub: String? {
        if app.store.down { return "Start it under Server." }
        if app.chat.model.isEmpty { return "Load one under Models, or pick one above." }
        return sees ? nil : "This one takes no images."
    }

    private func probe(_ title: String, _ body: String, _ pick: @escaping () -> Void) -> some View {
        Pressable(action: pick) {
            Text(title).sans(12, t.fg, weight: .semibold)
            Text(body).sans(11.5, t.fg2).lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)
        }
    }
}

// ── the composer ─────────────────────────────────────────────────────────

struct Composer: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let openModels: () -> Void

    private var entry: CatalogEntry? { app.store.entry(app.chat.model) }
    private var sees: Bool { entry?.sees == true }

    var body: some View {
        @Bindable var chat = app.chat
        VStack(alignment: .leading, spacing: 0) {
            if !chat.clips.isEmpty && !sees {
                Wall {
                    Text("\(Fmt.displayName(chat.model)) does not read images. ")
                        .sans(12, t.fg2, weight: .semibold)
                        + Text("Drop the clip, or switch to one that does.").sans(12, t.fg2)
                } action: {
                    ActionText(label: "Switch model ▾", action: openModels)
                }
                .padding(.bottom, 9)
            }

            LoadNote(app: app).padding(.bottom, 9)

            ForEach(chat.clips) { clip in
                ClipChip(clip: clip, sees: sees, drop: { chat.drop(clip) })
                    .padding(.bottom, 8)
            }

            HStack(alignment: .bottom, spacing: 9) {
                AttachButton(app: app, sees: sees)
                Draft(app: app)
                SendButton(app: app)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 9)
        .padding(.bottom, 10)
        .background(t.side)
        .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
    }
}

/// What is being typed, marked as it is typed.
///
/// An attributed editor and not the plain field it was: the markup stays on screen — a `*`
/// nobody can see is a `*` nobody can delete — and takes the shape it will have in the bubble.
/// `chat.draft` is still the plain string, which is what gets sent; this view holds the marked
/// copy of it and the two are kept level in one direction each.
///
/// Return sends and shift-return opens a line, which the plain field could not offer: it had
/// one key and a submit action. The panel is dismissed by a click outside it, so a composer
/// with no way to send from the keyboard would be a composer with no way to send at all.
struct Draft: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    @State private var written = AttributedString()
    @State private var caret = AttributedTextSelection()
    @State private var tall: CGFloat = 0
    @FocusState private var typing: Bool

    private static let size: CGFloat = 12.5
    private static let pad: CGFloat = 7
    /// Five lines, as the field held before. Past that the editor scrolls inside itself.
    private static let lines = 5

    var body: some View {
        let chat = app.chat
        ZStack(alignment: .topLeading) {
            TextEditor(text: $written, selection: $caret)
                .textEditorStyle(.plain)
                .font(.system(size: Self.size))
                .foregroundColor(t.fg)
                .scrollContentBackground(.hidden)
                .focused($typing)
                // On the editor and not on the box around it: `onKeyPress` is offered only
                // what the focused view has not consumed, and a text view an ancestor asked
                // about return would already have eaten it. Shift-return is left to the
                // editor, which opens a line with it — the plain field this replaced had one
                // key and a submit action, and no second key to type a newline with.
                .onKeyPress(.return, phases: .down) { press in
                    guard !press.modifiers.contains(.shift) else { return .ignored }
                    chat.send()
                    return .handled
                }
                .padding(.horizontal, Self.pad)
                .padding(.vertical, Self.pad)
            if written.characters.isEmpty {
                Text("Ask this model…").sans(Self.size, t.fg3)
                    .padding(.horizontal, 11)
                    .padding(.vertical, Self.pad)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: tall == 0 ? nil : tall)
        // A `TextEditor` never reports the height of what is in it; it takes whatever it is
        // given. So the same string is laid out behind it, unseen and unclamped, and the
        // editor is given what that came to — one line to five, and scrolling past that.
        .background(alignment: .topLeading) { ruler }
        .background(t.field)
        .overlay(RoundedRectangle(cornerRadius: 11).strokeBorder(t.hair2, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 11))
        // Typing: the characters go out to the draft, and the marks come back over them. The
        // second write is attributes only, so the guard below stops it here rather than
        // bouncing between the two.
        .onChange(of: written) { _, now in
            let plain = String(now.characters)
            guard plain != chat.draft else { return }
            chat.draft = plain
            written.transform(updating: &caret) { Markup.mark(&$0, t, Self.size) }
        }
        // The draft set from anywhere else — a probe filling it in, a send emptying it.
        .onChange(of: chat.draft, initial: true) { _, plain in
            guard plain != String(written.characters) else { return }
            var made = AttributedString(plain)
            Markup.mark(&made, t, Self.size)
            written = made
            caret = AttributedTextSelection(insertionPoint: written.endIndex)
        }
        // The marks are the palette's, so they are redrawn when it changes.
        .onChange(of: t.dark) { _, _ in
            written.transform(updating: &caret) { Markup.mark(&$0, t, Self.size) }
        }
    }

    /// What the editor is measured against: the same text at the same face and the same
    /// insets, taking its own height rather than the one it is offered.
    private var ruler: some View {
        Text(written.characters.isEmpty ? " " : AttributedString(String(written.characters)))
            .font(.system(size: Self.size))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 11)
            .padding(.vertical, Self.pad)
            .frame(maxWidth: .infinity, alignment: .leading)
            .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { measured in
                let one = Self.size * 1.25 + 2 * Self.pad
                tall = min(max(measured, one), one + (Self.size * 1.25) * CGFloat(Self.lines - 1))
            }
            .opacity(0)
    }
}

/// A line stuck to the composer: what the next turn will run into, said where the turn is
/// about to be typed rather than after it is refused.
struct Wall<Body: View, Action: View>: View {
    @Environment(\.tokens) private var t
    @ViewBuilder var content: Body
    @ViewBuilder var action: Action

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            content
                .lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            action
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(t.sel)
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.hair2, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// What answering will cost when the model in force is not in memory. It states the price
/// and not the setting that decided it — whoever is reading wants to know what the next turn
/// will do, and the policy that chose it is on the Server screen, read once.
struct LoadNote: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        let chat = app.chat
        let name = Fmt.displayName(chat.model)
        let entry = app.store.entry(chat.model)
        if chat.shouldShowLoadNote(resident: app.store.residentIds.contains(chat.model)) {
            if let entry {
                Wall {
                    Text("\(name) is not resident. ").sans(12, t.fg2, weight: .semibold)
                        + Text(
                            "The first turn loads \(Fmt.gb(Double(entry.bytesOnDisk))) GB into memory."
                        ).sans(12, t.fg2)
                } action: {
                    EmptyView()
                }
            } else {
                Wall {
                    Text("\(name) isn't on disk.").sans(12, t.fg2, weight: .semibold)
                } action: {
                    ActionText(label: "Find it in Models →") {
                        app.modelSearch = Fmt.displayName(chat.model)
                        app.go(.models)
                        app.modelsTab = .library
                    }
                }
            }
        }
    }
}

/// The multimodal door. Shut, with the reason, when the checkpoint in force has no tower —
/// the refusal belongs here and not after the send.
struct AttachButton: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let sees: Bool

    var body: some View {
        Menu {
            Button("Grab a region…") { app.grab() }
            Button("Choose a file…") { app.chat.attach(Attach.file()) }
            Button("Paste image") { app.chat.attach(Attach.pasteboard()) }
        } label: {
            Text("+").sans(15, sees ? t.fg2 : t.fg3)
                .frame(width: 26, height: 26)
                .background(sees ? t.elev : .clear)
                .overlay(
                    Circle().strokeBorder(
                        sees ? t.hair2 : t.hair,
                        style: StrokeStyle(lineWidth: 1, dash: sees ? [] : [2, 2])
                    )
                )
                .clipShape(Circle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(!sees)
        .help(sees ? "Attach an image" : "This checkpoint takes no images")
    }
}

struct SendButton: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        let chat = app.chat
        let live = chat.streaming
        let ready = live || chat.canSend
        RoundedRectangle(cornerRadius: 9)
            .fill(ready ? t.accent : t.sel)
            .frame(width: 30, height: 30)
            .overlay {
                if live {
                    RoundedRectangle(cornerRadius: 2).fill(t.onAccent).frame(width: 9, height: 9)
                } else {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(ready ? t.onAccent : t.fg3)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { live ? chat.stop() : chat.send() }
    }
}

/// A clip in hand, with what it will cost. The arithmetic is the daemon's — this draws the
/// answer and says "priced by the model" when there is none yet, never a guess.
struct ClipChip: View {
    @Environment(\.tokens) private var t
    let clip: Clip
    let sees: Bool
    let drop: () -> Void

    var body: some View {
        HStack(spacing: 9) {
            if let image = clip.thumbnail {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 52, height: 34)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(t.hair2, lineWidth: 1))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(clip.origin).sans(11.5, t.fg).lineLimit(1)
                Text(line).mono(10, t.fg3).lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 0)
            Text("✕").sans(10.5, t.fg3)
                .contentShape(Rectangle())
                .onTapGesture(perform: drop)
        }
        .opacity(sees ? 1 : 0.55)
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(t.elev)
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.hair, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var line: String {
        let held = "\(clip.width)×\(clip.height)"
        guard sees else { return "\(held) · held, not sent" }
        guard let cost = clip.cost else { return "\(held) · pricing…" }
        let read = cost.height == clip.height && cost.width == clip.width
            ? held
            : "\(held) → \(cost.width)×\(cost.height)"
        return "\(read) · \(Fmt.grouped(Double(cost.tokens))) tokens"
    }
}

// The two sheets the chat opens over its own transcript: which model answers, and how.
//
// They cover the transcript rather than pushing it, because both are answers to a question
// about the conversation that is on screen — and at 420 pt there is no second column to put
// them in.

import SwiftUI

/// Which checkpoint answers, split by the one property the chat cares about. Holding a
/// picture, the split is the whole point: the text-only half is dimmed and says why.
struct ModelSheet: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let close: () -> Void

    private var holding: Bool { !app.chat.clips.isEmpty }

    var body: some View {
        let store = app.store
        let listed = store.catalog.filter(\.supported)
        let seeing = listed.filter(\.sees)
        let reading = listed.filter { !$0.sees }

        // Two checkpoints of the same model under different repositories collapse to one
        // display name, and this list is exactly where somebody is choosing between them.
        let taken = Dictionary(grouping: listed, by: { Fmt.displayName($0.id) })
        let ambiguous = Set(taken.filter { $0.value.count > 1 }.keys)

        Sheet(title: holding ? "Switch model" : "Model", close: close) {
            if !seeing.isEmpty {
                SectionHead(title: "Sees images · \(seeing.count)")
                ForEach(sorted(seeing)) { entry in row(entry, dim: false, whole: ambiguous) }
            }
            SectionHead(title: "Text only · \(reading.count)")
            ForEach(sorted(reading)) { entry in row(entry, dim: holding, whole: ambiguous) }
            if listed.isEmpty {
                EmptyNote(message: "Nothing on disk this engine can load.")
            }
        }
    }

    /// Resident first, then the rest by name: what is already in memory costs nothing to
    /// switch to, and that is the only ordering the reader is choosing on.
    private func sorted(_ entries: [CatalogEntry]) -> [CatalogEntry] {
        entries.sorted { left, right in
            let mine = app.store.residentIds.contains(left.id)
            let theirs = app.store.residentIds.contains(right.id)
            if mine != theirs { return mine }
            return Fmt.displayName(left.id) < Fmt.displayName(right.id)
        }
    }

    private func row(_ entry: CatalogEntry, dim: Bool, whole: Set<String>) -> some View {
        let resident = app.store.residentIds.contains(entry.id)
        let current = entry.id == app.chat.model
        let name = Fmt.displayName(entry.id)
        return HStack(spacing: 9) {
            Circle()
                .strokeBorder(resident ? .clear : t.fg3, lineWidth: 1.5)
                .background(Circle().fill(resident ? t.mat(app.store.materials[entry.id] ?? 0) : .clear))
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 2) {
                Text(whole.contains(name) ? entry.id : name).sans(12.5, t.fg, weight: .medium)
                    .lineLimit(1).truncationMode(.middle)
                Text(under(entry, resident: resident, current: current)).mono(10, t.fg3).lineLimit(1)
            }
            Spacer(minLength: 0)
            if current {
                PillView(label: "Current", kind: resident ? "resident" : "disk")
            } else if resident {
                PillView(label: "Resident", kind: "resident")
            } else {
                PushButton(label: "Use", kind: "") { pick(entry) }
            }
        }
        // Dimmed says "this one cannot read what you are holding", not "you may not pick
        // it". Switching to it is allowed; the composer is where that costs something.
        .opacity(dim ? 0.42 : 1)
        .contentShape(Rectangle())
        .onTapGesture { pick(entry) }
        .padding(.vertical, 8)
        .padding(.horizontal, 2)
        .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
    }

    /// What switching to it costs. A resident model costs nothing; one on disk costs its
    /// weights and the load, and this is the only door in the chat that spends memory.
    private func under(_ entry: CatalogEntry, resident: Bool, current: Bool) -> String {
        var said: [String] = [resident ? "resident" : "on disk"]
        if current { said.append("in this conversation") }
        else if !resident { said.append(Fmt.gb(Double(entry.bytesOnDisk)) + " GB") }
        if let decode = app.store.decode(entry.id) { said.append(decode) }
        else if let label = entry.quantization { said.append(label) }
        return said.joined(separator: " · ")
    }

    private func pick(_ entry: CatalogEntry) {
        app.chat.use(entry.id, entry)
        // Loading it is the daemon's to do on the first request: a model named in a chat
        // request is loaded to serve it. Nothing is loaded here, which is why the row says
        // what it will cost rather than doing it now.
        close()
    }
}

/// The window's tuning panel at 392 pt. Same knobs, same resolution — what the panel names
/// beats the profile, and what it leaves unset the daemon decides.
struct TuningSheet: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let close: () -> Void

    var body: some View {
        @Bindable var chat = app.chat
        Sheet(title: "Tuning", close: close) {
            Knob(name: "temperature", value: $chat.knobs.temperature, low: 0, high: 2)
            Knob(name: "top_p", value: $chat.knobs.topP, low: 0.01, high: 1)
            Knob(name: "top_k", whole: $chat.knobs.topK, low: 0, high: 200)
            Knob(name: "min_p", value: $chat.knobs.minP, low: 0, high: 0.99)

            HStack(spacing: 10) {
                Text("effort").sans(12, t.fg2).frame(width: 78, alignment: .leading)
                Segmented(
                    options: Self.efforts.map { ($0, $0) },
                    chosen: chat.knobs.effort,
                    size: 11.5,
                    pick: { chat.knobs.effort = $0 }
                )
            }
            .padding(.vertical, 8)
            .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }

            SectionHead(title: "System prompt")
            TextField("What the model is told first", text: $chat.knobs.systemPrompt, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(2...4)
                .font(.system(size: 11.5))
                .foregroundColor(t.fg)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(t.field)
                .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(t.hair2, lineWidth: 1))
                .clipShape(RoundedRectangle(cornerRadius: 9))

            SectionHead(title: toolsLabel)
            TextField("[{\"type\": \"function\", \"function\": {…}}]", text: $chat.knobs.tools, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(2...5)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(t.fg)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(t.field)
                .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(t.hair2, lineWidth: 1))
                .clipShape(RoundedRectangle(cornerRadius: 9))

            Text("Nothing runs the function.").sans(11.5, t.fg3).padding(.top, 8)

            HStack(spacing: 8) {
                PushButton(label: "Reset to checkpoint", wide: true) {
                    chat.knobs = Knobs.of(app.store.entry(chat.model))
                }
            }
            .padding(.top, 12)
        }
    }

    static let efforts = ["default", "none", "low", "medium", "high"]

    /// How many the field declares, so a typo shows up here rather than as a model that
    /// never calls anything.
    private var toolsLabel: String {
        let text = app.chat.knobs.tools.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return "Tools" }
        guard let parsed = try? JSONSerialization.jsonObject(with: Data(text.utf8)),
              let list = parsed as? [Any]
        else { return "Tools · not a JSON array" }
        return "Tools · \(list.count) declared"
    }
}

/// One sampling knob: its name and an optional numeric override. An empty field means the
/// daemon's own preset decides.
struct Knob: View {
    @Environment(\.tokens) private var t
    let name: String
    var value: Binding<Double?>?
    var whole: Binding<Int?>?
    let low: Double
    let high: Double

    var body: some View {
        HStack(spacing: 10) {
            Text(name).sans(12, t.fg2).frame(width: 78, alignment: .leading)
            Group {
                if let value {
                    TextField(
                        "unset",
                        value: limited(value),
                        format: .number.precision(.fractionLength(0...2))
                    )
                } else if let whole {
                    TextField("unset", value: limited(whole), format: .number)
                }
            }
            .textFieldStyle(.plain)
            .font(.system(size: 11.5, design: .monospaced))
            .multilineTextAlignment(.trailing)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background(t.field)
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(t.hair2, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding(.vertical, 8)
        .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
    }

    private func limited(_ binding: Binding<Double?>) -> Binding<Double?> {
        Binding(
            get: { binding.wrappedValue },
            set: { binding.wrappedValue = $0.map { min(max($0, low), high) } }
        )
    }

    private func limited(_ binding: Binding<Int?>) -> Binding<Int?> {
        let range = Int(low)...Int(high)
        return Binding(
            get: { binding.wrappedValue },
            set: { binding.wrappedValue = $0.map { min(max($0, range.lowerBound), range.upperBound) } }
        )
    }
}

/// The chrome the sheets wear: a title, a way out, and a scrolling body.
struct Sheet<Content: View>: View {
    @Environment(\.tokens) private var t
    let title: String
    let close: () -> Void
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text(title).sans(13.5, t.fg, weight: .semibold)
                Spacer(minLength: 0)
                IconButton(symbol: "xmark", action: close)
            }
            .padding(.horizontal, Panel.pad)
            .padding(.top, 11)
            .padding(.bottom, 9)
            .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }

            ScrollView {
                VStack(alignment: .leading, spacing: 0) { content }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, Panel.pad)
                    .padding(.vertical, 12)
            }
            .scrollIndicators(.never)
        }
        .background(t.surface)
    }
}

// The conversations, in the three places the chat has room for them.
//
// A 504 pt panel has no room for a permanent sidebar — one would take a third of the
// transcript — so the list is where it is needed instead of where it would always be: the
// three most recent under the probes, which is the common case; this sheet behind the `↺`,
// with search, for the rest; and the name of the open one at the top of the thread.
//
// The rows carry the hour, the count and the model because the titles do not distinguish
// them: the title is the first thing said, and a machine somebody probes with one phrase
// ends up with a store full of the same sentence. Renaming is on the open one, at the top of
// its thread, for the same reason.

import SwiftUI

/// One conversation as a row, with the `✕` that drops it.
struct ConvoRow: View {
    @Environment(\.tokens) private var t
    let summary: SessionSummary
    let open: () -> Void
    let forget: () -> Void

    @State private var hovering = false

    var body: some View {
        HStack(spacing: 9) {
            VStack(alignment: .leading, spacing: 1) {
                Text(summary.title.isEmpty ? "New chat" : summary.title)
                    .sans(12.5, t.fg)
                    .lineLimit(1).truncationMode(.tail)
                Text(meta).mono(10, t.fg3).lineLimit(1).truncationMode(.tail)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Image(systemName: "xmark")
                .font(.system(size: 9, weight: .medium))
                .foregroundColor(t.fg3)
                .opacity(hovering ? 1 : 0.35)
                .frame(width: 18, height: 18)
                .contentShape(Rectangle())
                .onTapGesture(perform: forget)
        }
        .padding(.leading, 9)
        .padding(.trailing, 4)
        .padding(.vertical, 7)
        .background(hovering ? t.sel : .clear)
        .clipShape(RoundedRectangle(cornerRadius: 9))
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture(perform: open)
    }

    private var meta: String {
        var said = [Fmt.stamp(summary.updatedAt)]
        said.append(summary.messageCount == 0 ? "no messages" : "\(summary.messageCount) messages")
        if !summary.model.isEmpty { said.append(Fmt.displayName(summary.model)) }
        return said.joined(separator: " · ")
    }
}

/// Everything in the store, by the day it was last touched.
struct HistorySheet: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let close: () -> Void

    @State private var query = ""

    var body: some View {
        let sessions = app.chat.sessions
        let needle = query.lowercased()
        let found = sessions.summaries.filter { row in
            needle.isEmpty
                || row.title.lowercased().contains(needle)
                || row.model.lowercased().contains(needle)
        }

        Sheet(title: "History", close: close) {
            SearchBox(placeholder: "Search conversations…", text: $query)

            ForEach(Day.allCases, id: \.self) { day in
                let rows = found.filter { day.holds($0.updatedAt) }
                if !rows.isEmpty {
                    SectionHead(title: "\(day.label) · \(rows.count)")
                    ForEach(rows) { row in
                        ConvoRow(
                            summary: row,
                            open: { open(row) },
                            forget: { app.chat.act { await sessions.forget(row.id) } }
                        )
                    }
                }
            }

            if found.isEmpty {
                EmptyNote(
                    message: query.isEmpty
                        ? "No conversations yet. The first turn starts one."
                        : "Nothing matches."
                )
            } else {
                Text("\(sessions.count) in the daemon's store. The window shows the same list.")
                    .sans(11.5, t.fg3)
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 14)
                    .padding(.horizontal, 2)
            }
        }
        .task { await sessions.load() }
    }

    private func open(_ row: SessionSummary) {
        close()
        app.chat.act { await app.chat.reopen(row.id, app.store) }
    }
}

enum Day: CaseIterable {
    case today, yesterday, earlier

    var label: String {
        switch self {
        case .today: return "Today"
        case .yesterday: return "Yesterday"
        case .earlier: return "Earlier"
        }
    }

    func holds(_ epoch: Double) -> Bool {
        let moment = Date(timeIntervalSince1970: epoch)
        switch self {
        case .today: return Calendar.current.isDateInToday(moment)
        case .yesterday: return Calendar.current.isDateInYesterday(moment)
        case .earlier:
            return !Calendar.current.isDateInToday(moment)
                && !Calendar.current.isDateInYesterday(moment)
        }
    }
}

/// The name of the conversation on screen, over its first turn. It costs no chrome — it
/// scrolls with the text — and it is where renaming is, because a store full of the same
/// title is a store nothing is findable in.
struct ThreadHead: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    @State private var naming = false
    @State private var draft = ""
    @FocusState private var typing: Bool

    var body: some View {
        let chat = app.chat
        let summary = chat.sessions.summary(chat.session)
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if naming {
                TextField("Name this conversation", text: $draft)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(t.fg)
                    .focused($typing)
                    .onSubmit { rename() }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(t.field)
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(t.hair2, lineWidth: 1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                ActionText(label: "Save") { rename() }
            } else {
                Text(chat.title.isEmpty ? "New chat" : chat.title)
                    .sans(13, t.fg, weight: .semibold)
                    .lineLimit(1).truncationMode(.tail)
                Image(systemName: "pencil")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(t.fg3)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        draft = chat.title
                        naming = true
                        typing = true
                    }
                Spacer(minLength: 0)
                if let summary {
                    Text("\(Fmt.stamp(summary.updatedAt)) · \(summary.messageCount) messages")
                        .mono(10, t.fg3)
                }
            }
        }
        .padding(.bottom, 9)
        .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }
        .padding(.bottom, 5)
    }

    private func rename() {
        let name = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        naming = false
        guard !name.isEmpty, name != app.chat.title else { return }
        app.chat.title = name
        let chat = app.chat
        chat.act { await chat.sessions.rename(chat.session, to: name) }
    }
}

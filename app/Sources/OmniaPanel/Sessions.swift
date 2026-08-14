// The conversations, which live in the daemon and not in this app.
//
// `/admin/sessions` is the store the desktop window already reads, so a probe started in the
// bar is in the window's sidebar and the other way round. The panel gains no database of its
// own; it gains a client.
//
// Two rules follow from sharing it, and they are why the shapes below are exact rather than
// convenient. The message array is written whole, in the vocabulary both apps already use —
// `metrics`, `reasoning_content`, `error` — so a reopened turn keeps the numbers it was
// answered with instead of a blank line where they were. And the title is the one the window
// writes: the first thing said, whitespace collapsed, 48 characters, ellipsis.
//
// What it cannot carry: a picture. `content` is a string on the way in — the window's reader
// drops a message whose content is anything else — so a turn that held a clip is stored as
// the words that came with it.

import Foundation
import Observation

/// A row of the list: everything but the conversation itself.
struct SessionSummary: Decodable, Identifiable, Equatable {
    let id: String
    let title: String
    let model: String
    let createdAt: Double
    let updatedAt: Double
    let messageCount: Int
}

/// What a turn cost, beside the turn. The same seven keys `app/api/conversation.py` writes;
/// a key left out reads as null there, which is what an absent measure means.
struct StoredMetrics: Codable {
    var loadMs: Double?
    var ttftMs: Double?
    var prefillTokensPerSecond: Double?
    var tokensPerSecond: Double?
    var ceilingFraction: Double?
    var finish: String?
}

struct StoredMessage: Codable {
    var role: String
    var content: String
    var reasoningContent: String?
    var metrics: StoredMetrics?
    var error: String?
}

struct StoredSession: Decodable {
    let id: String
    let title: String
    let model: String
    let messages: [StoredMessage]
}

private struct SessionBody: Encodable {
    var title: String
    var model: String
}

private struct SessionPatch: Encodable {
    var title: String?
    var model: String?
}

private struct MessagesBody: Encodable {
    var messages: [StoredMessage]
}

private struct Listed: Decodable {
    let sessions: [SessionSummary]
}

@MainActor
@Observable
final class Sessions {
    /// Newest first, which is the daemon's own order — `updated_at DESC`.
    var summaries: [SessionSummary] = []
    /// What a write refused to do. The turn is on screen either way; this is how the reader
    /// finds out it will not be there tomorrow.
    var trouble = ""

    var count: Int { summaries.count }

    func recent(_ many: Int) -> [SessionSummary] { Array(summaries.prefix(many)) }

    func summary(_ id: String) -> SessionSummary? { summaries.first { $0.id == id } }

    // ── reading ──────────────────────────────────────────────────────────

    /// A list that cannot be fetched leaves the last one standing: the conversations are the
    /// daemon's, and a daemon that is restarting is not a history that was lost.
    func load() async {
        guard let listed: Listed = try? await Client.get("/admin/sessions") else { return }
        summaries = listed.sessions
    }

    func open(_ identifier: String) async -> StoredSession? {
        try? await Client.get("/admin/sessions/\(Client.at(identifier))")
    }

    // ── writing ──────────────────────────────────────────────────────────

    /// The row a conversation becomes on its first turn. Never before: an empty conversation
    /// is not a conversation, and the window's `+` is why the store already holds rows with
    /// no messages in them.
    func begin(title: String, model: String) async -> String? {
        do {
            let made: StoredSession? = try await Client.send(
                "POST", "/admin/sessions", body: SessionBody(title: title, model: model)
            )
            await load()
            return made?.id
        } catch {
            trouble = said(error)
            return nil
        }
    }

    func rename(_ identifier: String, to title: String) async {
        await patch(identifier, SessionPatch(title: title, model: nil))
    }

    func moved(_ identifier: String, to model: String) async {
        await patch(identifier, SessionPatch(title: nil, model: model))
    }

    /// The whole array and not an append, which is the route's own shape: what the panel
    /// holds is the conversation, so a turn that was stopped or dropped has to be able to
    /// leave the file too.
    func write(_ identifier: String, _ messages: [StoredMessage]) async {
        do {
            try await Client.send(
                "PUT", "/admin/sessions/\(Client.at(identifier))/messages",
                body: MessagesBody(messages: messages)
            )
            trouble = ""
            await load()
        } catch {
            trouble = "This turn was answered but not saved — \(said(error))"
        }
    }

    /// Immediate, and there is no bin to take it out of.
    func forget(_ identifier: String) async {
        do {
            try await Client.send("DELETE", "/admin/sessions/\(Client.at(identifier))")
            summaries.removeAll { $0.id == identifier }
            trouble = ""
        } catch {
            trouble = said(error)
        }
    }

    private func patch(_ identifier: String, _ body: SessionPatch) async {
        do {
            try await Client.send("PATCH", "/admin/sessions/\(Client.at(identifier))", body: body)
            await load()
        } catch {
            trouble = said(error)
        }
    }

    private func said(_ error: Error) -> String {
        (error as? Refused)?.detail ?? error.localizedDescription
    }
}

/// The title a conversation is listed under: the first thing said, whitespace collapsed, cut
/// at 48 with an ellipsis. Rows written before this panel existed carry a bare `[:48]` of the
/// same text, so the list is read across two shapes already and this adds no third.
func titled(_ text: String) -> String {
    let line = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    guard line.count > 48 else { return line }
    let cut = String(line.prefix(48))
    return cut.replacingOccurrences(
        of: "\\s+$", with: "", options: .regularExpression
    ) + "…"
}

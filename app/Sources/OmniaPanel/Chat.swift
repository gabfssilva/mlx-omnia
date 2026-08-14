// The conversation, and what it costs.
//
// This chat is an instrument and not a client: production traffic goes to the API, and what
// comes here is the turn somebody runs to check that a model still answers, still sees, and
// still writes a call in the shape its dialect declares. Three things follow from that, and
// they are the whole design.
//
// Every turn keeps its own numbers. The composer says what will be sent before it is sent,
// including what the picture costs. And nothing is offered that the model would refuse:
// `sees` decides whether the door opens at all.
//
// The transcript lives here rather than in a view because the panel is dismissed by a click
// anywhere else on the screen — a conversation held by the view is a conversation lost by
// looking away. It is written to `/admin/sessions` for the same reason one step further out:
// the panel is dismissed, and the daemon is not. That store is the window's too, so what is
// written there is written in the window's own vocabulary (`Sessions.swift`).

import Foundation
import Observation

struct ToolCall: Identifiable, Equatable {
    let index: Int
    var name = ""
    var arguments = ""
    var id: Int { index }
}

struct Turn: Identifiable, Equatable {
    let id = UUID()
    let role: String
    var text = ""
    var reasoning = ""
    var clips: [Clip] = []
    var calls: [ToolCall] = []
    /// What the turn cost, off the frame that closes it — or off nothing at all while it is
    /// still being written, which is why every field is optional.
    var timings: Timings?
    var completionTokens: Int?
    /// How long the reasoning took, measured from the first thinking token to the first
    /// answer token. The daemon does not report it; the wall clock is the honest source
    /// because what is being timed is the wait, not the arithmetic.
    var thoughtFor: Double?
    var finish: String?
    var failure: String?

    var writing = false

    static func == (lhs: Turn, rhs: Turn) -> Bool {
        lhs.id == rhs.id
            && lhs.text == rhs.text
            && lhs.reasoning == rhs.reasoning
            && lhs.calls == rhs.calls
            && lhs.writing == rhs.writing
            && lhs.failure == rhs.failure
            && lhs.completionTokens == rhs.completionTokens
    }

    func metricsLine(live: Sample?) -> String? {
        var said: [String] = []
        if let ttft = timings?.ttftSeconds ?? live?.ttft {
            said.append("\(Fmt.duration(ttft)) TTFT")
        }
        if let prefill = timings?.prefillTokensPerSecond ?? live?.prefillTokensPerSecond {
            said.append("\(Fmt.grouped(prefill)) prefill")
        }
        if let decode = timings?.tokensPerSecond ?? live?.tokensPerSecond {
            said.append(String(format: "%.1f tok/s", decode))
        }
        if let tokens = completionTokens ?? live?.completionTokens, tokens > 0 {
            said.append("\(Fmt.grouped(Double(tokens))) tokens")
        }
        if let spec = timings?.speculation ?? live?.speculation, spec.rounds > 0 {
            said.append(String(
                format: "%.1f of %.0f speculated",
                Double(spec.accepted) / Double(spec.rounds),
                Double(spec.proposed) / Double(spec.rounds)
            ))
        }
        return said.isEmpty ? nil : said.joined(separator: " · ")
    }
}

/// The knobs this panel names on a request — its overrides. A named knob beats the profile
/// the request also names, which beats the checkpoint's own defaults: the daemon's order,
/// and the reason an unnamed knob has to stay out of the body rather than be restated at
/// some default.
struct Knobs: Equatable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var minP: Double?
    /// `default` leaves the field out, which is the checkpoint's template deciding.
    var effort = "default"
    var systemPrompt = ""
    var tools = ""

    static func of(_ entry: CatalogEntry?) -> Knobs {
        var made = Knobs()
        guard let declared = entry?.defaults else { return made }
        // A declared 0.0 is kept: `do_sample: false` reaches the catalog as temperature 0,
        // which is the checkpoint saying greedy.
        made.temperature = declared.temperature
        made.topP = declared.topP
        made.topK = declared.topK
        made.minP = declared.minP
        return made
    }
}

@MainActor
@Observable
final class ChatModel {
    var turns: [Turn] = []
    var draft = ""
    var clips: [Clip] = []
    var knobs = Knobs()
    /// The conversations in the daemon's store, and this conversation's place in them: the
    /// row is empty until the first turn creates it.
    let sessions = Sessions()
    var session = ""
    var title = ""
    var restored = false
    /// Which checkpoint answers. Empty until a model is resident or one is picked.
    var model = ""
    /// Whether that checkpoint takes images, off the catalog. Kept here and not read from
    /// the store on the spot because it decides whether the turn can be sent at all.
    var sees = false
    var refusal = ""
    var grabbing = false

    private var writing: Task<Void, Never>?
    private var priced: Task<Void, Never>?

    var streaming: Bool { writing != nil }

    func shouldShowLoadNote(resident: Bool) -> Bool {
        restored && !model.isEmpty && !resident
    }

    func loadReceipt(after index: Int, liveLoad: Double?) -> Double? {
        guard turns.indices.contains(index), turns[index].role == "user" else { return nil }
        let answer = index + 1
        guard turns.indices.contains(answer), turns[answer].role == "assistant" else { return nil }
        return turns[answer].timings?.loadSeconds ?? (turns[answer].writing ? liveLoad : nil)
    }

    /// A clip the model in force cannot read is not something to send: the daemon would
    /// refuse the turn, and a lit button that produces a refusal is the panel promising
    /// what the engine has already said it will not do.
    var canSend: Bool {
        guard !model.isEmpty, clips.isEmpty || sees else { return false }
        return !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !clips.isEmpty
    }

    // ── what the turn will cost before it is sent ────────────────────────

    /// The rows the clips reserve, as the daemon priced them. Nil while any of them is
    /// still unpriced: a partial sum reads as the whole one.
    var clipTokens: Int? {
        let counted = clips.compactMap(\.cost?.tokens)
        return counted.count == clips.count ? counted.reduce(0, +) : nil
    }

    /// What the reading over the transcript says: every turn's text, roughly in tokens,
    /// plus what the pictures cost exactly. Rough because the panel does not have the
    /// tokenizer — 4 characters to the token is the ratio the window uses for the same
    /// line, and the number it produces is an order of magnitude, not a measurement.
    func used(_ pending: Bool) -> Int {
        let written = turns.reduce(0) { total, turn in
            total + turn.text.count / 4 + turn.reasoning.count / 4
        }
        let images = turns.flatMap(\.clips).compactMap(\.cost?.tokens).reduce(0, +)
        guard pending else { return written + images }
        return written + images + draft.count / 4 + (clipTokens ?? 0)
    }

    // ── the model in force ───────────────────────────────────────────────

    func adopt(_ store: EngineStore) {
        // The first resident model, and only while none has been picked: a panel that
        // reopened on a conversation keeps whatever it was talking to.
        guard model.isEmpty, let first = store.models.first?.id else { return }
        use(first, store.entry(first))
    }

    func use(_ identifier: String, _ entry: CatalogEntry?) {
        sees = entry?.sees == true
        guard identifier != model else { return }
        model = identifier
        // The knobs belong to the checkpoint they were tuned against, so another model is
        // a clean panel — the same rule the window's picker follows.
        knobs = Knobs.of(entry)
        refusal = ""
        // And so does the price: the arithmetic is the family's, and a count carried over
        // from the last model is a number about a different tower.
        for index in clips.indices { clips[index].cost = nil }
        price()
    }

    // ── the clips ────────────────────────────────────────────────────────

    func attach(_ clip: Clip?) {
        guard let clip else { return }
        clips.append(clip)
        price()
    }

    func drop(_ clip: Clip) {
        clips.removeAll { $0.id == clip.id }
    }

    /// Ask the daemon what each clip costs this checkpoint. One request per clip and per
    /// model, never per keystroke — and a refusal leaves the count nil rather than putting
    /// up a guess.
    func price() {
        priced?.cancel()
        let asked = model
        guard !asked.isEmpty, !clips.isEmpty else { return }
        priced = Task { [weak self] in
            guard let self else { return }
            for clip in clips where clip.cost == nil {
                let cost: ImageCost? = try? await Client.get(
                    "/admin/models/\(Client.at(asked))/image",
                    query: ["height": String(clip.height), "width": String(clip.width)]
                )
                guard !Task.isCancelled, asked == model else { return }
                if let cost, let index = clips.firstIndex(where: { $0.id == clip.id }) {
                    clips[index].cost = cost
                }
            }
        }
    }

    // ── the turn ─────────────────────────────────────────────────────────

    /// A new conversation. What was said stays in the store — this is the panel letting go
    /// of it, not the store.
    func clear() {
        writing?.cancel()
        writing = nil
        turns = []
        clips = []
        draft = ""
        refusal = ""
        session = ""
        title = ""
        restored = false
        sessions.trouble = ""
    }

    func stop() {
        let running = writing
        running?.cancel()
        writing = nil
        if let last = turns.indices.last, turns[last].writing {
            turns[last].writing = false
            turns[last].finish = "stopped"
        }
        // A stopped answer is still an answer somebody may want tomorrow, and the task that
        // would have filed it is the one just cancelled. Filed after it has unwound, because
        // what it does on the way out is paint the tokens it was still holding — filing first
        // would store an answer shorter than the one on screen.
        Task { [weak self] in
            _ = await running?.value
            await self?.keep()
        }
    }

    func send() {
        guard canSend, !streaming else { return }
        let said = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let opening = turns.isEmpty
        turns.append(Turn(role: "user", text: said, clips: clips))
        turns.append(Turn(role: "assistant", writing: true))
        draft = ""
        clips = []
        refusal = ""

        let body = request()
        writing = Task { [weak self] in
            // The row before the answer: a panel dismissed mid-turn leaves a conversation in
            // the list either way, and the list is the only way back to it.
            await self?.enrol(said, opening: opening)
            await self?.follow(body)
            self?.writing = nil
            guard !Task.isCancelled else { return }
            await self?.keep()
        }
    }

    // ── the store ────────────────────────────────────────────────────────

    /// Work a row or a sheet started, run where the conversation lives. A refusal is already
    /// the store's to report — it leaves it on `sessions.trouble`.
    func act(_ work: @escaping () async -> Void) {
        Task { await work() }
    }

    /// The row this conversation is, created on the first turn and named after it. A row that
    /// exists and has nothing in it yet — the window's `+` makes those — is renamed instead,
    /// which is the rule the window follows for the same case.
    private func enrol(_ said: String, opening: Bool) async {
        let name = titled(said)
        guard !session.isEmpty else {
            title = name
            session = await sessions.begin(title: name, model: model) ?? ""
            return
        }
        if opening {
            title = name
            await sessions.rename(session, to: name)
        }
    }

    /// The conversation as the store holds it: the whole array, every time.
    private func keep() async {
        guard !session.isEmpty else { return }
        await sessions.write(session, stored())
        if sessions.summary(session)?.model != model, !model.isEmpty {
            // Reopening has to give back the model the conversation was last answered by,
            // so the row follows whatever is in force when a turn lands.
            await sessions.moved(session, to: model)
        }
    }

    private func stored() -> [StoredMessage] {
        var written: [StoredMessage] = []
        let system = knobs.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            written.append(StoredMessage(role: "system", content: system))
        }
        for turn in turns where !turn.writing {
            written.append(
                StoredMessage(
                    role: turn.role,
                    content: turn.text,
                    reasoningContent: turn.reasoning.isEmpty ? nil : turn.reasoning,
                    metrics: measures(turn),
                    error: turn.failure
                )
            )
        }
        return written
    }

    private func measures(_ turn: Turn) -> StoredMetrics? {
        guard turn.role == "assistant", turn.timings != nil || turn.finish != nil else {
            return nil
        }
        return StoredMetrics(
            loadMs: turn.timings?.loadSeconds.map { $0 * 1000 },
            ttftMs: turn.timings?.ttftSeconds.map { $0 * 1000 },
            prefillTokensPerSecond: turn.timings?.prefillTokensPerSecond,
            tokensPerSecond: turn.timings?.tokensPerSecond,
            ceilingFraction: turn.timings?.ceilingFraction,
            finish: turn.finish
        )
    }

    /// Read one back. The model comes with it — a conversation is answered by what answered
    /// it last — and so does the system prompt, which is a knob here and a message there.
    func reopen(_ identifier: String, _ store: EngineStore) async {
        guard let held = await sessions.open(identifier) else {
            refusal = "That conversation is not in the store any more."
            return
        }
        writing?.cancel()
        writing = nil
        clips = []
        draft = ""
        refusal = ""
        session = held.id
        title = held.title
        restored = true
        if !held.model.isEmpty { use(held.model, store.entry(held.model)) }
        turns = held.messages.compactMap(Self.turn)
        if let system = held.messages.first(where: { $0.role == "system" }) {
            knobs.systemPrompt = system.content
        }
    }

    private static func turn(_ message: StoredMessage) -> Turn? {
        guard message.role != "system" else { return nil }
        var made = Turn(role: message.role, text: message.content)
        made.reasoning = message.reasoningContent ?? ""
        made.failure = message.error
        made.finish = message.metrics?.finish
        if let measured = message.metrics {
            made.timings = Timings(
                loadSeconds: measured.loadMs.map { $0 / 1000 },
                ttftSeconds: measured.ttftMs.map { $0 / 1000 },
                prefillTokensPerSecond: measured.prefillTokensPerSecond,
                tokensPerSecond: measured.tokensPerSecond,
                ceilingFraction: measured.ceilingFraction,
                // The store keeps the seven keys `conversation.py` writes, and the
                // acceptance of a speculated round is not one of them: a turn read back
                // off disk knows what it cost and not how it was drafted.
                speculation: nil
            )
        }
        return made
    }

    /// The conversation as the dialect reads it. The turn just appended is already in
    /// `turns`, so history and the question are one loop — and the assistant's empty
    /// placeholder falls out of it by being empty.
    private func request() -> ChatBody {
        var messages: [SentMessage] = []
        let system = knobs.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            messages.append(SentMessage(role: "system", content: .text(system)))
        }
        for turn in turns where turn.role == "user" || !turn.text.isEmpty {
            guard turn.role != "system" else { continue }
            if turn.clips.isEmpty {
                messages.append(SentMessage(role: turn.role, content: .text(turn.text)))
                continue
            }
            // The picture comes before the words, which is the order every VL template in
            // circulation renders and the order the question reads in: this, then what
            // about it.
            var parts: [ContentPart] = turn.clips.map { .image($0.uri) }
            if !turn.text.isEmpty { parts.append(.text(turn.text)) }
            messages.append(SentMessage(role: turn.role, content: .parts(parts)))
        }

        var body = ChatBody(model: model, messages: messages)
        body.temperature = knobs.temperature
        body.topP = knobs.topP
        body.topK = knobs.topK
        body.minP = knobs.minP
        if knobs.effort != "default" { body.reasoningEffort = knobs.effort }
        body.tools = declaredTools()
        return body
    }

    /// The tools field as typed, or nothing. Unreadable JSON is left out of the request and
    /// said out loud instead: a schema the daemon never saw is a tool call that never comes,
    /// and "the model did not call it" is the wrong lesson to draw from a typo.
    private func declaredTools() -> [AnyJSON]? {
        let text = knobs.tools.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        guard let parsed = try? JSONSerialization.jsonObject(with: Data(text.utf8)),
              let list = parsed as? [Any]
        else {
            refusal = "Tools is not a JSON array — it was left out of this request."
            return nil
        }
        return list.map(AnyJSON.init)
    }

    /// Twenty-five a second, which is what the eye reads. Every write to a turn invalidates
    /// the transcript, and the transcript is every turn in the conversation — a model that
    /// decodes at 80 tok/s would otherwise lay the whole screen out three times per frame the
    /// screen actually draws.
    private static let paint: TimeInterval = 1.0 / 25

    private func follow(_ body: ChatBody) async {
        let started = Date()
        var thinkingSince: Date?
        // Held out of `turns` between paints. Only the two that arrive per token are buffered;
        // a call, a finish reason and the closing numbers land once each and go straight in.
        var text = ""
        var thought = ""
        var painted = Date.distantPast

        func paint(_ force: Bool) {
            guard !text.isEmpty || !thought.isEmpty else { return }
            let now = Date()
            guard force || now.timeIntervalSince(painted) >= Self.paint else { return }
            painted = now
            guard let last = turns.indices.last else { return }
            turns[last].reasoning += thought
            turns[last].text += text
            text = ""
            thought = ""
        }

        do {
            for try await frame in Client.events("/api/openai/v1/chat/completions", body: body) {
                if String(data: frame, encoding: .utf8) == "[DONE]" { break }
                guard let chunk = try? Client.decoder.decode(ChatChunk.self, from: frame) else {
                    continue
                }
                guard let last = turns.indices.last else { break }
                if let choice = chunk.choices?.first {
                    if let said = choice.delta?.reasoningContent, !said.isEmpty {
                        if thinkingSince == nil { thinkingSince = Date() }
                        thought += said
                    }
                    if let said = choice.delta?.content, !said.isEmpty {
                        // The turn between the two channels, taken where it happens rather
                        // than at the next paint: what is being timed is the wait.
                        if turns[last].text.isEmpty, text.isEmpty, let since = thinkingSince {
                            turns[last].thoughtFor = Date().timeIntervalSince(since)
                        }
                        text += said
                    }
                    for call in choice.delta?.toolCalls ?? [] { write(call, into: last) }
                    if let finish = choice.finishReason { turns[last].finish = finish }
                }
                if let usage = chunk.usage { turns[last].completionTokens = usage.completionTokens }
                if let timings = chunk.xMlxOmnia { turns[last].timings = timings }
                paint(false)
            }
        } catch {
            if let last = turns.indices.last {
                turns[last].failure = (error as? Refused)?.detail ?? error.localizedDescription
            }
        }
        paint(true)
        if let last = turns.indices.last {
            turns[last].writing = false
            if turns[last].thoughtFor == nil, !turns[last].reasoning.isEmpty {
                turns[last].thoughtFor = Date().timeIntervalSince(started)
            }
        }
    }

    /// A call arrives in pieces: the index says which one, the name lands once, and the
    /// arguments accumulate as JSON text. Rebuilding it here is what lets the screen show a
    /// call being written rather than a call that has been.
    private func write(_ delta: ToolCallDelta, into turn: Int) {
        if let seat = turns[turn].calls.firstIndex(where: { $0.index == delta.index }) {
            if let name = delta.function?.name, !name.isEmpty { turns[turn].calls[seat].name = name }
            if let part = delta.function?.arguments { turns[turn].calls[seat].arguments += part }
            return
        }
        var made = ToolCall(index: delta.index)
        made.name = delta.function?.name ?? ""
        made.arguments = delta.function?.arguments ?? ""
        turns[turn].calls.append(made)
    }
}

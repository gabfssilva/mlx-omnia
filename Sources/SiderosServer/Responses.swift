import Foundation
import Hummingbird
import Sideros

/// `/v1/responses`. The conversation is a flat list of items rather than messages, the
/// system prompt is `instructions`, and the stream is a numbered sequence of typed events
/// the SDK replays into a Response. Stateless: `previous_response_id` is refused rather
/// than ignored, because ignoring it silently drops the history.
enum Responses {
    struct Payload: Decodable {
        let input: Input
        let instructions: String?
        let maxOutputTokens: Int?
        let temperature: Float?
        let topP: Float?
        let stream: Bool?
        let previousResponseId: String?

        /// A bare string is shorthand for a single user turn.
        struct Input: Decodable {
            let items: [Item]

            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let text = try? container.decode(String.self) {
                    items = [Item(role: "user", content: text)]
                } else {
                    items = try container.decode([Item].self)
                }
            }
        }

        struct Item: Decodable {
            let role: String
            let content: Content

            init(role: String, content: String) {
                self.role = role
                self.content = Content(content)
            }
        }
    }

    struct Reply: Encodable {
        let id: String
        let object = "response"
        let createdAt: Int
        let status: String
        let model: String
        let output: [Item]
        let usage: Usage?
        let incompleteDetails: Incomplete?
        let instructions: String?
        let temperature: Float?
        let topP: Float?
        // The SDK's model requires these, and this server has no tools yet.
        let tools: [String] = []
        let toolChoice = "auto"
        let parallelToolCalls = false

        struct Incomplete: Encodable {
            let reason: String
        }
    }

    struct Item: Encodable {
        let id: String
        let type = "message"
        let status: String
        let role = "assistant"
        let content: [Part]
    }

    struct Part: Encodable {
        let type = "output_text"
        let text: String
        let annotations: [String] = []
    }

    struct Usage: Encodable {
        let inputTokens: Int
        let outputTokens: Int
        let totalTokens: Int

        init(_ finish: Finish) {
            inputTokens = finish.promptTokens
            outputTokens = finish.completionTokens
            totalTokens = finish.promptTokens + finish.completionTokens
        }
    }

    static func create(_ request: Request, engine: Engine) async throws -> Response {
        let payload = try await JSON.decode(Payload.self, from: request)
        guard payload.previousResponseId == nil else {
            throw HTTPError(.badRequest, message: "previous_response_id is not supported")
        }

        let instructions = payload.instructions.map { [Message(role: .system, content: $0)] } ?? []
        let canonical = ChatRequest(
            messages: instructions + (try payload.input.items.map(canonicalize)),
            maxTokens: payload.maxOutputTokens,
            temperature: payload.temperature,
            topP: payload.topP,
            stream: payload.stream ?? false)

        let id = "resp_\(UUID().uuidString)"
        let itemId = "msg_\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let stream = try engine.stream(canonical)

        @Sendable func reply(_ text: String?, finish: Finish?) -> Reply {
            Reply(
                id: id, createdAt: created,
                status: status(finish), model: engine.name,
                output: text.map {
                    [Item(id: itemId, status: status(finish), content: [Part(text: $0)])]
                } ?? [],
                usage: finish.map(Usage.init),
                incompleteDetails: finish?.reason == .length
                    ? .init(reason: "max_output_tokens") : nil,
                instructions: payload.instructions, temperature: payload.temperature,
                topP: payload.topP)
        }

        guard canonical.stream else {
            var text = ""
            var finish: Finish?
            for try await item in stream {
                switch item {
                case .start: break
                case .delta(let delta): text += delta
                case .done(let done): finish = done
                }
            }
            guard let finish else { throw HTTPError(.internalServerError) }
            return try JSON.response(reply(text, finish: finish))
        }

        return SSE.response { writer in
            var sequence = 0
            func next() -> Int {
                defer { sequence += 1 }
                return sequence
            }

            var text = ""
            for try await item in stream {
                switch item {
                case .start:
                    for type in ["response.created", "response.in_progress"] {
                        try await writer.send(
                            ResponseFrame(type: type, sequenceNumber: next(), response: reply(nil, finish: nil)),
                            event: type)
                    }
                    try await writer.send(
                        ItemFrame(
                            type: "response.output_item.added", sequenceNumber: next(),
                            item: Item(id: itemId, status: "in_progress", content: [])),
                        event: "response.output_item.added")
                    try await writer.send(
                        PartFrame(
                            type: "response.content_part.added", sequenceNumber: next(),
                            itemId: itemId, part: Part(text: "")),
                        event: "response.content_part.added")
                case .delta(let delta):
                    text += delta
                    try await writer.send(
                        DeltaFrame(
                            type: "response.output_text.delta", sequenceNumber: next(),
                            itemId: itemId, delta: delta),
                        event: "response.output_text.delta")
                case .done(let done):
                    try await writer.send(
                        TextFrame(
                            type: "response.output_text.done", sequenceNumber: next(),
                            itemId: itemId, text: text),
                        event: "response.output_text.done")
                    try await writer.send(
                        PartFrame(
                            type: "response.content_part.done", sequenceNumber: next(),
                            itemId: itemId, part: Part(text: text)),
                        event: "response.content_part.done")
                    try await writer.send(
                        ItemFrame(
                            type: "response.output_item.done", sequenceNumber: next(),
                            item: Item(id: itemId, status: "completed", content: [Part(text: text)])),
                        event: "response.output_item.done")
                    let type = done.reason == .length ? "response.incomplete" : "response.completed"
                    try await writer.send(
                        ResponseFrame(
                            type: type, sequenceNumber: next(),
                            response: reply(text, finish: done)),
                        event: type)
                }
            }
        }
    }

    private struct ResponseFrame: Encodable {
        let type: String
        let sequenceNumber: Int
        let response: Reply
    }

    private struct ItemFrame: Encodable {
        let type: String
        let sequenceNumber: Int
        let outputIndex = 0
        let item: Item
    }

    private struct PartFrame: Encodable {
        let type: String
        let sequenceNumber: Int
        let itemId: String
        let outputIndex = 0
        let contentIndex = 0
        let part: Part
    }

    private struct DeltaFrame: Encodable {
        let type: String
        let sequenceNumber: Int
        let itemId: String
        let outputIndex = 0
        let contentIndex = 0
        let delta: String
    }

    private struct TextFrame: Encodable {
        let type: String
        let sequenceNumber: Int
        let itemId: String
        let outputIndex = 0
        let contentIndex = 0
        let text: String
    }

    private static func status(_ finish: Finish?) -> String {
        guard let finish else { return "in_progress" }
        return finish.reason == .length ? "incomplete" : "completed"
    }

    private static func canonicalize(_ item: Payload.Item) throws -> Message {
        guard let role = Message.Role(rawValue: item.role) else {
            throw HTTPError(.badRequest, message: "unsupported role: \(item.role)")
        }
        return Message(role: role, content: item.content.text)
    }
}

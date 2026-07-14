import Foundation
import Hummingbird
import Sideros

/// `/v1/chat/completions` and `/v1/models`: the dialect litellm speaks by default.
enum OpenAI {
    struct Payload: Decodable {
        let messages: [Turn]
        let maxTokens: Int?
        /// What the current SDK sends; `max_tokens` is the deprecated spelling.
        let maxCompletionTokens: Int?
        let temperature: Float?
        let topP: Float?
        let stop: Stop?
        let seed: UInt64?
        let stream: Bool?
        let streamOptions: StreamOptions?

        struct Turn: Decodable {
            let role: String
            let content: Content?
        }

        struct StreamOptions: Decodable {
            let includeUsage: Bool?
        }

        /// One string or a list of them.
        struct Stop: Decodable {
            let all: [String]

            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let one = try? container.decode(String.self) {
                    all = [one]
                } else {
                    all = try container.decode([String].self)
                }
            }
        }
    }

    struct Completion: Encodable {
        let id: String
        let object = "chat.completion"
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage

        struct Choice: Encodable {
            let index = 0
            let message: Turn
            let finishReason: String
        }

        struct Turn: Encodable {
            let role = "assistant"
            let content: String
        }
    }

    struct Chunk: Encodable {
        let id: String
        let object = "chat.completion.chunk"
        let created: Int
        let model: String
        let choices: [Choice]
        /// Last frame only, and only when the client asked for it.
        let usage: Usage?

        struct Choice: Encodable {
            let index = 0
            let delta: Delta
            let finishReason: String?
        }

        struct Delta: Encodable {
            let role: String?
            let content: String?
        }
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        init(_ finish: Finish) {
            promptTokens = finish.promptTokens
            completionTokens = finish.completionTokens
            totalTokens = finish.promptTokens + finish.completionTokens
        }
    }

    struct Models: Encodable {
        let object = "list"
        let data: [Entry]

        struct Entry: Encodable {
            let id: String
            let object = "model"
            let created = 0
            let ownedBy = "sideros"
        }
    }

    static func chat(_ request: Request, engine: Engine) async throws -> Response {
        let payload = try await JSON.decode(Payload.self, from: request)
        let canonical = ChatRequest(
            messages: try payload.messages.map(canonicalize),
            maxTokens: payload.maxCompletionTokens ?? payload.maxTokens,
            temperature: payload.temperature,
            topP: payload.topP,
            stop: payload.stop?.all ?? [],
            seed: payload.seed,
            stream: payload.stream ?? false)

        let id = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let stream = try engine.stream(canonical)

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
            return try JSON.response(
                Completion(
                    id: id, created: created, model: engine.name,
                    choices: [
                        .init(message: .init(content: text), finishReason: reason(finish.reason))
                    ],
                    usage: Usage(finish)))
        }

        let includeUsage = payload.streamOptions?.includeUsage ?? false
        return SSE.response { writer in
            func chunk(_ delta: Chunk.Delta, finish: String? = nil, usage: Usage? = nil) -> Chunk {
                Chunk(
                    id: id, created: created, model: engine.name,
                    choices: [.init(delta: delta, finishReason: finish)], usage: usage)
            }

            try await writer.send(chunk(.init(role: "assistant", content: nil)))
            for try await item in stream {
                switch item {
                case .start:
                    break
                case .delta(let delta):
                    try await writer.send(chunk(.init(role: nil, content: delta)))
                case .done(let done):
                    try await writer.send(
                        chunk(
                            .init(role: nil, content: nil), finish: reason(done.reason),
                            usage: includeUsage ? Usage(done) : nil))
                }
            }
            try await writer.send(raw: "[DONE]")
        }
    }

    static func models(engine: Engine) throws -> Response {
        try JSON.response(Models(data: [.init(id: engine.name)]))
    }

    private static func reason(_ reason: Finish.Reason) -> String {
        switch reason {
        case .stop: "stop"
        case .length: "length"
        }
    }

    private static func canonicalize(_ turn: Payload.Turn) throws -> SiderosServer.Message {
        guard let role = SiderosServer.Message.Role(rawValue: turn.role) else {
            throw HTTPError(.badRequest, message: "unsupported role: \(turn.role)")
        }
        return SiderosServer.Message(role: role, content: turn.content?.text ?? "")
    }
}

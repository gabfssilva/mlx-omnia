import Foundation
import Hummingbird
import Sideros

/// `/v1beta/models/{model}:generateContent` and `:streamGenerateContent`. The odd one out
/// twice over: the fields are camelCase, and the method rides in the path after a colon.
enum Gemini {
    struct Payload: Decodable {
        let contents: [Content]
        let systemInstruction: Content?
        let generationConfig: Config?

        struct Content: Decodable {
            let role: String?
            let parts: [Part]

            var text: String { parts.compactMap(\.text).joined() }
        }

        struct Part: Decodable {
            let text: String?
        }

        struct Config: Decodable {
            let temperature: Float?
            let topP: Float?
            let maxOutputTokens: Int?
            let stopSequences: [String]?
            let seed: UInt64?
        }
    }

    struct Reply: Encodable {
        let candidates: [Candidate]
        let usageMetadata: Usage?
        let modelVersion: String

        struct Candidate: Encodable {
            let content: Content
            let finishReason: String?
            let index = 0
        }

        struct Content: Encodable {
            let role = "model"
            let parts: [Part]
        }

        struct Part: Encodable {
            let text: String
        }

        struct Usage: Encodable {
            let promptTokenCount: Int
            let candidatesTokenCount: Int
            let totalTokenCount: Int

            init(_ finish: Finish) {
                promptTokenCount = finish.promptTokens
                candidatesTokenCount = finish.completionTokens
                totalTokenCount = finish.promptTokens + finish.completionTokens
            }
        }
    }

    static func generate(_ request: Request, engine: Engine) async throws -> Response {
        // ".../models/<name>:<method>" — the name is whatever the client asked for; this
        // server has one model and answers for it, like the other dialects do.
        guard let method = request.uri.path.split(separator: ":").last else {
            throw HTTPError(.notFound)
        }
        let streaming = method == "streamGenerateContent"
        guard streaming || method == "generateContent" else {
            throw HTTPError(.notFound, message: "unsupported method: \(method)")
        }

        let payload = try await JSON.decode(Payload.self, from: request, snakeCase: false)
        let config = payload.generationConfig
        let system = payload.systemInstruction.map { [Message(role: .system, content: $0.text)] }
        let canonical = ChatRequest(
            messages: (system ?? []) + payload.contents.map(canonicalize),
            maxTokens: config?.maxOutputTokens,
            temperature: config?.temperature,
            topP: config?.topP,
            stop: config?.stopSequences ?? [],
            seed: config?.seed,
            stream: streaming)

        let stream = try engine.stream(canonical)

        @Sendable func reply(_ text: String, finish: Finish?) -> Reply {
            Reply(
                candidates: [
                    .init(
                        content: .init(parts: [.init(text: text)]),
                        finishReason: finish.map(reason))
                ],
                usageMetadata: finish.map(Reply.Usage.init), modelVersion: engine.name)
        }

        guard streaming else {
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
            return try JSON.response(reply(text, finish: finish), snakeCase: false)
        }

        return SSE.response { writer in
            var last = ""
            for try await item in stream {
                switch item {
                case .start:
                    break
                case .delta(let delta):
                    // Held back one frame: the finish reason and the usage ride on the last
                    // candidate, and until the next delta arrives we do not know it is not it.
                    if !last.isEmpty {
                        try await writer.send(reply(last, finish: nil), snakeCase: false)
                    }
                    last = delta
                case .done(let done):
                    try await writer.send(reply(last, finish: done), snakeCase: false)
                }
            }
        }
    }

    private static func canonicalize(_ content: Payload.Content) -> Message {
        Message(role: content.role == "model" ? .assistant : .user, content: content.text)
    }

    private static func reason(_ finish: Finish) -> String {
        switch finish.reason {
        case .stop: "STOP"
        case .length: "MAX_TOKENS"
        }
    }
}

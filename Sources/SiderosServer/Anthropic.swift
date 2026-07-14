import Foundation
import Hummingbird
import Sideros

/// `/v1/messages`. Unlike OpenAI's, every SSE frame here is named and typed, and the system
/// prompt arrives beside the turns instead of as one of them.
enum Anthropic {
    struct Payload: Decodable {
        let messages: [Turn]
        let system: Content?
        let maxTokens: Int?
        let temperature: Float?
        let topP: Float?
        let stopSequences: [String]?
        let stream: Bool?

        struct Turn: Decodable {
            let role: String
            let content: Content?
        }
    }

    /// Also the body of `message_start`, where the content is empty and the stop is not
    /// known yet.
    struct Reply: Encodable {
        let id: String
        let type = "message"
        let role = "assistant"
        let content: [Block]
        let model: String
        let stopReason: String?
        let stopSequence: String?
        let usage: Usage
    }

    struct Block: Encodable {
        let type = "text"
        let text: String
    }

    struct Usage: Encodable {
        let inputTokens: Int
        let outputTokens: Int
    }

    static func messages(_ request: Request, engine: Engine) async throws -> Response {
        let payload = try await JSON.decode(Payload.self, from: request)
        let system = payload.system.map { [Message(role: .system, content: $0.text)] } ?? []
        let canonical = ChatRequest(
            messages: system + (try payload.messages.map(canonicalize)),
            maxTokens: payload.maxTokens,
            temperature: payload.temperature,
            topP: payload.topP,
            stop: payload.stopSequences ?? [],
            stream: payload.stream ?? false)

        let id = "msg_\(UUID().uuidString)"
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
                Reply(
                    id: id, content: [Block(text: text)], model: engine.name,
                    stopReason: reason(finish), stopSequence: finish.stopString,
                    usage: Usage(
                        inputTokens: finish.promptTokens, outputTokens: finish.completionTokens)))
        }

        return SSE.response { writer in
            for try await item in stream {
                switch item {
                case .start(let promptTokens):
                    try await writer.send(
                        Start(
                            message: Reply(
                                id: id, content: [], model: engine.name, stopReason: nil,
                                stopSequence: nil,
                                usage: Usage(inputTokens: promptTokens, outputTokens: 0))),
                        event: "message_start")
                    try await writer.send(BlockStart(), event: "content_block_start")
                case .delta(let delta):
                    try await writer.send(
                        BlockDelta(delta: .init(text: delta)), event: "content_block_delta")
                case .done(let done):
                    try await writer.send(BlockStop(), event: "content_block_stop")
                    try await writer.send(
                        MessageDelta(
                            delta: .init(stopReason: reason(done), stopSequence: done.stopString),
                            usage: .init(outputTokens: done.completionTokens)),
                        event: "message_delta")
                    try await writer.send(Stop(), event: "message_stop")
                }
            }
        }
    }

    private struct Start: Encodable {
        let type = "message_start"
        let message: Reply
    }

    private struct BlockStart: Encodable {
        let type = "content_block_start"
        let index = 0
        let contentBlock = Block(text: "")
    }

    private struct BlockDelta: Encodable {
        let type = "content_block_delta"
        let index = 0
        let delta: TextDelta

        struct TextDelta: Encodable {
            let type = "text_delta"
            let text: String
        }
    }

    private struct BlockStop: Encodable {
        let type = "content_block_stop"
        let index = 0
    }

    private struct MessageDelta: Encodable {
        let type = "message_delta"
        let delta: Reason
        let usage: Output

        struct Reason: Encodable {
            let stopReason: String
            let stopSequence: String?
        }

        struct Output: Encodable {
            let outputTokens: Int
        }
    }

    private struct Stop: Encodable {
        let type = "message_stop"
    }

    /// A stop string and the model's own end token are the same `.stop` to the engine, and
    /// two different reasons here.
    private static func reason(_ finish: Finish) -> String {
        switch finish.reason {
        case .length: "max_tokens"
        case .stop: finish.stopString == nil ? "end_turn" : "stop_sequence"
        }
    }

    private static func canonicalize(_ turn: Payload.Turn) throws -> Message {
        guard let role = Message.Role(rawValue: turn.role) else {
            throw HTTPError(.badRequest, message: "unsupported role: \(turn.role)")
        }
        return Message(role: role, content: turn.content?.text ?? "")
    }
}

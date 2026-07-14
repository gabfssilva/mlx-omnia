import Foundation
import Hummingbird

enum SSE {
    /// The streaming response every dialect writes into. Each one names its events
    /// differently — OpenAI sends anonymous data frames and a literal `[DONE]`, Anthropic
    /// types every frame — so the shape of the events stays with the dialect.
    static func response(
        _ write: @escaping @Sendable (inout any ResponseBodyWriter) async throws -> Void
    ) -> Response {
        Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
            body: .init { writer in
                try await write(&writer)
                try await writer.finish(nil)
            })
    }
}

extension ResponseBodyWriter {
    mutating func send(_ value: some Encodable, event: String? = nil, snakeCase: Bool = true)
        async throws
    {
        let data = String(buffer: try JSON.encode(value, snakeCase: snakeCase))
        let name = event.map { "event: \($0)\n" } ?? ""
        try await write(ByteBuffer(string: "\(name)data: \(data)\n\n"))
    }

    mutating func send(raw: String) async throws {
        try await write(ByteBuffer(string: "data: \(raw)\n\n"))
    }
}

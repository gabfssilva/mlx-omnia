import Foundation
import Hummingbird

/// Key casing is per dialect, not per server: OpenAI and Anthropic spell their fields in
/// snake_case, Gemini in camelCase. So the codec is chosen at the handler, not installed in
/// the request context.
enum JSON {
    static func decode<T: Decodable>(_ type: T.Type, from request: Request, snakeCase: Bool = true)
        async throws -> T
    {
        let body = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let decoder = JSONDecoder()
        if snakeCase { decoder.keyDecodingStrategy = .convertFromSnakeCase }
        do {
            return try decoder.decode(T.self, from: Data(buffer: body))
        } catch {
            throw HTTPError(.badRequest, message: "\(error)")
        }
    }

    static func encode(_ value: some Encodable, snakeCase: Bool = true) throws -> ByteBuffer {
        let encoder = JSONEncoder()
        if snakeCase { encoder.keyEncodingStrategy = .convertToSnakeCase }
        return ByteBuffer(data: try encoder.encode(value))
    }

    static func response(_ value: some Encodable, snakeCase: Bool = true) throws -> Response {
        Response(
            status: .ok, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: try encode(value, snakeCase: snakeCase)))
    }
}

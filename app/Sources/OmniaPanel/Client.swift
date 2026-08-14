// How the panel reaches the daemon, and the two shapes an answer arrives in: a body, or
// a stream of them. The house refusal is FastAPI's `{"detail": ...}`, which is what a
// screen shows rather than a status code.

import Foundation

struct Refused: LocalizedError {
    let detail: String
    var errorDescription: String? { detail }
}

enum Client {
    static let base: URL = {
        let text = ProcessInfo.processInfo.environment["OMNIA_DAEMON_URL"]
            ?? "http://127.0.0.1:8642"
        return URL(string: text.hasSuffix("/") ? String(text.dropLast()) : text)!
    }()

    static var port: String { base.port.map(String.init) ?? "8642" }

    /// A model id in a path segment. The route is a `:path`, so the slash in `org/name`
    /// stays a slash and only the rest is escaped.
    static func at(_ identifier: String) -> String {
        identifier.split(separator: "/", omittingEmptySubsequences: false)
            .map { part in
                part.addingPercentEncoding(withAllowedCharacters: .alphanumerics.union(
                    CharacterSet(charactersIn: "-._~")
                )) ?? String(part)
            }
            .joined(separator: "/")
    }

    static let decoder: JSONDecoder = {
        let made = JSONDecoder()
        made.keyDecodingStrategy = .convertFromSnakeCase
        return made
    }()

    static let encoder: JSONEncoder = {
        let made = JSONEncoder()
        made.keyEncodingStrategy = .convertToSnakeCase
        return made
    }()

    /// For a body whose keys are not this panel's to rename — the chat request, which
    /// carries a tools schema somebody pasted, and the quantize request, whose overrides
    /// are keyed by patterns read off a checkpoint. Both spell their own keys.
    static let plain = JSONEncoder()

    private static let calls: URLSession = {
        let shape = URLSessionConfiguration.ephemeral
        shape.timeoutIntervalForRequest = 30
        shape.waitsForConnectivity = false
        return URLSession(configuration: shape)
    }()

    /// The stream session has no request timeout: `/admin/events` is meant to stay open,
    /// and a timeout there is a reconnect loop nobody asked for.
    private static let streams: URLSession = {
        let shape = URLSessionConfiguration.ephemeral
        shape.timeoutIntervalForRequest = 86400
        shape.timeoutIntervalForResource = 86400
        return URLSession(configuration: shape)
    }()

    private static func url(_ path: String) -> URL {
        URL(string: base.absoluteString + path) ?? base
    }

    private static func detail(_ data: Data, _ answer: HTTPURLResponse) -> String {
        if let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let said = body["detail"] as? String { return said }
            if let error = body["error"] as? [String: Any],
               let said = error["message"] as? String { return said }
        }
        return HTTPURLResponse.localizedString(forStatusCode: answer.statusCode)
    }

    static func get<T: Decodable>(_ path: String) async throws -> T {
        let (data, answer) = try await calls.data(from: url(path))
        guard let answer = answer as? HTTPURLResponse else { throw Refused(detail: "no answer") }
        guard (200..<300).contains(answer.statusCode) else {
            throw Refused(detail: detail(data, answer))
        }
        return try decoder.decode(T.self, from: data)
    }

    static func get<T: Decodable>(_ path: String, query: [String: String]) async throws -> T {
        let items = query
            .sorted { $0.key < $1.key }
            .map { URLQueryItem(name: $0.key, value: $0.value) }
        var made = URLComponents(url: url(path), resolvingAgainstBaseURL: false)
        made?.queryItems = items
        guard let asked = made?.url else { throw Refused(detail: "the query cannot be spelled") }
        let (data, answer) = try await calls.data(from: asked)
        guard let answer = answer as? HTTPURLResponse else { throw Refused(detail: "no answer") }
        guard (200..<300).contains(answer.statusCode) else {
            throw Refused(detail: detail(data, answer))
        }
        return try decoder.decode(T.self, from: data)
    }

    /// A route that answers text rather than a body: the model card, which the daemon serves
    /// raw because rendering it is the client's. 404 is not a refusal here — a checkpoint
    /// that ships no README is the ordinary case, and it answers nil.
    static func text(_ path: String) async throws -> String? {
        let (data, answer) = try await calls.data(from: url(path))
        guard let answer = answer as? HTTPURLResponse else { throw Refused(detail: "no answer") }
        if answer.statusCode == 404 { return nil }
        guard (200..<300).contains(answer.statusCode) else {
            throw Refused(detail: detail(data, answer))
        }
        return String(decoding: data, as: UTF8.self)
    }

    /// `spelled` is for a body that writes its own keys and must not be renamed on the way
    /// out — see `plain`.
    @discardableResult
    static func send<T: Decodable>(
        _ method: String, _ path: String, body: (any Encodable)? = nil, spelled: Bool = false
    ) async throws -> T? {
        var request = URLRequest(url: url(path))
        request.httpMethod = method
        if let body {
            request.httpBody = try (spelled ? plain : encoder).encode(AnyEncodable(body))
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        // A POST that starts a job answers when the daemon has taken it, which for a
        // quantization plan is seconds of arithmetic.
        request.timeoutInterval = method == "POST" ? 300 : 30
        let (data, answer) = try await calls.data(for: request)
        guard let answer = answer as? HTTPURLResponse else { throw Refused(detail: "no answer") }
        guard (200..<300).contains(answer.statusCode) else {
            throw Refused(detail: detail(data, answer))
        }
        if answer.statusCode == 204 || data.isEmpty { return nil }
        return try decoder.decode(T.self, from: data)
    }

    static func send(
        _ method: String, _ path: String, body: (any Encodable)? = nil, spelled: Bool = false
    ) async throws {
        let _: Nothing? = try await send(method, path, body: body, spelled: spelled)
    }

    /// text/event-stream carrying whole JSON states on `data:` lines, plus `: keep-alive`
    /// comments that mean nothing but the connection is warm.
    ///
    /// One `data:` line is one frame here, and the blank line SSE ends a frame with is not
    /// what this waits for — `AsyncLineSequence` drops empty lines, so a parser that waits
    /// for one waits forever: the stream stays open, nothing is ever yielded, and the panel
    /// sits at "not answering" while the daemon answers. The daemon writes every frame as
    /// `data: {json}\n\n` and `json.dumps` never emits a raw newline, so the line is the
    /// frame.
    static func events(_ path: String) -> AsyncThrowingStream<Data, Error> {
        events(path, body: nil)
    }

    /// The same stream a POST opens — the chat dialect's, which ends with a `[DONE]` line
    /// this hands through as it came: what a caller does with the sentinel is the dialect's
    /// business and not the transport's.
    static func events(_ path: String, body: (any Encodable)?) -> AsyncThrowingStream<Data, Error> {
        AsyncThrowingStream { continuation in
            let work = Task {
                do {
                    var request = URLRequest(url: url(path))
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    if let body {
                        request.httpMethod = "POST"
                        request.httpBody = try plain.encode(AnyEncodable(body))
                        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    }
                    let (bytes, answer) = try await streams.bytes(for: request)
                    guard let answer = answer as? HTTPURLResponse else {
                        throw Refused(detail: "the stream refused to open")
                    }
                    guard (200..<300).contains(answer.statusCode) else {
                        // A refusal arrives as a body on the same connection, not as a
                        // frame: read it so the screen can say what was wrong with the
                        // request instead of "the stream refused to open".
                        var said = Data()
                        for try await byte in bytes { said.append(byte) }
                        throw Refused(detail: detail(said, answer))
                    }
                    for try await line in bytes.lines where line.hasPrefix("data:") {
                        let frame = String(line.dropFirst(5))
                            .trimmingCharacters(in: .whitespaces)
                        if !frame.isEmpty { continuation.yield(Data(frame.utf8)) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in work.cancel() }
        }
    }
}

/// A body a screen built out of a concrete request type, erased for `encode`.
struct AnyEncodable: Encodable {
    private let write: (Encoder) throws -> Void
    init(_ wrapped: any Encodable) {
        write = { encoder in try wrapped.encode(to: encoder) }
    }
    func encode(to encoder: Encoder) throws { try write(encoder) }
}

/// The answer of a route that returns nothing.
struct Nothing: Decodable {}

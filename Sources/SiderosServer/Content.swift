import Hummingbird

/// Every dialect lets a message body be either a bare string or a list of typed parts.
/// Only text parts survive here: there is no vision model behind this server yet, and a
/// silently dropped image is worse than a refused request.
struct Content: Decodable {
    private let parts: [String]

    init(_ text: String) {
        parts = [text]
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            parts = [text]
            return
        }
        parts = try container.decode([Part].self).map(\.text)
    }

    var text: String { parts.joined() }

    private struct Part: Decodable {
        let text: String

        private enum Key: String, CodingKey {
            case type, text
        }

        /// `text` in the chat dialects, `input_text` / `output_text` in Responses.
        private static let text = ["text", "input_text", "output_text"]

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: Key.self)
            let type = try container.decode(String.self, forKey: .type)
            guard Self.text.contains(type) else {
                throw HTTPError(.badRequest, message: "unsupported content part: \(type)")
            }
            text = try container.decode(String.self, forKey: .text)
        }
    }
}

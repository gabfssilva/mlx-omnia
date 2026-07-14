import Foundation
import Jinja

/// The checkpoint's own jinja template: the only thing that knows how this model spells a
/// conversation. Rendering it produces the literal text — `<|im_start|>` and all — which
/// the tokenizer then encodes back into the single ids of those added tokens.
public struct ChatTemplate: Sendable {
    public enum LoadError: Error {
        /// A base model. Nothing to fall back on: guessing a template produces fluent,
        /// wrong text.
        case noTemplate(URL)
    }

    private let template: Template
    private let bos: Value
    private let eos: Value

    /// Everything else in the file, which the templates read as scalars.
    private struct Config: Decodable {
        let chatTemplate: String?
        let bosToken: Token?
        let eosToken: Token?

        /// Either the literal string or `{"content": ..., ...}` from a slow tokenizer.
        struct Token: Decodable {
            let content: String?
            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let text = try? container.decode(String.self) {
                    content = text
                } else {
                    struct Object: Decodable { let content: String }
                    content = try? container.decode(Object.self).content
                }
            }
        }
    }

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(
            Config.self, from: Data(contentsOf: directory.appending(path: "tokenizer_config.json")))

        // Newer exports move the template out of the json and into its own file.
        let file = directory.appending(path: "chat_template.jinja")
        let source =
            (try? String(contentsOf: file, encoding: .utf8)) ?? config.chatTemplate
        guard let source else { throw LoadError.noTemplate(directory) }

        // transformers renders with both trims on; without them the templates' own
        // indentation lands in the prompt.
        template = try Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
        bos = config.bosToken?.content.map(Value.string) ?? .null
        eos = config.eosToken?.content.map(Value.string) ?? .null
    }

    public func callAsFunction(_ messages: [Message]) throws -> String {
        try template.render([
            "messages": try Value(any: messages.map(\.context)),
            "add_generation_prompt": true,
            "bos_token": bos,
            "eos_token": eos,
        ])
    }
}

extension Message {
    /// What a template indexes into: `message.role` and `message.content`.
    fileprivate var context: [String: String] {
        ["role": role.rawValue, "content": content]
    }
}

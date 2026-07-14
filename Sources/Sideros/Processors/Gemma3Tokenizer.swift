import Foundation

public struct Gemma3Tokenizer: Sendable {
    private static let space: Character = "\u{2581}"

    private let encoder: [String: Int32]
    private let decoder: [Int32: String]
    private let bpe: BPE
    private let added: AddedTokens
    private let bos: Int32
    /// Byte → id of its `<0xXX>` fallback token.
    private let byteFallbacks: [Int32]

    public init(directory: URL) throws {
        let file = try TokenizerFile(directory.appending(path: "tokenizer.json"))

        encoder = file.model.vocab
        decoder = Dictionary(uniqueKeysWithValues: file.model.vocab.map { ($1, $0) })
        bpe = BPE(file.pairs, vocab: file.model.vocab)
        added = AddedTokens(file.addedTokens)
        byteFallbacks = (UInt8.min...UInt8.max).map {
            file.model.vocab[String(format: "<0x%02X>", $0)]!
        }

        guard let bos = file.model.vocab["<bos>"] else {
            throw CocoaError(.coderValueNotFound)
        }
        self.bos = bos
    }

    /// The post-processor is a TemplateProcessing that prepends `<bos>` to every sequence.
    public func encode(_ text: String) -> [Int32] {
        [bos]
            + added.split(text).flatMap { piece in
                switch piece {
                case .token(let id): [id]
                case .text(let text): encode(text: text)
                }
            }
    }

    public func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            let token = decoder[id]!
            if let byte = Self.fallbackByte(token) {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: token.replacing(String(Self.space), with: " ").utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// There is no pre-tokenizer to speak of: the normalizer turns every space into `▁`,
    /// leaving the whole piece as one word for BPE. Characters outside the vocabulary
    /// fall back to their UTF-8 bytes, which are tokens in their own right and may merge.
    private func encode(text: String) -> [Int32] {
        let symbols = text.replacing(" ", with: String(Self.space)).flatMap { character -> [Int32] in
            let symbol = String(character)
            guard let id = encoder[symbol] else {
                return symbol.utf8.map { byteFallbacks[Int($0)] }
            }
            return [id]
        }
        return bpe(symbols)
    }

    private static func fallbackByte(_ token: String) -> UInt8? {
        guard token.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }
}

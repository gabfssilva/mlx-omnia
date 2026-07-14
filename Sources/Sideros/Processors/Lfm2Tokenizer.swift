import Foundation

public struct Lfm2Tokenizer: Sendable {
    private let encoder: [String: Int32]
    private let decoder: [Int32: String]
    private let bpe: BPE
    private let added: AddedTokens
    /// Byte → id of its single-character byte-level symbol, the alphabet BPE starts from.
    private let byteTokens: [Int32]

    /// The pre-tokenizer regex from tokenizer.json: Qwen2's, except digits group up to
    /// three at a time instead of one by one.
    nonisolated(unsafe) private static let pattern = try! NSRegularExpression(
        pattern:
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    )

    public init(directory: URL) throws {
        let file = try TokenizerFile(directory.appending(path: "tokenizer.json"))

        encoder = file.model.vocab
        bpe = BPE(file.pairs, vocab: file.model.vocab)
        added = AddedTokens(file.addedTokens)
        byteTokens = (UInt8.min...UInt8.max).map { file.model.vocab[String(ByteLevel.encoder[$0]!)]! }

        var decoder = Dictionary(uniqueKeysWithValues: file.model.vocab.map { ($1, $0) })
        for token in file.addedTokens { decoder[token.id] = token.content }
        self.decoder = decoder
    }

    public func encode(_ text: String) -> [Int32] {
        added.split(text).flatMap { piece in
            switch piece {
            case .token(let id): [id]
            case .text(let text): encode(text: text)
            }
        }
    }

    public func decode(_ ids: [Int32]) -> String {
        let mapped = ids.map { decoder[$0]! }.joined()
        return String(decoding: mapped.map { ByteLevel.decoder[$0]! }, as: UTF8.self)
    }

    private func encode(text: String) -> [Int32] {
        pretokenize(text.precomposedStringWithCanonicalMapping).flatMap { piece -> [Int32] in
            // `ignore_merges`: a pre-token that is already a vocabulary entry is that
            // entry, even where the merge walk would split it.
            let symbol = String(piece.utf8.map { ByteLevel.encoder[$0]! })
            if let id = encoder[symbol] { return [id] }
            return bpe(piece.utf8.map { byteTokens[Int($0)] })
        }
    }

    private func pretokenize(_ text: String) -> [String] {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return Self.pattern.matches(in: text, range: range).map { ns.substring(with: $0.range) }
    }
}

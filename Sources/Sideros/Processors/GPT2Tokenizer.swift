import Foundation

public struct GPT2Tokenizer: Sendable {
    private let encoder: [String: Int32]
    private let decoder: [Int32: String]
    private let bpe: BPE
    /// Byte → id of its single-character byte-level symbol, the alphabet BPE starts from.
    private let byteTokens: [Int32]

    // GPT-2's pre-tokenizer. Contractions first, then letters/digits/punctuation each
    // optionally carrying one leading space; `\s+(?!\S)` keeps trailing runs separate.
    nonisolated(unsafe) private static let pattern = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    )

    public init(vocab vocabURL: URL, merges mergesURL: URL) throws {
        let vocab = try JSONDecoder().decode([String: Int32].self, from: Data(contentsOf: vocabURL))
        encoder = vocab
        decoder = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })

        let lines = try String(contentsOf: mergesURL, encoding: .utf8)
            .split(separator: "\n")
            .dropFirst()  // "#version: 0.2"
        bpe = BPE(
            lines.compactMap { line in
                let parts = line.split(separator: " ")
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }, vocab: vocab)
        byteTokens = (UInt8.min...UInt8.max).map { vocab[String(ByteLevel.encoder[$0]!)]! }
    }

    public func encode(_ text: String) -> [Int32] {
        pretokenize(text).flatMap { piece in
            bpe(piece.utf8.map { byteTokens[Int($0)] })
        }
    }

    public func decode(_ ids: [Int32]) -> String {
        let mapped = ids.map { decoder[$0]! }.joined()
        return String(decoding: mapped.map { ByteLevel.decoder[$0]! }, as: UTF8.self)
    }

    private func pretokenize(_ text: String) -> [String] {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return Self.pattern.matches(in: text, range: range).map { ns.substring(with: $0.range) }
    }
}

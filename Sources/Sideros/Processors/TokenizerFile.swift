import Foundation

/// The parts of HuggingFace's `tokenizer.json` the BPE tokenizers read. The pipeline
/// (normalizer, pre-tokenizer, decoder) is transcribed into Swift per architecture
/// rather than interpreted from the file.
struct TokenizerFile: Decodable {
    struct Model: Decodable {
        let vocab: [String: Int32]
        let merges: [Merge]
    }

    /// Byte-level checkpoints write a merge as `"a b"`, SentencePiece ones as `["a", "b"]`.
    struct Merge: Decodable {
        let first: String
        let second: String

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            let parts: [String]
            if let joined = try? container.decode(String.self) {
                parts = joined.split(separator: " ", maxSplits: 1).map(String.init)
            } else {
                parts = try container.decode([String].self)
            }
            guard parts.count == 2 else {
                throw DecodingError.dataCorruptedError(
                    in: container, debugDescription: "merge is not a pair: \(parts)")
            }
            (first, second) = (parts[0], parts[1])
        }
    }

    struct AddedToken: Decodable {
        let id: Int32
        let content: String
    }

    let model: Model
    let addedTokens: [AddedToken]

    init(_ url: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self = try decoder.decode(Self.self, from: Data(contentsOf: url))
    }

    var pairs: [(String, String)] { model.merges.map { ($0.first, $0.second) } }
}

/// Added tokens are matched against the raw text before any normalization, longest first,
/// and never take part in a merge.
struct AddedTokens: Sendable {
    enum Piece {
        case token(Int32)
        case text(String)
    }

    private let ids: [String: Int32]
    private let lengths: [Int]
    /// First characters of every added token: almost no position can start one, and the
    /// check is what keeps the scan from slicing a candidate substring per position.
    private let starts: Set<Character>

    init(_ tokens: [TokenizerFile.AddedToken]) {
        ids = Dictionary(uniqueKeysWithValues: tokens.map { ($0.content, $0.id) })
        lengths = Set(tokens.map(\.content.count)).sorted(by: >)
        starts = Set(tokens.compactMap(\.content.first))
    }

    func split(_ text: String) -> [Piece] {
        let characters = Array(text)
        var pieces: [Piece] = []
        var pending = ""
        var i = 0

        while i < characters.count {
            guard starts.contains(characters[i]) else {
                pending.append(characters[i])
                i += 1
                continue
            }
            let matched = lengths.lazy
                .filter { i + $0 <= characters.count }
                .compactMap { length -> (Int, Int32)? in
                    ids[String(characters[i..<(i + length)])].map { (length, $0) }
                }
                .first

            guard let (length, id) = matched else {
                pending.append(characters[i])
                i += 1
                continue
            }

            if !pending.isEmpty {
                pieces.append(.text(pending))
                pending = ""
            }
            pieces.append(.token(id))
            i += length
        }

        if !pending.isEmpty { pieces.append(.text(pending)) }
        return pieces
    }
}

import Foundation

/// The merge loop every BPE shares: repeatedly join the adjacent pair with the lowest
/// rank. It runs over vocabulary ids, not strings: every symbol a merge can produce is
/// itself in the vocabulary, so a pair of ids maps straight to the joined symbol's id.
struct BPE: Sendable {
    private let ranks: [UInt64: Int32]
    private let joined: [UInt64: Int32]

    init(_ merges: [(String, String)], vocab: [String: Int32]) {
        var ranks: [UInt64: Int32] = [:]
        var joined: [UInt64: Int32] = [:]
        ranks.reserveCapacity(merges.count)
        joined.reserveCapacity(merges.count)

        for (rank, merge) in merges.enumerated() {
            guard let first = vocab[merge.0], let second = vocab[merge.1],
                let whole = vocab[merge.0 + merge.1]
            else { continue }
            let pair = Self.pack(first, second)
            guard ranks[pair] == nil else { continue }
            ranks[pair] = Int32(rank)
            joined[pair] = whole
        }
        self.ranks = ranks
        self.joined = joined
    }

    private static func pack(_ first: Int32, _ second: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: first)) << 32 | UInt64(UInt32(bitPattern: second))
    }

    func callAsFunction(_ symbols: [Int32]) -> [Int32] {
        var word = symbols
        while word.count > 1 {
            var bestRank = Int32.max
            var bestPair: UInt64 = 0
            for i in 0..<(word.count - 1) {
                let pair = Self.pack(word[i], word[i + 1])
                if let rank = ranks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            guard bestRank < Int32.max else { break }

            let whole = joined[bestPair]!
            var merged: [Int32] = []
            merged.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1, Self.pack(word[i], word[i + 1]) == bestPair {
                    merged.append(whole)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        return word
    }
}

/// Reversible map from the 256 bytes onto printable scalars, so BPE never sees
/// whitespace or control characters. GPT-2 and Qwen2 both tokenize over this alphabet.
enum ByteLevel {
    static let encoder: [UInt8: Character] = {
        var bytes = Array(UInt8(ascii: "!")...UInt8(ascii: "~"))
        bytes += Array(UInt8(0xA1)...UInt8(0xAC)) + Array(UInt8(0xAE)...UInt8(0xFF))

        var scalars = bytes.map { UInt32($0) }
        var next: UInt32 = 0
        for byte in UInt8.min...UInt8.max where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(256 + next)
            next += 1
        }

        return Dictionary(
            uniqueKeysWithValues: zip(bytes, scalars.map { Character(UnicodeScalar($0)!) }))
    }()

    static let decoder: [Character: UInt8] = Dictionary(
        uniqueKeysWithValues: encoder.map { ($1, $0) })
}

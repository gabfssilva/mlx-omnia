/// What a caller that has to report back needs and `callAsFunction` does not carry: why
/// generation ended and what it cost. The deltas are the same text, chunk for chunk.
public enum Completion: Sendable {
    /// Emitted before the first token: the prompt is tokenized by then, and a dialect that
    /// opens its stream with the input count (Anthropic's `message_start`) has to know it
    /// before it knows anything else.
    case start(promptTokens: Int)
    case delta(String)
    case done(Finish)
}

public struct Finish: Sendable {
    public enum Reason: Sendable {
        /// A stop token or a stop string.
        case stop
        /// `maxTokens` ran out first.
        case length
    }

    public let reason: Reason
    /// The stop string that ended it, when one did. A stop *token* leaves this nil — the
    /// Anthropic dialect reports the two differently.
    public let stopString: String?
    public let promptTokens: Int
    public let completionTokens: Int
}

/// Holds back the tail of the stream that could still turn out to be the start of a stop
/// string — a stop can straddle two chunks, and a chunk once emitted cannot be recalled.
struct StopFilter {
    private let stops: [String]
    private var held = ""

    init(_ stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
    }

    /// The text safe to emit now, and the stop string that closed the stream if one did.
    /// The stop string itself is never emitted.
    mutating func push(_ delta: String) -> (emit: String, halt: String?) {
        guard !stops.isEmpty else { return (delta, nil) }
        held += delta

        // The earliest stop wins: two of them can match, and the text before the first is
        // all the caller was ever going to see.
        if let hit = stops.compactMap({ stop in held.range(of: stop).map { ($0.lowerBound, stop) } })
            .min(by: { $0.0 < $1.0 })
        {
            defer { held = "" }
            return (String(held[held.startIndex..<hit.0]), hit.1)
        }

        // Any suffix of what we hold that opens a stop string has to wait for the token
        // that either completes it or rules it out.
        let ambiguous = held.indices.first { index in
            let suffix = held[index...]
            return stops.contains { $0.hasPrefix(suffix) }
        }
        let boundary = ambiguous ?? held.endIndex
        defer { held = String(held[boundary...]) }
        return (String(held[held.startIndex..<boundary]), nil)
    }
}

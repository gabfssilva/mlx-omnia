import MLX

/// How one token is drawn from a row of logits. The pick stays a lazy array: the
/// generation loop pipelines the next step's graph before it syncs this one's token.
public struct Sampler: Sendable {
    let pick: @Sendable (MLXArray) -> MLXArray
    /// Speculative decoding needs to know: only a deterministic pick lets a draft's
    /// token be verified by equality against the target's own choice.
    let isGreedy: Bool

    init(isGreedy: Bool = false, _ pick: @escaping @Sendable (MLXArray) -> MLXArray) {
        self.pick = pick
        self.isGreedy = isGreedy
    }

    public static let greedy = Sampler(isGreedy: true) { argMax($0, axis: -1) }

    public static func temperature(_ value: Float) -> Sampler {
        Sampler { MLXRandom.categorical($0 * (1 / value)) }
    }

    /// Nucleus sampling: keep the shortest set of tokens whose mass reaches `p`. Sorting
    /// ascending puts that set at the tail, which is why the cut compares against `1 - p`.
    public static func topP(_ p: Float, temperature: Float = 1) -> Sampler {
        Sampler { logits in
            let probabilities = softmax(logits * (1 / temperature), axis: -1)
            let order = argSort(probabilities, axis: -1)
            let sorted = probabilities[order]

            let kept = which(cumsum(sorted, axis: -1) .> (1 - p), sorted, zeros(like: sorted))
            return order[MLXRandom.categorical(log(kept))]
        }
    }
}

public struct GenerationOptions: Sendable {
    public var maxTokens: Int
    public var sampler: Sampler
    /// `nil` means the model's own end-of-sequence token.
    public var stop: Set<Int32>?
    /// Text that ends the generation wherever it appears, never a token boundary. Only
    /// `stream` honors it — the stop string is cut out of the output.
    public var stopStrings: [String]
    public var seed: UInt64?
    /// Draft tokens proposed per speculation round when the model carries a draft
    /// (`ModelOptions.draft`). Higher rides longer acceptance runs; every rejected
    /// token was a wasted draft forward, and past 7 the verification forward outgrows
    /// the skinny gemm kernel and falls back to the slow tile path.
    public var speculated: Int

    public init(
        maxTokens: Int = 256, sampler: Sampler = .greedy, stop: Set<Int32>? = nil,
        stopStrings: [String] = [], seed: UInt64? = nil, speculated: Int = 4
    ) {
        self.maxTokens = maxTokens
        self.sampler = sampler
        self.stop = stop
        self.stopStrings = stopStrings
        self.seed = seed
        self.speculated = speculated
    }
}

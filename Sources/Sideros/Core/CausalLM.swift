import MLX

/// A decoder over embeddings. `embed` is where a modality enters — text today, image
/// segments spliced into the same causal stream later — and `logits` never learns which.
public protocol CausalLM {
    /// `offset` is how many tokens precede these ids. Only a learned positional table
    /// (GPT-2's wpe) reads it; rotary models take their offset from the cache. Taking a
    /// lazy array of ids is what lets a sampled token feed the next step without a sync.
    func embed(_ tokens: MLXArray, offset: Int) -> MLXArray
    func logits(_ embeddings: MLXArray, cache: [KVCache]) -> (logits: MLXArray, cache: [KVCache])
    func makeCache() -> [KVCache]
}

/// Where the rotation reads its position. A scalar covers text and every decode step of every
/// model: the position is the row count, growing by one. It stops covering a prompt with an
/// image in it — there the three MRoPE sections read (t, h, w) and diverge inside the vision
/// run, and after it the position resumes past the image's *width*, not past its token count.
public enum RopePositions {
    case offset(Int)
    /// `[3, L]` int32, one column per token.
    case explicit(MLXArray)
}

/// A decoder whose rotary position is not implied by the cache's row count. Qwen3.6 alone: the
/// other six read `KVCache.offset` and are right to.
public protocol PositionedLM: CausalLM {
    func logits(_ embeddings: MLXArray, positions: RopePositions, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
}

extension CausalLM {
    public func embed(_ ids: [Int32], offset: Int = 0) -> MLXArray {
        embed(MLXArray(ids, [1, ids.count]), offset: offset)
    }
    /// Recomputes the whole prefix each step: a fresh cache per token is the same thing as
    /// no cache. This is the reference the streamed, cached path is checked against.
    public func generate(_ ids: [Int32], maxNewTokens: Int) -> [Int32] {
        var tokens = ids
        for _ in 0..<maxNewTokens {
            let (logits, _) = logits(embed(tokens, offset: 0), cache: makeCache())
            tokens.append(argMax(logits[0, -1], axis: -1).item(Int32.self))
        }
        return tokens
    }
}

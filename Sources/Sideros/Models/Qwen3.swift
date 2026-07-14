import Foundation
import MLX

public struct Qwen3Activations {
    public let embeddings: MLXArray
    public let blocks: [MLXArray]
    public let norm: MLXArray
    public let logits: MLXArray
}

/// Qwen2 with three deltas: q/k are rms-normed per head between the projection and the
/// rotation, the projections carry no bias, and heads * headDim is decoupled from
/// hiddenSize (the small sizes widen, o_proj narrows back).
public struct Qwen3: CausalLM {
    private let config: Qwen3Config
    private let parameters: [String: MLXArray]
    private let quantization: Quantization?
    private let skinny: Bool
    /// One compiled closure per layer: the trace fuses silu(gate) · up into one kernel.
    let mlps: [@Sendable (MLXArray) -> MLXArray]

    public init(
        config: Qwen3Config, parameters: [String: MLXArray], quantization: Quantization? = nil,
        skinny: Bool = false
    ) {
        self.config = config
        self.quantization = quantization
        self.skinny = skinny

        // One fused projection per layer instead of three, as in Qwen2 — row-aligned in
        // every representation. No bias to fuse here.
        var parameters = parameters
        for layer in 0..<config.numHiddenLayers {
            let p = "model.layers.\(layer).self_attn."
            for suffix in ["weight", "scales", "biases"] {
                let parts = ["q_proj.", "k_proj.", "v_proj."].compactMap {
                    parameters[p + $0 + suffix]
                }
                guard parts.count == 3 else { continue }
                parameters[p + "qkv_proj." + suffix] = concatenated(parts, axis: 0)
            }
        }
        self.parameters = parameters

        mlps = (0..<config.numHiddenLayers).map { layer in
            let p = "model.layers.\(layer).mlp."
            let gate = Linear(
                p + "gate_proj", parameters: parameters, quantization: quantization, skinny: skinny)
            let up = Linear(
                p + "up_proj", parameters: parameters, quantization: quantization, skinny: skinny)
            let down = Linear(
                p + "down_proj", parameters: parameters, quantization: quantization, skinny: skinny)
            // silu spelled out: MLXNN's is itself a compiled function with its own lock,
            // and entering it while this trace holds the global eval lock inverts the
            // order against any thread calling it directly — an ABBA deadlock.
            return compile { x in
                let gated = gate(x)
                return down(gated * sigmoid(gated) * up(x))
            }
        }
    }

    /// Total by construction: loadQwen3Parameters rejects any checkpoint missing a name.
    private subscript(name: String) -> MLXArray { parameters[name]! }

    private func linear(_ name: String) -> Linear {
        Linear(name, parameters: parameters, quantization: quantization, skinny: skinny)
    }

    /// No positional embedding table: position enters only through RoPE.
    public func embed(_ tokens: MLXArray, offset: Int = 0) -> MLXArray {
        let table = Embedding("model.embed_tokens", parameters: parameters, quantization: quantization)
        return table(tokens)
    }

    public func makeCache() -> [KVCache] { (0..<config.numHiddenLayers).map { _ in KVCache() } }

    public func logits(_ embeddings: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
    {
        let decoded = decode(embeddings, cache: cache)
        return (decoded.logits, decoded.cache)
    }

    public func callAsFunction(_ ids: [Int32]) -> Qwen3Activations {
        let embeddings = embed(ids)
        let decoded = decode(embeddings, cache: makeCache())
        return Qwen3Activations(
            embeddings: embeddings,
            blocks: decoded.blocks,
            norm: decoded.norm,
            logits: decoded.logits
        )
    }

    private func decode(_ embeddings: MLXArray, cache: [KVCache])
        -> (blocks: [MLXArray], norm: MLXArray, logits: MLXArray, cache: [KVCache])
    {
        var cache = cache
        var x = embeddings
        var blocks: [MLXArray] = []
        for layer in 0..<config.numHiddenLayers {
            (x, cache[layer]) = block(x, layer: layer, cache: cache[layer])
            blocks.append(x)
        }

        let normed = rmsNorm(x, prefix: "model.norm")
        let head = config.tieWordEmbeddings ? "model.embed_tokens" : "lm_head"
        return (blocks, normed, linear(head)(normed), cache)
    }

    func block(_ x: MLXArray, layer: Int, cache: KVCache) -> (MLXArray, KVCache) {
        let prefix = "model.layers.\(layer)."
        let (context, grown) = attention(
            rmsNorm(x, prefix: prefix + "input_layernorm"), prefix: prefix, cache: cache)
        let attended = x + context
        return (
            attended
                + mlps[layer](rmsNorm(attended, prefix: prefix + "post_attention_layernorm")),
            grown
        )
    }

    func attention(_ x: MLXArray, prefix: String, cache: KVCache) -> (MLXArray, KVCache) {
        let length = x.dim(1)
        let offset = cache.offset
        let p = prefix + "self_attn."

        let queryWidth = config.numAttentionHeads * config.headDim
        let keyValueWidth = config.numKeyValueHeads * config.headDim
        let fused = linear(p + "qkv_proj")(x)
        let parts = split(fused, indices: [queryWidth, queryWidth + keyValueWidth], axis: -1)

        func heads(_ part: MLXArray, _ count: Int) -> MLXArray {
            part.reshaped(1, length, count, config.headDim).transposed(0, 2, 1, 3)
        }

        // The norm weight is one headDim vector shared by every head and position.
        let queries = rope(
            rmsNorm(heads(parts[0], config.numAttentionHeads), prefix: p + "q_norm"),
            offset: offset)
        let (keys, values, grown) = cache.appending(
            keys: rope(
                rmsNorm(heads(parts[1], config.numKeyValueHeads), prefix: p + "k_norm"),
                offset: offset),
            values: heads(parts[2], config.numKeyValueHeads))

        // GQA: the kernel broadcasts the kv heads across the query heads itself. See
        // Qwen2 for why `.causal` and why a single query goes unmasked.
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(config.headDim)), mask: length == 1 ? .none : .causal)

        let projected = linear(p + "o_proj")(
            attended.transposed(0, 2, 1, 3).reshaped(1, length, queryWidth))
        return (projected, grown)
    }

    func rmsNorm(_ x: MLXArray, prefix: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: self[prefix + ".weight"], eps: config.rmsNormEps)
    }

    /// `traditional: false` is the interleaving HF calls `rotate_half`: the vector is
    /// split in two halves rather than into adjacent pairs.
    func rope(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: config.headDim, traditional: false, base: config.ropeTheta,
            scale: 1, offset: offset)
    }
}

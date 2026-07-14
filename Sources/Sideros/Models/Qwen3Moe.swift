import Foundation
import MLX

public struct Qwen3MoeActivations {
    public let embeddings: MLXArray
    public let blocks: [MLXArray]
    public let norm: MLXArray
    public let logits: MLXArray
}

/// Qwen3's attention — q/k normed per head between the projection and the rotation, no
/// bias — with every MLP replaced by a mixture of 128 experts, 8 per token. The router
/// softmaxes over all of them and renormalizes the eight it keeps; there is no shared
/// expert and no selection bias, so LFM2.5's sigmoid routing does not carry over.
public struct Qwen3Moe: CausalLM {
    private let config: Qwen3MoeConfig
    private let parameters: [String: MLXArray]
    private let quantization: Quantization?
    /// One closure per layer: routing plus the two gathered matmuls.
    let mlps: [(MLXArray) -> MLXArray]
    /// The one-token step, compiled in two chains around the cache: `pre` is
    /// [x, offset] → [queries, keys, values] (norm, qkv, heads, q/k norms, rotation) and
    /// `post` is [attended, x] → [block output] (o_proj, residual, norm, MoE, residual).
    /// The rotary offset rides in as an array — as an op attribute the trace would
    /// freeze it at its first value. Only T=1 compiles: the shape never changes, so it
    /// traces once; the cache stays outside because its shape grows.
    let steps: [(pre: ([MLXArray]) -> [MLXArray], post: ([MLXArray]) -> [MLXArray])]

    public init(
        config: Qwen3MoeConfig, parameters: [String: MLXArray], quantization: Quantization? = nil
    ) {
        self.config = config
        self.quantization = quantization

        var parameters = parameters
        for layer in 0..<config.numHiddenLayers {
            let p = "model.layers.\(layer).self_attn."
            for suffix in ["weight", "scales", "biases"] {
                let parts = ["q_proj.", "k_proj.", "v_proj."].compactMap {
                    parameters[p + $0 + suffix]
                }
                guard parts.count == 3 else { continue }
                // Materialized like gate_up below: `compile` pins captured arrays, and
                // an unevaluated concatenation would re-run inside every traced call.
                let fused = concatenated(parts, axis: 0)
                eval(fused)
                parameters[p + "qkv_proj." + suffix] = fused
            }
        }
        // Gate and up fused row-interleaved ([g₀,u₀,g₁,u₁,…]) so one gather reads both
        // and each (gate, up) pair lands in one simdgroup of the one-token kernel.
        // Row-aligned in all three representations, so the packed weight interleaves
        // exactly like the dense one. Materialized here, originals dropped.
        for layer in 0..<config.numHiddenLayers {
            let p = "model.layers.\(layer).mlp.switch_mlp."
            for suffix in ["weight", "scales", "biases"] {
                let parts = ["gate_proj.", "up_proj."].compactMap { parameters[p + $0 + suffix] }
                guard parts.count == 2 else { continue }
                var shape = parts[0].shape
                shape[1] *= 2
                let fused = stacked(parts, axis: 2).reshaped(shape)
                eval(fused)
                parameters[p + "gate_up_proj." + suffix] = fused
                parameters.removeValue(forKey: p + "gate_proj." + suffix)
                parameters.removeValue(forKey: p + "up_proj." + suffix)
            }
        }
        self.parameters = parameters

        let eps = config.rmsNormEps
        let headDim = config.headDim
        let ropeTheta = config.ropeTheta
        let numHeads = config.numAttentionHeads
        let numKVHeads = config.numKeyValueHeads
        let queryWidth = numHeads * headDim
        let keyValueWidth = numKVHeads * headDim
        let freqs = exp(
            MLXArray(Array(stride(from: Float(0), to: Float(headDim / 2), by: 1)))
                * (-2 / Float(headDim) * log(ropeTheta)))
        let epsArray = MLXArray([eps])
        eval(freqs, epsArray)

        var mlps: [(MLXArray) -> MLXArray] = []
        var steps: [(pre: ([MLXArray]) -> [MLXArray], post: ([MLXArray]) -> [MLXArray])] = []
        for layer in 0..<config.numHiddenLayers {
            let p = "model.layers.\(layer).mlp."
            let router = Linear(p + "gate", parameters: parameters, quantization: quantization)
            let gateUp = Experts(
                p + "switch_mlp.gate_up_proj", parameters: parameters, quantization: quantization)
            let down = Experts(
                p + "switch_mlp.down_proj", parameters: parameters, quantization: quantization)
            let k = config.numExpertsPerTok
            let split = config.numExperts - k
            let hidden = config.hiddenSize
            let inner = config.moeIntermediateSize
            let normTopk = config.normTopkProb

            // The softmax spans all 128 experts, so the eight kept weights depend on the
            // ones dropped — unlike a sigmoid, where selection and weighting are
            // independent. Renormalizing after the cut is what makes the partition
            // equivalent to transformers' sorted topk. `precise` accumulates in float32
            // and rounds back: the normalization runs in the checkpoint's dtype, as in
            // mlx-lm, and float32 here would be a different graph than the reference.
            func route(_ x: MLXArray) -> (chosen: MLXArray, weights: MLXArray) {
                let probs = softmax(router(x), axis: -1, precise: true)
                let chosen = argPartition(probs, kth: split, axis: -1)[0..., 0..., split...]
                var weights = takeAlong(probs, chosen, axis: -1)
                if normTopk {
                    weights = weights / weights.sum(axis: -1, keepDims: true)
                }
                return (chosen, weights)
            }

            func experts(_ tokens: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
                let fused = gateUp(tokens, indices: indices, sorted: sorted)
                let pairs = fused.reshaped(Array(fused.shape.dropLast()) + [inner, 2])
                let gated = pairs[.ellipsis, 0]
                let up = pairs[.ellipsis, 1]
                return down(gated * sigmoid(gated) * up, indices: indices, sorted: sorted)
            }

            mlps.append { x in
                let (chosen, weights) = route(x)
                let length = x.dim(-2)
                let routed: MLXArray
                if length * k >= 64 {
                    // Prefill: group the rows by expert so the gather streams each
                    // expert's weight once. A pure reorder — every row is still its own
                    // gemv, so the numbers cannot move.
                    let flat = chosen.reshaped(-1)
                    let order = argSort(flat)
                    let tokens =
                        x.reshaped(length, 1, hidden)[floorDivide(order, MLXArray(Int32(k)))]
                    let out = experts(tokens, flat[order], sorted: true)
                    routed = out[argSort(order)].reshaped(1, length, k, hidden)
                } else {
                    let tokens = expandedDimensions(x, axes: [-2, -3])
                    routed = squeezed(experts(tokens, chosen, sorted: false), axis: -2)
                }
                return (routed * expandedDimensions(weights, axis: -1)).sum(axis: -2)
            }

            let attn = "model.layers.\(layer).self_attn."
            let qkv = Linear(attn + "qkv_proj", parameters: parameters, quantization: quantization)
            let oProj = Linear(attn + "o_proj", parameters: parameters, quantization: quantization)
            let inputNorm = parameters["model.layers.\(layer).input_layernorm.weight"]!
            let postNorm = parameters["model.layers.\(layer).post_attention_layernorm.weight"]!
            let qNorm = parameters[attn + "q_norm.weight"]!
            let kNorm = parameters[attn + "k_norm.weight"]!

            let pre = compile { (inputs: [MLXArray]) -> [MLXArray] in
                let fused = qkv(MLXFast.rmsNorm(inputs[0], weight: inputNorm, eps: eps))
                let (queries, keys) = ropeEpilogue(
                    fused.reshaped(-1), queryHeads: numHeads, kvHeads: numKVHeads,
                    headDim: headDim, qNorm: qNorm, kNorm: kNorm, offset: inputs[1],
                    freqs: freqs, eps: epsArray)
                return [
                    queries.reshaped(1, numHeads, 1, headDim),
                    keys.reshaped(1, numKVHeads, 1, headDim),
                    fused[.ellipsis, (queryWidth + keyValueWidth)...]
                        .reshaped(1, numKVHeads, 1, headDim),
                ]
            }

            let post: ([MLXArray]) -> [MLXArray]
            // `softmaxTopK` renormalizes unconditionally, as both MoE checkpoints here ask
            // it to; a router that keeps the raw softmax weights has to take the op chain.
            if normTopk,
                case .quantized(let guW, let guS, let guB, let guGroup, let guBits) = gateUp,
                case .quantized(let dnW, let dnS, let dnB, let dnGroup, let dnBits) = down,
                moeGemvApplies(
                    hidden: hidden, inner: inner, gateUpGroup: guGroup, downGroup: dnGroup)
            {
                post = compile { (inputs: [MLXArray]) -> [MLXArray] in
                    let projected = oProj(
                        inputs[0].transposed(0, 2, 1, 3).reshaped(1, 1, queryWidth))
                    let (attended, h) = addRmsNorm(
                        inputs[1].reshaped(-1), projected.reshaped(-1), weight: postNorm,
                        eps: epsArray)
                    let (chosen, weights) = softmaxTopK(
                        router(h.reshaped(1, 1, hidden)).reshaped(-1), k: k)
                    let act = moeGateUpAct(
                        h, weight: guW, scales: guS, biases: guB, indices: chosen,
                        groupSize: guGroup, bits: guBits)
                    return [
                        moeDownCombine(
                            act.reshaped(-1), weight: dnW, scales: dnS, biases: dnB,
                            indices: chosen, routing: weights, residual: attended,
                            groupSize: dnGroup, bits: dnBits
                        ).reshaped(1, 1, hidden)
                    ]
                }
            } else {
                post = compile { (inputs: [MLXArray]) -> [MLXArray] in
                    let attended =
                        inputs[1]
                        + oProj(inputs[0].transposed(0, 2, 1, 3).reshaped(1, 1, queryWidth))
                    let h = MLXFast.rmsNorm(attended, weight: postNorm, eps: eps)
                    let (chosen, weights) = route(h)
                    let tokens = expandedDimensions(h, axes: [-2, -3])
                    let routed = squeezed(experts(tokens, chosen, sorted: false), axis: -2)
                    return [
                        attended
                            + (routed * expandedDimensions(weights, axis: -1)).sum(axis: -2)
                    ]
                }
            }
            steps.append((pre: pre, post: post))
        }
        self.mlps = mlps
        self.steps = steps
    }

    /// Total by construction: loadQwen3MoeParameters rejects any checkpoint missing a name.
    private subscript(name: String) -> MLXArray { parameters[name]! }

    private func linear(_ name: String) -> Linear {
        Linear(name, parameters: parameters, quantization: quantization)
    }

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

    public func callAsFunction(_ ids: [Int32]) -> Qwen3MoeActivations {
        let embeddings = embed(ids)
        let decoded = decode(embeddings, cache: makeCache())
        return Qwen3MoeActivations(
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
        // One rotary offset for the whole token: every layer's cache is at the same
        // position, and 48 four-byte uploads per token are 47 too many.
        let offset = x.dim(1) == 1 ? MLXArray([Int32(cache[0].offset)]) : nil
        for layer in 0..<config.numHiddenLayers {
            (x, cache[layer]) = block(x, layer: layer, cache: cache[layer], offset: offset)
            blocks.append(x)
        }

        let normed = rmsNorm(x, prefix: "model.norm")
        let head = config.tieWordEmbeddings ? "model.embed_tokens" : "lm_head"
        return (blocks, normed, linear(head)(normed), cache)
    }

    func block(_ x: MLXArray, layer: Int, cache: KVCache, offset: MLXArray? = nil)
        -> (MLXArray, KVCache)
    {
        if x.dim(1) == 1 {
            let step = steps[layer]
            let projected = step.pre([x, offset ?? MLXArray([Int32(cache.offset)])])
            let (keys, values, grown) = cache.appending(
                keys: projected[1], values: projected[2])
            let attended = MLXFast.scaledDotProductAttention(
                queries: projected[0], keys: keys, values: values,
                scale: 1 / sqrt(Float(config.headDim)), mask: .none)
            return (step.post([attended, x])[0], grown)
        }

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

        let queries = rope(
            rmsNorm(heads(parts[0], config.numAttentionHeads), prefix: p + "q_norm"),
            offset: offset)
        let (keys, values, grown) = cache.appending(
            keys: rope(
                rmsNorm(heads(parts[1], config.numKeyValueHeads), prefix: p + "k_norm"),
                offset: offset),
            values: heads(parts[2], config.numKeyValueHeads))

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

    func rope(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: config.headDim, traditional: false, base: config.ropeTheta,
            scale: 1, offset: offset)
    }
}

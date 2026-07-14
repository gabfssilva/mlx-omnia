import Foundation
import MLX

public struct Lfm2MoeActivations {
    public let embeddings: MLXArray
    public let blocks: [MLXArray]
    public let norm: MLXArray
    public let logits: MLXArray
}

/// LFM2.5: a hybrid decoder where most layers are a gated short conv and every fourth
/// is GQA, and the MLP past the first two layers is a mixture of 32 experts, 4 per
/// token. The conv layers cache a sliding window instead of keys and values.
public struct Lfm2Moe: CausalLM {
    private let config: Lfm2MoeConfig
    private let parameters: [String: MLXArray]
    /// One closure per layer: the dense MLPs compiled as in Qwen, the sparse ones
    /// routing through gatherMM.
    let mlps: [(MLXArray) -> MLXArray]
    /// One compiled (x, window) → (y, window) closure per conv layer: 18 layers of
    /// tiny elementwise ops per step is launch overhead a fused trace erases.
    private let convs: [(([MLXArray]) -> [MLXArray])?]

    public init(config: Lfm2MoeConfig, parameters: [String: MLXArray]) {
        self.config = config

        // One fused projection per attention layer instead of three, as in Qwen3.
        var parameters = parameters
        for (layer, type) in config.layerTypes.enumerated() where type == "full_attention" {
            let p = "model.layers.\(layer).self_attn."
            let parts = ["q_proj.", "k_proj.", "v_proj."].map { parameters[p + $0 + "weight"]! }
            parameters[p + "qkv_proj.weight"] = concatenated(parts, axis: 0)
        }
        // Gate and up fused the same way, row-aligned along each expert's output
        // axis: one gather reads both. Materialized here (the concat is real work,
        // and a compiled closure would otherwise re-run it every token) and the
        // separate stacks dropped, or both copies would stay resident.
        for layer in config.numDenseLayers..<config.numHiddenLayers {
            let p = "model.layers.\(layer).feed_forward.experts."
            let fused = concatenated(
                [parameters[p + "w1.weight"]!, parameters[p + "w3.weight"]!], axis: 1)
            eval(fused)
            parameters[p + "w13.weight"] = fused
            parameters.removeValue(forKey: p + "w1.weight")
            parameters.removeValue(forKey: p + "w3.weight")
        }
        self.parameters = parameters

        convs = config.layerTypes.enumerated().map { layer, type in
            guard type == "conv" else { return nil }
            let p = "model.layers.\(layer).conv."
            let inProj = Linear(p + "in_proj", parameters: parameters, quantization: nil)
            let outProj = Linear(p + "out_proj", parameters: parameters, quantization: nil)
            let weight = parameters[p + "conv.weight"]!
            // The conv kernel unrolled: `convLCache` taps of [hidden].
            let taps = (0..<config.convLCache).map { weight[0..., 0, $0] }

            // The gated short conv: in_proj splits into B, C, x; a causal depthwise
            // conv runs over B·x; C gates the result. The taps are shifted elementwise
            // products accumulated in float32 like the conv kernel would, rounding
            // once — per-tap bfloat16 rounding is what would diverge from mlx-lm.
            let body: ([MLXArray]) -> [MLXArray] = { inputs in
                let (x, window) = (inputs[0], inputs[1])
                let length = x.dim(1)
                let parts = split(inProj(x), parts: 3, axis: -1)
                let bx = parts[0] * parts[2]
                let padded = concatenated([window, bx], axis: 1)
                let lifted = padded.asType(.float32)
                var conv = lifted[0..., ..<length, 0...] * taps[0]
                for j in 1..<taps.count {
                    conv = conv + lifted[0..., j..<(j + length), 0...] * taps[j]
                }
                return [
                    outProj(parts[1] * conv.asType(bx.dtype)),
                    padded[0..., length..., 0...],
                ]
            }
            // Only the one-token step compiles: its shape never changes, so it traces
            // once. Compiling prefill would re-trace per prompt length. With the
            // standard kernel of 3 the step goes through the fused conv dispatch
            // (see ConvMix.swift) and only out_proj stays a matmul.
            let step: ([MLXArray]) -> [MLXArray]
            if config.convLCache == 3 {
                let inW = parameters[p + "in_proj.weight"]!
                let flatTaps = weight.squeezed(axis: 1).reshaped(-1)
                eval(flatTaps)
                let hidden = config.hiddenSize
                step = compile { inputs in
                    let (gated, window) = convMix(
                        inputs[0].reshaped(-1), weights: inW, taps: flatTaps,
                        window: inputs[1].reshaped(2, hidden))
                    return [outProj(gated.reshaped(1, 1, hidden)), window.reshaped(1, 2, hidden)]
                }
            } else {
                step = compile(body)
            }
            return { inputs in inputs[0].dim(1) == 1 ? step(inputs) : body(inputs) }
        }

        mlps = (0..<config.numHiddenLayers).map { layer in
            let p = "model.layers.\(layer).feed_forward."
            guard layer >= config.numDenseLayers else {
                let gate = Linear(p + "w1", parameters: parameters, quantization: nil)
                let up = Linear(p + "w3", parameters: parameters, quantization: nil)
                let down = Linear(p + "w2", parameters: parameters, quantization: nil)
                // silu spelled out for the same ABBA-deadlock reason as in Qwen.
                return compile { x in
                    let gated = gate(x)
                    return down(gated * sigmoid(gated) * up(x))
                }
            }

            let router = Linear(p + "gate", parameters: parameters, quantization: nil)
            let bias = config.useExpertBias ? parameters[p + "expert_bias"]! : nil
            // The fused stack in checkpoint layout for the custom kernels; gatherMM
            // wants [experts, in, out], so it gets the swapped view.
            let gateUpW = parameters[p + "experts.w13.weight"]!
            let downW = parameters[p + "experts.w2.weight"]!
            let k = config.numExpertsPerTok
            let split = config.numExperts - k
            let hidden = config.hiddenSize
            let inner = config.moeIntermediateSize
            let normTopk = config.normTopkProb
            let scaling = config.routedScalingFactor

            func route(_ x: MLXArray) -> (chosen: MLXArray, weights: MLXArray) {
                // Sigmoid scores route; the bias joins in float32 for selection only,
                // so it shifts which experts win but never their weights.
                let scores = sigmoid(router(x))
                let selector = bias.map { scores.asType(.float32) + $0 } ?? scores
                let chosen = argPartition(selector, kth: split, axis: -1)[0..., 0..., split...]
                var weights = takeAlong(scores, chosen, axis: -1)
                if normTopk {
                    weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-6)
                }
                return (chosen, weights)
            }

            let body: (MLXArray) -> MLXArray = { x in
                let (chosen, weights) = route(x)
                let length = x.dim(-2)
                let experts: MLXArray
                if length * k >= 64 {
                    // Prefill: group the rows by expert so gatherMM streams each
                    // expert's weights once. A pure reorder — every row is still its
                    // own gemv, so the numbers cannot move.
                    let flat = chosen.reshaped(-1)
                    let order = argSort(flat)
                    let tokens = x.reshaped(length, 1, hidden)[floorDivide(order, MLXArray(Int32(k)))]
                    let sorted = flat[order]
                    let gateUp = gatherMM(
                        tokens, gateUpW.swappedAxes(-2, -1), rhsIndices: sorted,
                        sortedIndices: true)
                    let gated = gateUp[.ellipsis, ..<inner]
                    let up = gateUp[.ellipsis, inner...]
                    let routed = gatherMM(
                        gated * sigmoid(gated) * up, downW.swappedAxes(-2, -1),
                        rhsIndices: sorted, sortedIndices: true)
                    experts = routed[argSort(order)].reshaped(1, length, k, hidden)
                } else {
                    let tokens = expandedDimensions(x, axes: [-2, -3])
                    let gateUp = gatherMM(
                        tokens, gateUpW.swappedAxes(-2, -1), rhsIndices: chosen)
                    let gated = gateUp[.ellipsis, ..<inner]
                    let up = gateUp[.ellipsis, inner...]
                    let routed = gatherMM(
                        gated * sigmoid(gated) * up, downW.swappedAxes(-2, -1),
                        rhsIndices: chosen)
                    experts = squeezed(routed, axis: -2)
                }
                return (experts * expandedDimensions(weights * scaling, axis: -1)).sum(axis: -2)
            }
            // The one-token step is three custom dispatches: routing in a single
            // simdgroup, then the gemv pair — the down kernel applies silu and the
            // routing weights while it streams, so the routed MLP runs at ~700 GB/s
            // where gatherMM manages ~450 (see MoeGemv.swift).
            let routerW = parameters[p + "gate.weight"]!
            let routeBias = bias ?? zeros([config.numExperts], dtype: .float32)
            let routeScale = MLXArray([scaling])
            let step = compile { (x: MLXArray) -> MLXArray in
                let (chosen, weights) = moeRoute(
                    x.reshaped(-1), router: routerW, bias: routeBias, scale: routeScale,
                    k: k, normalized: normTopk)
                let gateUp = moeGateUp(x.reshaped(-1), weights: gateUpW, indices: chosen)
                let routed = moeDown(
                    gateUp, weights: downW, indices: chosen, routing: weights)
                return routed.sum(axis: 0).reshaped(1, 1, hidden)
            }
            return { x in x.dim(-2) == 1 ? step(x) : body(x) }
        }
    }

    /// Total by construction: loadLfm2MoeParameters rejects any checkpoint missing a name.
    private subscript(name: String) -> MLXArray { parameters[name]! }

    private func linear(_ name: String) -> Linear {
        Linear(name, parameters: parameters, quantization: nil)
    }

    public func embed(_ tokens: MLXArray, offset: Int = 0) -> MLXArray {
        let table = Embedding("model.embed_tokens", parameters: parameters, quantization: nil)
        return table(tokens)
    }

    public func makeCache() -> [KVCache] { (0..<config.numHiddenLayers).map { _ in KVCache() } }

    public func logits(_ embeddings: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
    {
        let decoded = decode(embeddings, cache: cache)
        return (decoded.logits, decoded.cache)
    }

    public func callAsFunction(_ ids: [Int32]) -> Lfm2MoeActivations {
        let embeddings = embed(ids)
        let decoded = decode(embeddings, cache: makeCache())
        return Lfm2MoeActivations(
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

        let normed = rmsNorm(x, prefix: "model.embedding_norm")
        return (blocks, normed, linear("model.embed_tokens")(normed), cache)
    }

    func block(_ x: MLXArray, layer: Int, cache: KVCache) -> (MLXArray, KVCache) {
        let prefix = "model.layers.\(layer)."
        let normed = rmsNorm(x, prefix: prefix + "operator_norm")
        let (mixed, grown) =
            config.layerTypes[layer] == "full_attention"
            ? attention(normed, prefix: prefix, cache: cache)
            : shortConv(normed, layer: layer, cache: cache)
        let residual = x + mixed
        return (
            residual + mlps[layer](rmsNorm(residual, prefix: prefix + "ffn_norm")),
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

        // As in Qwen3: one headDim norm vector per projection, between it and the rope.
        let queries = rope(
            rmsNorm(heads(parts[0], config.numAttentionHeads), prefix: p + "q_layernorm"),
            offset: offset)
        let (keys, values, grown) = cache.appending(
            keys: rope(
                rmsNorm(heads(parts[1], config.numKeyValueHeads), prefix: p + "k_layernorm"),
                offset: offset),
            values: heads(parts[2], config.numKeyValueHeads))

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(config.headDim)), mask: length == 1 ? .none : .causal)

        let projected = linear(p + "out_proj")(
            attended.transposed(0, 2, 1, 3).reshaped(1, length, queryWidth))
        return (projected, grown)
    }

    func shortConv(_ x: MLXArray, layer: Int, cache: KVCache) -> (MLXArray, KVCache) {
        let window =
            cache.window ?? zeros([1, config.convLCache - 1, config.hiddenSize], dtype: x.dtype)
        let out = convs[layer]!([x, window])
        return (out[0], cache.slid(window: out[1], by: x.dim(1)))
    }

    func rmsNorm(_ x: MLXArray, prefix: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: self[prefix + ".weight"], eps: config.normEps)
    }

    func rope(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: config.headDim, traditional: false, base: config.ropeTheta,
            scale: 1, offset: offset)
    }
}

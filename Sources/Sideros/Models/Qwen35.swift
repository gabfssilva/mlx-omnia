import Foundation
import MLX

public struct Qwen35Activations {
    public let embeddings: MLXArray
    public let blocks: [MLXArray]
    public let norm: MLXArray
    public let logits: MLXArray
}

/// Qwen3.5/3.6: a hybrid decoder where three of every four layers are a gated DeltaNet
/// (linear attention — a short causal conv over q/k/v feeding a recurrent delta rule at
/// float32) and the fourth is GQA with an output gate fused into q_proj and RoPE over a
/// quarter of the head. The linear layers cache a conv window and the recurrent state
/// instead of keys and values.
public struct Qwen35: PositionedLM {
    private let config: Qwen35Config
    private let parameters: [String: MLXArray]
    private let quantization: Quantization?
    /// One closure per layer. Dense, it is compiled: the trace fuses silu(gate) · up into
    /// one kernel. Sparse, it routes and gathers, and stays out of `compile` — the two
    /// gather paths below trace differently and prefill would re-trace per length.
    let mlps: [(MLXArray) -> MLXArray]

    /// The one-token step. A DeltaNet layer is one compiled chain: its conv window and
    /// recurrent state ride in and out as arrays, and the twenty-odd elementwise ops
    /// around the recurrence — the unrolled conv, two l2 norms, the decay, the output
    /// gate — collapse into a handful of kernels. An attention layer keeps its projections
    /// eager, because `MLXFast.RoPE` takes the offset as an op attribute and a trace would
    /// freeze it at the first token; only its tail is compiled.
    ///
    /// Both tails are the same: the residual join, the routing and the two fused gemvs.
    /// Only the sparse variant builds any of this — the dense 0.8B and 27B keep the eager
    /// block, whose numbers are pinned against a float32 fixture.
    enum Step {
        case attention((MLXArray, MLXArray) -> MLXArray)
        case linear(([MLXArray]) -> [MLXArray])
    }
    let steps: [Step]

    public init(
        config: Qwen35Config, parameters: [String: MLXArray], quantization: Quantization? = nil
    ) {
        self.config = config
        self.quantization = quantization

        // One fused projection per mixer instead of three or four, as in Qwen3 —
        // row-aligned in every representation, so dense and quantized fuse alike.
        // Materialized here: `compile` pins captured arrays, and an unevaluated
        // concatenation would re-run inside every traced call.
        var parameters = parameters
        for (layer, type) in config.layerTypes.enumerated() {
            let names =
                type == "full_attention"
                ? ("model.layers.\(layer).self_attn.", ["q_proj.", "k_proj.", "v_proj."])
                : (
                    "model.layers.\(layer).linear_attn.",
                    ["in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a."]
                )
            for suffix in ["weight", "scales", "biases"] {
                let parts = names.1.compactMap { parameters[names.0 + $0 + suffix] }
                guard parts.count == names.1.count else { continue }
                let fused = concatenated(parts, axis: 0)
                eval(fused)
                parameters[names.0 + "fused_proj." + suffix] = fused
                for part in names.1 { parameters.removeValue(forKey: names.0 + part + suffix) }
            }
        }
        // Gate and up fused row-interleaved ([g₀,u₀,g₁,u₁,…]) so one gather reads both and
        // each (gate, up) pair lands in one simdgroup of the one-token kernel. Row-aligned
        // in all three representations — the groups run along the row, so interleaving rows
        // leaves every one of them intact and no number moves.
        for layer in 0..<config.numHiddenLayers where config.mixtureOfExperts {
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

        let mlps: [(MLXArray) -> MLXArray] = (0..<config.numHiddenLayers).map { layer in
            let p = "model.layers.\(layer).mlp."
            if config.mixtureOfExperts {
                return Self.mixture(
                    p, config: config, parameters: parameters, quantization: quantization)
            }
            let gate = Linear(p + "gate_proj", parameters: parameters, quantization: quantization)
            let up = Linear(p + "up_proj", parameters: parameters, quantization: quantization)
            let down = Linear(p + "down_proj", parameters: parameters, quantization: quantization)
            // silu spelled out for the same ABBA-deadlock reason as in Qwen.
            return compile { x in
                let gated = gate(x)
                return down(gated * sigmoid(gated) * up(x))
            }
        }
        self.mlps = mlps

        guard config.mixtureOfExperts else {
            self.steps = []
            return
        }

        let eps = config.rmsNormEps
        let epsArray = MLXArray([eps])
        let zero = MLXArray(Float(0))
        eval(epsArray, zero)

        self.steps = (0..<config.numHiddenLayers).map { layer in
            let prefix = "model.layers.\(layer)."
            let inputNorm = parameters[prefix + "input_layernorm.weight"]!
            let postNorm = parameters[prefix + "post_attention_layernorm.weight"]!
            let tail = Self.tail(
                layer, config: config, parameters: parameters, quantization: quantization,
                mlp: mlps[layer], postNorm: postNorm, eps: epsArray)

            if config.layerTypes[layer] == "full_attention" {
                let compiled = compile { (inputs: [MLXArray]) in [tail(inputs[0], inputs[1])] }
                return .attention { x, projected in compiled([x, projected])[0] }
            }

            let p = prefix + "linear_attn."
            let proj = Linear(p + "fused_proj", parameters: parameters, quantization: quantization)
            let outProj = Linear(p + "out_proj", parameters: parameters, quantization: quantization)
            let gateNorm = parameters[p + "norm.weight"]!
            // Constants of the decay, folded once: exp(A_log) and the bias are the same
            // numbers every step, and the conv taps are read a row at a time.
            let taps = (0..<config.linearConvKernelDim).map {
                parameters[p + "conv1d.weight"]![0..., $0].asType(.float32)
            }
            let decay = exp(parameters[p + "A_log"]!)
            let dtBias = parameters[p + "dt_bias"]!.asType(.float32)
            eval(taps + [decay, dtBias])

            let (keyDim, valueDim) = (config.keyDim, config.valueDim)
            let heads = config.linearNumValueHeads
            let (keyHeads, keyHeadDim) = (config.linearNumKeyHeads, config.linearKeyHeadDim)
            let valueHeadDim = config.linearValueHeadDim
            let scale = 1 / sqrt(Float(keyHeadDim))

            // l2norm as in transformers: eps inside the sum, not the mean; the query also
            // takes the 1/√Dk attention scale here, the key stays unit-length.
            func l2(_ v: MLXArray, scale: Float = 1) -> MLXArray {
                let lifted = v.asType(.float32)
                let norm = rsqrt(lifted.square().sum(axis: -1, keepDims: true) + 1e-6)
                return (lifted * norm * scale).asType(v.dtype)
            }

            let step = compile { (inputs: [MLXArray]) -> [MLXArray] in
                let (x, window, state) = (inputs[0], inputs[1], inputs[2])
                let fused = proj(MLXFast.rmsNorm(x, weight: inputNorm, eps: eps))
                let parts = split(
                    fused,
                    indices: [config.convDim, config.convDim + valueDim,
                              config.convDim + valueDim + heads], axis: -1)
                let z = parts[1].reshaped(1, 1, heads, valueHeadDim)

                // The causal conv, unrolled into its taps and accumulated at float32 as the
                // conv kernel would. At one token the window is the whole history.
                let padded = concatenated([window, parts[0]], axis: 1)
                let lifted = padded.asType(.float32)
                var conv = lifted[0..., ..<1, 0...] * taps[0]
                for j in 1..<taps.count {
                    conv = conv + lifted[0..., j..<(j + 1), 0...] * taps[j]
                }
                let mixed = conv.asType(x.dtype)
                let convOut = mixed * sigmoid(mixed)

                let qkv = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
                let q = qkv[0].reshaped(1, 1, keyHeads, keyHeadDim)
                let k = qkv[1].reshaped(1, 1, keyHeads, keyHeadDim)
                let v = qkv[2].reshaped(1, 1, heads, valueHeadDim)

                // g = exp(−exp(A_log) · softplus(a + dt_bias)) at float32; beta in dtype.
                let g = exp(-decay * logAddExp(parts[3].asType(.float32) + dtBias, zero))
                let (y, grown) = gatedDelta(
                    q: l2(q, scale: scale), k: l2(k), v: v, g: g,
                    beta: sigmoid(parts[2]), state: state)

                // Gated RMSNorm per value head: the norm at float32, silu(z) joining there.
                let normed = MLXFast.rmsNorm(y, weight: gateNorm, eps: eps)
                let zLifted = z.asType(.float32)
                let gated = (normed.asType(.float32) * zLifted * sigmoid(zLifted)).asType(x.dtype)
                let projected = outProj(gated.reshaped(1, 1, valueDim))

                return [tail(x, projected), padded[0..., 1..., 0...], grown]
            }
            return .linear(step)
        }
    }

    /// The tail every one-token layer ends in: the residual join, the routing and the two
    /// fused gemvs. The shared expert rides in as the residual the down projection folds —
    /// the reference spends an add there either way, and this way it costs no dispatch of
    /// its own. Falls back to the op chain when the weights are not quantized to a shape
    /// the gemvs tile.
    private static func tail(
        _ layer: Int, config: Qwen35Config, parameters: [String: MLXArray],
        quantization: Quantization?, mlp: @escaping (MLXArray) -> MLXArray,
        postNorm: MLXArray, eps: MLXArray
    ) -> (MLXArray, MLXArray) -> MLXArray {
        let p = "model.layers.\(layer).mlp."
        let hidden = config.hiddenSize
        let inner = config.moeIntermediateSize
        let k = config.numExpertsPerTok
        let gateUp = Experts(
            p + "switch_mlp.gate_up_proj", parameters: parameters, quantization: quantization)
        let down = Experts(
            p + "switch_mlp.down_proj", parameters: parameters, quantization: quantization)

        guard case .quantized(let guW, let guS, let guB, let guGroup, let guBits) = gateUp,
            case .quantized(let dnW, let dnS, let dnB, let dnGroup, let dnBits) = down,
            moeGemvApplies(
                hidden: hidden, inner: inner, gateUpGroup: guGroup, downGroup: dnGroup)
        else {
            let scalar = config.rmsNormEps
            return { x, projected in
                let attended = x + projected
                return attended
                    + mlp(MLXFast.rmsNorm(attended, weight: postNorm, eps: scalar))
            }
        }

        let router = Linear(p + "gate", parameters: parameters, quantization: quantization)
        let sharedGate = Linear(
            p + "shared_expert.gate_proj", parameters: parameters, quantization: quantization)
        let sharedUp = Linear(
            p + "shared_expert.up_proj", parameters: parameters, quantization: quantization)
        let sharedDown = Linear(
            p + "shared_expert.down_proj", parameters: parameters, quantization: quantization)
        let sharedSwitch = Linear(
            p + "shared_expert_gate", parameters: parameters, quantization: quantization)

        return { x, projected in
            let (residual, h) = addRmsNorm(
                x.reshaped(-1), projected.reshaped(-1), weight: postNorm, eps: eps)
            let row = h.reshaped(1, 1, hidden)
            let (chosen, weights) = softmaxTopK(router(row).reshaped(-1), k: k)

            let gated = sharedGate(row)
            let shared = sharedDown(gated * sigmoid(gated) * sharedUp(row))
            let bias = residual + (sigmoid(sharedSwitch(row)) * shared).reshaped(-1)

            let act = moeGateUpAct(
                h, weight: guW, scales: guS, biases: guB, indices: chosen,
                groupSize: guGroup, bits: guBits)
            return moeDownCombine(
                act.reshaped(-1), weight: dnW, scales: dnS, biases: dnB, indices: chosen,
                routing: weights, residual: bias, groupSize: dnGroup, bits: dnBits
            ).reshaped(1, 1, hidden)
        }
    }

    /// The sparse MLP of the 35B-A3B: 256 experts, 8 per token, plus a shared expert every
    /// token pays for, scaled by its own sigmoid gate. The router softmaxes over all 256 —
    /// so the eight kept weights depend on the ones dropped — and renormalizes the eight,
    /// which is what makes the partition equivalent to transformers' sorted topk.
    private static func mixture(
        _ p: String, config: Qwen35Config, parameters: [String: MLXArray],
        quantization: Quantization?
    ) -> (MLXArray) -> MLXArray {
        let router = Linear(p + "gate", parameters: parameters, quantization: quantization)
        let gateUp = Experts(
            p + "switch_mlp.gate_up_proj", parameters: parameters, quantization: quantization)
        let down = Experts(
            p + "switch_mlp.down_proj", parameters: parameters, quantization: quantization)
        let sharedGate = Linear(
            p + "shared_expert.gate_proj", parameters: parameters, quantization: quantization)
        let sharedUp = Linear(p + "shared_expert.up_proj", parameters: parameters, quantization: quantization)
        let sharedDown = Linear(
            p + "shared_expert.down_proj", parameters: parameters, quantization: quantization)
        let sharedSwitch = Linear(
            p + "shared_expert_gate", parameters: parameters, quantization: quantization)
        let k = config.numExpertsPerTok
        let split = config.numExperts - k
        let hidden = config.hiddenSize
        let inner = config.moeIntermediateSize

        func experts(_ tokens: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
            let fused = gateUp(tokens, indices: indices, sorted: sorted)
            let pairs = fused.reshaped(Array(fused.shape.dropLast()) + [inner, 2])
            let gated = pairs[.ellipsis, 0]
            let up = pairs[.ellipsis, 1]
            return down(gated * sigmoid(gated) * up, indices: indices, sorted: sorted)
        }

        return { x in
            // `precise` accumulates in float32 and rounds back: the normalization runs in
            // the checkpoint's dtype, as in mlx-lm and transformers.
            let probs = softmax(router(x), axis: -1, precise: true)
            let chosen = argPartition(probs, kth: split, axis: -1)[0..., 0..., split...]
            var weights = takeAlong(probs, chosen, axis: -1)
            weights = weights / weights.sum(axis: -1, keepDims: true)

            let length = x.dim(-2)
            let routed: MLXArray
            if length * k >= 64 {
                // Prefill: group the rows by expert so the gather streams each expert's
                // weight once. A pure reorder — every row is still its own gemv, so the
                // numbers cannot move.
                let flat = chosen.reshaped(-1)
                let order = argSort(flat)
                let tokens = x.reshaped(length, 1, hidden)[floorDivide(order, MLXArray(Int32(k)))]
                let out = experts(tokens, flat[order], sorted: true)
                routed = out[argSort(order)].reshaped(1, length, k, hidden)
            } else {
                let tokens = expandedDimensions(x, axes: [-2, -3])
                routed = squeezed(experts(tokens, chosen, sorted: false), axis: -2)
            }

            let gated = sharedGate(x)
            let shared = sharedDown(gated * sigmoid(gated) * sharedUp(x))
            return (routed * expandedDimensions(weights, axis: -1)).sum(axis: -2)
                + sigmoid(sharedSwitch(x)) * shared
        }
    }

    /// Total by construction: loadQwen35Parameters rejects any checkpoint missing a name.
    private subscript(name: String) -> MLXArray { parameters[name]! }

    private func linear(_ name: String) -> Linear {
        Linear(name, parameters: parameters, quantization: quantization)
    }

    public func embed(_ tokens: MLXArray, offset: Int = 0) -> MLXArray {
        let table = Embedding("model.embed_tokens", parameters: parameters, quantization: quantization)
        return table(tokens)
    }

    public func makeCache() -> [KVCache] { (0..<config.numHiddenLayers).map { _ in KVCache() } }

    /// Declared here rather than defaulted in an extension: `Generation` calls it through
    /// `any CausalLM`, and that witness has to be this one.
    public func logits(_ embeddings: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
    {
        logits(embeddings, positions: .offset(cache[0].offset), cache: cache)
    }

    public func logits(_ embeddings: MLXArray, positions: RopePositions, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
    {
        let decoded = decode(embeddings, positions: positions, cache: cache)
        return (decoded.logits, decoded.cache)
    }

    public func callAsFunction(_ ids: [Int32]) -> Qwen35Activations {
        let embeddings = embed(ids)
        let decoded = decode(embeddings, positions: .offset(0), cache: makeCache())
        return Qwen35Activations(
            embeddings: embeddings,
            blocks: decoded.blocks,
            norm: decoded.norm,
            logits: decoded.logits
        )
    }

    func decode(_ embeddings: MLXArray, positions: RopePositions, cache: [KVCache])
        -> (blocks: [MLXArray], norm: MLXArray, logits: MLXArray, cache: [KVCache])
    {
        var cache = cache
        var x = embeddings
        var blocks: [MLXArray] = []
        for layer in 0..<config.numHiddenLayers {
            (x, cache[layer]) = block(x, layer: layer, positions: positions, cache: cache[layer])
            blocks.append(x)
        }

        let normed = rmsNorm(x, prefix: "model.norm")
        let head = config.tied ? "model.embed_tokens" : "lm_head"
        return (blocks, normed, linear(head)(normed), cache)
    }

    func block(_ x: MLXArray, layer: Int, positions: RopePositions, cache: KVCache)
        -> (MLXArray, KVCache)
    {
        let prefix = "model.layers.\(layer)."
        if x.dim(1) == 1, !steps.isEmpty {
            switch steps[layer] {
            case .attention(let tail):
                let (context, grown) = attention(
                    rmsNorm(x, prefix: prefix + "input_layernorm"), prefix: prefix,
                    positions: positions, cache: cache)
                return (tail(x, context), grown)
            case .linear(let step):
                let out = step([x, window(cache, like: x), state(cache)])
                return (out[0], cache.slid(window: out[1], state: out[2], by: 1))
            }
        }

        let normed = rmsNorm(x, prefix: prefix + "input_layernorm")
        let (mixed, grown) =
            config.layerTypes[layer] == "full_attention"
            ? attention(normed, prefix: prefix, positions: positions, cache: cache)
            : deltaNet(normed, prefix: prefix, cache: cache)
        let residual = x + mixed
        return (
            residual + mlps[layer](rmsNorm(residual, prefix: prefix + "post_attention_layernorm")),
            grown
        )
    }

    private func window(_ cache: KVCache, like x: MLXArray) -> MLXArray {
        cache.window ?? zeros([1, config.linearConvKernelDim - 1, config.convDim], dtype: x.dtype)
    }

    private func state(_ cache: KVCache) -> MLXArray {
        cache.state
            ?? zeros(
                [1, config.linearNumValueHeads, config.linearValueHeadDim,
                 config.linearKeyHeadDim], dtype: .float32)
    }

    func attention(_ x: MLXArray, prefix: String, positions: RopePositions, cache: KVCache)
        -> (MLXArray, KVCache)
    {
        let length = x.dim(1)
        let p = prefix + "self_attn."

        // q_proj carries [query ‖ gate] per head: twice the query width up front.
        let queryWidth = config.numAttentionHeads * config.headDim
        let keyValueWidth = config.numKeyValueHeads * config.headDim
        let fused = linear(p + "fused_proj")(x)
        let parts = split(
            fused, indices: [2 * queryWidth, 2 * queryWidth + keyValueWidth], axis: -1)

        func heads(_ part: MLXArray, _ count: Int) -> MLXArray {
            part.reshaped(1, length, count, config.headDim).transposed(0, 2, 1, 3)
        }

        let qGate = split(
            parts[0].reshaped(1, length, config.numAttentionHeads, 2 * config.headDim),
            parts: 2, axis: -1)
        let gate = qGate[1].reshaped(1, length, queryWidth)

        // q/k-norm over the full head as in Qwen3, but the rotation touches only the
        // first quarter of it (partial rotary at theta 1e7).
        let queries = rotated(qGate[0], norm: p + "q_norm", positions: positions)
        let (keys, values, grown) = cache.appending(
            keys: rotated(
                parts[1].reshaped(1, length, config.numKeyValueHeads, config.headDim),
                norm: p + "k_norm", positions: positions),
            values: heads(parts[2], config.numKeyValueHeads))

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(config.headDim)), mask: length == 1 ? .none : .causal)

        // The gate multiplies before o_proj, sigmoid in the model dtype.
        let projected = linear(p + "o_proj")(
            attended.transposed(0, 2, 1, 3).reshaped(1, length, queryWidth) * sigmoid(gate))
        return (projected, grown)
    }

    func deltaNet(_ x: MLXArray, prefix: String, cache: KVCache) -> (MLXArray, KVCache) {
        let length = x.dim(1)
        let p = prefix + "linear_attn."
        let (keyDim, valueDim, convDim) = (config.keyDim, config.valueDim, config.convDim)
        let heads = config.linearNumValueHeads

        let fused = linear(p + "fused_proj")(x)
        let parts = split(
            fused,
            indices: [convDim, convDim + valueDim, convDim + valueDim + heads], axis: -1)
        let z = parts[1].reshaped(1, length, heads, config.linearValueHeadDim)

        // The causal conv over q‖k‖v, unrolled into kernel-size taps accumulated in
        // float32 like the conv kernel would, then silu — z, a and b bypass it.
        let window =
            cache.window ?? zeros([1, config.linearConvKernelDim - 1, convDim], dtype: x.dtype)
        let padded = concatenated([window, parts[0]], axis: 1)
        let taps = (0..<config.linearConvKernelDim).map { self[p + "conv1d.weight"][0..., $0] }
        let lifted = padded.asType(.float32)
        var conv = lifted[0..., ..<length, 0...] * taps[0].asType(.float32)
        for j in 1..<taps.count {
            conv = conv + lifted[0..., j..<(j + length), 0...] * taps[j].asType(.float32)
        }
        let mixed = conv.asType(x.dtype)
        let convOut = mixed * sigmoid(mixed)

        let qkv = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = qkv[0].reshaped(1, length, config.linearNumKeyHeads, config.linearKeyHeadDim)
        let k = qkv[1].reshaped(1, length, config.linearNumKeyHeads, config.linearKeyHeadDim)
        let v = qkv[2].reshaped(1, length, heads, config.linearValueHeadDim)

        // l2norm as in transformers: eps inside the sum, not the mean; the query also
        // takes the 1/√Dk attention scale here, the key stays unit-length.
        func l2(_ x: MLXArray, scale: Float = 1) -> MLXArray {
            let lifted = x.asType(.float32)
            let norm = rsqrt(lifted.square().sum(axis: -1, keepDims: true) + 1e-6)
            return (lifted * norm * scale).asType(x.dtype)
        }

        // Decay and write strength: g = exp(−exp(A_log) · softplus(a + dt_bias)) in
        // float32 — A_log never leaves float32 — and beta = sigmoid(b) in model dtype.
        let a = parts[3].asType(.float32) + self[p + "dt_bias"].asType(.float32)
        let g = exp(-exp(self[p + "A_log"]) * logAddExp(a, MLXArray(Float(0))))
        let beta = sigmoid(parts[2])

        let state =
            cache.state
            ?? zeros(
                [1, heads, config.linearValueHeadDim, config.linearKeyHeadDim], dtype: .float32)
        let (y, newState) = gatedDelta(
            q: l2(q, scale: 1 / sqrt(Float(config.linearKeyHeadDim))), k: l2(k), v: v,
            g: g, beta: beta, state: state)

        // Gated RMSNorm per value head: the norm in float32, silu(z) joining there too.
        let normed = MLXFast.rmsNorm(y, weight: self[p + "norm.weight"], eps: config.rmsNormEps)
        let zLifted = z.asType(.float32)
        let gated = (normed.asType(.float32) * zLifted * sigmoid(zLifted)).asType(x.dtype)

        let projected = linear(p + "out_proj")(gated.reshaped(1, length, valueDim))
        return (projected, cache.slid(window: padded[0..., length..., 0...], state: newState, by: length))
    }

    func rmsNorm(_ x: MLXArray, prefix: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: self[prefix + ".weight"], eps: config.rmsNormEps)
    }

    /// Norm and rotation over `[1, L, H, D]`, handed back as `[1, H, L, D]` for the attention.
    private func rotated(_ x: MLXArray, norm: String, positions: RopePositions) -> MLXArray {
        switch positions {
        case .offset(let offset):
            // What text has always done, character for character: the norm over the transposed
            // view, the fast kernel with a scalar. Every decode step of every prompt lands here.
            return rope(rmsNorm(x.transposed(0, 2, 1, 3), prefix: norm), offset: offset)

        case .explicit(let table):
            // The same kernel takes one position per *batch row* (`mlx_fast_rope_dynamic`), so
            // viewing the tokens as batch rows — [1, L, H, D] as [L, H, 1, D], a free reshape:
            // identical element order — gives every token a position of its own. Three calls,
            // one per MRoPE section, then a select by dimension. No new arithmetic anywhere,
            // which is what makes this bit-identical to the scalar path when the three sections
            // carry the same position — a hand-rolled cos/sin could not be (the rope kernel
            // uses metal::fast::cos, the elementwise op metal::precise::cos).
            let (length, heads, dim) = (x.dim(1), x.dim(2), x.dim(3))
            let rows = rmsNorm(x, prefix: norm).reshaped(length, heads, 1, dim)
            let sections = (0..<3).map { rope(rows, offset: table[$0]) }

            var rotated = sections[0]
            for section in 1..<3 {
                rotated = MLX.where(owned(by: section, dim: dim), sections[section], rotated)
            }
            return rotated.reshaped(1, length, heads, dim).transposed(0, 2, 1, 3)
        }
    }

    /// Which dimensions each MRoPE section rotates. The layout is interleaved: section 1 takes
    /// the frequencies 1, 4, 7…, section 2 takes 2, 5, 8…, and section 0 keeps what is left — so
    /// with `mrope_section` [11, 11, 10] over 32 frequency pairs, frequency i simply reads
    /// section i % 3. In the half-rotation layout frequency i owns dims i and i + dims/2, and
    /// past `ropeDimensions` nothing rotates at all: the three sections agree there, so the
    /// select is free to pick any of them.
    private func owned(by section: Int, dim: Int) -> MLXArray {
        let half = config.ropeDimensions / 2
        var owns = [Bool](repeating: false, count: dim)
        let bound = min(3 * config.mropeSection[section], half)
        for frequency in stride(from: section, to: bound, by: 3) {
            owns[frequency] = true
            owns[frequency + half] = true
        }
        return MLXArray(owns)
    }

    /// Text-only MRoPE is a plain partial rope: every section reads the same position,
    /// so the interleave rewrites each frequency with the value it already had.
    func rope(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: config.ropeDimensions, traditional: false, base: config.ropeTheta,
            scale: 1, offset: offset)
    }

    /// One position per row of the batch axis: the overload that makes a 3-D position sayable.
    func rope(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: config.ropeDimensions, traditional: false, base: config.ropeTheta,
            scale: 1, offset: offset)
    }
}

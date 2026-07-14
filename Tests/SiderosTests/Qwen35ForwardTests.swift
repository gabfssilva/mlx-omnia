import Foundation
import MLX
import Testing

@testable import Sideros

private func loadModel() throws -> (Qwen35Config, Qwen35, [String: MLXArray]) {
    let directory = try qwen35Directory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: .float32)
    return (config, Qwen35(config: config, parameters: parameters), parameters)
}

/// The reference is itself only a float32 approximation; `noise.*` is how far it drifts
/// from float64, measured when the fixture was generated. Three times it and no more.
@Test func qwen35ForwardMatchesTransformers() throws {
    let (config, model, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    func expectWithinNoiseFloor(_ ours: MLXArray, _ name: String) {
        let floor = golden["noise.\(name)"]!.item(Float.self)
        #expect(relativeDiff(ours, golden[name]!) < 3 * floor, "\(name)")
    }

    #expect(relativeDiff(activations.embeddings, golden["embeddings"]!) == 0)
    for layer in 0..<config.numHiddenLayers {
        expectWithinNoiseFloor(activations.blocks[layer], "block_\(layer)")
    }
    expectWithinNoiseFloor(activations.norm, "norm")
    expectWithinNoiseFloor(activations.logits, "logits")
}

/// Numeric closeness is not the goal; identical predictions are.
@Test func qwen35GreedyPredictionsMatchTransformers() throws {
    let (_, model, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    let ours = argMax(activations.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// Twenty decode steps: a wrong conv window, recurrent state, rope offset or gate
/// diverges the sequence.
@Test func qwen35GreedyGenerationMatchesTransformers() throws {
    let (_, model, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let expected = golden["greedy_ids"]!.asArray(Int32.self)

    let ids = golden["input_ids"]!.asArray(Int32.self)
    #expect(model.generate(ids, maxNewTokens: expected.count - ids.count) == expected)
}

/// Layer 0 is a DeltaNet layer: block-level submodules driven from the golden input
/// above them, so a failure names the culprit.
@Test func qwen35LinearBlockInternalsMatchTransformers() throws {
    let (_, model, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let embeddings = golden["embeddings"]!

    let ln1 = model.rmsNorm(embeddings, prefix: "model.layers.0.input_layernorm")
    let (mixer, _) = model.deltaNet(ln1, prefix: "model.layers.0.", cache: KVCache())
    let residual = embeddings + mixer
    let ln2 = model.rmsNorm(residual, prefix: "model.layers.0.post_attention_layernorm")
    let mlp = model.mlps[0](ln2)

    #expect(relativeDiff(ln1, golden["b0_ln_1"]!) < 1e-6)
    #expect(relativeDiff(mixer, golden["b0_mixer"]!) < 1e-5)
    #expect(relativeDiff(ln2, golden["b0_ln_2"]!) < 1e-5)
    #expect(relativeDiff(mlp, golden["b0_mlp"]!) < 1e-5)
}

/// The conv stage alone: the golden in_proj output through the causal conv and silu
/// must land on the q/k/v the reference fed its delta rule.
@Test func qwen35ConvStageMatchesTransformers() throws {
    let (config, _, parameters) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let qkv = golden["b0_qkv"]!
    let length = qkv.dim(1)

    let weight = parameters["model.layers.0.linear_attn.conv1d.weight"]!
    let window = zeros([1, config.linearConvKernelDim - 1, config.convDim], dtype: qkv.dtype)
    let padded = concatenated([window, qkv], axis: 1)
    var conv = padded[0..., ..<length, 0...] * weight[0..., 0]
    for j in 1..<config.linearConvKernelDim {
        conv = conv + padded[0..., j..<(j + length), 0...] * weight[0..., j]
    }
    let convOut = conv * sigmoid(conv)

    let keyDim = config.keyDim
    let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
    #expect(
        relativeDiff(
            parts[0].reshaped(1, length, config.linearNumKeyHeads, config.linearKeyHeadDim),
            golden["b0_rule_q"]!) < 1e-5)
    #expect(
        relativeDiff(
            parts[1].reshaped(1, length, config.linearNumKeyHeads, config.linearKeyHeadDim),
            golden["b0_rule_k"]!) < 1e-5)
    #expect(
        relativeDiff(
            parts[2].reshaped(1, length, config.linearNumValueHeads, config.linearValueHeadDim),
            golden["b0_rule_v"]!) < 1e-5)
}

/// The Metal kernel against the reference's chunked delta rule, from the same inputs:
/// same math, different accumulation order, so float32 disagreement stays tiny. The
/// final recurrent state must agree too — the fixture stores it as [B, Hv, Dk, Dv].
@Test func qwen35GatedDeltaKernelMatchesChunkedRule() throws {
    let (config, _, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")

    func l2(_ x: MLXArray, scale: Float = 1) -> MLXArray {
        x * rsqrt(x.square().sum(axis: -1, keepDims: true) + 1e-6) * scale
    }

    let state = zeros(
        [1, config.linearNumValueHeads, config.linearValueHeadDim, config.linearKeyHeadDim],
        dtype: .float32)
    let (y, newState) = gatedDelta(
        q: l2(golden["b0_rule_q"]!, scale: 1 / sqrt(Float(config.linearKeyHeadDim))),
        k: l2(golden["b0_rule_k"]!),
        v: golden["b0_rule_v"]!,
        // The fixture captures g in log space — the reference rule exponentiates it
        // inside; the kernel takes the decay already exponentiated.
        g: exp(golden["b0_rule_g"]!),
        beta: golden["b0_rule_beta"]!,
        state: state)

    #expect(relativeDiff(y, golden["b0_rule_out"]!) < 1e-5)
    #expect(relativeDiff(newState.transposed(0, 1, 3, 2), golden["b0_rule_state"]!) < 1e-5)
}

/// The 0.8B has as many key heads as value heads, so its fixtures cannot see the
/// kernel's broadcast map (hk = hv / (Hv/Hk), interleaved like repeat_interleave).
/// Synthetic ratio-3 inputs against a sequential ops reference cover it.
@Test func qwen35GatedDeltaKernelBroadcastsKeyHeads() throws {
    MLXRandom.seed(11)
    let (t, keyHeads, valueHeads, keyDim, valueDim) = (4, 2, 6, 32, 16)
    let q = MLXRandom.normal([1, t, keyHeads, keyDim])
    let k = MLXRandom.normal([1, t, keyHeads, keyDim])
    let v = MLXRandom.normal([1, t, valueHeads, valueDim])
    let g = sigmoid(MLXRandom.normal([1, t, valueHeads]))
    let beta = sigmoid(MLXRandom.normal([1, t, valueHeads]))
    let initial = zeros([1, valueHeads, valueDim, keyDim], dtype: .float32)

    let (y, state) = gatedDelta(q: q, k: k, v: v, g: g, beta: beta, state: initial)

    // Each key head serves Hv/Hk consecutive value heads.
    let ratio = valueHeads / keyHeads
    let qFull = repeated(q, count: ratio, axis: 2)
    let kFull = repeated(k, count: ratio, axis: 2)
    var reference = initial
    var rows: [MLXArray] = []
    for step in 0..<t {
        reference = reference * expandedDimensions(g[0..., step], axes: [-1, -2])
        let key = expandedDimensions(kFull[0..., step], axis: -2)
        let memory = (reference * key).sum(axis: -1)
        let delta = (v[0..., step] - memory) * expandedDimensions(beta[0..., step], axis: -1)
        reference = reference + key * expandedDimensions(delta, axis: -1)
        rows.append((reference * expandedDimensions(qFull[0..., step], axis: -2)).sum(axis: -1))
    }

    #expect(relativeDiff(y, stacked(rows, axis: 1)) < 1e-6)
    #expect(relativeDiff(state, reference) < 1e-6)
}

/// Layer 3 is the first full-attention layer: q‖gate split, q/k-norm, partial rope,
/// sigmoid gate before o_proj.
@Test func qwen35AttentionBlockInternalsMatchTransformers() throws {
    let (_, model, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let input = golden["block_2"]!

    let ln1 = model.rmsNorm(input, prefix: "model.layers.3.input_layernorm")
    let (attention, _) = model.attention(
        ln1, prefix: "model.layers.3.", positions: .offset(0), cache: KVCache())
    let residual = input + attention
    let ln2 = model.rmsNorm(residual, prefix: "model.layers.3.post_attention_layernorm")
    let mlp = model.mlps[3](ln2)

    #expect(relativeDiff(ln1, golden["b3_ln_1"]!) < 1e-5)
    #expect(relativeDiff(attention, golden["b3_attn"]!) < 1e-5)
    #expect(relativeDiff(ln2, golden["b3_ln_2"]!) < 1e-5)
    #expect(relativeDiff(mlp, golden["b3_mlp"]!) < 1e-5)
}

/// Only the first 64 of 256 dims rotate, at theta 1e7. HF captures the normed
/// projections pre-transpose: (1, len, heads, headDim).
@Test func qwen35PartialRopeMatchesTransformers() throws {
    let (_, model, _) = try loadModel()
    let golden = try fixture("qwen3_5_forward.safetensors")

    #expect(
        relativeDiff(
            model.rope(golden["b3_q_norm"]!.transposed(0, 2, 1, 3)), golden["b3_q_rope"]!) < 1e-6)
    #expect(
        relativeDiff(
            model.rope(golden["b3_k_norm"]!.transposed(0, 2, 1, 3)), golden["b3_k_rope"]!) < 1e-6)
}

/// Prefill once then one token per step: the conv window, the recurrent state and the
/// KV cache must agree with recomputing the whole prefix every step.
@Test func qwen35CachedGenerationMatchesRecomputation() throws {
    let (_, model, _) = try loadModel()
    let ids = try fixture("qwen3_5_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

/// Argmax can hide a stale window or state (the degenerate greedy loop barely moves);
/// the logits themselves cannot. Every stepwise row must match the prefill's.
@Test func qwen35StepwiseLogitsMatchPrefill() throws {
    let (_, model, _) = try loadModel()
    let ids = try fixture("qwen3_5_forward.safetensors")["greedy_ids"]!.asArray(Int32.self)

    let full = model.logits(model.embed(ids), cache: model.makeCache()).logits

    var cache = model.makeCache()
    for (position, id) in ids.enumerated() {
        let (logits, grown) = model.logits(model.embed([id]), cache: cache)
        cache = grown
        #expect(relativeDiff(logits[0, -1], full[0, position]) < 1e-5, "position \(position)")
    }
}

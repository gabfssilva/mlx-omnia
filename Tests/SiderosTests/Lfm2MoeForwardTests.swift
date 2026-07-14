import Foundation
import MLX
import Testing

@testable import Sideros

private func loadModel() throws -> (Lfm2MoeConfig, Lfm2Moe) {
    let directory = try lfm2MoeDirectory()
    let config = try Lfm2MoeConfig(directory: directory)
    let parameters = try loadLfm2MoeParameters(
        directory: directory, config: config, precision: .float32)
    return (config, Lfm2Moe(config: config, parameters: parameters))
}

/// The reference is itself only a float32 approximation; `noise.*` is how far it drifts
/// from float64, measured when the fixture was generated. Three times it and no more.
@Test func lfm2MoeForwardMatchesTransformers() throws {
    let (config, model) = try loadModel()
    let golden = try fixture("lfm2moe_forward.safetensors")
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
@Test func lfm2MoeGreedyPredictionsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("lfm2moe_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    let ours = argMax(activations.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// Twenty decode steps: a wrong conv window, rope offset or expert pick diverges the
/// sequence.
@Test func lfm2MoeGreedyGenerationMatchesTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("lfm2moe_forward.safetensors")
    let expected = golden["greedy_ids"]!.asArray(Int32.self)

    let ids = golden["input_ids"]!.asArray(Int32.self)
    #expect(model.generate(ids, maxNewTokens: expected.count - ids.count) == expected)
}

/// Layer 0 is a short-conv layer with the dense MLP: each submodule driven from the
/// golden input above it, so a failure names the culprit.
@Test func lfm2MoeConvBlockInternalsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("lfm2moe_forward.safetensors")
    let embeddings = golden["embeddings"]!

    let ln1 = model.rmsNorm(embeddings, prefix: "model.layers.0.operator_norm")
    let (conv, _) = model.shortConv(ln1, layer: 0, cache: KVCache())
    let residual = embeddings + conv
    let ln2 = model.rmsNorm(residual, prefix: "model.layers.0.ffn_norm")
    let mlp = model.mlps[0](ln2)

    #expect(relativeDiff(ln1, golden["b0_ln_1"]!) < 1e-6)
    #expect(relativeDiff(conv, golden["b0_conv"]!) < 1e-5)
    #expect(relativeDiff(ln2, golden["b0_ln_2"]!) < 1e-5)
    #expect(relativeDiff(mlp, golden["b0_mlp"]!) < 1e-5)
}

/// Layer 2 is the first attention layer and the first routed MLP.
@Test func lfm2MoeAttentionBlockInternalsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("lfm2moe_forward.safetensors")
    let input = golden["block_1"]!

    let ln1 = model.rmsNorm(input, prefix: "model.layers.2.operator_norm")
    let (attention, _) = model.attention(ln1, prefix: "model.layers.2.", cache: KVCache())
    let residual = input + attention
    let ln2 = model.rmsNorm(residual, prefix: "model.layers.2.ffn_norm")
    let moe = model.mlps[2](ln2)

    #expect(relativeDiff(ln1, golden["b2_ln_1"]!) < 1e-5)
    #expect(relativeDiff(attention, golden["b2_attn"]!) < 1e-5)
    #expect(relativeDiff(ln2, golden["b2_ln_2"]!) < 1e-5)
    #expect(relativeDiff(moe, golden["b2_moe"]!) < 1e-5)
}

/// The q/k norms sit between the projections and the rotation; then rope with theta
/// 5e6. HF captures them pre-transpose: (1, len, heads, headDim).
@Test func lfm2MoeQKNormAndRopeMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("lfm2moe_forward.safetensors")

    #expect(
        relativeDiff(
            model.rope(golden["b2_q_norm"]!.transposed(0, 2, 1, 3)), golden["b2_q_rope"]!) < 1e-6)
    #expect(
        relativeDiff(
            model.rope(golden["b2_k_norm"]!.transposed(0, 2, 1, 3)), golden["b2_k_rope"]!) < 1e-6)
}

/// Prefill once then one token per step: the conv window and the KV cache must agree
/// with recomputing the whole prefix every step.
@Test func lfm2MoeCachedGenerationMatchesRecomputation() throws {
    let (_, model) = try loadModel()
    let ids = try fixture("lfm2moe_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

/// The custom gemv pair (MoeGemv.swift) against gatherMM over the same routed MLP,
/// in float32 where reduction order is the only daylight between them.
@Test func lfm2MoeCustomGemvMatchesGatherMM() throws {
    MLXRandom.seed(7)
    let (experts, k, hidden, inner) = (8, 4, 64, 32)
    let w13 = MLXRandom.normal([experts, 2 * inner, hidden])
    let w2 = MLXRandom.normal([experts, hidden, inner])
    let x = MLXRandom.normal([hidden])
    let indices = MLXArray([Int32(1), 3, 4, 6])
    let routing = MLXArray([Float(0.4), 0.3, 0.2, 0.1])

    let gateUp = moeGateUp(x, weights: w13, indices: indices)
    let ours = moeDown(gateUp, weights: w2, indices: indices, routing: routing)
        .sum(axis: 0)

    let tokens = expandedDimensions(x.reshaped(1, 1, hidden), axes: [-2, -3])
    let gu = gatherMM(
        tokens, w13.swappedAxes(-2, -1), rhsIndices: indices.reshaped(1, 1, k))
    let g = gu[.ellipsis, ..<inner]
    let u = gu[.ellipsis, inner...]
    let routed = gatherMM(
        g * sigmoid(g) * u, w2.swappedAxes(-2, -1), rhsIndices: indices.reshaped(1, 1, k))
    let expected = (squeezed(routed, axis: -2)
        * expandedDimensions(routing.reshaped(1, 1, k), axis: -1)).sum(axis: -2)

    #expect(relativeDiff(ours, expected.reshaped(hidden)) < 1e-6)
}

/// Argmax can hide a stale conv window (the degenerate greedy loop barely moves);
/// the logits themselves cannot. Every stepwise row must match the prefill's. The 25
/// greedy ids also push the prefill past the 64-index threshold where the MoE routes
/// through the expert-sorted gather, pinning it to the decode path's numbers.
@Test func lfm2MoeStepwiseLogitsMatchPrefill() throws {
    let (_, model) = try loadModel()
    let ids = try fixture("lfm2moe_forward.safetensors")["greedy_ids"]!.asArray(Int32.self)

    let full = model.logits(model.embed(ids), cache: model.makeCache()).logits

    var cache = model.makeCache()
    for (position, id) in ids.enumerated() {
        let (logits, grown) = model.logits(model.embed([id]), cache: cache)
        cache = grown
        #expect(relativeDiff(logits[0, -1], full[0, position]) < 1e-5, "position \(position)")
    }
}

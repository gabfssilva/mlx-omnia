import Foundation
import MLX
import Testing

@testable import Sideros

private func loadModel() throws -> (Qwen3Config, Qwen3) {
    let directory = try qwen3Directory()
    let config = try Qwen3Config(directory: directory)
    let parameters = try loadQwen3Parameters(directory: directory, config: config, precision: .float32)
    return (config, Qwen3(config: config, parameters: parameters))
}

/// The reference is itself only a float32 approximation; `noise.*` is how far it drifts
/// from float64, measured when the fixture was generated. Three times it and no more.
@Test func qwen3ForwardMatchesTransformers() throws {
    let (config, model) = try loadModel()
    let golden = try fixture("qwen3_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    func expectWithinNoiseFloor(_ ours: MLXArray, _ name: String) {
        let floor = golden["noise.\(name)"]!.item(Float.self)
        #expect(relativeDiff(ours, golden[name]!) < 3 * floor)
    }

    #expect(relativeDiff(activations.embeddings, golden["embeddings"]!) == 0)
    for layer in 0..<config.numHiddenLayers {
        expectWithinNoiseFloor(activations.blocks[layer], "block_\(layer)")
    }
    expectWithinNoiseFloor(activations.norm, "norm")
    expectWithinNoiseFloor(activations.logits, "logits")
}

/// Numeric closeness is not the goal; identical predictions are.
@Test func qwen3GreedyPredictionsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("qwen3_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    let ours = argMax(activations.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// Twenty decode steps: a single wrong rotation or norm placement diverges the sequence.
@Test func qwen3GreedyGenerationMatchesTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("qwen3_forward.safetensors")
    let expected = golden["greedy_ids"]!.asArray(Int32.self)

    let ids = golden["input_ids"]!.asArray(Int32.self)
    #expect(model.generate(ids, maxNewTokens: expected.count - ids.count) == expected)
}

/// Bisects a block-level divergence: each submodule is driven from the golden input
/// above it, so a failure names the culprit instead of the layer it surfaced in.
@Test func qwen3BlockZeroInternalsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("qwen3_forward.safetensors")
    let embeddings = golden["embeddings"]!

    let ln1 = model.rmsNorm(embeddings, prefix: "model.layers.0.input_layernorm")
    let (attention, _) = model.attention(ln1, prefix: "model.layers.0.", cache: KVCache())
    let residual = embeddings + attention
    let ln2 = model.rmsNorm(residual, prefix: "model.layers.0.post_attention_layernorm")
    let mlp = model.mlps[0](ln2)

    #expect(relativeDiff(ln1, golden["b0_ln_1"]!) < 1e-6)
    #expect(relativeDiff(attention, golden["b0_attn"]!) < 1e-5)
    #expect(relativeDiff(ln2, golden["b0_ln_2"]!) < 3 * golden["noise.b0_ln_2"]!.item(Float.self))
    #expect(relativeDiff(mlp, golden["b0_mlp"]!) < 1e-5)
}

/// The q/k norms sit between the projections and the rotation, one headDim vector shared
/// by every head; then rope. Driven from the golden normed projections, before attention
/// can average the error away. HF captures them pre-transpose: (1, len, heads, headDim).
@Test func qwen3QKNormAndRopeMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("qwen3_forward.safetensors")

    #expect(
        relativeDiff(
            model.rope(golden["b0_q_norm"]!.transposed(0, 2, 1, 3)), golden["b0_q_rope"]!) < 1e-6)
    #expect(
        relativeDiff(
            model.rope(golden["b0_k_norm"]!.transposed(0, 2, 1, 3)), golden["b0_k_rope"]!) < 1e-6)
}

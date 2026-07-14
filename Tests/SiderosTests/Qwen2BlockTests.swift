import Foundation
import MLX
import Testing

@testable import Sideros

/// Bisects a block-level divergence: each submodule is driven from the golden input
/// above it, so a failure names the culprit instead of the layer it surfaced in.
@Test func qwen2BlockZeroInternalsMatchTransformers() throws {
    let directory = try qwen2Directory()
    let config = try Qwen2Config(directory: directory)
    let model = Qwen2(
        config: config, parameters: try loadQwen2Parameters(directory: directory, config: config, precision: .float32))

    let golden = try fixture("qwen2_forward.safetensors")
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

/// The half-split and adjacent-pair rotations agree on shape and on norm, so only the
/// values distinguish them. Driven from the golden projections, before attention can
/// average the error away.
@Test func qwen2RopeMatchesTransformers() throws {
    let directory = try qwen2Directory()
    let config = try Qwen2Config(directory: directory)
    let model = Qwen2(
        config: config, parameters: try loadQwen2Parameters(directory: directory, config: config, precision: .float32))

    let golden = try fixture("qwen2_forward.safetensors")
    let length = golden["embeddings"]!.dim(1)

    func heads(_ name: String, _ count: Int) -> MLXArray {
        golden[name]!.reshaped(1, length, count, config.headDim).transposed(0, 2, 1, 3)
    }

    #expect(
        relativeDiff(
            model.rope(heads("b0_q_proj", config.numAttentionHeads)), golden["b0_q_rope"]!) < 1e-6)
    #expect(
        relativeDiff(
            model.rope(heads("b0_k_proj", config.numKeyValueHeads)), golden["b0_k_rope"]!) < 1e-6)
}

import Foundation
import MLX
import Testing

@testable import Sideros

/// Bisects a block-level divergence: each submodule is driven from the golden input
/// above it, so a failure names the culprit instead of the layer it surfaced in.
@Test func blockZeroInternalsMatchTransformers() throws {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let model = GPT2(config: config, parameters: try loadGPT2Parameters(directory: directory, config: config, precision: .float32))

    let golden = try fixture("gpt2_forward.safetensors")
    let embeddings = golden["embeddings"]!

    let ln1 = model.layerNorm(embeddings, prefix: "h.0.ln_1")
    let length = embeddings.dim(1)
    let (attention, _) = model.attention(
        ln1, mask: GPT2.causalMask(queries: length, keys: length), prefix: "h.0.",
        cache: KVCache())
    let residual = embeddings + attention
    let ln2 = model.layerNorm(residual, prefix: "h.0.ln_2")
    let mlp = model.mlp(ln2, prefix: "h.0.")

    #expect(relativeDiff(ln1, golden["b0_ln_1"]!) < 1e-7)
    #expect(relativeDiff(attention, golden["b0_attn"]!) < 1e-5)
    #expect(relativeDiff(ln2, golden["b0_ln_2"]!) < 1e-6)
    #expect(relativeDiff(mlp, golden["b0_mlp"]!) < 1e-5)
}

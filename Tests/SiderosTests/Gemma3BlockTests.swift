import Foundation
import MLX
import Testing

@testable import Sideros

/// A query may attend to itself and to the `window - 1` keys before it. One position
/// either way is invisible in the forward pass until the sequence outgrows the window.
@Test func gemma3SlidingMaskKeepsExactlyTheLastWindowKeys() {
    let mask = Gemma3.causalMask(queries: 8, keys: 8, window: 3).asArray(Bool.self)
    let row5 = Array(mask[(5 * 8)..<(6 * 8)])

    #expect(row5 == [false, false, false, true, true, true, false, false])
    #expect(Gemma3.causalMask(queries: 8, keys: 8, window: nil).asArray(Bool.self)[(5 * 8)..<(6 * 8)].filter { $0 }.count == 6)
}

/// Bisects a block-level divergence: each submodule is driven from the golden input
/// above it, so a failure names the culprit instead of the layer it surfaced in.
@Test func gemma3BlockZeroInternalsMatchTransformers() throws {
    let directory = try gemma3Directory()
    let config = try Gemma3Config(directory: directory)
    let parameters = try loadGemma3Parameters(directory: directory, config: config, precision: .float32)
    let model = Gemma3(config: config, parameters: parameters)

    let golden = try fixture("gemma3_forward.safetensors")
    let embeddings = golden["embeddings"]!
    let length = embeddings.dim(1)
    let prefix = "model.layers.0."

    // Layer 0 slides, so it rotates with the local base rather than ropeTheta.
    #expect(config.layerTypes[0] == .sliding)
    let context = Gemma3.Context(
        full: Gemma3.causalMask(queries: length, keys: length, window: nil),
        sliding: Gemma3.causalMask(queries: length, keys: length, window: config.slidingWindow),
        cache: model.makeCache(), offset: 0)

    let (ln1, _) = model.norm(prefix + "input_layernorm")(embeddings, context)
    let (attention, _) = model.attention(layer: 0, prefix: prefix, type: .sliding)(ln1, context)
    let (postAttention, _) = model.norm(prefix + "post_attention_layernorm")(attention, context)
    let hidden = embeddings + postAttention
    let (preFeedForward, _) = model.norm(prefix + "pre_feedforward_layernorm")(hidden, context)
    let mlp = model.mlp(prefix)(preFeedForward)

    #expect(relativeDiff(ln1, golden["b0_input_layernorm"]!) < 1e-6)
    #expect(relativeDiff(attention, golden["b0_self_attn"]!) < 1e-5)
    #expect(relativeDiff(postAttention, golden["b0_post_attention_layernorm"]!) < 1e-6)
    #expect(relativeDiff(preFeedForward, golden["b0_pre_feedforward_layernorm"]!) < 1e-6)
    #expect(relativeDiff(mlp, golden["b0_mlp"]!) < 1e-5)
    #expect(relativeDiff(model.norm(prefix + "post_feedforward_layernorm")(mlp, context).0,
                         golden["b0_post_feedforward_layernorm"]!) < 1e-5)

    // q_norm and k_norm are the only weights that sit between a projection and the kernel.
    let p = prefix + "self_attn."
    func projected(_ name: String, heads: Int) -> MLXArray {
        matmul(ln1, parameters[p + name + ".weight"]!.T)
            .reshaped(1, length, heads, config.headDim).transposed(0, 2, 1, 3)
    }
    #expect(relativeDiff(
        model.norm(p + "q_norm")(projected("q_proj", heads: config.numAttentionHeads), context).0,
        golden["b0_q_norm"]!) < 1e-6)
    #expect(relativeDiff(
        model.norm(p + "k_norm")(projected("k_proj", heads: config.numKeyValueHeads), context).0,
        golden["b0_k_norm"]!) < 1e-6)
}

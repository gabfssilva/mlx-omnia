import Foundation
import MLX
import Testing

@testable import Sideros

private func loadModel() throws -> (Gemma3Config, [String: MLXArray], Gemma3) {
    let directory = try gemma3Directory()
    let config = try Gemma3Config(directory: directory)
    let parameters = try loadGemma3Parameters(directory: directory, config: config, precision: .float32)
    return (config, parameters, Gemma3(config: config, parameters: parameters))
}

/// `noise.*` is how far the float32 reference drifts from float64, measured per tensor
/// when the fixture was generated; we allow three times it and no more.
@Test func gemma3ForwardMatchesTransformers() throws {
    let (config, _, model) = try loadModel()
    let golden = try fixture("gemma3_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    func expectWithinNoiseFloor(_ ours: MLXArray, _ name: String) {
        #expect(relativeDiff(ours, golden[name]!) < 3 * golden["noise.\(name)"]!.item(Float.self))
    }

    expectWithinNoiseFloor(activations.embeddings, "embeddings")
    for layer in 0..<config.numHiddenLayers {
        expectWithinNoiseFloor(activations.blocks[layer], "block_\(layer)")
    }
    expectWithinNoiseFloor(activations.norm, "norm")
    expectWithinNoiseFloor(activations.logits, "logits")
}

/// Below 512 tokens the sliding mask and the full mask are the same matrix. Only here do
/// layers 0 and 5 take different paths.
@Test func gemma3ForwardMatchesTransformersBeyondTheSlidingWindow() throws {
    let (_, parameters, model) = try loadModel()
    let golden = try fixture("gemma3_forward.safetensors")

    let ids = golden["long_input_ids"]!.asArray(Int32.self)
    #expect(ids.count > 512)
    let activations = model(ids)

    func expectWithinNoiseFloor(_ ours: MLXArray, _ name: String) {
        #expect(relativeDiff(ours, golden[name]!) < 3 * golden["noise.\(name)"]!.item(Float.self))
    }

    expectWithinNoiseFloor(activations.blocks[0], "long_block_0")  // sliding
    expectWithinNoiseFloor(activations.blocks[5], "long_block_5")  // full

    // The full logits would be a 630 MB tensor; the last row is the one that generates.
    let last = activations.norm[0..., (ids.count - 1)...]
    let logits = matmul(last, parameters["model.embed_tokens.weight"]!.T)
    expectWithinNoiseFloor(logits, "long_logits_last")
}

/// Numeric closeness is not the goal; identical predictions are.
@Test func gemma3GreedyPredictionsMatchTransformers() throws {
    let (_, _, model) = try loadModel()
    let golden = try fixture("gemma3_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    let ours = argMax(activations.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// Twenty decode steps: a single wrong rope base or mask entry diverges the sequence.
@Test func gemma3GreedyGenerationMatchesTransformers() throws {
    let (_, _, model) = try loadModel()
    let golden = try fixture("gemma3_forward.safetensors")
    let expected = golden["greedy_ids"]!.asArray(Int32.self)

    let ids = golden["input_ids"]!.asArray(Int32.self)
    #expect(model.generate(ids, maxNewTokens: expected.count - ids.count) == expected)
}

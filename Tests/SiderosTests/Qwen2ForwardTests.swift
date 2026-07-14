import Foundation
import MLX
import Testing

@testable import Sideros

private func loadModel() throws -> (Qwen2Config, Qwen2) {
    let directory = try qwen2Directory()
    let config = try Qwen2Config(directory: directory)
    let parameters = try loadQwen2Parameters(directory: directory, config: config, precision: .float32)
    return (config, Qwen2(config: config, parameters: parameters))
}

/// The reference is itself only a float32 approximation, and how far it drifts varies by
/// three orders of magnitude across blocks. `noise.*` is that drift, measured against
/// float64 when the fixture was generated; we allow three times it and no more.
@Test func qwen2ForwardMatchesTransformers() throws {
    let (config, model) = try loadModel()
    let golden = try fixture("qwen2_forward.safetensors")
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

/// mlx-lm runs the same kernels on the same GPU, so a gap here is ours alone. `gap.*` is
/// how far mlx-lm lands from torch; we must be at least that close to mlx-lm.
@Test func qwen2ForwardMatchesMLXLM() throws {
    let (config, model) = try loadModel()
    let reference = try fixture("qwen2_mlxlm.safetensors")
    let activations = model(reference["input_ids"]!.asArray(Int32.self))

    func expectCloserThanMLXLMIsToTorch(_ ours: MLXArray, _ name: String) {
        #expect(relativeDiff(ours, reference[name]!) < reference["gap.\(name)"]!.item(Float.self))
    }

    #expect(relativeDiff(activations.embeddings, reference["embeddings"]!) == 0)
    for layer in 0..<config.numHiddenLayers {
        expectCloserThanMLXLMIsToTorch(activations.blocks[layer], "block_\(layer)")
    }
    expectCloserThanMLXLMIsToTorch(activations.norm, "norm")
    expectCloserThanMLXLMIsToTorch(activations.logits, "logits")
}

/// Numeric closeness is not the goal; identical predictions are.
@Test func qwen2GreedyPredictionsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("qwen2_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    let ours = argMax(activations.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// Twenty decode steps: a single wrong rotation or mask entry diverges the sequence.
@Test func qwen2GreedyGenerationMatchesTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("qwen2_forward.safetensors")
    let expected = golden["greedy_ids"]!.asArray(Int32.self)

    let ids = golden["input_ids"]!.asArray(Int32.self)
    #expect(model.generate(ids, maxNewTokens: expected.count - ids.count) == expected)
}

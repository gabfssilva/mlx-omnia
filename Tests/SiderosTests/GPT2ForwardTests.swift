import Foundation
import MLX
import Testing

@testable import Sideros

private func loadModel() throws -> (GPT2Config, GPT2) {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let parameters = try loadGPT2Parameters(directory: directory, config: config, precision: .float32)
    return (config, GPT2(config: config, parameters: parameters))
}

/// Torch and MLX disagree at roughly one float32 ULP; the residual stream's scale reaches
/// thousands by block 5, so only the relative deviation is meaningful.
@Test func forwardMatchesTransformers() throws {
    let (config, model) = try loadModel()
    let golden = try fixture("gpt2_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    #expect(relativeDiff(activations.embeddings, golden["embeddings"]!) == 0)
    for layer in 0..<config.nLayer {
        #expect(relativeDiff(activations.blocks[layer], golden["block_\(layer)"]!) < 1e-5)
    }
    #expect(relativeDiff(activations.lnF, golden["ln_f"]!) < 1e-5)
    #expect(relativeDiff(activations.logits, golden["logits"]!) < 1e-5)
}

/// mlx-lm runs the same kernels on the same GPU, so any gap here is ours alone.
@Test func forwardMatchesMLXLM() throws {
    let (config, model) = try loadModel()
    let reference = try fixture("gpt2_mlxlm.safetensors")
    let activations = model(reference["input_ids"]!.asArray(Int32.self))

    #expect(relativeDiff(activations.embeddings, reference["embeddings"]!) == 0)
    for layer in 0..<config.nLayer {
        #expect(relativeDiff(activations.blocks[layer], reference["block_\(layer)"]!) < 1e-6)
    }
    #expect(relativeDiff(activations.lnF, reference["ln_f"]!) < 1e-6)
    #expect(relativeDiff(activations.logits, reference["logits"]!) < 1e-6)
}

/// Numeric closeness is not the goal; identical predictions are.
@Test func greedyPredictionsMatchTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("gpt2_forward.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))

    let ours = argMax(activations.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// Twenty decode steps: a single wrong position or mask entry diverges the sequence.
@Test func greedyGenerationMatchesTransformers() throws {
    let (_, model) = try loadModel()
    let golden = try fixture("gpt2_forward.safetensors")
    let expected = golden["greedy_ids"]!.asArray(Int32.self)

    let ids = golden["input_ids"]!.asArray(Int32.self)
    #expect(model.generate(ids, maxNewTokens: expected.count - ids.count) == expected)
}

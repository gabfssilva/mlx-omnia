import Foundation
import MLX
import Testing

@testable import Sideros

func qwen2Directory() throws -> URL { try hubSnapshot("Qwen--Qwen2.5-0.5B") }

@Test func qwen2ConfigMatchesCheckpoint() throws {
    let config = try Qwen2Config(directory: qwen2Directory())

    #expect(config.numHiddenLayers == 24)
    #expect(config.numAttentionHeads == 14)
    #expect(config.numKeyValueHeads == 2)
    #expect(config.hiddenSize == 896)
    #expect(config.intermediateSize == 4864)
    #expect(config.vocabSize == 151936)
    #expect(config.headDim == 64)
    #expect(config.rmsNormEps == 1e-6)
    // Not the usual 10000: reading a default instead of the config would only show up
    // on long sequences.
    #expect(config.ropeTheta == 1_000_000)
}

@Test func qwen2ParametersMatchExpectedSetAndShapes() throws {
    let directory = try qwen2Directory()
    let config = try Qwen2Config(directory: directory)
    let parameters = try loadQwen2Parameters(directory: directory, config: config, precision: .float32)

    #expect(parameters.count == 2 + 12 * config.numHiddenLayers)
    #expect(parameters["lm_head.weight"] == nil)
    #expect(parameters["model.layers.0.self_attn.o_proj.bias"] == nil)
    #expect(parameters.values.allSatisfy { $0.dtype == .float32 })

    // 2 kv heads x 64 = 128, against 896 for the 14 query heads.
    #expect(parameters["model.layers.0.self_attn.k_proj.weight"]!.shape == [128, 896])
    #expect(parameters["model.layers.0.self_attn.q_proj.weight"]!.shape == [896, 896])
    #expect(parameters["model.layers.0.mlp.down_proj.weight"]!.shape == [896, 4864])
}

import Foundation
import MLX
import Testing

@testable import Sideros

func gemma3Directory() throws -> URL { try hubSnapshot("google--gemma-3-270m") }

@Test func gemma3ConfigMatchesCheckpoint() throws {
    let config = try Gemma3Config(directory: gemma3Directory())

    #expect(config.numHiddenLayers == 18)
    #expect(config.numAttentionHeads == 4)
    #expect(config.numKeyValueHeads == 1)  // multi-query, not grouped
    #expect(config.hiddenSize == 640)
    #expect(config.vocabSize == 262144)
    #expect(config.slidingWindow == 512)

    // 4 * 256 = 1024, which is not the hidden size. Deriving headDim would be wrong.
    #expect(config.headDim == 256)
    #expect(config.numAttentionHeads * config.headDim != config.hiddenSize)

    #expect(config.ropeTheta == 1_000_000)
    #expect(config.ropeLocalBaseFreq == 10_000)
    #expect(config.queryPreAttnScalar == 256)

    // Every sixth layer attends to the whole prefix; the rest slide.
    #expect(config.layerTypes.count == 18)
    #expect(config.layerTypes.enumerated().allSatisfy { ($0.element == .full) == (($0.offset + 1) % 6 == 0) })
}

@Test func gemma3ParametersMatchExpectedSetAndShapes() throws {
    let directory = try gemma3Directory()
    let config = try Gemma3Config(directory: directory)
    let parameters = try loadGemma3Parameters(directory: directory, config: config, precision: .float32)

    #expect(parameters.count == 2 + 13 * config.numHiddenLayers)
    #expect(parameters["lm_head.weight"] == nil)
    #expect(parameters.values.allSatisfy { $0.dtype == .float32 })

    #expect(parameters["model.layers.0.self_attn.q_proj.weight"]!.shape == [1024, 640])
    #expect(parameters["model.layers.0.self_attn.k_proj.weight"]!.shape == [256, 640])
    #expect(parameters["model.layers.0.self_attn.o_proj.weight"]!.shape == [640, 1024])
    #expect(parameters["model.layers.0.self_attn.q_norm.weight"]!.shape == [256])
    #expect(parameters["model.layers.0.post_feedforward_layernorm.weight"]!.shape == [640])
}

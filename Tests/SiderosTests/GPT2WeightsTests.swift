import Foundation
import MLX
import Testing

@testable import Sideros

func gpt2Directory() throws -> URL { try hubSnapshot("openai-community--gpt2") }

@Test func configMatchesCheckpoint() throws {
    let config = try GPT2Config(directory: gpt2Directory())

    #expect(config.nLayer == 12)
    #expect(config.nHead == 12)
    #expect(config.nEmbd == 768)
    #expect(config.nPositions == 1024)
    #expect(config.vocabSize == 50257)
    #expect(config.layerNormEpsilon == 1e-5)
}

@Test func parametersMatchExpectedSetAndShapes() throws {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let parameters = try loadGPT2Parameters(directory: directory, config: config, precision: .float32)

    #expect(parameters.count == 4 + 12 * config.nLayer)
    #expect(parameters["lm_head.weight"] == nil)
    #expect(parameters.keys.allSatisfy { !$0.hasSuffix(".attn.bias") })

    #expect(parameters["h.0.attn.c_attn.weight"]!.shape == [768, 2304])
    #expect(parameters["h.0.mlp.c_proj.weight"]!.shape == [3072, 768])
}

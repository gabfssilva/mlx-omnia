import Foundation
import MLX
import Testing

@testable import Sideros

func lfm2MoeDirectory() throws -> URL { try hubSnapshot("LiquidAI--LFM2.5-8B-A1B") }

@Test func lfm2MoeConfigMatchesCheckpoint() throws {
    let config = try Lfm2MoeConfig(directory: lfm2MoeDirectory())

    #expect(config.numHiddenLayers == 24)
    #expect(config.numAttentionHeads == 32)
    #expect(config.numKeyValueHeads == 8)
    #expect(config.headDim == 64)
    #expect(config.hiddenSize == 2048)
    #expect(config.intermediateSize == 7168)
    #expect(config.moeIntermediateSize == 1792)
    #expect(config.numExperts == 32)
    #expect(config.numExpertsPerTok == 4)
    #expect(config.numDenseLayers == 2)
    #expect(config.normTopkProb)
    #expect(config.useExpertBias)
    #expect(config.routedScalingFactor == 1)
    #expect(config.convLCache == 3)
    #expect(config.layerTypes.count == 24)
    #expect(config.layerTypes.filter { $0 == "full_attention" }.count == 6)
    #expect(config.ropeTheta == 5_000_000)
    #expect(config.vocabSize == 128_000)
    #expect(config.tieWordEmbeddings)
}

@Test func lfm2MoeParametersStackExpertsPerLayer() throws {
    let directory = try lfm2MoeDirectory()
    let config = try Lfm2MoeConfig(directory: directory)
    let parameters = try loadLfm2MoeParameters(directory: directory, config: config, precision: nil)

    // embed + final norm, 2 norms every layer, 3 conv tensors x 18, 6 attention x 6,
    // 3 dense-MLP tensors x 2, and gate + bias + 3 stacked expert tensors x 22.
    #expect(parameters.count == 2 + 24 * 2 + 18 * 3 + 6 * 6 + 2 * 3 + 22 * 5)
    #expect(
        parameters["model.layers.2.feed_forward.experts.w1.weight"]!.shape == [32, 1792, 2048])
    #expect(
        parameters["model.layers.2.feed_forward.experts.w2.weight"]!.shape == [32, 2048, 1792])
    #expect(parameters["model.layers.2.feed_forward.experts.0.w1.weight"] == nil)

    // Expert 7 of the stack must be expert 7 of the checkpoint, bit for bit.
    let raw = try loadCheckpoint(directory: directory)
    #expect(
        allClose(
            parameters["model.layers.2.feed_forward.experts.w1.weight"]![7],
            raw["model.layers.2.feed_forward.experts.7.w1.weight"]!, rtol: 0, atol: 0
        ).item(Bool.self))

    // The router bias ships float32 and stays float32 among the bfloat16 weights.
    #expect(parameters["model.layers.2.feed_forward.expert_bias"]!.dtype == .float32)
    #expect(parameters["model.layers.2.feed_forward.gate.weight"]!.dtype == .bfloat16)
}

/// The bias exemption holds at any requested precision, not just the checkpoint's.
@Test func lfm2MoeExpertBiasSurvivesPrecisionCast() throws {
    let directory = try lfm2MoeDirectory()
    let config = try Lfm2MoeConfig(directory: directory)
    let parameters = try loadLfm2MoeParameters(
        directory: directory, config: config, precision: .float32)

    #expect(parameters["model.layers.2.feed_forward.expert_bias"]!.dtype == .float32)
    #expect(parameters["model.embed_tokens.weight"]!.dtype == .float32)
}

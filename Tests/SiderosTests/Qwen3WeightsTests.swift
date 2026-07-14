import Foundation
import MLX
import Testing

@testable import Sideros

func qwen3Directory() throws -> URL { try hubSnapshot("Qwen--Qwen3-0.6B") }

@Test func qwen3ConfigMatchesCheckpoint() throws {
    let config = try Qwen3Config(directory: qwen3Directory())

    #expect(config.numHiddenLayers == 28)
    #expect(config.numAttentionHeads == 16)
    #expect(config.numKeyValueHeads == 8)
    // Explicit in the config and decoupled from hiddenSize: 16 x 128 = 2048, not 1024.
    #expect(config.headDim == 128)
    #expect(config.hiddenSize == 1024)
    #expect(config.intermediateSize == 3072)
    #expect(config.vocabSize == 151936)
    #expect(config.rmsNormEps == 1e-6)
    #expect(config.ropeTheta == 1_000_000)
    #expect(config.tieWordEmbeddings)
}

@Test func qwen3ParametersMatchExpectedSetAndShapes() throws {
    let directory = try qwen3Directory()
    let config = try Qwen3Config(directory: directory)
    let parameters = try loadQwen3Parameters(directory: directory, config: config, precision: .float32)

    // The checkpoint serializes lm_head despite tying it; the loader drops it like
    // transformers does, leaving embed + norm + 11 tensors per layer.
    #expect(parameters.count == 2 + 11 * config.numHiddenLayers)
    #expect(parameters["lm_head.weight"] == nil)
    #expect(parameters["model.layers.0.self_attn.q_proj.bias"] == nil)
    #expect(parameters.values.allSatisfy { $0.dtype == .float32 })

    #expect(parameters["model.layers.0.self_attn.q_proj.weight"]!.shape == [2048, 1024])
    #expect(parameters["model.layers.0.self_attn.k_proj.weight"]!.shape == [1024, 1024])
    #expect(parameters["model.layers.0.self_attn.o_proj.weight"]!.shape == [1024, 2048])
    #expect(parameters["model.layers.0.self_attn.q_norm.weight"]!.shape == [128])
}

/// A synthetic two-shard checkpoint: the loader must read exactly the files the index
/// names and merge them; single-file loading is covered by every other model test.
@Test func shardedCheckpointMergesAllShards() throws {
    let directory = FileManager.default.temporaryDirectory
        .appending(path: "sideros-sharded-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    try save(
        arrays: ["a.weight": MLXArray([1, 2] as [Float])],
        url: directory.appending(path: "model-00001-of-00002.safetensors"))
    try save(
        arrays: ["b.weight": MLXArray([3] as [Float])],
        url: directory.appending(path: "model-00002-of-00002.safetensors"))
    let index = """
        {"weight_map": {"a.weight": "model-00001-of-00002.safetensors",
                        "b.weight": "model-00002-of-00002.safetensors"}}
        """
    try index.data(using: .utf8)!.write(to: directory.appending(path: "model.safetensors.index.json"))

    let parameters = try loadCheckpoint(directory: directory)
    #expect(parameters.count == 2)
    #expect(parameters["a.weight"]!.asArray(Float.self) == [1, 2])
    #expect(parameters["b.weight"]!.asArray(Float.self) == [3])
}

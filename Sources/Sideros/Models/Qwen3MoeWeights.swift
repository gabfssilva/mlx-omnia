import Foundation
import MLX

public struct Qwen3MoeConfig: Decodable, Sendable {
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool
    /// The experts are narrow: 768 against the 6144 a dense MLP of this width would use.
    public let moeIntermediateSize: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let normTopkProb: Bool
    /// Present when the checkpoint itself is quantized (mlx-community exports).
    public let quantization: Quantization?

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        self = try decoder.decode(Self.self, from: data)
    }
}

/// Qwen3's attention with the MLP replaced by 128 experts, 8 per token, on every layer.
/// The expert weights arrive stacked as `mlp.switch_mlp.*` (mlx's layout), one matrix per
/// expert on the leading axis.
public func loadQwen3MoeParameters(
    directory: URL, config: Qwen3MoeConfig, precision: Precision?, quantization: Quantization? = nil
) throws -> [String: MLXArray] {
    if config.quantization != nil {
        guard precision == nil, quantization == nil else { throw Qwen2LoadError.alreadyQuantized }
    }
    var parameters = try loadCheckpoint(directory: directory)
        .mapValues { array in precision.map { array.asType($0.dtype) } ?? array }

    if config.tieWordEmbeddings {
        parameters["lm_head.weight"] = nil
        parameters["lm_head.scales"] = nil
        parameters["lm_head.biases"] = nil
    }

    let expected = expectedShapes(config)
    let missing = expected.keys.filter { parameters[$0] == nil }.sorted()
    let unexpected = parameters.keys.filter { expected[$0] == nil }.sorted()
    guard missing.isEmpty, unexpected.isEmpty else {
        throw Qwen2LoadError.parameterMismatch(missing: missing, unexpected: unexpected)
    }

    for (name, shape) in expected where parameters[name]!.shape != shape {
        throw Qwen2LoadError.shapeMismatch(
            name: name, expected: shape, actual: parameters[name]!.shape)
    }

    if let quantization {
        parameters = quantizing(parameters, with: quantization) { name in
            name == "model.embed_tokens.weight" || name == "lm_head.weight"
                || name.hasSuffix("proj.weight") || name.hasSuffix("mlp.gate.weight")
        }
    }
    return parameters
}

private func expectedShapes(_ config: Qwen3MoeConfig) -> [String: [Int]] {
    let hidden = config.hiddenSize
    let queries = config.numAttentionHeads * config.headDim
    let kv = config.numKeyValueHeads * config.headDim
    let experts = config.numExperts
    let inner = config.moeIntermediateSize
    var shapes: [String: [Int]] = [
        "model.embed_tokens.weight": [config.vocabSize, hidden],
        "model.norm.weight": [hidden],
    ]
    if !config.tieWordEmbeddings {
        shapes["lm_head.weight"] = [config.vocabSize, hidden]
    }

    for layer in 0..<config.numHiddenLayers {
        let p = "model.layers.\(layer)."
        shapes[p + "input_layernorm.weight"] = [hidden]
        shapes[p + "post_attention_layernorm.weight"] = [hidden]
        shapes[p + "self_attn.q_proj.weight"] = [queries, hidden]
        shapes[p + "self_attn.k_proj.weight"] = [kv, hidden]
        shapes[p + "self_attn.v_proj.weight"] = [kv, hidden]
        shapes[p + "self_attn.o_proj.weight"] = [hidden, queries]
        shapes[p + "self_attn.q_norm.weight"] = [config.headDim]
        shapes[p + "self_attn.k_norm.weight"] = [config.headDim]
        shapes[p + "mlp.gate.weight"] = [experts, hidden]
        shapes[p + "mlp.switch_mlp.gate_proj.weight"] = [experts, inner, hidden]
        shapes[p + "mlp.switch_mlp.up_proj.weight"] = [experts, inner, hidden]
        shapes[p + "mlp.switch_mlp.down_proj.weight"] = [experts, hidden, inner]
    }

    // Packing runs along the last axis at any rank, so a stacked expert weight quantizes
    // exactly like a 2D one: only the row length changes.
    if let q = config.quantization {
        for (name, shape) in shapes where shape.count >= 2 {
            let base = String(name.dropLast(".weight".count))
            let last = shape.count - 1
            shapes[name] = shape.enumerated().map { $0.offset == last ? $0.element * q.bits / 32 : $0.element }
            let grouped = shape.enumerated().map { $0.offset == last ? $0.element / q.groupSize : $0.element }
            shapes[base + ".scales"] = grouped
            shapes[base + ".biases"] = grouped
        }
    }
    return shapes
}

import Foundation
import MLX

public struct Lfm2MoeConfig: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    /// Width of the two dense MLPs; the routed experts use `moeIntermediateSize`.
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    /// The first layers keep an ordinary MLP; routing starts after them.
    public let numDenseLayers: Int
    public let normTopkProb: Bool
    public let useExpertBias: Bool
    public let routedScalingFactor: Float
    public let normEps: Float
    public let convBias: Bool
    /// Kernel size of the short conv; the cache window keeps one row less.
    public let convLCache: Int
    /// "conv" or "full_attention" per layer — 18 conv, 6 attention in the 8B-A1B.
    public let layerTypes: [String]
    public let tieWordEmbeddings: Bool

    private struct RopeParameters: Decodable { let ropeTheta: Float }
    private let ropeParameters: RopeParameters
    public var ropeTheta: Float { ropeParameters.ropeTheta }

    /// Implicit, unlike Qwen3: always hidden / heads.
    public var headDim: Int { hiddenSize / numAttentionHeads }

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        self = try decoder.decode(Self.self, from: data)
    }
}

/// The checkpoint is bfloat16 except `expert_bias`, which ships float32 and stays
/// float32 at every precision — the router adds it in float32 regardless.
public func loadLfm2MoeParameters(
    directory: URL, config: Lfm2MoeConfig, precision: Precision?
) throws -> [String: MLXArray] {
    var parameters = try loadCheckpoint(directory: directory)
        .map { name, array in
            name.hasSuffix("expert_bias")
                ? (name, array) : (name, precision.map { array.asType($0.dtype) } ?? array)
        }
        .reduce(into: [String: MLXArray]()) { $0[$1.0] = $1.1 }

    // One stacked [experts, out, in] tensor per projection is what gatherMM routes over.
    for layer in config.numDenseLayers..<config.numHiddenLayers {
        let p = "model.layers.\(layer).feed_forward.experts."
        for w in ["w1", "w2", "w3"] {
            let slices = (0..<config.numExperts).compactMap {
                parameters.removeValue(forKey: p + "\($0).\(w).weight")
            }
            guard slices.count == config.numExperts else {
                throw Qwen2LoadError.parameterMismatch(
                    missing: [p + "*.\(w).weight"], unexpected: [])
            }
            parameters[p + w + ".weight"] = stacked(slices)
        }
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
    return parameters
}

private func expectedShapes(_ config: Lfm2MoeConfig) -> [String: [Int]] {
    let hidden = config.hiddenSize
    let queries = config.numAttentionHeads * config.headDim
    let kv = config.numKeyValueHeads * config.headDim
    var shapes: [String: [Int]] = [
        "model.embed_tokens.weight": [config.vocabSize, hidden],
        "model.embedding_norm.weight": [hidden],
    ]

    for (layer, type) in config.layerTypes.enumerated() {
        let p = "model.layers.\(layer)."
        shapes[p + "operator_norm.weight"] = [hidden]
        shapes[p + "ffn_norm.weight"] = [hidden]

        if type == "full_attention" {
            shapes[p + "self_attn.q_proj.weight"] = [queries, hidden]
            shapes[p + "self_attn.k_proj.weight"] = [kv, hidden]
            shapes[p + "self_attn.v_proj.weight"] = [kv, hidden]
            shapes[p + "self_attn.out_proj.weight"] = [hidden, queries]
            shapes[p + "self_attn.q_layernorm.weight"] = [config.headDim]
            shapes[p + "self_attn.k_layernorm.weight"] = [config.headDim]
        } else {
            shapes[p + "conv.conv.weight"] = [hidden, 1, config.convLCache]
            shapes[p + "conv.in_proj.weight"] = [3 * hidden, hidden]
            shapes[p + "conv.out_proj.weight"] = [hidden, hidden]
            if config.convBias {
                shapes[p + "conv.conv.bias"] = [hidden]
                shapes[p + "conv.in_proj.bias"] = [3 * hidden]
                shapes[p + "conv.out_proj.bias"] = [hidden]
            }
        }

        if layer < config.numDenseLayers {
            shapes[p + "feed_forward.w1.weight"] = [config.intermediateSize, hidden]
            shapes[p + "feed_forward.w3.weight"] = [config.intermediateSize, hidden]
            shapes[p + "feed_forward.w2.weight"] = [hidden, config.intermediateSize]
        } else {
            let inner = config.moeIntermediateSize
            shapes[p + "feed_forward.gate.weight"] = [config.numExperts, hidden]
            shapes[p + "feed_forward.experts.w1.weight"] = [config.numExperts, inner, hidden]
            shapes[p + "feed_forward.experts.w3.weight"] = [config.numExperts, inner, hidden]
            shapes[p + "feed_forward.experts.w2.weight"] = [config.numExperts, hidden, inner]
            if config.useExpertBias {
                shapes[p + "feed_forward.expert_bias"] = [config.numExperts]
            }
        }
    }
    return shapes
}

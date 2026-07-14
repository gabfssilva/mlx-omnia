import Foundation
import MLX

public struct Gemma3Config: Decodable, Sendable {
    public enum LayerType: String, Decodable, Sendable {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    /// Not hiddenSize / numAttentionHeads: 4 x 256 = 1024 against a hidden size of 640.
    public let headDim: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeLocalBaseFreq: Float
    /// The attention scale is (this)^-0.5, which only coincides with headDim here.
    public let queryPreAttnScalar: Float
    public let slidingWindow: Int
    public let layerTypes: [LayerType]

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        self = try decoder.decode(Self.self, from: data)
    }
}

public enum Gemma3LoadError: Error {
    case parameterMismatch(missing: [String], unexpected: [String])
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
}

/// The checkpoint is bfloat16; upcasting to float32 is lossless and matches the reference.
public func loadGemma3Parameters(
    directory: URL, config: Gemma3Config, precision: Precision?, quantization: Quantization? = nil
) throws -> [String: MLXArray] {
    var parameters = try loadArrays(url: directory.appending(path: "model.safetensors"))
        .mapValues { array in precision.map { array.asType($0.dtype) } ?? array }

    let expected = expectedShapes(config)
    let missing = expected.keys.filter { parameters[$0] == nil }.sorted()
    let unexpected = parameters.keys.filter { expected[$0] == nil }.sorted()
    guard missing.isEmpty, unexpected.isEmpty else {
        throw Gemma3LoadError.parameterMismatch(missing: missing, unexpected: unexpected)
    }

    for (name, shape) in expected where parameters[name]!.shape != shape {
        throw Gemma3LoadError.shapeMismatch(
            name: name, expected: shape, actual: parameters[name]!.shape)
    }

    if let quantization {
        parameters = quantizing(parameters, with: quantization) { name in
            name == "model.embed_tokens.weight" || name.hasSuffix("proj.weight")
        }
    }
    return parameters
}

/// lm_head is absent from the checkpoint: it is tied to embed_tokens.
private func expectedShapes(_ config: Gemma3Config) -> [String: [Int]] {
    let hidden = config.hiddenSize
    let queries = config.numAttentionHeads * config.headDim
    let kv = config.numKeyValueHeads * config.headDim
    var shapes: [String: [Int]] = [
        "model.embed_tokens.weight": [config.vocabSize, hidden],
        "model.norm.weight": [hidden],
    ]

    for layer in 0..<config.numHiddenLayers {
        let p = "model.layers.\(layer)."
        // Four norms per block, not two: each residual branch is normed on entry and exit.
        shapes[p + "input_layernorm.weight"] = [hidden]
        shapes[p + "post_attention_layernorm.weight"] = [hidden]
        shapes[p + "pre_feedforward_layernorm.weight"] = [hidden]
        shapes[p + "post_feedforward_layernorm.weight"] = [hidden]
        shapes[p + "self_attn.q_proj.weight"] = [queries, hidden]
        shapes[p + "self_attn.k_proj.weight"] = [kv, hidden]
        shapes[p + "self_attn.v_proj.weight"] = [kv, hidden]
        shapes[p + "self_attn.o_proj.weight"] = [hidden, queries]
        // Per-head norms live inside attention; no other architecture here has weights
        // between the projections and the kernel.
        shapes[p + "self_attn.q_norm.weight"] = [config.headDim]
        shapes[p + "self_attn.k_norm.weight"] = [config.headDim]
        shapes[p + "mlp.gate_proj.weight"] = [config.intermediateSize, hidden]
        shapes[p + "mlp.up_proj.weight"] = [config.intermediateSize, hidden]
        shapes[p + "mlp.down_proj.weight"] = [hidden, config.intermediateSize]
    }
    return shapes
}

import Foundation
import MLX

public struct Qwen2Config: Decodable, Sendable {
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    /// Present when the checkpoint itself is quantized (mlx-community exports).
    public let quantization: Quantization?

    public var headDim: Int { hiddenSize / numAttentionHeads }

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        self = try decoder.decode(Self.self, from: data)
    }
}

public enum Qwen2LoadError: Error, Equatable {
    case parameterMismatch(missing: [String], unexpected: [String])
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
    /// The checkpoint is already quantized; requantizing or converting its precision
    /// would silently produce a different model than the one on disk.
    case alreadyQuantized
}

/// The checkpoint is bfloat16. Upcasting to float32 is lossless and is what
/// `torch_dtype=float32` does on the reference side, so parity stays exact.
public func loadQwen2Parameters(
    directory: URL, config: Qwen2Config, precision: Precision?, quantization: Quantization? = nil
) throws -> [String: MLXArray] {
    if config.quantization != nil {
        guard precision == nil, quantization == nil else { throw Qwen2LoadError.alreadyQuantized }
    }
    var parameters = try loadArrays(url: directory.appending(path: "model.safetensors"))
        .mapValues { array in precision.map { array.asType($0.dtype) } ?? array }

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
            name == "model.embed_tokens.weight" || name.hasSuffix("proj.weight")
        }
    }
    return parameters
}

/// lm_head is absent from the checkpoint: tie_word_embeddings ties it to embed_tokens.
private func expectedShapes(_ config: Qwen2Config) -> [String: [Int]] {
    let hidden = config.hiddenSize
    let kv = config.numKeyValueHeads * config.headDim
    var shapes: [String: [Int]] = [
        "model.embed_tokens.weight": [config.vocabSize, hidden],
        "model.norm.weight": [hidden],
    ]

    for layer in 0..<config.numHiddenLayers {
        let p = "model.layers.\(layer)."
        shapes[p + "input_layernorm.weight"] = [hidden]
        shapes[p + "post_attention_layernorm.weight"] = [hidden]
        // Linear weights are [out, in]; GPT-2's Conv1D stored them transposed.
        // Only the qkv projections carry a bias, and only Qwen2 has them at all.
        shapes[p + "self_attn.q_proj.weight"] = [hidden, hidden]
        shapes[p + "self_attn.q_proj.bias"] = [hidden]
        shapes[p + "self_attn.k_proj.weight"] = [kv, hidden]
        shapes[p + "self_attn.k_proj.bias"] = [kv]
        shapes[p + "self_attn.v_proj.weight"] = [kv, hidden]
        shapes[p + "self_attn.v_proj.bias"] = [kv]
        shapes[p + "self_attn.o_proj.weight"] = [hidden, hidden]
        shapes[p + "mlp.gate_proj.weight"] = [config.intermediateSize, hidden]
        shapes[p + "mlp.up_proj.weight"] = [config.intermediateSize, hidden]
        shapes[p + "mlp.down_proj.weight"] = [hidden, config.intermediateSize]
    }

    // A quantized checkpoint stores every 2D weight as a triple: the packed uint32
    // rows (bits/32 of the dense columns) plus per-group scales and biases.
    if let q = config.quantization {
        for (name, shape) in shapes where shape.count == 2 {
            let base = String(name.dropLast(".weight".count))
            shapes[name] = [shape[0], shape[1] * q.bits / 32]
            shapes[base + ".scales"] = [shape[0], shape[1] / q.groupSize]
            shapes[base + ".biases"] = [shape[0], shape[1] / q.groupSize]
        }
    }
    return shapes
}

import Foundation
import MLX

public struct GPT2Config: Decodable, Sendable {
    public let nLayer: Int
    public let nHead: Int
    public let nEmbd: Int
    public let nPositions: Int
    public let vocabSize: Int
    public let layerNormEpsilon: Float

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        self = try decoder.decode(Self.self, from: data)
    }
}

public enum GPT2LoadError: Error {
    case parameterMismatch(missing: [String], unexpected: [String])
    case shapeMismatch(name: String, expected: [Int], actual: [Int])
}

/// `precision` of nil leaves the checkpoint's own dtype, which for GPT-2 is float32.
public func loadGPT2Parameters(
    directory: URL, config: GPT2Config, precision: Precision?, quantization: Quantization? = nil
) throws -> [String: MLXArray] {
    let all = try loadArrays(url: directory.appending(path: "model.safetensors"))
    // h.N.attn.bias is HF's pre-materialized causal mask persisted as a buffer, not a
    // learned parameter. We build the mask from nPositions instead.
    var parameters = all
        .filter { !$0.key.hasSuffix(".attn.bias") }
        .mapValues { array in precision.map { array.asType($0.dtype) } ?? array }

    let expected = expectedShapes(config)
    let missing = expected.keys.filter { parameters[$0] == nil }.sorted()
    let unexpected = parameters.keys.filter { expected[$0] == nil }.sorted()
    guard missing.isEmpty, unexpected.isEmpty else {
        throw GPT2LoadError.parameterMismatch(missing: missing, unexpected: unexpected)
    }

    for (name, shape) in expected where parameters[name]!.shape != shape {
        throw GPT2LoadError.shapeMismatch(name: name, expected: shape, actual: parameters[name]!.shape)
    }

    // wpe stays dense: positions are read as rows, and 1024 of them cost nothing.
    if let quantization {
        let conv = { (name: String) in name.contains(".c_") && name.hasSuffix(".weight") }
        parameters = quantizing(
            parameters, with: quantization,
            where: { name in name == "wte.weight" || conv(name) },
            transposing: conv)
    }
    return parameters
}

/// lm_head is absent from the checkpoint: GPT-2 ties it to wte.
private func expectedShapes(_ config: GPT2Config) -> [String: [Int]] {
    let embd = config.nEmbd
    var shapes: [String: [Int]] = [
        "wte.weight": [config.vocabSize, embd],
        "wpe.weight": [config.nPositions, embd],
        "ln_f.weight": [embd],
        "ln_f.bias": [embd],
    ]

    for layer in 0..<config.nLayer {
        let p = "h.\(layer)."
        shapes[p + "ln_1.weight"] = [embd]
        shapes[p + "ln_1.bias"] = [embd]
        shapes[p + "ln_2.weight"] = [embd]
        shapes[p + "ln_2.bias"] = [embd]
        // The projections come from Conv1D, so weights are [in, out] — transposed
        // relative to what MLX's Linear expects.
        shapes[p + "attn.c_attn.weight"] = [embd, 3 * embd]
        shapes[p + "attn.c_attn.bias"] = [3 * embd]
        shapes[p + "attn.c_proj.weight"] = [embd, embd]
        shapes[p + "attn.c_proj.bias"] = [embd]
        shapes[p + "mlp.c_fc.weight"] = [embd, 4 * embd]
        shapes[p + "mlp.c_fc.bias"] = [4 * embd]
        shapes[p + "mlp.c_proj.weight"] = [4 * embd, embd]
        shapes[p + "mlp.c_proj.bias"] = [embd]
    }
    return shapes
}

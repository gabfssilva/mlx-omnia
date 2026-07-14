import Foundation
import MLX
import MLXNN

public struct GPT2Activations {
    public let embeddings: MLXArray
    public let blocks: [MLXArray]
    public let lnF: MLXArray
    public let logits: MLXArray
}

public struct GPT2: CausalLM {
    private let config: GPT2Config
    private let parameters: [String: MLXArray]
    private let quantization: Quantization?

    public init(
        config: GPT2Config, parameters: [String: MLXArray], quantization: Quantization? = nil
    ) {
        self.config = config
        self.parameters = parameters
        self.quantization = quantization
    }

    /// Total by construction: loadGPT2Parameters rejects any checkpoint missing a name.
    private subscript(name: String) -> MLXArray { parameters[name]! }

    /// Dense Conv1D weights are (in, out) and apply directly; quantize-on-load transposes
    /// them to (out, in) before packing, so the quantized case is the standard layout.
    private func conv(_ name: String) -> Linear {
        guard parameters[name + ".scales"] == nil else {
            return Linear(name, parameters: parameters, quantization: quantization)
        }
        return .dense(self[name + ".weight"], bias: parameters[name + ".bias"], skinny: false)
    }

    /// The only learned positional table here: a cached step must read the row for its
    /// absolute position, not for position zero. wpe stays dense even when quantized.
    public func embed(_ tokens: MLXArray, offset: Int = 0) -> MLXArray {
        let wte = Embedding("wte", parameters: parameters, quantization: quantization)
        return wte(tokens) + self["wpe.weight"][offset..<(offset + tokens.dim(1))]
    }

    public func makeCache() -> [KVCache] { (0..<config.nLayer).map { _ in KVCache() } }

    public func logits(_ embeddings: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
    {
        let decoded = decode(embeddings, cache: cache)
        return (decoded.logits, decoded.cache)
    }

    public func callAsFunction(_ ids: [Int32]) -> GPT2Activations {
        let embeddings = embed(ids)
        let decoded = decode(embeddings, cache: makeCache())
        return GPT2Activations(
            embeddings: embeddings,
            blocks: decoded.blocks,
            lnF: decoded.lnF,
            logits: decoded.logits
        )
    }

    private func decode(_ embeddings: MLXArray, cache: [KVCache])
        -> (blocks: [MLXArray], lnF: MLXArray, logits: MLXArray, cache: [KVCache])
    {
        var cache = cache
        let queries = embeddings.dim(1)
        let mask = Self.causalMask(
            queries: queries, keys: cache[0].offset + queries, dtype: embeddings.dtype)

        var x = embeddings
        var blocks: [MLXArray] = []
        for layer in 0..<config.nLayer {
            (x, cache[layer]) = block(x, mask: mask, prefix: "h.\(layer).", cache: cache[layer])
            blocks.append(x)
        }

        let lnF = layerNorm(x, prefix: "ln_f")
        let head = Linear("wte", parameters: parameters, quantization: quantization)
        return (blocks, lnF, head(lnF), cache)  // lm_head is tied to wte
    }

    /// Additive, unlike the boolean masks of Qwen2 and Gemma 3: it is summed into the
    /// scores, so a float32 mask would drag the whole forward back up to float32. The
    /// queries sit at the end of the keys, which is what lets one query attend to a cache.
    static func causalMask(queries: Int, keys: Int, dtype: DType = .float32) -> MLXArray {
        triu(
            full([queries, keys], values: -Float.greatestFiniteMagnitude),
            k: keys - queries + 1
        ).asType(dtype)
    }

    func block(_ x: MLXArray, mask: MLXArray, prefix: String, cache: KVCache)
        -> (MLXArray, KVCache)
    {
        let (context, grown) = attention(
            layerNorm(x, prefix: prefix + "ln_1"), mask: mask, prefix: prefix, cache: cache)
        let attended = x + context
        return (attended + mlp(layerNorm(attended, prefix: prefix + "ln_2"), prefix: prefix), grown)
    }

    func attention(_ x: MLXArray, mask: MLXArray, prefix: String, cache: KVCache)
        -> (MLXArray, KVCache)
    {
        let length = x.dim(1)
        let headDim = config.nEmbd / config.nHead

        let qkv = conv(prefix + "attn.c_attn")(x)
        let heads = split(qkv, parts: 3, axis: -1).map {
            $0.reshaped(1, length, config.nHead, headDim).transposed(0, 2, 1, 3)
        }
        let (keys, values, grown) = cache.appending(keys: heads[1], values: heads[2])

        let scores = matmul(heads[0], keys.transposed(0, 1, 3, 2)) / sqrt(Float(headDim))
        let weights = softmax(scores + mask, axis: -1)

        let context = matmul(weights, values).transposed(0, 2, 1, 3).reshaped(1, length, config.nEmbd)
        return (conv(prefix + "attn.c_proj")(context), grown)
    }

    func mlp(_ x: MLXArray, prefix: String) -> MLXArray {
        conv(prefix + "mlp.c_proj")(geluApproximate(conv(prefix + "mlp.c_fc")(x)))
    }

    func layerNorm(_ x: MLXArray, prefix: String) -> MLXArray {
        MLXFast.layerNorm(
            x, weight: self[prefix + ".weight"], bias: self[prefix + ".bias"],
            eps: config.layerNormEpsilon)
    }
}

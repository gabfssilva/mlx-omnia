import Foundation
import MLX

public struct Gemma3Activations {
    public let embeddings: MLXArray
    public let blocks: [MLXArray]
    public let norm: MLXArray
    public let logits: MLXArray
}

public struct Gemma3: CausalLM {
    /// Everything the layers need per call and cannot capture at build time. Five of every
    /// six layers see only the last slidingWindow keys, so a layer picks its own mask.
    /// The caches are the only thing a layer writes to, one per layer, indexed by it.
    struct Context {
        let full: MLXArray
        let sliding: MLXArray
        var cache: [KVCache]
        let offset: Int
    }

    private let config: Gemma3Config
    private let parameters: [String: MLXArray]
    private let quantization: Quantization?
    private var layers: [Layer<Context>] = []

    public init(
        config: Gemma3Config, parameters: [String: MLXArray], quantization: Quantization? = nil
    ) {
        self.config = config
        self.parameters = parameters
        self.quantization = quantization
        self.layers = config.layerTypes.indices.map(block)
    }

    /// Total by construction: loadGemma3Parameters rejects any checkpoint missing a name.
    private subscript(name: String) -> MLXArray { parameters[name]! }

    /// Gemma scales the embeddings by sqrt(hiddenSize). mlx-lm rounds that scalar to
    /// bfloat16 (25.25); transformers keeps it in the model dtype (25.298...).
    public func embed(_ tokens: MLXArray, offset: Int = 0) -> MLXArray {
        let table = Embedding("model.embed_tokens", parameters: parameters, quantization: quantization)
        return table(tokens) * sqrt(Float(config.hiddenSize))
    }

    public func makeCache() -> [KVCache] { config.layerTypes.map { _ in KVCache() } }

    public func logits(_ embeddings: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, cache: [KVCache])
    {
        let decoded = decode(embeddings, cache: cache)
        return (decoded.logits, decoded.cache)
    }

    public func callAsFunction(_ ids: [Int32]) -> Gemma3Activations {
        let embeddings = embed(ids)
        let decoded = decode(embeddings, cache: makeCache())
        return Gemma3Activations(
            embeddings: embeddings,
            blocks: decoded.blocks,
            norm: decoded.norm,
            logits: decoded.logits
        )
    }

    private func decode(_ embeddings: MLXArray, cache: [KVCache])
        -> (blocks: [MLXArray], norm: MLXArray, logits: MLXArray, cache: [KVCache])
    {
        let queries = embeddings.dim(1)
        let offset = cache[0].offset
        var context = Context(
            full: Self.causalMask(queries: queries, keys: offset + queries, window: nil),
            sliding: Self.causalMask(
                queries: queries, keys: offset + queries, window: config.slidingWindow),
            cache: cache, offset: offset)

        var x = embeddings
        var blocks: [MLXArray] = []
        for layer in layers {
            (x, context) = layer(x, context)
            blocks.append(x)
        }

        let (normed, _) = norm("model.norm")(x, context)
        // tied lm_head
        let head = Linear("model.embed_tokens", parameters: parameters, quantization: quantization)
        return (blocks, normed, head(normed), context.cache)
    }

    /// A boolean mask of the keys each query may attend to: everything up to itself, and
    /// within `window` positions of it when the layer slides. The queries are the last
    /// `queries` of the `keys` positions, which is what lets one query attend to a cache.
    static func causalMask(queries: Int, keys: Int, window: Int?) -> MLXArray {
        let rows = MLXArray(Array((keys - queries)..<keys).map(Int32.init)).reshaped(queries, 1)
        let columns = MLXArray(Array(0..<keys).map(Int32.init)).reshaped(1, keys)

        let causal = rows .>= columns
        guard let window else { return causal }
        return causal .&& (rows .< columns + Int32(window))
    }

    func block(_ layer: Int) -> Layer<Context> {
        let prefix = "model.layers.\(layer)."
        let type = config.layerTypes[layer]

        // Each residual branch is normed on the way out as well as on the way in.
        return residual {
            norm(prefix + "input_layernorm")
                >>> attention(layer: layer, prefix: prefix, type: type)
                >>> norm(prefix + "post_attention_layernorm")
        } >>> residual {
            norm(prefix + "pre_feedforward_layernorm")
                >>> mlp(prefix)
                >>> norm(prefix + "post_feedforward_layernorm")
        }
    }

    func attention(layer: Int, prefix: String, type: Gemma3Config.LayerType) -> Layer<Context> {
        let p = prefix + "self_attn."
        let base = type == .sliding ? config.ropeLocalBaseFreq : config.ropeTheta

        // q_norm and k_norm sit between the projections and the rotation.
        return graph {
            let q = proj(p + "q_proj") >>> heads(config.numAttentionHeads) >>> norm(p + "q_norm") >>> rope(base)
            let k = proj(p + "k_proj") >>> heads(config.numKeyValueHeads) >>> norm(p + "k_norm") >>> rope(base)
            let v = proj(p + "v_proj") >>> heads(config.numKeyValueHeads)
            return sdpa(q, k, v, layer: layer, type: type) >>> merge >>> proj(p + "o_proj")
        }
    }

    /// The interior is a pure chain, so `graph` compiles it and the gelu(gate) · up
    /// pair fuses into one kernel.
    func mlp(_ prefix: String) -> Chain {
        let p = prefix + "mlp."
        return graph { (proj(p + "gate_proj") >>> gelu) * proj(p + "up_proj") >>> proj(p + "down_proj") }
    }

    /// Gemma stores the norm weight centred on zero, so the scale is `1 + w`.
    func norm(_ name: String) -> Layer<Context> {
        let weight = 1 + self[name + ".weight"]
        let eps = config.rmsNormEps
        return stateless { x, _ in MLXFast.rmsNorm(x, weight: weight, eps: eps) }
    }

    func proj(_ name: String) -> Chain {
        let linear = Linear(name, parameters: parameters, quantization: quantization)
        return { x in linear(x) }
    }

    /// The tanh gelu written raw, not MLXNN's — this chain runs inside a traced graph
    /// (see the note on `graph`).
    var gelu: Chain {
        { x in 0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x ** 3))) }
    }

    func heads(_ count: Int) -> Layer<Context> {
        let dim = config.headDim
        return stateless { x, _ in x.reshaped(1, x.dim(1), count, dim).transposed(0, 2, 1, 3) }
    }

    func rope(_ base: Float) -> Layer<Context> {
        let dim = config.headDim
        return stateless { x, context in
            MLXFast.RoPE(
                x, dimensions: dim, traditional: false, base: base, scale: 1,
                offset: context.offset)
        }
    }

    /// The join: every branch reads the same input, threaded in order. The one layer
    /// that writes to the context. The cache is grown after the rotation, so what it
    /// stores is the rotated keys and a cached key is never rotated twice.
    func sdpa(
        _ query: @escaping Layer<Context>, _ key: @escaping Layer<Context>,
        _ value: @escaping Layer<Context>, layer: Int, type: Gemma3Config.LayerType
    ) -> Layer<Context> {
        let scale = pow(config.queryPreAttnScalar, -0.5)
        return { x, context in
            let (q, afterQ) = query(x, context)
            let (k, afterK) = key(x, afterQ)
            let (v, context) = value(x, afterK)

            let (keys, values, grown) = context.cache[layer].appending(keys: k, values: v)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale,
                mask: .array(type == .sliding ? context.sliding : context.full))

            var next = context
            next.cache[layer] = grown
            return (attended, next)
        }
    }

    /// The attention output is heads * headDim = 1024 wide, not hiddenSize.
    var merge: Layer<Context> {
        let width = config.numAttentionHeads * config.headDim
        return stateless { x, _ in x.transposed(0, 2, 1, 3).reshaped(1, x.dim(2), width) }
    }
}

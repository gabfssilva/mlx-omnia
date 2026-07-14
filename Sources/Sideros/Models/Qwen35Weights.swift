import Foundation
import MLX

/// The ViT of the Qwen3.6 checkpoints — one architecture at every size (12 blocks of 768 on
/// the 0.8B, 27 of 1152 on the 27B and the 35B-A3B). It ships bfloat16 even inside a quantized
/// checkpoint: no mlx conversion packs it. `deepstack_visual_indexes` is empty on all of them,
/// so the tower's output enters the trunk only through the merger — there is no fan-in to
/// intermediate layers, and the field is not read.
public struct VisionConfig: Decodable, Sendable {
    public let depth: Int
    public let hiddenSize: Int
    public let numHeads: Int
    public let intermediateSize: Int
    /// The trunk's hidden size: what the merger projects into.
    public let outHiddenSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let numPositionEmbeddings: Int
    public let inChannels: Int

    public var headDim: Int { hiddenSize / numHeads }
    /// One patch, flattened the way the processor emits it: [channels, temporal, row, column].
    public var patchDim: Int { inChannels * temporalPatchSize * patchSize * patchSize }
    /// The merger folds a spatial block of patches into one trunk token.
    public var mergeUnit: Int { spatialMergeSize * spatialMergeSize }
    public var mergerHidden: Int { hiddenSize * mergeUnit }
    /// The learned position grid is square (2304 = 48²) and interpolated to the image's.
    public var gridPerSide: Int { Int(Double(numPositionEmbeddings).squareRoot()) }
}

public struct Qwen35Config: Decodable, Sendable {
    /// The interesting fields live under `text_config`; the top level carries the vision
    /// tower's config and, on mlx exports, the quantization block.
    private struct TextConfig: Decodable {
        let vocabSize: Int
        let hiddenSize: Int
        /// Absent on the MoE variant, where the experts carry their own width.
        let intermediateSize: Int?
        let numExperts: Int?
        let numExpertsPerTok: Int?
        let moeIntermediateSize: Int?
        let sharedExpertIntermediateSize: Int?
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let headDim: Int
        let rmsNormEps: Float
        /// "linear_attention" (gated DeltaNet) or "full_attention" per layer.
        let layerTypes: [String]
        let linearNumKeyHeads: Int
        let linearNumValueHeads: Int
        let linearKeyHeadDim: Int
        let linearValueHeadDim: Int
        let linearConvKernelDim: Int
        let tieWordEmbeddings: Bool?
        let eosTokenId: Int32
        let ropeParameters: RopeParameters
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float
        let partialRotaryFactor: Float
        let mropeSection: [Int]
    }

    private let textConfig: TextConfig
    /// The 0.8B states tying only at the top level; the 27B in both places.
    private let tieWordEmbeddings: Bool?
    /// Absent on a text-only export; present on every Qwen3.6 checkpoint shipped so far.
    public let visionConfig: VisionConfig?
    public let imageTokenId: Int32?
    public let visionStartTokenId: Int32?
    public let visionEndTokenId: Int32?
    /// Present when the checkpoint itself is quantized (mlx-community exports). Read again
    /// with the raw key strategy in `init`: its per-module overrides are weight paths, not
    /// snake_case names, and the conversion would rewrite them.
    public private(set) var quantization: Quantization?

    public var vocabSize: Int { textConfig.vocabSize }
    public var hiddenSize: Int { textConfig.hiddenSize }
    public var intermediateSize: Int? { textConfig.intermediateSize }
    public var numHiddenLayers: Int { textConfig.numHiddenLayers }
    public var numAttentionHeads: Int { textConfig.numAttentionHeads }
    public var numKeyValueHeads: Int { textConfig.numKeyValueHeads }
    public var headDim: Int { textConfig.headDim }
    public var rmsNormEps: Float { textConfig.rmsNormEps }
    public var layerTypes: [String] { textConfig.layerTypes }
    public var linearNumKeyHeads: Int { textConfig.linearNumKeyHeads }
    public var linearNumValueHeads: Int { textConfig.linearNumValueHeads }
    public var linearKeyHeadDim: Int { textConfig.linearKeyHeadDim }
    public var linearValueHeadDim: Int { textConfig.linearValueHeadDim }
    public var linearConvKernelDim: Int { textConfig.linearConvKernelDim }
    public var ropeTheta: Float { textConfig.ropeParameters.ropeTheta }
    public var eosTokenId: Int32 { textConfig.eosTokenId }
    public var tied: Bool { textConfig.tieWordEmbeddings ?? tieWordEmbeddings ?? false }

    /// MRoPE degenerates to a plain partial rope for text: the three sections read the
    /// same position, so the interleave rewrites each frequency with its own value. Inside a
    /// vision run they read (t, h, w) and the interleave stops being the identity.
    public var ropeDimensions: Int {
        Int(Float(textConfig.headDim) * textConfig.ropeParameters.partialRotaryFactor)
    }

    /// How many of the `ropeDimensions / 2` frequencies each section owns, in the interleaved
    /// layout: [11, 11, 10] means the frequency at index i reads section i % 3.
    public var mropeSection: [Int] { textConfig.ropeParameters.mropeSection }

    public var keyDim: Int { linearNumKeyHeads * linearKeyHeadDim }
    public var valueDim: Int { linearNumValueHeads * linearValueHeadDim }
    public var convDim: Int { 2 * keyDim + valueDim }

    /// The 35B-A3B replaces every dense MLP with 256 experts, 8 per token, plus a shared
    /// expert every token pays for. The dense 27B and 0.8B leave these out of the config.
    public var numExperts: Int { textConfig.numExperts ?? 0 }
    public var numExpertsPerTok: Int { textConfig.numExpertsPerTok ?? 0 }
    public var moeIntermediateSize: Int { textConfig.moeIntermediateSize ?? 0 }
    public var sharedExpertIntermediateSize: Int { textConfig.sharedExpertIntermediateSize ?? 0 }
    public var mixtureOfExperts: Bool { numExperts > 0 }

    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        self = try decoder.decode(Self.self, from: data)
        quantization = try JSONDecoder().decode(QuantizationBlock.self, from: data).quantization
    }
}

private struct QuantizationBlock: Decodable {
    let quantization: Quantization?
}

/// Two checkpoint dialects load here. Raw HF (`model.language_model.*`, tower under
/// `model.visual.*`): conv weight in torch layout `[dim, 1, kernel]`, zero-centered RMSNorms,
/// patch embedding in torch Conv3d layout, MTP tower serialized. mlx exports
/// (`language_model.model.*`, tower under `vision_tower.*`): conv already `[dim, kernel, 1]`,
/// norms already shifted by +1, patch embedding channels-last, MTP dropped. Both normalize to
/// the house names (`model.*`, `lm_head.*`, `visual.*`) — so the model never branches.
public func loadQwen35Parameters(
    directory: URL, config: Qwen35Config, precision: Precision?
) throws -> [String: MLXArray] {
    if config.quantization != nil {
        guard precision == nil else { throw Qwen2LoadError.alreadyQuantized }
    }

    var parameters: [String: MLXArray] = [:]
    for (name, array) in try loadCheckpoint(directory: directory) {
        if name.hasPrefix("mtp.") { continue }
        var name = name
        if name.hasPrefix("model.language_model.") {
            name = "model." + name.dropFirst("model.language_model.".count)
        } else if name.hasPrefix("language_model.model.") {
            name = "model." + name.dropFirst("language_model.model.".count)
        } else if name.hasPrefix("language_model.") {
            name = String(name.dropFirst("language_model.".count))
        }
        // After the trunk's rename, so a tower spelled `model.language_model.visual.*` lands
        // here too instead of slipping through as an unexpected `model.visual.*`.
        if name.hasPrefix("model.visual.") {
            name = "visual." + name.dropFirst("model.visual.".count)
        } else if name.hasPrefix("vision_tower.") {
            name = "visual." + name.dropFirst("vision_tower.".count)
        }
        // A_log stays float32 at every precision, as in mlx-lm: g is computed there.
        parameters[name] =
            name.hasSuffix("A_log") ? array : precision.map { array.asType($0.dtype) } ?? array
    }

    if config.tied {
        parameters["lm_head.weight"] = nil
        parameters["lm_head.scales"] = nil
        parameters["lm_head.biases"] = nil
    }

    // The torch conv layout marks a raw HF checkpoint: its norms are still zero-centered
    // (scale = 1 + w, as in Gemma), so the shift bakes in here — after the cast, exactly
    // like transformers' float32 `1.0 + weight`.
    let firstLinear = config.layerTypes.firstIndex(of: "linear_attention")!
    let conv = "model.layers.\(firstLinear).linear_attn.conv1d.weight"
    let rawHF = parameters[conv]?.shape[1] == 1
    for (name, array) in parameters {
        if name.hasSuffix("conv1d.weight") {
            parameters[name] = array.squeezed(axis: rawHF ? 1 : 2)
        }
        if rawHF, isZeroCentered(name) {
            parameters[name] = array + 1
        }
    }

    // MLX Conv3d consumes [out, T, H, W, C]. Keep that layout in memory; raw HF stores
    // [out, C, T, H, W] and needs the same moveaxis mlx-vlm applies while loading.
    if let vision = config.visionConfig, let weight = parameters["visual.patch_embed.proj.weight"]
    {
        let mlx = weight.shape[1] == vision.temporalPatchSize
        parameters["visual.patch_embed.proj.weight"] =
            mlx ? weight : weight.transposed(0, 2, 3, 4, 1)
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

/// The RMSNormGated inside the DeltaNet is the one norm of the family that is not
/// zero-centered, so plain `norm.weight` suffixes must not match. Neither must the tower's,
/// which are LayerNorms with a bias and no shift at all — hence the exact match on the trunk's
/// final norm rather than a suffix.
private func isZeroCentered(_ name: String) -> Bool {
    name == "model.norm.weight"
        || name.hasSuffix("input_layernorm.weight")
        || name.hasSuffix("post_attention_layernorm.weight")
        || name.hasSuffix("q_norm.weight")
        || name.hasSuffix("k_norm.weight")
}

/// The depthwise conv and the whole vision tower ship bfloat16 even inside a quantized
/// checkpoint — no mlx conversion packs either — so they stay out of the packed-triple
/// rewrite below. The day a conversion does pack the tower, dropping the prefix here is the
/// only change: `Linear` keys off the presence of `.scales` per tensor, and
/// `Quantization.overrides` picks up new module keys on its own.
private func shipsDense(_ name: String) -> Bool {
    name.hasSuffix("conv1d.weight") || name.hasPrefix("visual.")
}

private func expectedShapes(_ config: Qwen35Config) -> [String: [Int]] {
    let hidden = config.hiddenSize
    // q_proj carries the output gate fused per head: twice the query width.
    let queries = config.numAttentionHeads * config.headDim
    let kv = config.numKeyValueHeads * config.headDim
    var shapes: [String: [Int]] = [
        "model.embed_tokens.weight": [config.vocabSize, hidden],
        "model.norm.weight": [hidden],
    ]
    if !config.tied {
        shapes["lm_head.weight"] = [config.vocabSize, hidden]
    }

    for (layer, type) in config.layerTypes.enumerated() {
        let p = "model.layers.\(layer)."
        shapes[p + "input_layernorm.weight"] = [hidden]
        shapes[p + "post_attention_layernorm.weight"] = [hidden]

        if type == "full_attention" {
            shapes[p + "self_attn.q_proj.weight"] = [queries * 2, hidden]
            shapes[p + "self_attn.k_proj.weight"] = [kv, hidden]
            shapes[p + "self_attn.v_proj.weight"] = [kv, hidden]
            shapes[p + "self_attn.o_proj.weight"] = [hidden, queries]
            shapes[p + "self_attn.q_norm.weight"] = [config.headDim]
            shapes[p + "self_attn.k_norm.weight"] = [config.headDim]
        } else {
            let heads = config.linearNumValueHeads
            shapes[p + "linear_attn.in_proj_qkv.weight"] = [config.convDim, hidden]
            shapes[p + "linear_attn.in_proj_z.weight"] = [config.valueDim, hidden]
            shapes[p + "linear_attn.in_proj_b.weight"] = [heads, hidden]
            shapes[p + "linear_attn.in_proj_a.weight"] = [heads, hidden]
            shapes[p + "linear_attn.conv1d.weight"] = [config.convDim, config.linearConvKernelDim]
            shapes[p + "linear_attn.A_log"] = [heads]
            shapes[p + "linear_attn.dt_bias"] = [heads]
            shapes[p + "linear_attn.norm.weight"] = [config.linearValueHeadDim]
            shapes[p + "linear_attn.out_proj.weight"] = [hidden, config.valueDim]
        }

        if config.mixtureOfExperts {
            let experts = config.numExperts
            let inner = config.moeIntermediateSize
            let shared = config.sharedExpertIntermediateSize
            shapes[p + "mlp.gate.weight"] = [experts, hidden]
            shapes[p + "mlp.switch_mlp.gate_proj.weight"] = [experts, inner, hidden]
            shapes[p + "mlp.switch_mlp.up_proj.weight"] = [experts, inner, hidden]
            shapes[p + "mlp.switch_mlp.down_proj.weight"] = [experts, hidden, inner]
            shapes[p + "mlp.shared_expert.gate_proj.weight"] = [shared, hidden]
            shapes[p + "mlp.shared_expert.up_proj.weight"] = [shared, hidden]
            shapes[p + "mlp.shared_expert.down_proj.weight"] = [hidden, shared]
            shapes[p + "mlp.shared_expert_gate.weight"] = [1, hidden]
        } else {
            let inner = config.intermediateSize!
            shapes[p + "mlp.gate_proj.weight"] = [inner, hidden]
            shapes[p + "mlp.up_proj.weight"] = [inner, hidden]
            shapes[p + "mlp.down_proj.weight"] = [hidden, inner]
        }
    }

    if let vision = config.visionConfig {
        let dim = vision.hiddenSize
        let inner = vision.intermediateSize
        shapes["visual.patch_embed.proj.weight"] = [
            dim, vision.temporalPatchSize, vision.patchSize, vision.patchSize, vision.inChannels,
        ]
        shapes["visual.patch_embed.proj.bias"] = [dim]
        shapes["visual.pos_embed.weight"] = [vision.numPositionEmbeddings, dim]

        for block in 0..<vision.depth {
            let p = "visual.blocks.\(block)."
            for norm in ["norm1", "norm2"] {
                shapes[p + norm + ".weight"] = [dim]
                shapes[p + norm + ".bias"] = [dim]
            }
            shapes[p + "attn.qkv.weight"] = [3 * dim, dim]
            shapes[p + "attn.qkv.bias"] = [3 * dim]
            shapes[p + "attn.proj.weight"] = [dim, dim]
            shapes[p + "attn.proj.bias"] = [dim]
            shapes[p + "mlp.linear_fc1.weight"] = [inner, dim]
            shapes[p + "mlp.linear_fc1.bias"] = [inner]
            shapes[p + "mlp.linear_fc2.weight"] = [dim, inner]
            shapes[p + "mlp.linear_fc2.bias"] = [dim]
        }

        // The merger normalizes each patch, then reads a spatial block of them as one row.
        let merged = vision.mergerHidden
        shapes["visual.merger.norm.weight"] = [dim]
        shapes["visual.merger.norm.bias"] = [dim]
        shapes["visual.merger.linear_fc1.weight"] = [merged, merged]
        shapes["visual.merger.linear_fc1.bias"] = [merged]
        shapes["visual.merger.linear_fc2.weight"] = [vision.outHiddenSize, merged]
        shapes["visual.merger.linear_fc2.bias"] = [vision.outHiddenSize]
    }

    // A quantized checkpoint stores every projection as the packed triple; what `shipsDense`
    // names does not. Packing runs along the last axis at any rank, so a stacked expert weight
    // quantizes like a 2D one — only the row length changes — and the bits are the module's
    // own: the routers of the MoE ride at 8 over a 6-bit body.
    if let q = config.quantization {
        for (name, shape) in shapes where shape.count >= 2 && !shipsDense(name) {
            let base = String(name.dropLast(".weight".count))
            let module = q.module(base)
            let last = shape.count - 1
            shapes[name] = shape.enumerated().map {
                $0.offset == last ? $0.element * module.bits / 32 : $0.element
            }
            let grouped = shape.enumerated().map {
                $0.offset == last ? $0.element / module.groupSize : $0.element
            }
            shapes[base + ".scales"] = grouped
            shapes[base + ".biases"] = grouped
        }
    }
    return shapes
}

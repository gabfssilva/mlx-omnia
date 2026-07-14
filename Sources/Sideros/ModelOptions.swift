import Foundation
import MLX

public enum Precision: Sendable {
    case float32
    case bfloat16
    case float16

    var dtype: DType {
        switch self {
        case .float32: .float32
        case .bfloat16: .bfloat16
        case .float16: .float16
        }
    }
}

/// MLX's affine scheme: per-group scale and bias along the contraction axis.
/// Also what a quantized checkpoint declares in its config.json — where the block may
/// carry, next to the defaults, one entry per module that was quantized differently
/// (Qwen3.6-35B-A3B ships its routers at 8 bits over a 6-bit body: a single wrong expert
/// costs more than the row it saves). `module` resolves a weight against them.
public struct Quantization: Decodable, Sendable {
    public var groupSize: Int
    public var bits: Int
    /// Keyed by the checkpoint's own path, which prefixes the house name — `module`
    /// matches on the suffix.
    public var overrides: [String: Quantization] = [:]

    public init(groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits
    }

    public func module(_ name: String) -> Quantization {
        overrides.first { $0.key.hasSuffix(name) }?.value ?? self
    }

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    /// The config is read with both key strategies — the raw one for the paths, which are
    /// not snake_case names but weight names — so both spellings of the two scalars are
    /// accepted. Every key that decodes as a nested block is an override.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Key.self)
        func scalar(_ name: String, _ converted: String) throws -> Int {
            if let value = try container.decodeIfPresent(Int.self, forKey: Key(stringValue: name)!) {
                return value
            }
            return try container.decode(Int.self, forKey: Key(stringValue: converted)!)
        }
        groupSize = try scalar("group_size", "groupSize")
        bits = try scalar("bits", "bits")
        for key in container.allKeys {
            if let override = try? container.decode(Quantization.self, forKey: key) {
                overrides[key.stringValue] = override
            }
        }
    }
}

/// Fixed for the lifetime of the model, unlike `GenerationOptions`.
public struct ModelOptions: Sendable {
    /// `nil` keeps whatever the checkpoint stores: bfloat16 for Qwen2 and Gemma 3,
    /// float32 for GPT-2. Converting is never free — bfloat16 has 8 bits of mantissa,
    /// so downcasting a float32 checkpoint moves the greedy path.
    public var precision: Precision?

    /// Quantizes a dense checkpoint at load. `nil` keeps whatever the checkpoint stores —
    /// asking for it on an already-quantized checkpoint is an error, not a requantization.
    public var quantization: Quantization?

    /// A smaller checkpoint that speculates for this one — it must share the tokenizer
    /// (Qwen3-0.6B drafting Qwen3-14B). Greedy generation then verifies several draft
    /// tokens per forward of the big model: the speedup comes from reading the big
    /// weights once per accepted run, and every emitted token is the target's own
    /// argmax. Setting a draft also swaps the target's 2–8-row forwards onto the skinny
    /// gemm kernel, which reads weights at gemv bandwidth where MLX's tile gemm loses
    /// ~40%. The output is deterministic and does not depend on the draft, but the
    /// kernel's last bit differs from `matmul`'s — at an exact logit tie the undrafted
    /// path can break the other way. Samplers other than `.greedy` ignore the draft.
    public var draft: URL?

    public init(
        precision: Precision? = nil, quantization: Quantization? = nil, draft: URL? = nil
    ) {
        self.precision = precision
        self.quantization = quantization
        self.draft = draft
    }
}

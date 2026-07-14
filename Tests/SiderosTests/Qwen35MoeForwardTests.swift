import Foundation
import MLX
import Testing

@testable import Sideros

func qwen35MoeDirectory() throws -> URL { try hubSnapshot("mlx-community--Qwen3.6-35B-A3B-6bit") }

private func loadQwen35Moe() throws -> (Qwen35Config, Qwen35, [String: MLXArray]) {
    let directory = try qwen35MoeDirectory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: nil)
    return (
        config,
        Qwen35(config: config, parameters: parameters, quantization: config.quantization),
        parameters
    )
}

/// The routers of this checkpoint sit at 8 bits over a 6-bit body — an override the
/// quantization block spells per module, and the only way the packed rows add up.
@Test func qwen35MoeConfigMatchesCheckpoint() throws {
    let config = try Qwen35Config(directory: qwen35MoeDirectory())

    #expect(config.numHiddenLayers == 40)
    #expect(config.mixtureOfExperts)
    #expect(config.numExperts == 256)
    #expect(config.numExpertsPerTok == 8)
    #expect(config.moeIntermediateSize == 512)
    #expect(config.sharedExpertIntermediateSize == 512)
    #expect(config.intermediateSize == nil)

    let quantization = try #require(config.quantization)
    #expect(quantization.bits == 6)
    #expect(quantization.groupSize == 64)
    #expect(quantization.module("model.layers.7.mlp.gate").bits == 8)
    #expect(quantization.module("model.layers.7.mlp.shared_expert_gate").bits == 8)
    #expect(quantization.module("model.layers.7.mlp.switch_mlp.down_proj").bits == 6)
    #expect(quantization.module("lm_head").bits == 6)
}

/// No transformers ground truth exists at this size (float32 of 35B is ~140GB), so mlx-lm
/// over the same packed weights is the reference, bounded by what bfloat16 rounding costs
/// mlx-lm itself against the exact forward of the same graph. Per layer, so a divergence
/// names where it started: the recurrent state of a DeltaNet layer carries drift forward.
/// Each layer is held to its own floor — early in the trunk the residual stream is small
/// and the same absolute rounding is a much larger fraction of it (31% at block 0, 0.1% at
/// the end), so one floor for all of them would be vacuous at one end and impossible at
/// the other.
///
/// Plus four ulps of bfloat16, and this one is not slack: the l2norm of the DeltaNet is
/// deliberately not mlx-lm's (eps inside the sum, as transformers spells it, against their
/// rms_norm times a scale — the 0.8B's float32 fixture is what pins that choice). The gap
/// it opens is below the last bit of the input, and forty layers of recurrence carry it to
/// a couple of ulps of the block's largest element. A wrong expert, a dropped shared
/// expert or a mis-rotated head moves these blocks by order one, not by a bit.
@Test func qwen35MoeLogitsMatchMLXLM() throws {
    let (config, model, _) = try loadQwen35Moe()
    let golden = try fixture("qwen3_5_moe_mlxlm.safetensors")
    let activations = model(golden["input_ids"]!.asArray(Int32.self))
    let ulp: Float = 4 * 0x1p-8

    for layer in 0..<config.numHiddenLayers {
        let floor = golden["noise.block_\(layer)"]!.item(Float.self)
        #expect(
            relativeDiff(activations.blocks[layer], golden["block_\(layer)"]!)
                < max(3 * floor, ulp),
            "block \(layer)")
    }
    #expect(
        relativeDiff(activations.logits.asType(.float32), golden["logits"]!)
            < golden["noise.logits"]!.item(Float.self))
    #expect(
        argMax(activations.logits, axis: -1).asArray(Int32.self)
            == argMax(golden["logits"]!, axis: -1).asArray(Int32.self))
}

/// The sparse block spelled out from the golden input above it: the router over all 256
/// experts, the eight it keeps, their renormalized weights, and the shared expert every
/// token pays for. A failure names the piece instead of the layer.
@Test func qwen35MoeSparseBlockInternalsMatchMLXLM() throws {
    let (config, model, parameters) = try loadQwen35Moe()
    let golden = try fixture("qwen3_5_moe_mlxlm.safetensors")
    let quantization = config.quantization
    let p = "model.layers.0.mlp."
    let x = golden["b0_ln_2"]!.asType(.bfloat16)

    func linear(_ name: String) -> Linear {
        Linear(p + name, parameters: parameters, quantization: quantization)
    }

    let probs = softmax(linear("gate")(x), axis: -1, precise: true)
    let split = config.numExperts - config.numExpertsPerTok
    let chosen = argPartition(probs, kth: split, axis: -1)[0..., 0..., split...]
    let kept = takeAlong(probs, chosen, axis: -1)
    let weights = kept / kept.sum(axis: -1, keepDims: true)

    let gated = linear("shared_expert.gate_proj")(x)
    let shared =
        sigmoid(linear("shared_expert_gate")(x))
        * linear("shared_expert.down_proj")(
            gated * sigmoid(gated) * linear("shared_expert.up_proj")(x))

    #expect(relativeDiff(probs, golden["b0_moe_probs"]!) < 1e-5)
    #expect(
        chosen.asType(.int32).asArray(Int32.self)
            == golden["b0_moe_indices"]!.asType(.int32).asArray(Int32.self))
    #expect(relativeDiff(weights, golden["b0_moe_weights"]!) < 1e-5)
    #expect(relativeDiff(shared, golden["b0_moe_shared"]!) < 1e-5)
    #expect(relativeDiff(model.mlps[0](x), golden["b0_mlp"]!) < 1e-5)
}

/// Two routing paths exist: the plain gather, and the argsort/unsort reorder the prefill
/// takes once `length * k >= 64`. The 5-token prompt routes 40 rows and never reaches it;
/// these 21 route 168 and do, while stepping the same ids through the cache runs them one
/// by one down the plain path. A broken unsort shows up here and leaves the greedy argmax
/// of the other tests untouched.
///
/// The bound is not zero and cannot be: in bfloat16 a 21-row matmul rounds differently from
/// 21 one-row matmuls, so mlx-lm disagrees with itself across these same two paths by
/// `noise.batching`. That measured gap is the floor; the house tolerance of 3x sits on top.
@Test func qwen35MoeSortedGatherMatchesTheStepwisePath() throws {
    let golden = try fixture("qwen3_5_moe_mlxlm.safetensors")
    let ids = golden["greedy_ids"]!.asArray(Int32.self)
    #expect(ids.count * 8 >= 64)

    let (_, model, _) = try loadQwen35Moe()
    let prefill = model(ids).logits

    var cache = model.makeCache()
    var stepwise: [MLXArray] = []
    for (offset, id) in ids.enumerated() {
        let (logits, grown) = model.logits(model.embed([id], offset: offset), cache: cache)
        cache = grown
        stepwise.append(logits)
    }

    let gap = relativeDiff(prefill, concatenated(stepwise, axis: 1))
    #expect(gap < 3 * golden["noise.batching"]!.item(Float.self))
}

/// A stale conv window, recurrent state or KV cache does not have to move a degenerate
/// argmax; whole logits do not survive it.
@Test func qwen35MoeCachedGenerationMatchesRecomputation() throws {
    let ids = try fixture("qwen3_5_moe_mlxlm.safetensors")["input_ids"]!.asArray(Int32.self)
    let (_, model, _) = try loadQwen35Moe()

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

/// End to end through the facade: routing, the shared expert, the quantized gathers and
/// the hybrid cache together, against mlx-lm's own greedy continuation.
@Test func qwen35MoeGreedyThroughTheFacadeMatchesMLXLM() async throws {
    let directory = try qwen35MoeDirectory()
    let golden = try fixture("qwen3_5_moe_mlxlm.safetensors")
    let input = golden["input_ids"]!.asArray(Int32.self)
    let greedy = golden["greedy_ids"]!.asArray(Int32.self)
    let tokenizer = try Qwen2Tokenizer(directory: directory)

    let model = try LanguageModel(directory: directory)
    let generated = try await model.generate(
        tokenizer.decode(input),
        options: GenerationOptions(maxTokens: greedy.count - input.count, stop: []))

    #expect(generated == tokenizer.decode(Array(greedy.dropFirst(input.count))))
}

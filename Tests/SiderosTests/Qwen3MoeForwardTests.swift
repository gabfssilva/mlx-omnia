import Foundation
import MLX
import Testing

@testable import Sideros

func qwen3MoeDirectory() throws -> URL { try hubSnapshot("mlx-community--Qwen3-30B-A3B-4bit") }

private func loadQwen3Moe() throws -> Qwen3Moe {
let directory = try qwen3MoeDirectory()
let config = try Qwen3MoeConfig(directory: directory)
return Qwen3Moe(
    config: config,
    parameters: try loadQwen3MoeParameters(
        directory: directory, config: config, precision: nil),
    quantization: config.quantization)
}

/// No transformers ground truth exists at this size (float32 of 30B is ~120GB), so
/// mlx-lm over the same packed weights is the reference, bounded by what bfloat16
/// rounding costs mlx-lm itself against the exact forward of the same graph.
@Test func qwen3MoeLogitsMatchMLXLM() throws {
    let golden = try fixture("qwen3_moe_mlxlm.safetensors")
    let model = try loadQwen3Moe()
    let logits = model(golden["input_ids"]!.asArray(Int32.self)).logits.asType(.float32)

    #expect(relativeDiff(logits, golden["logits"]!) < golden["noise.logits"]!.item(Float.self))
}

/// Two routing paths exist: the plain gather, and the argsort/unsort reorder the
/// prefill takes once `length * k >= 64`. A 5-token prompt routes 40 rows and never
/// reaches it; these 21 route 168 and do, while stepping the same ids through the
/// cache runs them one by one down the plain path. A broken unsort shows up as a logit
/// gap here and leaves the greedy argmax of the other tests untouched.
///
/// The bound is not zero and cannot be: in bfloat16 a 21-row matmul rounds differently
/// from 21 one-row matmuls, so mlx-lm disagrees with itself across these same two
/// paths by `noise.batching`. That measured gap is the floor, and the house tolerance
/// of 3x sits on top of it — a dropped unsort feeds every token another token's
/// experts and misses by order 1, not by a factor of three.
@Test func qwen3MoeSortedGatherMatchesTheStepwisePath() throws {
    let golden = try fixture("qwen3_moe_mlxlm.safetensors")
    let ids = golden["greedy_ids"]!.asArray(Int32.self)
    #expect(ids.count * 8 >= 64)

    let model = try loadQwen3Moe()
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

/// A cache that mismatches the prefill does not have to move a degenerate argmax;
/// whole logits do not survive it.
@Test func qwen3MoeCachedGenerationMatchesRecomputation() throws {
    let ids = try fixture("qwen3_moe_mlxlm.safetensors")["input_ids"]!.asArray(Int32.self)

    let model = try loadQwen3Moe()

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

/// End to end through the facade: routing, the quantized gathers and the cache
/// together.
@Test func qwen3MoeGreedyThroughTheFacadeMatchesMLXLM() async throws {
    let directory = try qwen3MoeDirectory()
    let golden = try fixture("qwen3_moe_mlxlm.safetensors")
    let input = golden["input_ids"]!.asArray(Int32.self)
    let greedy = golden["greedy_ids"]!.asArray(Int32.self)
    let tokenizer = try Qwen2Tokenizer(directory: directory)

    let model = try LanguageModel(directory: directory)
    let generated = try await model.generate(
        tokenizer.decode(input),
        options: GenerationOptions(maxTokens: greedy.count - input.count, stop: []))

    #expect(generated == tokenizer.decode(Array(greedy.dropFirst(input.count))))
}

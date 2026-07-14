import Foundation
import MLX
import Testing

@testable import Sideros

/// Prefill once, then one token per step against the cache.
func cachedGreedy(_ model: some CausalLM, _ ids: [Int32], newTokens: Int) -> [Int32] {
    var cache = model.makeCache()
    var tokens = ids
    var pending = ids

    for _ in 0..<newTokens {
        let offset = cache[0].offset
        let (logits, grown) = model.logits(model.embed(pending, offset: offset), cache: cache)
        cache = grown

        let next = argMax(logits[0, -1], axis: -1).item(Int32.self)
        tokens.append(next)
        pending = [next]
    }
    return tokens
}

@Test func gpt2CachedGenerationMatchesRecomputation() throws {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let model = GPT2(
        config: config,
        parameters: try loadGPT2Parameters(directory: directory, config: config, precision: .float32))
    let ids = try fixture("gpt2_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

@Test func qwen2CachedGenerationMatchesRecomputation() throws {
    let directory = try qwen2Directory()
    let config = try Qwen2Config(directory: directory)
    let model = Qwen2(
        config: config,
        parameters: try loadQwen2Parameters(directory: directory, config: config, precision: .float32))
    let ids = try fixture("qwen2_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

@Test func gemma3CachedGenerationMatchesRecomputation() throws {
    let directory = try gemma3Directory()
    let config = try Gemma3Config(directory: directory)
    let model = Gemma3(
        config: config,
        parameters: try loadGemma3Parameters(directory: directory, config: config, precision: .float32))
    let ids = try fixture("gemma3_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

/// Past the sliding window a cached query must still see exactly the last 512 keys, and
/// the mask is the only thing enforcing it: the cache itself holds every key.
@Test func gemma3CachedGenerationMatchesRecomputationBeyondTheSlidingWindow() throws {
    let directory = try gemma3Directory()
    let config = try Gemma3Config(directory: directory)
    let model = Gemma3(
        config: config,
        parameters: try loadGemma3Parameters(directory: directory, config: config, precision: .float32))
    let ids = try fixture("gemma3_forward.safetensors")["long_input_ids"]!.asArray(Int32.self)

    #expect(ids.count > config.slidingWindow)
    #expect(cachedGreedy(model, ids, newTokens: 4) == model.generate(ids, maxNewTokens: 4))
}

/// The cache buffer grows in blocks of 256, and growth has to carry the rows already in
/// it. Nothing else reaches past one block: the model tests prefill a few hundred tokens
/// at most and then step a handful, so the copy on the block boundary would go untested
/// and a generation that crossed 256 would silently read zeros.
@Test func kvCacheKeepsItsRowsAcrossBlockGrowth() {
    let heads = 2, dim = 4
    func row(_ i: Int) -> MLXArray {
        MLXArray(Array(repeating: Float(i), count: heads * dim)).reshaped(1, heads, 1, dim)
    }

    var cache = KVCache()
    var appended: [MLXArray] = []
    var keys = MLXArray([] as [Float])
    for i in 0..<300 {
        appended.append(row(i))
        (keys, _, cache) = cache.appending(keys: row(i), values: row(i))
    }

    #expect(cache.offset == 300)
    #expect(keys.shape == [1, heads, 300, dim])
    #expect(allClose(keys, concatenated(appended, axis: 2)).item(Bool.self))
}

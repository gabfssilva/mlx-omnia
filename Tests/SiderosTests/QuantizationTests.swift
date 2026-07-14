import Foundation
import MLX
import Testing

@testable import Sideros

private let q4 = Quantization()

func qwen2Q4Directory() throws -> URL { try hubSnapshot("mlx-community--Qwen2.5-0.5B-Instruct-4bit") }

/// Bit-exact against mx.quantize of the same tensors: packing is either identical or
/// wrong. This is what catches a wrong group size or an untransposed Conv1D — mistakes
/// that still generate fluent text.
@Test func quantizeOnLoadPacksBitExactly() throws {
    let golden = try fixture("quantize.safetensors")

    let qwen2Dir = try qwen2Directory()
    let qwen2Config = try Qwen2Config(directory: qwen2Dir)
    let qwen2 = try loadQwen2Parameters(
        directory: qwen2Dir, config: qwen2Config, precision: nil, quantization: q4)

    let gemma3Dir = try gemma3Directory()
    let gemma3Config = try Gemma3Config(directory: gemma3Dir)
    let gemma3 = try loadGemma3Parameters(
        directory: gemma3Dir, config: gemma3Config, precision: nil, quantization: q4)

    let gpt2Dir = try gpt2Directory()
    let gpt2Config = try GPT2Config(directory: gpt2Dir)
    let gpt2 = try loadGPT2Parameters(
        directory: gpt2Dir, config: gpt2Config, precision: nil, quantization: q4)

    let cases: [(String, [String: MLXArray], String)] = [
        ("qwen2", qwen2, "model.layers.0.self_attn.q_proj"),
        ("qwen2", qwen2, "model.layers.0.mlp.down_proj"),
        ("gemma3", gemma3, "model.layers.0.self_attn.q_proj"),
        ("gpt2", gpt2, "h.0.attn.c_attn"),
        ("gpt2", gpt2, "h.0.mlp.c_fc"),
    ]
    for (prefix, parameters, base) in cases {
        for suffix in [".weight", ".scales", ".biases"] {
            let expected = golden["\(prefix).\(base)\(suffix)"]!
            let actual = parameters[base + suffix]!
            #expect(
                arrayEqual(actual, expected).item(Bool.self),
                "\(prefix).\(base)\(suffix)")
        }
    }
}

/// Same checkpoint, same kernels, both forwards in float16: we must land at least as
/// close to mlx-lm as its own float16 forward lands from the exact float32 one
/// (`noise.logits`, measured at fixture generation). Measured: 6.8e-3 against a floor
/// of 2.8e-2.
@Test func quantizedCheckpointLogitsMatchMLXLM() throws {
    let directory = try qwen2Q4Directory()
    let config = try Qwen2Config(directory: directory)
    #expect(config.quantization != nil)

    let model = Qwen2(
        config: config,
        parameters: try loadQwen2Parameters(directory: directory, config: config, precision: nil),
        quantization: config.quantization)

    let golden = try fixture("qwen2_q4_mlxlm.safetensors")
    let ids = golden["input_ids"]!.asArray(Int32.self)
    let logits = model(ids).logits.asType(.float32)

    #expect(relativeDiff(logits, golden["logits"]!) < golden["noise.logits"]!.item(Float.self))
}

/// End to end: config-declared quantization flows through the facade untouched.
@Test func quantizedCheckpointGreedyThroughTheFacadeMatchesMLXLM() async throws {
    let directory = try qwen2Q4Directory()
    let golden = try fixture("qwen2_q4_mlxlm.safetensors")
    let input = golden["input_ids"]!.asArray(Int32.self)
    let greedy = golden["greedy_ids"]!.asArray(Int32.self)
    let tokenizer = try Qwen2Tokenizer(directory: directory)

    let model = try LanguageModel(directory: directory)
    let generated = try await model.generate(
        tokenizer.decode(input),
        options: GenerationOptions(maxTokens: greedy.count - input.count, stop: []))

    #expect(generated == tokenizer.decode(Array(greedy.dropFirst(input.count))))
}

/// The quantized kernels and the cache agree on GPT-2's transposed Conv1D path.
@Test func quantizeOnLoadCachedGenerationMatchesRecomputation() throws {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let model = GPT2(
        config: config,
        parameters: try loadGPT2Parameters(
            directory: directory, config: config, precision: nil, quantization: q4),
        quantization: q4)
    let ids = try fixture("gpt2_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    #expect(cachedGreedy(model, ids, newTokens: 8) == model.generate(ids, maxNewTokens: 8))
}

/// Requantizing or reconverting a quantized checkpoint would silently produce a
/// different model than the one on disk.
@Test func touchingAQuantizedCheckpointThrows() throws {
    let directory = try qwen2Q4Directory()
    let config = try Qwen2Config(directory: directory)

    #expect(throws: Qwen2LoadError.alreadyQuantized) {
        try loadQwen2Parameters(
            directory: directory, config: config, precision: nil, quantization: q4)
    }
    #expect(throws: Qwen2LoadError.alreadyQuantized) {
        try loadQwen2Parameters(directory: directory, config: config, precision: .float32)
    }
}

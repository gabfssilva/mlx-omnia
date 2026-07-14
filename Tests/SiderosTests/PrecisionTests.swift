import Foundation
import MLX
import Testing

@testable import Sideros

/// Qwen2 and Gemma 3 ship as bfloat16, so loading them as bfloat16 *is* the checkpoint and
/// float32 is its lossless upcast. Neither the logits nor the greedy path may move.
private func expectGreedyIsPrecisionInvariant(directory: URL, prompt: String) async throws {
    let options = GenerationOptions(maxTokens: 20, stop: [])

    let exact = try await LanguageModel(directory: directory, options: fp32)
        .generate(prompt, options: options)
    let reduced = try await LanguageModel(
        directory: directory, options: ModelOptions(precision: .bfloat16)
    ).generate(prompt, options: options)

    #expect(reduced == exact)
}

@Test func qwen2GreedySurvivesBfloat16() async throws {
    try await expectGreedyIsPrecisionInvariant(
        directory: qwen2Directory(), prompt: "The capital of France is")
}

@Test func gemma3GreedySurvivesBfloat16() async throws {
    try await expectGreedyIsPrecisionInvariant(
        directory: gemma3Directory(), prompt: "The capital of France is")
}

/// Asking for nothing converts nothing: GPT-2 is stored in float32, the other two in
/// bfloat16, and that is what a default `LanguageModel` runs in.
@Test func defaultPrecisionIsWhateverTheCheckpointStores() throws {
    let gpt2Directory = try gpt2Directory()
    let qwen2Directory = try qwen2Directory()
    let gemma3Directory = try gemma3Directory()

    let gpt2 = try loadGPT2Parameters(
        directory: gpt2Directory, config: try GPT2Config(directory: gpt2Directory), precision: nil)
    let qwen2 = try loadQwen2Parameters(
        directory: qwen2Directory, config: try Qwen2Config(directory: qwen2Directory), precision: nil)
    let gemma3 = try loadGemma3Parameters(
        directory: gemma3Directory, config: try Gemma3Config(directory: gemma3Directory),
        precision: nil)

    #expect(gpt2["wte.weight"]!.dtype == .float32)
    #expect(qwen2["model.embed_tokens.weight"]!.dtype == .bfloat16)
    #expect(gemma3["model.embed_tokens.weight"]!.dtype == .bfloat16)
}

/// A float32 constant anywhere in the graph — an additive mask, a scale — silently drags
/// the forward back up, and the greedy tests above would not notice.
@Test func bfloat16ForwardStaysBfloat16() throws {
    let gpt2Directory = try gpt2Directory()
    let gpt2Config = try GPT2Config(directory: gpt2Directory)
    let gpt2 = GPT2(
        config: gpt2Config,
        parameters: try loadGPT2Parameters(
            directory: gpt2Directory, config: gpt2Config, precision: .bfloat16))

    let qwen2Directory = try qwen2Directory()
    let qwen2Config = try Qwen2Config(directory: qwen2Directory)
    let qwen2 = Qwen2(
        config: qwen2Config,
        parameters: try loadQwen2Parameters(
            directory: qwen2Directory, config: qwen2Config, precision: .bfloat16))

    let gemma3Directory = try gemma3Directory()
    let gemma3Config = try Gemma3Config(directory: gemma3Directory)
    let gemma3 = Gemma3(
        config: gemma3Config,
        parameters: try loadGemma3Parameters(
            directory: gemma3Directory, config: gemma3Config, precision: .bfloat16))

    #expect(gpt2([15496, 11]).logits.dtype == .bfloat16)
    #expect(qwen2([785, 6722]).logits.dtype == .bfloat16)
    #expect(gemma3([2, 818]).logits.dtype == .bfloat16)
}

/// GPT-2 is the one checkpoint stored in float32, so asking for bfloat16 discards 16 bits
/// of mantissa. The greedy path does drift — top-1 and top-2 sit ~0.19 logit apart, well
/// inside the error — which is precision, not a bug: the logits land at 8.0e-3 relative,
/// about twice bfloat16's epsilon (2^-8 = 3.9e-3) after twelve blocks of accumulation,
/// and the first token still agrees. The bound is 3x the epsilon, as everywhere else.
@Test func gpt2Bfloat16LogitsStayWithinItsEpsilon() throws {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let ids = try fixture("gpt2_forward.safetensors")["input_ids"]!.asArray(Int32.self)

    let exact = GPT2(
        config: config,
        parameters: try loadGPT2Parameters(directory: directory, config: config, precision: .float32))
    let reduced = GPT2(
        config: config,
        parameters: try loadGPT2Parameters(directory: directory, config: config, precision: .bfloat16))

    let exactLogits = exact(ids).logits[0, -1]
    let reducedLogits = reduced(ids).logits[0, -1]

    #expect(exactLogits.dtype == .float32)
    #expect(reducedLogits.dtype == .bfloat16)
    #expect(relativeDiff(exactLogits, reducedLogits.asType(.float32)) < 3 * 3.9e-3)
    #expect(
        argMax(exactLogits, axis: -1).item(Int32.self)
            == argMax(reducedLogits, axis: -1).item(Int32.self))
}

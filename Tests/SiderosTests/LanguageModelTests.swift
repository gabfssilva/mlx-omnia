import Foundation
import MLX
import Testing

@testable import Sideros

/// The goldens are float32; the runtime default is bfloat16.
let fp32 = ModelOptions(precision: .float32)

/// End to end through the facade: the prompt must tokenize to the fixture's input_ids and
/// the greedy stream must spell out the tokens transformers generated from them.
private func expectGreedyMatchesFixture(
    directory: URL, fixture name: String, prompt: String, tokenizer: any Tokenizer
) async throws {
    let golden = try fixture(name)
    let input = golden["input_ids"]!.asArray(Int32.self)
    let greedy = golden["greedy_ids"]!.asArray(Int32.self)

    #expect(tokenizer.encode(prompt) == input)

    let model = try LanguageModel(directory: directory, options: fp32)
    let generated = try await model.generate(
        prompt, options: GenerationOptions(maxTokens: greedy.count - input.count, stop: []))

    #expect(generated == tokenizer.decode(Array(greedy.dropFirst(input.count))))
}

@Test func gpt2GreedyThroughTheFacadeMatchesTransformers() async throws {
    let directory = try gpt2Directory()
    try await expectGreedyMatchesFixture(
        directory: directory, fixture: "gpt2_forward.safetensors", prompt: "Hello, my name is",
        tokenizer: try GPT2Tokenizer(
            vocab: directory.appending(path: "vocab.json"),
            merges: directory.appending(path: "merges.txt")))
}

@Test func qwen2GreedyThroughTheFacadeMatchesTransformers() async throws {
    let directory = try qwen2Directory()
    try await expectGreedyMatchesFixture(
        directory: directory, fixture: "qwen2_forward.safetensors",
        prompt: "The capital of France is", tokenizer: try Qwen2Tokenizer(directory: directory))
}

@Test func gemma3GreedyThroughTheFacadeMatchesTransformers() async throws {
    let directory = try gemma3Directory()
    try await expectGreedyMatchesFixture(
        directory: directory, fixture: "gemma3_forward.safetensors",
        prompt: "The capital of France is", tokenizer: try Gemma3Tokenizer(directory: directory))
}

/// A stop token ends the stream instead of being spelled out: generation halts at the
/// third greedy token, so only the two before it reach the caller.
@Test func streamStopsOnTheStopToken() async throws {
    let golden = try fixture("gpt2_forward.safetensors")
    let greedy = golden["greedy_ids"]!.asArray(Int32.self)
    let prompt = golden["input_ids"]!.asArray(Int32.self).count

    let tokenizer = try GPT2Tokenizer(
        vocab: gpt2Directory().appending(path: "vocab.json"),
        merges: gpt2Directory().appending(path: "merges.txt"))
    let model = try LanguageModel(directory: gpt2Directory(), options: fp32)

    let text = try await model.generate(
        "Hello, my name is",
        options: GenerationOptions(maxTokens: 20, stop: [greedy[prompt + 2]]))

    #expect(text == tokenizer.decode(Array(greedy[prompt..<(prompt + 2)])))
}

/// Collects the whole `stream`: the text it spelled and the `.done` that closed it.
private func drain(_ stream: AsyncThrowingStream<Completion, Error>) async throws -> (
    text: String, finish: Finish?
) {
    var text = ""
    var finish: Finish?
    for try await item in stream {
        switch item {
        case .start: break
        case .delta(let delta): text += delta
        case .done(let done): finish = done
        }
    }
    return (text, finish)
}

/// What an API server reports back: `maxTokens` ran out, not a stop token, and the token
/// counts are the ones it bills.
@Test func streamReportsUsageAndLength() async throws {
    let prompt = "Hello, my name is"
    let tokenizer = try GPT2Tokenizer(
        vocab: gpt2Directory().appending(path: "vocab.json"),
        merges: gpt2Directory().appending(path: "merges.txt"))
    let model = try LanguageModel(directory: gpt2Directory(), options: fp32)

    let (text, finish) = try await drain(
        model.stream(prompt, options: GenerationOptions(maxTokens: 5, stop: [])))

    #expect(!text.isEmpty)
    #expect(finish?.reason == .length)
    #expect(finish?.completionTokens == 5)
    #expect(finish?.promptTokens == tokenizer.encode(prompt).count)
}

/// A stop string ends the stream at a text boundary the tokens know nothing about, and is
/// itself cut out of the output.
@Test func streamCutsTheStopString() async throws {
    let prompt = "Hello, my name is"
    let model = try LanguageModel(directory: gpt2Directory(), options: fp32)
    let options = GenerationOptions(maxTokens: 20, stop: [])

    let full = try await model.generate(prompt, options: options)
    let stop = String(full.dropFirst(6).prefix(5))
    let cut = full.range(of: stop)!.lowerBound

    var stopped = options
    stopped.stopStrings = [stop]
    let (text, finish) = try await drain(model.stream(prompt, options: stopped))

    #expect(text == String(full[full.startIndex..<cut]))
    #expect(finish?.reason == .stop)
    #expect(finish!.completionTokens < 20)
}

/// The draft only proposes; every emitted token is the target's own argmax, so swapping
/// the draft — the target itself (all-accept path: trim boundaries, the draft running a
/// token behind) for a quantized one (rejection mid-run) — must not move one token.
/// Equality against the *undrafted* output is not promised: with a draft the target's
/// narrow forwards run the skinny kernel, whose last bit can break an exact logit tie
/// the other way. The scripted-draft test pins the acceptance rule itself.
@Test func speculatedOutputDoesNotDependOnTheDraft() async throws {
    let directory = try qwen2Directory()
    let options = GenerationOptions(maxTokens: 24, stop: [])

    var outputs: [String] = []
    for draft in try [directory, hubSnapshot("mlx-community--Qwen2.5-0.5B-Instruct-4bit")] {
        outputs.append(
            try await LanguageModel(directory: directory, options: ModelOptions(draft: draft))
                .generate("The capital of France is", options: options))
    }

    #expect(!outputs[0].isEmpty)
    #expect(outputs[0] == outputs[1])
}

/// Sampling is only reproducible through the seed; without one the draw is free.
@Test func seededSamplingIsDeterministic() async throws {
    let model = try LanguageModel(directory: gpt2Directory(), options: fp32)
    let options = GenerationOptions(
        maxTokens: 8, sampler: .topP(0.9, temperature: 0.8), stop: [], seed: 42)

    let first = try await model.generate("Hello, my name is", options: options)
    let second = try await model.generate("Hello, my name is", options: options)

    #expect(first == second)
    #expect(!first.isEmpty)
}

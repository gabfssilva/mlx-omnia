import Foundation
import MLX
import Testing

@testable import Sideros

/// Emits a predetermined sequence, one token per forward: the number of ids it is given
/// is the step. Lets the generation loop be driven without a checkpoint behind it.
private struct ScriptedDecoder: CausalLM {
    let script: [Int32]
    let vocabSize: Int

    func embed(_ tokens: MLXArray, offset: Int) -> MLXArray { tokens }

    func makeCache() -> [KVCache] { [KVCache()] }

    /// Grows the cache like a real layer would, so the offset counts the tokens consumed
    /// and doubles as the index into the script — one row per input position, which is
    /// what speculative verification reads. The clamp keeps a speculation round that
    /// overshoots the script's end deterministic instead of out of bounds.
    func logits(_ embeddings: MLXArray, cache: [KVCache]) -> (logits: MLXArray, cache: [KVCache]) {
        let step = cache[0].offset
        let length = embeddings.dim(1)
        let consumed = MLXArray.zeros([1, 1, length, 1])
        let (_, _, grown) = cache[0].appending(keys: consumed, values: consumed)

        var rows = [Float](repeating: 0, count: length * vocabSize)
        for position in 0..<length {
            rows[position * vocabSize + Int(script[min(step + position, script.count - 1)])] = 1
        }
        return (MLXArray(rows, [1, length, vocabSize]), [grown])
    }
}

/// Gemma spells a character outside its vocabulary as one token per UTF-8 byte. None of
/// them decodes on its own, so the stream must stay silent until the last one lands.
@Test func bytesOfOneCharacterAreHeldBackUntilItDecodes() throws {
    let directory = try gemma3Directory()
    let tokenizer = try Gemma3Tokenizer(directory: directory)
    let config = try Gemma3Config(directory: directory)

    let bytes = Array(tokenizer.encode("🜁").dropFirst())  // the <bos> is not generated
    #expect(bytes.count == 4)

    let generation = Generation(
        decoder: ScriptedDecoder(script: bytes, vocabSize: config.vocabSize),
        tokenizer: tokenizer, prompt: Prompt(ids: [0]), stop: [],
        options: GenerationOptions(maxTokens: bytes.count))

    var chunks: [String] = []
    while let chunk = generation.next() { chunks.append(chunk) }

    #expect(chunks == ["🜁"])
}

/// The draft disagrees with the target's script at chosen positions — at a round's first
/// draft, mid-run, and in runs that survive whole. Whatever it proposes, the stream must
/// spell the target's script token for token: speculation may only change speed.
@Test func speculationNeverChangesTheTargetScript() throws {
    let directory = try qwen2Directory()
    let tokenizer = try Qwen2Tokenizer(directory: directory)
    let config = try Qwen2Config(directory: directory)

    let target = tokenizer.encode(" Paris is the capital of the region and the seat")
    var noise = target
    noise[3] = target[2]
    noise[7] = target[0]
    // Shifted by one: wrong at every position, so every round rejects its first draft.
    let hostile = Array(target.dropFirst()) + [target[0]]

    for draft in [noise, hostile] {
        let generation = Generation(
            decoder: ScriptedDecoder(script: target, vocabSize: config.vocabSize),
            draft: ScriptedDecoder(script: draft, vocabSize: config.vocabSize),
            tokenizer: tokenizer, prompt: Prompt(ids: [0]), stop: [],
            options: GenerationOptions(maxTokens: target.count))

        var chunks: [String] = []
        while let chunk = generation.next() { chunks.append(chunk) }

        #expect(chunks.joined() == tokenizer.decode(target))
    }
}

/// Each chunk is the delta, not the text so far.
@Test func chunksConcatenateIntoTheDecodedText() throws {
    let directory = try qwen2Directory()
    let tokenizer = try Qwen2Tokenizer(directory: directory)
    let config = try Qwen2Config(directory: directory)

    let script = tokenizer.encode(" Paris is the capital")
    let generation = Generation(
        decoder: ScriptedDecoder(script: script, vocabSize: config.vocabSize),
        tokenizer: tokenizer, prompt: Prompt(ids: [0]), stop: [],
        options: GenerationOptions(maxTokens: script.count))

    var chunks: [String] = []
    while let chunk = generation.next() { chunks.append(chunk) }

    #expect(chunks.count == script.count)
    #expect(chunks.joined() == " Paris is the capital")
}

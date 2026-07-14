import Foundation
import MLX
import Testing

@testable import Sideros

private func loadDecoder() throws -> (Qwen35Config, Qwen35) {
    let directory = try qwen35Directory()
    let config = try Qwen35Config(directory: directory)
    let model = Qwen35(
        config: config,
        parameters: try loadQwen35Parameters(
            directory: directory, config: config, precision: .float32),
        quantization: config.quantization)
    return (config, model)
}

private func loadFacade() throws -> LanguageModel {
    try LanguageModel(
        directory: try qwen35Directory(), options: ModelOptions(precision: .float32))
}

private func fixtureImage(_ golden: [String: MLXArray]) -> Segment {
    let shape = golden["image_rgb"]!.shape
    return .image(
        .pixels(
            Pixels(
                width: shape[1], height: shape[0],
                rgb: golden["image_rgb"]!.asArray(UInt8.self))))
}

/// With the three sections carrying the same position the select is provably a no-op, so the
/// explicit path and the scalar one are not close — they are *identical*. They have to be: both
/// are the same rope kernel, called with the same integers. A hand-rolled cos/sin could not
/// pass this (the kernel uses `metal::fast::cos`, the elementwise op `metal::precise::cos`), and
/// that alone is the argument for the three-call trick.
@Test func qwen35ExplicitPositionsDegenerateToTheScalarPath() throws {
    let (_, model) = try loadDecoder()
    let golden = try fixture("qwen3_5_forward.safetensors")
    let ids = golden["input_ids"]!.asArray(Int32.self)
    let embeddings = model.embed(ids)

    let scalar = model.logits(embeddings, cache: model.makeCache()).logits

    let counted = (0..<ids.count).map { Int32($0) }
    let explicit = model.logits(
        embeddings,
        positions: .explicit(MLXArray(counted + counted + counted, [3, ids.count])),
        cache: model.makeCache()
    ).logits

    #expect(abs(scalar - explicit).max().item(Float.self) == 0)
}

/// The ids, the 3-D positions and the delta, against transformers' own `get_rope_index`. All
/// three are integers: they match or the port is wrong, and there is no tolerance to hide in.
///
/// This is the only test that catches advancing the position by the image's token count instead
/// of by its longer side. That mutation leaves the delta at zero, prefill and decode go on
/// agreeing with each other, and every self-consistency check in the suite passes.
@Test func qwen35VisionPromptMatchesTransformers() throws {
    let facade = try loadFacade()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let prompt = try facade.compose([fixtureImage(golden), .text("Describe the image.")])

    #expect(prompt.ids == golden["input_ids"]!.asArray(Int32.self))
    #expect(prompt.positions!.asArray(Int32.self) == golden["position_ids"]!.asArray(Int32.self))
    #expect(Int32(prompt.delta) == golden["rope_delta"]!.item(Int32.self))
    // The image spends 154 rows and 14 positions, so everything after it sits 140 behind.
    #expect(prompt.delta == -140)

    // The tower's rows, scattered over the pad run. The text rows around them are a gather and
    // come out exact; the image rows carry the tower's float32 drift.
    let floor = golden["noise.embeddings"]!.item(Float.self)
    #expect(relativeDiff(prompt.embeddings!, golden["embeddings"]!) < 3 * floor)
}

/// The whole multimodal forward: the tower's rows in the stream, the 3-D rotation over them,
/// the trunk on top.
@Test func qwen35VisionLogitsMatchTransformers() throws {
    let facade = try loadFacade()
    let (config, model) = try loadDecoder()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let prompt = try facade.compose([fixtureImage(golden), .text("Describe the image.")])
    let decoded = model.decode(
        prompt.embeddings!, positions: .explicit(prompt.positions!), cache: model.makeCache())

    for layer in 0..<config.numHiddenLayers {
        let floor = golden["noise.block_\(layer)"]!.item(Float.self)
        #expect(relativeDiff(decoded.blocks[layer], golden["block_\(layer)"]!) < 3 * floor, "\(layer)")
    }
    let floor = golden["noise.logits"]!.item(Float.self)
    #expect(relativeDiff(decoded.logits, golden["logits"]!) < 3 * floor)

    let ours = argMax(decoded.logits, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["logits"]!, axis: -1).asArray(Int32.self)
    #expect(ours == theirs)
}

/// The house rule, with an image in the prompt: a cache that disagrees with a full prefill is a
/// wrong cache, and a greedy sequence can survive one while full logits cannot.
@Test func qwen35VisionStepwiseLogitsMatchPrefill() throws {
    let facade = try loadFacade()
    let (config, model) = try loadDecoder()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let prompt = try facade.compose([fixtureImage(golden), .text("Describe the image.")])
    let prefilled = model.logits(
        prompt.embeddings!, positions: .explicit(prompt.positions!), cache: model.makeCache()
    ).logits

    var cache = model.makeCache()
    for index in 0..<prompt.ids.count {
        let (logits, grown) = model.logits(
            prompt.embeddings![0..., index..<(index + 1), 0...],
            positions: .explicit(prompt.positions![0..., index..<(index + 1)]),
            cache: cache)
        cache = grown
        #expect(relativeDiff(logits[0, -1], prefilled[0, index]) < 1e-5, "token \(index)")
    }
}

/// The decode positions, as integers, against transformers' own `get_rope_index` over the
/// prompt *and* its continuation. Everything generated after an image sits `delta` behind the
/// cache's row count, forever, and this is the shortest statement of that rule.
@Test func qwen35VisionDecodePositionsMatchTransformers() throws {
    let facade = try loadFacade()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let prompt = try facade.compose([fixtureImage(golden), .text("Describe the image.")])
    let positions = golden["forced_positions"]!
    let rows = positions.dim(1)

    for row in prompt.ids.count..<rows {
        guard case .offset(let ours) = prompt.rotation(atRow: row) else {
            Issue.record("a generated token rotates by a scalar position, not a table")
            return
        }
        // Past the image the three sections agree again, so one integer says it all.
        #expect(positions[0..., row].asArray(Int32.self) == [Int32(ours), Int32(ours), Int32(ours)])
    }
}

/// Teacher forcing through the real generation loop. The sampler is where `Generation` hands
/// over the logits row it just computed, so a sampler that keeps the row and answers with the
/// golden's token — instead of its own argmax — turns the loop itself into the thing under test:
/// its cache, its offsets, its rotation.
///
/// This is the golden that *sees* the decode positions. Greedy does not: 18 of the 24 layers are
/// DeltaNet and carry position through the recurrence rather than the rope, and the rope that is
/// left rotates 64 of 256 head dims — so a decode step rotating by the cache's row count instead
/// of the position, 140 adrift, still writes fluent, on-topic, wrong text and the argmax never
/// notices.
@Test func qwen35VisionForcedLogitsMatchTransformers() async throws {
    let facade = try loadFacade()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let ids = golden["forced_ids"]!.asArray(Int32.self)
    let continuation = Array(ids.dropFirst(golden["input_ids"]!.dim(0)))
    let forcing = TeacherForcing(continuation)

    _ = try await facade.generate(
        [fixtureImage(golden), .text("Describe the image.")],
        options: GenerationOptions(
            maxTokens: continuation.count, sampler: forcing.sampler, stop: []))

    let ours = expandedDimensions(stacked(forcing.rows, axis: 0), axis: 0)
    let floor = golden["noise.forced_logits"]!.item(Float.self)
    #expect(ours.shape == golden["forced_logits"]!.shape)
    #expect(relativeDiff(ours, golden["forced_logits"]!) < 3 * floor)
}

/// Feeds the loop the golden's tokens and keeps every logits row it was asked to sample from.
/// The row has to be evaluated here: the token going back is a constant, so nothing downstream
/// would ever force the forward that produced it.
private final class TeacherForcing: @unchecked Sendable {
    private let tokens: [Int32]
    private var index = 0
    var rows: [MLXArray] = []

    init(_ tokens: [Int32]) {
        self.tokens = tokens
    }

    var sampler: Sampler {
        Sampler { logits in
            eval(logits)
            self.rows.append(logits)
            defer { self.index += 1 }
            return MLXArray(self.tokens[self.index])
        }
    }
}

/// End to end through the facade, against transformers' own greedy: the ids, not the floats.
@Test func qwen35VisionGreedyMatchesTransformers() async throws {
    let facade = try loadFacade()
    let tokenizer = try Qwen2Tokenizer(directory: try qwen35Directory())
    let golden = try fixture("qwen3_5_vision.safetensors")

    let greedy = golden["greedy_ids"]!.asArray(Int32.self)
    let prompt = golden["input_ids"]!.asArray(Int32.self)
    let expected = Array(greedy.dropFirst(prompt.count))

    let text = try await facade.generate(
        [fixtureImage(golden), .text("Describe the image.")],
        options: GenerationOptions(maxTokens: expected.count, sampler: .greedy, stop: []))

    #expect(text == tokenizer.decode(expected))
}

/// A picture against a text-only checkpoint is refused, not quietly dropped.
@Test func qwen35VisionRefusesAnImageWithoutATower() throws {
    let model = try LanguageModel(directory: try qwen3Directory())
    let pixels = Pixels(width: 32, height: 32, rgb: [UInt8](repeating: 128, count: 32 * 32 * 3))

    #expect(throws: LanguageModel.PromptError.self) {
        _ = try model.compose([.image(.pixels(pixels)), .text("what is this?")])
    }
}

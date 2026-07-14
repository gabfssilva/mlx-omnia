import Foundation
import MLX
import Testing

@testable import Sideros

private func load27B() throws -> (URL, Qwen35Config, [String: MLXArray]) {
    let directory = try qwen3527BDirectory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: nil)
    return (directory, config, parameters)
}

private func load27BTower() throws -> (VisionConfig, Qwen35Vision) {
    let (_, config, parameters) = try load27B()
    let visual = parameters.filter { $0.key.hasPrefix("visual.") }
    let vision = config.visionConfig!
    return (vision, Qwen35Vision(config: vision, parameters: visual))
}

@Test func qwen35Vision27BFixtureHasOneGreedyIdPerStep() throws {
    let golden = try fixture("qwen3_5_vision_27b_mlxvlm.safetensors")

    #expect(golden["greedy_ids"]!.dim(0) == 64)
}

@Test func qwen35Vision27BEndToEndMatchesMLXVLM() throws {
    let golden = try fixture("qwen3_5_vision_27b_mlxvlm.safetensors")
    let directory = try qwen3527BDirectory()
    let prompt: Prompt
    do {
        let facade = try LanguageModel(directory: directory)
        let image = Bundle.module.resourceURL!.appending(path: "Fixtures/qwen3_5_vision.png")
        prompt = try facade.compose([.image(.file(image)), .text("Describe the image.")])
    }

    #expect(prompt.ids == golden["input_ids"]!.asArray(Int32.self))
    #expect(prompt.positions!.asArray(Int32.self) == golden["position_ids"]!.asArray(Int32.self))
    #expect(prompt.delta == Int(golden["rope_delta"]!.item(Int32.self)))
    #expect(relativeDiff(prompt.embeddings!, golden["embeddings"]!) < 0x1p-4)

    let (_, config, parameters) = try load27B()
    let decoder = Qwen35(
        config: config, parameters: parameters, quantization: config.quantization)
    let logits = decoder.logits(
        prompt.embeddings!, positions: .explicit(prompt.positions!), cache: decoder.makeCache()
    ).logits
    #expect(
        argMax(logits[0, -1], axis: -1).item(Int32.self)
            == argMax(golden["logits"]![0, -1], axis: -1).item(Int32.self))
}

private final class LargeTeacherForcing: @unchecked Sendable {
    private let ids: [Int32]
    private var index = 0
    var rows: [MLXArray] = []

    init(_ ids: [Int32]) {
        self.ids = ids
    }

    var sampler: Sampler {
        Sampler { logits in
            eval(logits)
            self.rows.append(logits)
            defer { self.index += 1 }
            return MLXArray(self.ids[self.index])
        }
    }
}

private func forcedLogprobs(
    model: LanguageModel, image: URL, forcing: LargeTeacherForcing
) async throws -> MLXArray {
    _ = try await model.generate(
        [.image(.file(image)), .text("Describe the image.")],
        options: GenerationOptions(maxTokens: 64, sampler: forcing.sampler, stop: []))
    let logits = stacked(forcing.rows, axis: 0)
    return logits - logSumExp(logits, axis: -1, keepDims: true)
}

@Test func qwen35Vision27BForcedLogitsMatchMLXVLM() async throws {
    let golden = try fixture("qwen3_5_vision_27b_mlxvlm.safetensors")
    let directory = try qwen3527BDirectory()
    let model = try LanguageModel(directory: directory)
    let image = Bundle.module.resourceURL!.appending(path: "Fixtures/qwen3_5_vision.png")
    let forcing = LargeTeacherForcing(golden["greedy_ids"]!.asArray(Int32.self))

    let logprobs = try await forcedLogprobs(model: model, image: image, forcing: forcing)
    #expect(logprobs.shape == golden["greedy_logits"]!.shape)
    #expect(relativeDiff(logprobs, golden["greedy_logits"]!) == 0)
    #expect(
        argMax(logprobs, axis: -1).asArray(Int32.self)
            == argMax(golden["greedy_logits"]!, axis: -1).asArray(Int32.self))
}

@Test func qwen35Vision27BDecoderMatchesMLXVLMWithGoldenEmbeddings() throws {
    let golden = try fixture("qwen3_5_vision_27b_mlxvlm.safetensors")
    let (_, config, parameters) = try load27B()
    let decoder = Qwen35(
        config: config, parameters: parameters, quantization: config.quantization)
    let ids = golden["greedy_ids"]!.asArray(Int32.self)
    let positions = golden["position_ids"]!
    let delta = Int(golden["rope_delta"]!.item(Int32.self))

    var cache = decoder.makeCache()
    var next = decoder.logits(
        golden["embeddings"]!.asType(.bfloat16), positions: .explicit(positions), cache: cache)
    cache = next.cache
    var rows: [MLXArray] = []
    for index in ids.indices {
        rows.append(next.logits[0, -1])
        guard index + 1 < ids.count else { break }
        next = decoder.logits(
            decoder.embed([ids[index]], offset: positions.dim(1) + index),
            positions: .offset(positions.dim(1) + index + delta), cache: cache)
        cache = next.cache
    }

    let logits = stacked(rows, axis: 0)
    let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
    let ours = argMax(logprobs, axis: -1).asArray(Int32.self)
    let theirs = argMax(golden["greedy_logits"]!, axis: -1).asArray(Int32.self)
    for index in ours.indices where ours[index] != theirs[index] {
        print(
            "decoder mismatch \(index): \(ours[index]) \(theirs[index]), "
                + "ours \(logprobs[index, Int(ours[index])].item(Float.self)) "
                + "\(logprobs[index, Int(theirs[index])].item(Float.self)), "
                + "reference \(golden["greedy_logits"]![index, Int(ours[index])].item(Float.self)) "
                + "\(golden["greedy_logits"]![index, Int(theirs[index])].item(Float.self))")
    }
    print("decoder relative diff \(relativeDiff(logprobs, golden["greedy_logits"]!))")
    #expect(ours == theirs)
}

@Test func qwen35Vision27BMatchesMLXVLM() throws {
    let (config, tower) = try load27BTower()
    let golden = try fixture("qwen3_5_vision_27b_mlxvlm.safetensors")
    let thw = golden["image_grid_thw"]!.asArray(Int32.self)
    let grid = Grid(t: Int(thw[0]), h: Int(thw[1]), w: Int(thw[2]))

    #expect(grid == Grid(t: 1, h: 22, w: 28))
    #expect(config.headDim == 72)

    let (cos, sin) = tower.rotation(grid)
    var x = tower.patchEmbed(golden["pixel_values"]!.asType(.bfloat16)) + tower.positions(grid)
    for index in 0..<config.depth {
        x = tower.block(x, index, cos: cos, sin: sin)
        let bound = index == 26 ? Float(0x1p-4) : Float(0x1p-5)
        #expect(relativeDiff(x, golden["vision_block_\(index)"]!) < bound, "block \(index)")
    }

    let merged = tower.merge(x)
    #expect(relativeDiff(merged, golden["vision_merged"]!) < 0x1p-4)
}

import Foundation
import MLX
import Testing

@testable import Sideros

private func loadTower() throws -> (VisionConfig, Qwen35Vision) {
    let directory = try qwen35Directory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: .float32)
    let vision = config.visionConfig!
    return (vision, Qwen35Vision(config: vision, parameters: parameters))
}

private func imageGrid(_ golden: [String: MLXArray]) -> Grid {
    let thw = golden["image_grid_thw"]!.asArray(Int32.self)
    return Grid(t: Int(thw[0]), h: Int(thw[1]), w: Int(thw[2]))
}

/// The tower is driven from the *transformers* pixel tensor, not from ours: its goldens carry
/// none of the resampler's error, which is measured on its own in the processor's fixture.
///
/// The image is 352x448, so the merged grid is 11x14 — not square. On a square grid a row/column
/// transposition in the 2D rope or in the block ordering is invisible, and the mutation passes.
@Test func qwen35VisionMatchesTransformers() throws {
    let (config, tower) = try loadTower()
    let golden = try fixture("qwen3_5_vision.safetensors")
    let grid = imageGrid(golden)

    func expectWithinNoiseFloor(_ ours: MLXArray, _ name: String) {
        let floor = golden["noise.\(name)"]!.item(Float.self)
        #expect(relativeDiff(ours, golden[name]!) < 3 * floor, "\(name)")
    }

    #expect(grid == Grid(t: 1, h: 22, w: 28))
    #expect(grid.tokens(merging: config.spatialMergeSize) == 154)

    let patched = tower.patchEmbed(golden["pixel_values"]!)
    expectWithinNoiseFloor(patched, "vision_patch")

    // The interpolated position table and the 2D rope table are closed-form, not propagated
    // activations: two float32 evaluations of the same expression, agreeing op for op, so they
    // take the house's internals bound rather than a noise floor. What can be wrong in them is
    // structural — raster instead of block order, row and column transposed, the wrong base —
    // and that misses by the values themselves, not by ulps.
    let interpolated = tower.positions(grid)
    #expect(relativeDiff(interpolated, golden["vision_pos"]!) < 2e-6)

    let (cos, sin) = tower.rotation(grid)
    let angles = concatenated([golden["vision_rope"]!, golden["vision_rope"]!], axis: -1)
    #expect(relativeDiff(cos, MLX.cos(angles)) < 1e-6)
    #expect(relativeDiff(sin, MLX.sin(angles)) < 1e-6)

    var x = patched + interpolated
    for index in 0..<config.depth {
        x = tower.block(x, index, cos: cos, sin: sin)
        expectWithinNoiseFloor(x, "vision_block_\(index)")
    }
    expectWithinNoiseFloor(tower.merge(x), "vision_merged")
}

/// What the trunk actually consumes: one row per merged block of four patches.
@Test func qwen35VisionMergesFourPatchesIntoOneToken() throws {
    let (config, tower) = try loadTower()
    let golden = try fixture("qwen3_5_vision.safetensors")
    let grid = imageGrid(golden)

    let merged = tower(golden["pixel_values"]!, grid: grid)
    #expect(merged.shape == [154, config.outHiddenSize])

    let floor = golden["noise.vision_merged"]!.item(Float.self)
    #expect(relativeDiff(merged, golden["vision_merged"]!) < 3 * floor)
}

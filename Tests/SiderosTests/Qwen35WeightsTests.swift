import Foundation
import MLX
import Testing

@testable import Sideros

func qwen35Directory() throws -> URL { try hubSnapshot("Qwen--Qwen3.5-0.8B") }
func qwen3527BDirectory() throws -> URL { try hubSnapshot("mlx-community--Qwen3.6-27B-6bit") }

@Test func qwen35ConfigMatchesCheckpoint() throws {
    let config = try Qwen35Config(directory: qwen35Directory())

    #expect(config.numHiddenLayers == 24)
    #expect(config.numAttentionHeads == 8)
    #expect(config.numKeyValueHeads == 2)
    #expect(config.headDim == 256)
    #expect(config.hiddenSize == 1024)
    #expect(config.intermediateSize == 3584)
    #expect(config.vocabSize == 248320)
    #expect(config.rmsNormEps == 1e-6)
    #expect(config.ropeTheta == 10_000_000)
    // A quarter of the head rotates; the rest passes through.
    #expect(config.ropeDimensions == 64)
    #expect(config.layerTypes.count == 24)
    #expect(config.layerTypes.filter { $0 == "full_attention" } .count == 6)
    #expect(config.layerTypes[3] == "full_attention" && config.layerTypes[0] == "linear_attention")
    #expect(config.linearNumKeyHeads == 16)
    #expect(config.linearNumValueHeads == 16)
    #expect(config.linearKeyHeadDim == 128)
    #expect(config.linearValueHeadDim == 128)
    #expect(config.linearConvKernelDim == 4)
    #expect(config.tied)
    #expect(config.eosTokenId == 248044)
    #expect(config.quantization == nil)
    #expect(config.mropeSection == [11, 11, 10])

    // The 0.8B carries the same tower as the 27B and the 35B-A3B, narrower: 12 blocks of 768.
    let vision = config.visionConfig!
    #expect(vision.depth == 12)
    #expect(vision.hiddenSize == 768)
    #expect(vision.numHeads == 12)
    #expect(vision.headDim == 64)
    #expect(vision.intermediateSize == 3072)
    #expect(vision.patchSize == 16)
    #expect(vision.temporalPatchSize == 2)
    #expect(vision.spatialMergeSize == 2)
    // A patch is C·T·H·W; a merged block of four of them is what the trunk reads.
    #expect(vision.patchDim == 1536)
    #expect(vision.mergerHidden == 3072)
    #expect(vision.gridPerSide == 48)
    #expect(vision.outHiddenSize == config.hiddenSize)
    #expect(config.imageTokenId == 248056)
    #expect(config.visionStartTokenId == 248053)
    #expect(config.visionEndTokenId == 248054)
}

@Test func qwen35ParametersMatchExpectedSetAndShapes() throws {
    let directory = try qwen35Directory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: .float32)

    // MTP tower dropped; `model.language_model.*` renamed to `model.*` and `model.visual.*`
    // to `visual.*`. Trunk: embed + final norm + 14 tensors per linear layer + 11 per full.
    // Tower: patch embed (weight + bias) + position table, 12 per block, 6 in the merger.
    #expect(parameters.count == 2 + 14 * 18 + 11 * 6 + 3 + 12 * 12 + 6)
    #expect(parameters["lm_head.weight"] == nil)
    #expect(parameters["mtp.fc.weight"] == nil)

    // The conv weight is squeezed out of its torch layout; A_log alone stays float32.
    #expect(parameters["model.layers.0.linear_attn.conv1d.weight"]!.shape == [6144, 4])
    #expect(parameters["model.layers.0.linear_attn.A_log"]!.dtype == .float32)
    #expect(
        parameters.allSatisfy { name, array in
            name.hasSuffix("A_log") || array.dtype == .float32
        })

    #expect(parameters["model.layers.0.linear_attn.in_proj_qkv.weight"]!.shape == [6144, 1024])
    #expect(parameters["model.layers.0.linear_attn.in_proj_z.weight"]!.shape == [2048, 1024])
    #expect(parameters["model.layers.0.linear_attn.in_proj_b.weight"]!.shape == [16, 1024])
    #expect(parameters["model.layers.0.linear_attn.norm.weight"]!.shape == [128])
    // q_proj carries the fused output gate: twice the query width.
    #expect(parameters["model.layers.3.self_attn.q_proj.weight"]!.shape == [4096, 1024])
    #expect(parameters["model.layers.3.self_attn.k_proj.weight"]!.shape == [512, 1024])
    #expect(parameters["model.layers.3.self_attn.q_norm.weight"]!.shape == [256])

    // The tower keeps MLX Conv3d's channels-last layout.
    let raw = try loadCheckpoint(directory: directory)
    let loaded = parameters["visual.patch_embed.proj.weight"]!
    #expect(loaded.shape == [768, 2, 16, 16, 3])
    let torch = raw["model.visual.patch_embed.proj.weight"]!.asType(.float32)
        .transposed(0, 2, 3, 4, 1)
    #expect(abs(loaded - torch).max().item(Float.self) == 0)

    #expect(parameters["visual.pos_embed.weight"]!.shape == [2304, 768])
    #expect(parameters["visual.blocks.0.attn.qkv.weight"]!.shape == [2304, 768])
    #expect(parameters["visual.blocks.0.attn.qkv.bias"]!.shape == [2304])
    #expect(parameters["visual.blocks.11.mlp.linear_fc1.weight"]!.shape == [3072, 768])
    #expect(parameters["visual.merger.linear_fc1.weight"]!.shape == [3072, 3072])
    #expect(parameters["visual.merger.linear_fc2.weight"]!.shape == [1024, 3072])

    // The tower's LayerNorms are not the trunk's zero-centered RMSNorms: no +1 shift.
    let rawNorm = raw["model.visual.blocks.0.norm1.weight"]!.asType(.float32)
    #expect(abs(parameters["visual.blocks.0.norm1.weight"]! - rawNorm).max().item(Float.self) == 0)
}

/// The 27B is where the tower's two other loader paths live, and the 0.8B exercises neither:
/// the mlx conversion ships `patch_embed.proj.weight` channels-last, and the tower is dense
/// bfloat16 inside a 6-bit checkpoint.
@Test func qwen35VisionLoadsFromAQuantizedCheckpoint() throws {
    let directory = try qwen3527BDirectory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: nil)

    let vision = config.visionConfig!
    #expect(vision.depth == 27)
    #expect(vision.hiddenSize == 1152)
    #expect(vision.numHeads == 16)
    // Not a multiple of 8, unlike the 0.8B's 64.
    #expect(vision.headDim == 72)
    #expect(vision.outHiddenSize == 5120)
    #expect(config.quantization != nil)

    // The tower carries no scales: it stays dense while the trunk around it is packed. Were
    // the loader's quantization rewrite to reach it, the load would throw before this.
    #expect(parameters["visual.blocks.0.attn.qkv.scales"] == nil)
    #expect(parameters["visual.blocks.0.attn.qkv.weight"]!.dtype == .bfloat16)
    #expect(parameters["model.layers.0.linear_attn.in_proj_qkv.scales"] != nil)

    // The mlx conversion already stores MLX Conv3d's [out, T, H, W, C] layout.
    let raw = try loadCheckpoint(directory: directory)
    let shipped = raw["vision_tower.patch_embed.proj.weight"]!
    #expect(shipped.shape == [1152, 2, 16, 16, 3])

    let loaded = parameters["visual.patch_embed.proj.weight"]!
    #expect(loaded.shape == shipped.shape)
    #expect(abs(loaded - shipped).max().item(Float.self) == 0)
}

/// The raw HF checkpoint stores the shiftable norms zero-centered (scale = 1 + w); the
/// loader bakes the +1 in, so their weights land near one, while the DeltaNet's gated
/// norm is ordinary and must not shift.
@Test func qwen35LoaderShiftsZeroCenteredNorms() throws {
    let directory = try qwen35Directory()
    let config = try Qwen35Config(directory: directory)
    let parameters = try loadQwen35Parameters(
        directory: directory, config: config, precision: .float32)

    let shifted = parameters["model.layers.0.input_layernorm.weight"]!.mean().item(Float.self)
    #expect(abs(shifted - 1) < 0.5)

    // Raw file value + 1 for one spot-checked norm: the shift is exactly one.
    let raw = try loadCheckpoint(directory: directory)
    let rawNorm = raw["model.language_model.layers.0.input_layernorm.weight"]!
        .asType(.float32)
    let diff = (parameters["model.layers.0.input_layernorm.weight"]! - rawNorm - 1)
    #expect(abs(diff).max().item(Float.self) == 0)

    let gated = parameters["model.layers.0.linear_attn.norm.weight"]!
    let rawGated = raw["model.language_model.layers.0.linear_attn.norm.weight"]!.asType(.float32)
    #expect(abs(gated - rawGated).max().item(Float.self) == 0)
}

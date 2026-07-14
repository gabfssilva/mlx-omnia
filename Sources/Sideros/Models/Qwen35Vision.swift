import Foundation
import MLX
import MLXNN

/// What one processed image occupies, in patches: temporal, rows, columns. The merger folds
/// each `spatialMergeSize`² block of patches into one token, so the trunk sees
/// `t · h · w / mergeUnit` of them.
public struct Grid: Sendable, Equatable {
    public let t: Int
    public let h: Int
    public let w: Int

    public init(t: Int, h: Int, w: Int) {
        (self.t, self.h, self.w) = (t, h, w)
    }

    public var patches: Int { t * h * w }
    public func tokens(merging merge: Int) -> Int { patches / (merge * merge) }
}

/// The Qwen3.6 vision tower. A ViT with bidirectional attention, run once per image at
/// prefill: it never sees a `KVCache` and nothing in it is compiled. It ships bfloat16 even
/// inside a quantized checkpoint, so every projection here is dense.
///
/// It is the first part of the house bound by compute rather than bandwidth — a 1MP image is
/// ~3900 patches and ~5 TFLOP against 0.86 GB of weights, read once — so the ceiling it is
/// measured against is the machine's bf16 matmul peak, not its 490 GB/s.
struct Qwen35Vision {
    let config: VisionConfig
    private let parameters: [String: MLXArray]

    init(config: VisionConfig, parameters: [String: MLXArray]) {
        self.config = config
        self.parameters = parameters
    }

    private subscript(name: String) -> MLXArray { parameters["visual." + name]! }

    private func linear(_ name: String) -> Linear {
        Linear("visual." + name, parameters: parameters, quantization: nil)
    }

    /// LayerNorm with a bias, not the trunk's RMSNorm — and no zero-centering. The epsilon is
    /// hardcoded in transformers, not a config field.
    private func layerNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: self[name + ".weight"], bias: self[name + ".bias"], eps: 1e-6)
    }

    /// `patches` is `[t·h·w, patchDim]` in the 2×2-block order the processor emits; the result
    /// is one row per trunk token.
    func callAsFunction(_ patches: MLXArray, grid: Grid) -> MLXArray {
        var x = patchEmbed(patches) + positions(grid)
        let (cos, sin) = rotation(grid)
        for index in 0..<config.depth {
            x = block(x, index, cos: cos, sin: sin)
        }
        return merge(x)
    }

    /// The checkpoint keeps the Conv3d in MLX's channels-last kernel layout.
    func patchEmbed(_ patches: MLXArray) -> MLXArray {
        let weight = self["patch_embed.proj.weight"]
        let volume = patches
            .reshaped(
                -1, config.inChannels, config.temporalPatchSize, config.patchSize,
                config.patchSize)
            .transposed(0, 2, 3, 4, 1)
        return (
            conv3d(
                volume, weight,
                stride: [config.temporalPatchSize, config.patchSize, config.patchSize])
                + self["patch_embed.proj.bias"]
        ).reshaped(-1, config.hiddenSize)
    }

    func block(_ x: MLXArray, _ index: Int, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let p = "blocks.\(index)."
        let residual = x + attention(layerNorm(x, p + "norm1"), p, cos: cos, sin: sin)
        return residual + mlp(layerNorm(residual, p + "norm2"), p)
    }

    /// Bidirectional: every patch of the image sees every other, so there is no mask at all.
    private func attention(_ x: MLXArray, _ p: String, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(0)
        let dim = config.headDim
        let qkv = linear(p + "attn.qkv")(x)
            .reshaped(length, 3, config.numHeads, dim)
            .transposed(1, 0, 2, 3)

        func prepared(_ part: MLXArray, rotated: Bool) -> MLXArray {
            let heads = rotated ? rotate(part, cos: cos, sin: sin) : part
            return expandedDimensions(heads.transposed(1, 0, 2), axis: 0)
        }

        var queries = prepared(qkv[0], rotated: true)
        var keys = prepared(qkv[1], rotated: true)
        var values = prepared(qkv[2], rotated: false)
        if dim == 72 {
            let widths: [IntOrPair] = [0, 0, 0, [0, 8]]
            queries = padded(queries, widths: widths)
            keys = padded(keys, widths: widths)
            values = padded(values, widths: widths)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(dim)), mask: .none)[.ellipsis, ..<dim]

        return linear(p + "attn.proj")(
            attended.transposed(0, 2, 1, 3).reshaped(length, config.numHeads * dim))
    }

    /// The rotation runs in float32 and rounds once on the way out, as it does in transformers
    /// (which casts q, k, cos and sin to float32) and in mlx-lm.
    private func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let heads = expandedDimensions(cos, axis: 1)
        let sines = expandedDimensions(sin, axis: 1)
        let half = x.dim(-1) / 2
        let rotated = concatenated(
            [-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return (x * heads + rotated * sines).asType(x.dtype)
    }

    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {
        linear(p + "mlp.linear_fc2")(geluTanh(linear(p + "mlp.linear_fc1")(x)))
    }

    /// The merger normalizes each patch and then reads a whole spatial block of them as one
    /// row — which is only a spatial block because the processor already grouped them that way.
    func merge(_ x: MLXArray) -> MLXArray {
        let grouped = layerNorm(x, "merger.norm").reshaped(-1, config.mergerHidden)
        return linear("merger.linear_fc2")(gelu(linear("merger.linear_fc1")(grouped)))
    }

    /// The tower's MLP takes the tanh approximation (`gelu_pytorch_tanh`), the merger the exact
    /// one (`nn.GELU()`). Both raw rather than through MLXNN, as elsewhere in the house.
    private func geluTanh(_ x: MLXArray) -> MLXArray {
        MLXNN.geluApproximate(x)
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        MLXNN.gelu(x)
    }

    /// The learned position table is a square grid, bilinearly interpolated by the same
    /// `linspace` and four-corner expression as mlx-vlm, then put in 2×2-block order.
    func positions(_ grid: Grid) -> MLXArray {
        let table = self["pos_embed.weight"]
        let side = config.gridPerSide
        let rows = linspace(Float(0), Float(side - 1), count: grid.h)
        let columns = linspace(Float(0), Float(side - 1), count: grid.w)
        let rowFloor = rows.asType(.int32)
        let columnFloor = columns.asType(.int32)
        let rowCeil = minimum(rowFloor + 1, side - 1)
        let columnCeil = minimum(columnFloor + 1, side - 1)
        let dh = rows - rowFloor.asType(.float32)
        let dw = columns - columnFloor.asType(.float32)
        let base = rowFloor * side
        let baseCeil = rowCeil * side

        let indices = [
            (expandedDimensions(base, axis: 1) + expandedDimensions(columnFloor, axis: 0)).flattened(),
            (expandedDimensions(base, axis: 1) + expandedDimensions(columnCeil, axis: 0)).flattened(),
            (expandedDimensions(baseCeil, axis: 1) + expandedDimensions(columnFloor, axis: 0)).flattened(),
            (expandedDimensions(baseCeil, axis: 1) + expandedDimensions(columnCeil, axis: 0)).flattened(),
        ]
        let weights = [
            expandedDimensions(1 - dh, axis: 1) * expandedDimensions(1 - dw, axis: 0),
            expandedDimensions(1 - dh, axis: 1) * expandedDimensions(dw, axis: 0),
            expandedDimensions(dh, axis: 1) * expandedDimensions(1 - dw, axis: 0),
            expandedDimensions(dh, axis: 1) * expandedDimensions(dw, axis: 0),
        ]

        var sum = table[indices[0]] * expandedDimensions(weights[0].flattened(), axis: 1).asType(table.dtype)
        for index in 1..<4 {
            sum = sum + table[indices[index]]
                * expandedDimensions(weights[index].flattened(), axis: 1).asType(table.dtype)
        }
        return blocked(sum, grid: grid)
    }

    /// Raster order in, 2×2-block order out — the order the patches themselves arrive in.
    private func blocked(_ raster: MLXArray, grid: Grid) -> MLXArray {
        let merge = config.spatialMergeSize
        let frames =
            grid.t == 1 ? raster : concatenated(Array(repeating: raster, count: grid.t), axis: 0)
        return
            frames
            .reshaped(grid.t, grid.h / merge, merge, grid.w / merge, merge, -1)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(grid.patches, -1)
    }

    /// A 2D rope: the first half of the frequencies carries the patch's row, the second its
    /// column, and the pair is duplicated so the half-rotation covers the whole head.
    func rotation(_ grid: Grid) -> (MLXArray, MLXArray) {
        let merge = config.spatialMergeSize
        let dimensions = config.headDim / 2
        let exponents = arange(0, dimensions, step: 2, dtype: .float32) / Float(dimensions)
        let frequencies = 1 / (Float(10_000) ** exponents)
        let table = outer(arange(max(grid.h, grid.w), dtype: .float32), frequencies)

        var rows: [Int32] = []
        var columns: [Int32] = []
        rows.reserveCapacity(grid.patches)
        columns.reserveCapacity(grid.patches)
        for _ in 0..<grid.t {
            for blockRow in 0..<(grid.h / merge) {
                for blockColumn in 0..<(grid.w / merge) {
                    for intraRow in 0..<merge {
                        for intraColumn in 0..<merge {
                            rows.append(Int32(blockRow * merge + intraRow))
                            columns.append(Int32(blockColumn * merge + intraColumn))
                        }
                    }
                }
            }
        }

        let theta = concatenated([table[MLXArray(rows)], table[MLXArray(columns)]], axis: -1)
        let doubled = concatenated([theta, theta], axis: -1)
        return (cos(doubled), sin(doubled))
    }
}

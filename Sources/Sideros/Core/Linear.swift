import MLX

/// A weight application, dense or quantized, behind one call. The asymmetry of layouts
/// lives here: dense holds the matrix ready for `matmul` (in, out), quantized holds the
/// (out, in) triple the kernel wants with `transpose: true`. A projection bias, when the
/// checkpoint has one, rides along — fused into the matmul where a kernel exists for it.
enum Linear {
    case dense(MLXArray, bias: MLXArray?, skinny: Bool)
    case quantized(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int,
        bias: MLXArray?)

    /// Standard HF layout, (out, in) on disk. Quantized when the triple is present —
    /// per-tensor, so a dense table (GPT-2's wpe) inside a quantized model stays dense.
    /// `skinny` opts the projection into the narrow-batch kernel; only a model that
    /// verifies drafts wants it, because its rounding differs from `matmul`'s in the
    /// last bit and the plain path promises mlx-lm's exact numerics.
    init(
        _ name: String, parameters: [String: MLXArray], quantization: Quantization?,
        skinny: Bool = false
    ) {
        let bias = parameters[name + ".bias"]
        if let quantization = quantization?.module(name), let scales = parameters[name + ".scales"] {
            self = .quantized(
                weight: parameters[name + ".weight"]!, scales: scales,
                biases: parameters[name + ".biases"]!,
                groupSize: quantization.groupSize, bits: quantization.bits, bias: bias)
        } else {
            self = .dense(parameters[name + ".weight"]!.T, bias: bias, skinny: skinny)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight, let bias, let skinny):
            // Speculative verification runs 2–8 rows through every projection, a width
            // MLX's tile gemm wastes ~40% of the bandwidth on. `weight.T` recovers the
            // contiguous [out, in] checkpoint buffer the kernel streams.
            if skinny, x.dtype == .bfloat16, weight.dtype == .bfloat16,
                skinnyRows.contains(x.dim(-2)), weight.dim(0) % 4 == 0, weight.dim(1) % 2 == 0
            {
                let projected = skinnyMatmul(x, weight: weight.T)
                return bias.map { projected + $0 } ?? projected
            }
            return bias.map { addMM($0, x, weight) } ?? matmul(x, weight)
        case .quantized(let weight, let scales, let biases, let groupSize, let bits, let bias):
            let projected = quantizedMatmul(
                x, weight, scales: scales, biases: biases, transpose: true,
                groupSize: groupSize, bits: bits)
            return bias.map { projected + $0 } ?? projected
        }
    }
}

/// `Linear` over a weight stacked per expert: the router names a matrix per row, and the
/// gather kernel streams it. Dense wants (in, out) and so swaps the checkpoint's axes;
/// quantized reads (out, in) in place, as in `Linear`.
enum Experts {
    case dense(MLXArray)
    case quantized(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int)

    init(_ name: String, parameters: [String: MLXArray], quantization: Quantization?) {
        if let quantization = quantization?.module(name), let scales = parameters[name + ".scales"] {
            self = .quantized(
                weight: parameters[name + ".weight"]!, scales: scales,
                biases: parameters[name + ".biases"]!,
                groupSize: quantization.groupSize, bits: quantization.bits)
        } else {
            self = .dense(parameters[name + ".weight"]!)
        }
    }

    /// `indices` carries one expert id per row of `x`. `sorted` promises the kernel that
    /// they ascend, so each expert's weight is streamed once instead of per row.
    func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray {
        switch self {
        case .dense(let weight):
            gatherMM(
                x, weight.swappedAxes(-2, -1), rhsIndices: indices, sortedIndices: sorted)
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            gatherQuantizedMM(
                x, weight, scales: scales, biases: biases, rhsIndices: indices,
                transpose: true, groupSize: groupSize, bits: bits, sortedIndices: sorted)
        }
    }
}

/// A token-id gather over the same storage `Linear` projects with; the tied lm_head is
/// `Linear` over the same name. Groups run along a row, so gathering rows of the packed
/// weight keeps each group intact and dequantizes only the rows the batch touches.
enum Embedding {
    case dense(MLXArray)
    case quantized(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int)

    init(_ name: String, parameters: [String: MLXArray], quantization: Quantization?) {
        if let quantization = quantization?.module(name), let scales = parameters[name + ".scales"] {
            self = .quantized(
                weight: parameters[name + ".weight"]!, scales: scales,
                biases: parameters[name + ".biases"]!,
                groupSize: quantization.groupSize, bits: quantization.bits)
        } else {
            self = .dense(parameters[name + ".weight"]!)
        }
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight):
            weight[tokens]
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            dequantized(
                weight[tokens], scales: scales[tokens], biases: biases[tokens],
                groupSize: groupSize, bits: bits)
        }
    }
}

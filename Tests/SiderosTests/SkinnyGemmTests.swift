import MLX
import Testing

@testable import Sideros

/// Each output row of the skinny kernel reads only its own row of x, so the same row
/// through a different batch width is the same arithmetic, bit for bit. matmul offers
/// no such guarantee — its tile internals change with shape.
@Test func skinnyGemmRowsDoNotDependOnBatchWidth() {
    MLXRandom.seed(7)
    let x = MLXRandom.normal([8, 1024]).asType(.bfloat16)
    let w = MLXRandom.normal([512, 1024]).asType(.bfloat16)

    let wide = skinnyMatmul(x, weight: w)
    for start in stride(from: 0, to: 8, by: 2) {
        let narrow = skinnyMatmul(x[start..<(start + 2)], weight: w)
        #expect(allClose(narrow, wide[start..<(start + 2)], rtol: 0, atol: 0).item(Bool.self))
    }
}

/// Against the fp32 ground truth, kernel and matmul are both one bf16 rounding away:
/// fp32 accumulation error at K=1024 is orders of magnitude below a bf16 ulp, so the
/// tolerance is one ulp (2⁻⁸ relative) at the result's ~√K scale.
@Test func skinnyGemmMatchesMatmul() {
    MLXRandom.seed(11)
    let exact = { (x: MLXArray, w: MLXArray) in
        matmul(x.asType(.float32), w.asType(.float32).T)
    }
    for rows in skinnyRows {
        let x = MLXRandom.normal([rows, 1024]).asType(.bfloat16)
        let w = MLXRandom.normal([512, 1024]).asType(.bfloat16)
        let close = allClose(
            skinnyMatmul(x, weight: w).asType(.float32), exact(x, w),
            rtol: 0.0039, atol: 0.125)
        #expect(close.item(Bool.self), "rows \(rows)")
    }
}

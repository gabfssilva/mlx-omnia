import Foundation
import MLX
import Testing

@testable import Sideros

/// The one-token kernels against the op chains they replace, over random quantized
/// weights — no checkpoint. softmaxTopK must be bit-exact including ties; the fused
/// gemvs and the epilogue re-round at every op boundary the chain has, so they sit
/// within a couple of bf16 ulps of it.

private func randomQuantized(
    _ rows: Int, _ cols: Int, bits: Int, seed: UInt64
) -> (MLXArray, MLXArray, MLXArray) {
    MLXRandom.seed(seed)
    // Checkpoint-scale weights: at N(0, 1) the dots sit in the thousands, where one
    // bf16 ulp is 16 and the comparison stops meaning anything.
    let dense = (MLXRandom.normal([rows, cols]) * 0.02).asType(.bfloat16)
    let (weight, scales, biases) = quantized(dense, groupSize: 64, bits: bits)
    eval(weight, scales, biases!)
    return (weight, scales, biases!)
}

/// 128 experts is the 30B, 256 the 35B: four scores per lane and eight.
@Test(arguments: [128, 256])
func softmaxTopKMatchesTheOpChainBitForBit(experts: Int) {
    MLXRandom.seed(3)
    for trial in 0..<200 {
        var logits = MLXRandom.normal([experts]).asType(.bfloat16)
        if trial % 4 == 0 {
            // Force exact ties, including at the cut: after bf16 rounding of the
            // probabilities, coarsened logits collide constantly.
            logits = (logits * 4).round() / 4
        }
        let (indices, weights) = softmaxTopK(logits, k: 8)

        // The reference in the model's shapes — a 1-d sum reduces differently.
        let split = experts - 8
        let probs = softmax(logits.reshaped(1, 1, experts), axis: -1, precise: true)
        let chosen = argPartition(probs, kth: split, axis: -1)[0..., 0..., split...]
        var expected = takeAlong(probs, chosen, axis: -1)
        expected = expected / expected.sum(axis: -1, keepDims: true)

        let got = Dictionary(
            uniqueKeysWithValues: zip(
                indices.asArray(UInt32.self), weights.asType(.float32).asArray(Float.self)))
        let want = Dictionary(
            uniqueKeysWithValues: zip(
                chosen.reshaped(-1).asType(.uint32).asArray(UInt32.self),
                expected.reshaped(-1).asType(.float32).asArray(Float.self)))
        #expect(got == want, "trial \(trial)")
    }
}

@Test func ropeEpilogueMatchesNormAndRope() {
    MLXRandom.seed(5)
    let (numHeads, kvHeads, headDim) = (32, 4, 128)
    let qNorm = MLXRandom.normal([headDim]).asType(.bfloat16)
    let kNorm = MLXRandom.normal([headDim]).asType(.bfloat16)
    let freqs = exp(
        MLXArray(Array(stride(from: Float(0), to: Float(headDim / 2), by: 1)))
            * (-2 / Float(headDim) * log(Float(1_000_000))))
    let eps = MLXArray([Float(1e-6)])
    eval(freqs, eps)

    for offset in [0, 1, 430, 4097] {
        let fused = MLXRandom.normal([(numHeads + 2 * kvHeads) * headDim]).asType(.bfloat16)
        let (q, k) = ropeEpilogue(
            fused, queryHeads: numHeads, kvHeads: kvHeads, headDim: headDim,
            qNorm: qNorm, kNorm: kNorm, offset: MLXArray([Int32(offset)]), freqs: freqs,
            eps: eps)

        func reference(_ part: MLXArray, _ count: Int, _ norm: MLXArray) -> MLXArray {
            MLXFast.RoPE(
                MLXFast.rmsNorm(
                    part.reshaped(1, count, 1, headDim), weight: norm, eps: 1e-6),
                dimensions: headDim, traditional: false, base: 1_000_000, scale: 1,
                offset: offset)
        }
        let qRef = reference(fused[..<(numHeads * headDim)], numHeads, qNorm)
        let kRef = reference(
            fused[(numHeads * headDim)..<((numHeads + kvHeads) * headDim)], kvHeads, kNorm)

        // Each side is a couple of rounded ops away from the other; what a broken
        // rotation or norm produces is off by the values themselves, not by ulps.
        #expect(
            allClose(q.reshaped(1, numHeads, 1, headDim), qRef, rtol: 0.05, atol: 0.05)
                .item(Bool.self), "q, offset \(offset)")
        #expect(
            allClose(k.reshaped(1, kvHeads, 1, headDim), kRef, rtol: 0.05, atol: 0.05)
                .item(Bool.self), "k, offset \(offset)")
    }
}

/// Over every width MLX packs, not just the nibble one: the kernel walks the bit stream,
/// so 6 bits (4 values per 3 bytes) has to come out of it exactly as 4 does.
@Test(arguments: [2, 3, 4, 5, 6, 8])
func fusedMoeGemvsMatchTheGatherChain(bits: Int) {
    let (experts, inner, hidden, k) = (16, 768, 2048, 8)
    func words(_ cols: Int) -> Int { cols * bits / 32 }
    // Interleaved [g₀,u₀,…] stack for the kernel and the same weights split back into
    // the plain fused layout for the reference chain.
    let (giW, giS, giB) = randomQuantized(experts * 2 * inner, hidden, bits: bits, seed: 21)
    let interleaved = (
        giW.reshaped(experts, 2 * inner, words(hidden)),
        giS.reshaped(experts, 2 * inner, hidden / 64),
        giB.reshaped(experts, 2 * inner, hidden / 64)
    )
    let (dnW, dnS, dnB) = randomQuantized(experts * hidden, inner, bits: bits, seed: 22)
    let down = (
        dnW.reshaped(experts, hidden, words(inner)),
        dnS.reshaped(experts, hidden, inner / 64),
        dnB.reshaped(experts, hidden, inner / 64)
    )

    MLXRandom.seed(23)
    let x = (MLXRandom.normal([hidden]) * 0.3).asType(.bfloat16)
    let residual = MLXRandom.normal([hidden]).asType(.bfloat16)
    let routing = MLXRandom.uniform(0.0 ..< 0.3, [k]).asType(.bfloat16)
    let indices = MLXArray([2, 5, 5, 7, 9, 11, 14, 15].map { UInt32($0) })

    let act = moeGateUpAct(
        x, weight: interleaved.0, scales: interleaved.1, biases: interleaved.2,
        indices: indices, groupSize: 64, bits: bits)
    let out = moeDownCombine(
        act.reshaped(-1), weight: down.0, scales: down.1, biases: down.2,
        indices: indices, routing: routing, residual: residual, groupSize: 64, bits: bits)

    let fused = gatherQuantizedMM(
        x.reshaped(1, 1, 1, hidden), interleaved.0, scales: interleaved.1,
        biases: interleaved.2, rhsIndices: indices.reshaped(1, 1, k), transpose: true,
        groupSize: 64, bits: bits)
    let pairs = fused.reshaped(1, 1, k, inner, 2)
    let gated = pairs[.ellipsis, 0]
    let up = pairs[.ellipsis, 1]
    let actRef = gated * sigmoid(gated) * up
    #expect(
        allClose(act, actRef.reshaped(k, inner), rtol: 0.05, atol: 0.02).item(Bool.self))

    let routed = gatherQuantizedMM(
        actRef.reshaped(k, 1, 1, inner), down.0, scales: down.1, biases: down.2,
        rhsIndices: indices.reshaped(k, 1), transpose: true, groupSize: 64, bits: bits)
    let outRef =
        residual
        + (routed.reshaped(k, hidden) * routing.reshaped(k, 1)).sum(axis: 0)
    #expect(allClose(out, outRef, rtol: 0.05, atol: 0.05).item(Bool.self))
}

@Test func addRmsNormMatchesTheOpPair() {
    MLXRandom.seed(31)
    let weight = MLXRandom.normal([2048]).asType(.bfloat16)
    let eps = MLXArray([Float(1e-6)])
    for _ in 0..<5 {
        let x = MLXRandom.normal([2048]).asType(.bfloat16)
        let projected = MLXRandom.normal([2048]).asType(.bfloat16)
        let (added, normed) = addRmsNorm(x, projected, weight: weight, eps: eps)
        let addedRef = x + projected
        let normedRef = MLXFast.rmsNorm(addedRef, weight: weight, eps: 1e-6)
        #expect(allClose(added, addedRef, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(normed, normedRef, rtol: 0.01, atol: 0.01).item(Bool.self))
    }
}

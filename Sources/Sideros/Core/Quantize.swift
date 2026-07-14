import MLX

/// Replaces each selected `<name>.weight` with the MLX affine triple. `quantized` packs
/// groups along the last axis, so a weight stored (in, out) — GPT-2's Conv1D — must be
/// flipped to (out, in) first or the groups land on the output axis, a layout that runs
/// but matches nothing.
func quantizing(
    _ parameters: [String: MLXArray], with quantization: Quantization,
    where isQuantized: (String) -> Bool, transposing: (String) -> Bool = { _ in false }
) -> [String: MLXArray] {
    var result = parameters
    var produced: [MLXArray] = []
    for (name, weight) in parameters where name.hasSuffix(".weight") && isQuantized(name) {
        let (packed, scales, biases) = quantized(
            transposing(name) ? weight.T : weight,
            groupSize: quantization.groupSize, bits: quantization.bits)
        let base = String(name.dropLast(".weight".count))
        result[name] = packed
        result[base + ".scales"] = scales
        result[base + ".biases"] = biases!
        produced += [packed, scales, biases!]
    }
    // Materialized here, not lazily on first forward: `compile` embeds captured weights
    // as constants, and a constant that is still a lazy quantize node re-executes on
    // every call of the compiled function — requantizing the MLP once per decoded token
    // (measured: 653 → 221 tok/s on the 0.5B).
    eval(produced)
    return result
}

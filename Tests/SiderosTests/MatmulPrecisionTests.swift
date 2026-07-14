import Foundation
import MLX
import Testing

@testable import Sideros

/// Guards the Makefile's TEST_RUNNER_MLX_ENABLE_TF32=0. Without it MLX runs float32
/// matmul on the reduced-precision Metal path and every parity test silently loosens
/// by three decimal digits.
@Test func float32MatmulIsExactOnGPU() throws {
    let directory = try gpt2Directory()
    let config = try GPT2Config(directory: directory)
    let parameters = try loadGPT2Parameters(directory: directory, config: config, precision: .float32)

    let x = try fixture("gpt2_forward.safetensors")["b0_ln_1"]!
    let weight = parameters["h.0.attn.c_attn.weight"]!

    #expect(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0")
    #expect(abs(matmul(x, weight, stream: .gpu) - matmul(x, weight, stream: .cpu)).max().item(Float.self) == 0)
}

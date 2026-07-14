import MLX
import Testing

@testable import Sideros

@Test func gpuIsAvailable() {
    #expect(Sideros.gpuIsAvailable())
}

@Test func matmulEvaluatesOnGPU() {
    let a = MLXArray(converting: [1.0, 2.0, 3.0, 4.0], [2, 2])
    let b = MLXArray(converting: [5.0, 6.0, 7.0, 8.0], [2, 2])

    let c = matmul(a, b, stream: .gpu)
    eval(c)

    #expect(c.asArray(Float.self) == [19, 22, 43, 50])
}

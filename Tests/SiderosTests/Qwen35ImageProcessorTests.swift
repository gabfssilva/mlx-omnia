import Foundation
import MLX
import Testing

@testable import Sideros

private func loadProcessor() throws -> Qwen35ImageProcessor {
    try Qwen35ImageProcessor(directory: qwen35Directory())
}

private func image(_ golden: [String: MLXArray], _ name: String) -> Image {
    let shape = golden[name]!.shape
    return .pixels(
        Pixels(width: shape[1], height: shape[0], rgb: golden[name]!.asArray(UInt8.self)))
}

/// Integer arithmetic: it matches or it is broken. The table walks both branches — an area over
/// `max_pixels` shrinks, one under `min_pixels` grows — and the plain snap in between, where
/// Python rounds half to even and a half-up port would miss by a whole patch block.
@Test func qwen35SmartResizeMatchesTransformers() throws {
    let processor = try loadProcessor()
    let golden = try fixture("qwen3_5_vision.safetensors")
    let table = golden["smart_resize"]!

    for row in 0..<table.dim(0) {
        let expected = table[row].asArray(Int32.self)
        let resized = try processor.resized(height: Int(expected[0]), width: Int(expected[1]))
        #expect(resized.height == Int(expected[2]), "height for \(expected[0])x\(expected[1])")
        #expect(resized.width == Int(expected[3]), "width for \(expected[0])x\(expected[1])")
    }
}

/// The image's sides are multiples of patch·merge and its area is inside the bounds, so
/// smart_resize is the identity and no resampling runs. Everything that is left — the fused
/// rescale-and-normalize, the patch cut, the block ordering — is exact arithmetic, and the
/// pixel tensor has to come out bit for bit.
@Test func qwen35ImageProcessorIsExactWhenNoResampleRuns() throws {
    let processor = try loadProcessor()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let (patches, grid) = try processor(image(golden, "image_rgb"))
    #expect(grid == Grid(t: 1, h: 22, w: 28))
    #expect(patches.shape == golden["pixel_values"]!.shape)
    #expect(abs(patches - golden["pixel_values"]!).max().item(Float.self) == 0)
}

/// The one step of the port without guaranteed bit parity. Our resampler and torchvision's
/// agree byte for byte except where a value lands on a rounding tie, and a tie moves a byte by
/// one — so this bound is not a tolerance chosen to pass, it is the claim itself: after the
/// normalize, one byte is 1/127.5.
@Test func qwen35ImageProcessorResamplesWithinOneByte() throws {
    let processor = try loadProcessor()
    let golden = try fixture("qwen3_5_vision.safetensors")

    let (patches, grid) = try processor(image(golden, "odd_rgb"))
    #expect(grid == Grid(t: 1, h: 18, w: 26))

    let reference = golden["odd_pixel_values"]!
    #expect(patches.shape == reference.shape)

    let gap = abs(patches - reference)
    #expect(gap.max().item(Float.self) <= 1 / 127.5 + 1e-6)

    // Measured on this image: 976 of 718,848 bytes, 0.14%. The guard sits just above it — the
    // fixture is deterministic, so the count only moves if the resampler does.
    let moved = (gap .> 1e-6).sum().item(Int32.self)
    #expect(Float(moved) / Float(patches.size) < 0.002, "\(moved) of \(patches.size) bytes moved")
}

/// ImageIO against PIL: the same bytes out of the same png, which pins the colour space. (They
/// part only on transparency, which this image has none of — see the note on `Image.decode`.)
@Test func qwen35ImageDecodeMatchesPIL() throws {
    let golden = try fixture("qwen3_5_vision.safetensors")
    let url = Bundle.module.resourceURL!.appending(path: "Fixtures/qwen3_5_vision.png")

    let decoded = try Image.file(url).decoded()
    #expect(decoded.width == 448 && decoded.height == 352)
    #expect(decoded.rgb == golden["image_rgb"]!.asArray(UInt8.self))
}

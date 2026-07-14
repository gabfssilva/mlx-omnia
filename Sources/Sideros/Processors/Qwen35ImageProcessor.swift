import Foundation
import MLX

/// The image frontend of Qwen3.6: an image in, the patch tensor the tower reads and the grid it
/// occupies out. A port of `Qwen2VLImageProcessorFast` — the one the checkpoint's
/// preprocessor_config.json names — and every constant here comes from that file.
public struct Qwen35ImageProcessor: Sendable {
    public struct Config: Decodable, Sendable {
        /// Not edges at all: the bounds are on the pixel *count*, and smart_resize keeps the
        /// area between them.
        private struct Size: Decodable {
            let shortestEdge: Int
            let longestEdge: Int
        }

        private let size: Size
        let patchSize: Int
        let temporalPatchSize: Int
        let mergeSize: Int
        let imageMean: [Float]
        let imageStd: [Float]

        var minPixels: Int { size.shortestEdge }
        var maxPixels: Int { size.longestEdge }

        public init(directory: URL) throws {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            self = try decoder.decode(
                Self.self,
                from: Data(contentsOf: directory.appending(path: "preprocessor_config.json")))
        }
    }

    private let config: Config

    public init(_ config: Config) {
        self.config = config
    }

    public init(directory: URL) throws {
        self.init(try Config(directory: directory))
    }

    /// `[t·h·w, patchDim]` in the tower's block order, and the grid the image occupies.
    public func callAsFunction(_ image: Image) throws -> (patches: MLXArray, grid: Grid) {
        let pixels = try image.decoded()
        let (height, width) = try resized(height: pixels.height, width: pixels.width)
        let bytes =
            (height, width) == (pixels.height, pixels.width)
            ? pixels.rgb
            : resample(pixels, height: height, width: width)
        return patchify(bytes, height: height, width: width)
    }

    /// Both sides to a multiple of patch·merge, the area inside the bounds, the aspect ratio as
    /// close as it can stay. `round` here is Python's — half to even — and only the shrinking
    /// branch floors the result to at least one patch block.
    func resized(height: Int, width: Int) throws -> (height: Int, width: Int) {
        let ratio = Double(max(height, width)) / Double(min(height, width))
        guard ratio <= 200 else { throw ImageError.aspectRatio(ratio) }

        let factor = config.patchSize * config.mergeSize
        func snapped(_ value: Double) -> Int {
            Int((value / Double(factor)).rounded(.toNearestOrEven)) * factor
        }
        var bars = (height: snapped(Double(height)), width: snapped(Double(width)))
        let area = Double(height * width)

        if bars.height * bars.width > config.maxPixels {
            let beta = (area / Double(config.maxPixels)).squareRoot()
            func shrunk(_ value: Int) -> Int {
                max(factor, Int((Double(value) / beta / Double(factor)).rounded(.down)) * factor)
            }
            bars = (shrunk(height), shrunk(width))
        } else if bars.height * bars.width < config.minPixels {
            let beta = (Double(config.minPixels) / area).squareRoot()
            func grown(_ value: Int) -> Int {
                Int((Double(value) * beta / Double(factor)).rounded(.up)) * factor
            }
            bars = (grown(height), grown(width))
        }
        return bars
    }

    /// torchvision's `resize(..., BICUBIC, antialias=True)` over bytes, which is Pillow's
    /// algorithm and *not* torchvision's own float one. Four things make it what it is, and all
    /// four were measured against the reference before this was written:
    ///
    ///  * the cubic filter takes a = −0.5, Pillow's, not the −0.75 of plain bicubic;
    ///  * the support widens with the scale when shrinking, and the taps renormalize to one;
    ///  * the width pass runs first;
    ///  * the intermediate is rounded and **clamped back to bytes** between the passes, which
    ///    is where the cubic's negative lobes get cut. Keeping it in float instead — the
    ///    obvious thing to do — is wrong by up to 21/255, not by a rounding.
    ///
    /// What is left after all four is a tie: 99.6% of bytes come out identical to the
    /// reference and the rest differ by one. That residue is what `noise.pixels` carries.
    func resample(_ pixels: Pixels, height: Int, width: Int) -> [UInt8] {
        let horizontal = pass(
            pixels.rgb, rows: pixels.height, columns: pixels.width, to: width)
        // The pass reads rows and writes columns, so running it twice transposes twice.
        return pass(horizontal, rows: width, columns: pixels.height, to: height)
    }

    /// One separable pass: `[rows, columns, 3]` in, `[columns, out, 3]` out — transposed, so
    /// that two of them put the image back the way it was.
    private func pass(_ source: [UInt8], rows: Int, columns: Int, to out: Int) -> [UInt8] {
        let taps = kernel(columns, out)
        var result = [UInt8](repeating: 0, count: columns == 0 ? 0 : rows * out * 3)

        source.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                for row in 0..<rows {
                    let base = row * columns * 3
                    for column in 0..<out {
                        let start = taps.start[column]
                        let weights = taps.weights[column]
                        for channel in 0..<3 {
                            var sum: Float = 0
                            for (offset, weight) in weights.enumerated() {
                                sum += Float(input[base + (start + offset) * 3 + channel]) * weight
                            }
                            let clamped = min(max(sum, 0), 255).rounded(.toNearestOrEven)
                            output[(column * rows + row) * 3 + channel] = UInt8(clamped)
                        }
                    }
                }
            }
        }
        return result
    }

    private struct Kernel {
        var start: [Int] = []
        var weights: [[Float]] = []
    }

    private func kernel(_ input: Int, _ output: Int) -> Kernel {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale

        func cubic(_ x: Double) -> Double {
            let a = -0.5
            let x = abs(x)
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }

        var kernel = Kernel()
        for index in 0..<output {
            let center = (Double(index) + 0.5) * scale
            let first = max(0, Int(center - support + 0.5))
            let last = min(input, Int(center + support + 0.5))
            let raw = (first..<last).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = raw.reduce(0, +)
            kernel.start.append(first)
            kernel.weights.append(raw.map { Float($0 / total) })
        }
        return kernel
    }

    /// Bytes to the tower's input: rescaled, normalized, and cut into patches grouped by merge
    /// block — the grouping is what makes four consecutive rows of the merger a spatial block.
    private func patchify(_ bytes: [UInt8], height: Int, width: Int) -> (MLXArray, Grid) {
        let patch = config.patchSize
        let merge = config.mergeSize
        let temporal = config.temporalPatchSize
        let grid = Grid(t: 1, h: height / patch, w: width / patch)

        // The reference fuses the rescale into the normalize — (x − 255·mean) / (255·std) over
        // the raw byte — so the byte is divided once, not scaled and then divided.
        let mean = MLXArray(config.imageMean.map { $0 * 255 })
        let deviation = MLXArray(config.imageStd.map { $0 * 255 })
        let image = ((MLXArray(bytes, [height, width, 3]).asType(.float32) - mean) / deviation)
            .transposed(2, 0, 1)

        // A single image is one temporal patch of two identical frames.
        let frames = concatenated(
            Array(repeating: expandedDimensions(image, axis: 0), count: temporal), axis: 0)

        return (
            frames
                .reshaped(
                    grid.t, temporal, 3, grid.h / merge, merge, patch, grid.w / merge, merge, patch
                )
                .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
                .reshaped(grid.patches, 3 * temporal * patch * patch),
            grid
        )
    }
}

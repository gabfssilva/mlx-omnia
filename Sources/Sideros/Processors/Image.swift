import CoreGraphics
import Foundation
import ImageIO

/// Bytes, row-major, three per pixel. What the image processor consumes.
public struct Pixels: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]

    public init(width: Int, height: Int, rgb: [UInt8]) {
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

/// An image on its way into a prompt.
public enum Image: Sendable {
    case file(URL)
    /// An encoded image — png, jpeg, anything ImageIO reads.
    case data(Data)
    case pixels(Pixels)
}

public enum ImageError: Error {
    case undecodable
    /// transformers refuses these too: the resize would collapse one side to nothing.
    case aspectRatio(Double)
}

extension Image {
    public func decoded() throws -> Pixels {
        switch self {
        case .pixels(let pixels): return pixels
        case .file(let url): return try Self.decode(Data(contentsOf: url))
        case .data(let data): return try Self.decode(data)
        }
    }

    /// Drawn through sRGB, alpha dropped. Note the one place this parts from PIL, which the
    /// reference uses: `convert("RGB")` there discards the alpha channel and keeps the colour
    /// underneath, while a Core Graphics draw composites it against the empty context. They
    /// agree on every opaque image, and differ only inside transparent regions.
    private static func decode(_ data: Data) throws -> Pixels {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
            let space = CGColorSpace(name: CGColorSpace.sRGB)
        else { throw ImageError.undecodable }

        let width = image.width
        let height = image.height
        var padded = [UInt8](repeating: 0, count: width * height * 4)
        guard
            let context = padded.withUnsafeMutableBytes({
                CGContext(
                    data: $0.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            })
        else { throw ImageError.undecodable }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            rgb[3 * pixel] = padded[4 * pixel]
            rgb[3 * pixel + 1] = padded[4 * pixel + 1]
            rgb[3 * pixel + 2] = padded[4 * pixel + 2]
        }
        return Pixels(width: width, height: height, rgb: rgb)
    }
}

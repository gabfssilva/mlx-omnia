import Foundation
import MLX

/// A piece of a prompt. A string literal is one, so a text prompt stays a string.
public enum Segment: Sendable {
    case text(String)
    case image(Image)
}

extension Segment: ExpressibleByStringLiteral, ExpressibleByStringInterpolation {
    public init(stringLiteral value: String) {
        self = .text(value)
    }
}

/// What the model is actually fed. Text is the degenerate case: ids, and nothing else.
///
/// With an image the ids still carry the whole `<|image_pad|>` run — that is what the stop
/// logic and the prompt-token count are counted in — but the rows beneath those ids come from
/// the vision tower, the rotation reads a 3-D position per token, and every step after the
/// prompt shifts by `delta`.
///
/// That last field is why this is a value and not an `[Int32]`. MRoPE advances the position
/// past an image by the image's *longer side*, not by the number of tokens it took: a 22x28
/// image spends 154 rows and 14 positions, so from there on the position runs 140 behind the
/// row count and stays there. A decode step that rotates by the row count instead — which is
/// what `KVCache.offset` would hand it — writes fluent, wrong text.
struct Prompt {
    let ids: [Int32]
    /// The tower's rows, already scattered over the pad run. nil: text only.
    let embeddings: MLXArray?
    /// `[3, L]` int32. nil: text only, where all three sections read the row index.
    let positions: MLXArray?
    let delta: Int

    init(ids: [Int32]) {
        self.ids = ids
        self.embeddings = nil
        self.positions = nil
        self.delta = 0
    }

    init(ids: [Int32], embeddings: MLXArray, positions: MLXArray, delta: Int) {
        self.ids = ids
        self.embeddings = embeddings
        self.positions = positions
        self.delta = delta
    }

    /// Where the rotation reads once the prompt is behind us. Not `row` — that is the cache's
    /// count of what it holds, and an image spends more rows than it does positions. Everything
    /// generated after one stays exactly that far behind, forever.
    func rotation(atRow row: Int) -> RopePositions {
        .offset(row + delta)
    }
}

import SwiftUI
import XCTest
@testable import OmniaPanel

/// Drawing an answer that is still arriving.
///
/// What these hold is the one property the split rests on: a piece, once closed, is the same
/// piece when the next token lands. That is what lets SwiftUI leave it alone, and it is the
/// whole of the saving — without it the panel draws the answer from the top on every frame.
@MainActor
final class MarkupTests: XCTestCase {
    private let fenced = """
        Here is the shape of it:

        ```python
        def step(model):
            return model.forward()

        step(model)
        ```

        And that is the step.

        """

    func testABlankLineInsideAFenceIsNotABoundary() {
        let opened = Markup.boundaries(fenced).map { String(fenced[$0...].prefix(9)) }

        XCTAssertEqual(opened, ["```python", "And that "])
    }

    func testAListIsNotCutIntoTwoLists() {
        let loose = """
            The reasons are these.

            1. the weights are read whole

            2. the cache grows with the answer

            3. the head is read once
            """

        for cut in Markup.boundaries(loose) {
            XCTAssertFalse(loose[cut...].hasPrefix("2."), "an ordered list cut in two starts at 1")
            XCTAssertFalse(loose[cut...].hasPrefix("3."))
        }
    }

    func testAnIndentedLineBelongsToWhatIsAboveIt() {
        // What the first item goes on to say, which is inside the item and not after the list.
        let carried = "- one\n\n    and what it goes on to say\n\n- two"

        XCTAssertNil(Markup.written(carried))
    }

    func testARuleIsItsOwnBlockAndNotAListItem() {
        let ruled = "Before it.\n\n---\n\nAfter it."
        let cuts = Markup.boundaries(ruled).map { ruled[$0...] }

        XCTAssertTrue(cuts.contains { $0.hasPrefix("---") })
        XCTAssertEqual(Markup.gap(after: "Before it."[...], before: "---"[...]), Markup.Gap.apart)
        // And below itself it asks for more than the paragraph under it does, which is the one
        // place the two sides of a seam disagree.
        XCTAssertEqual(Markup.gap(after: "A.\n\n---"[...], before: "After it."[...]), Markup.Gap.apart)
    }

    /// The invariant under the whole split: a boundary, once there, is there in every longer
    /// text. Read a character at a time — which is how an answer arrives — because a line the
    /// answer stops inside can look like one kind of block and turn out to be another.
    func testABoundaryFoundIsABoundaryKept() {
        let whole = """
            Two things follow.

            1. the weights are read whole

            ---

            ```python
            step(model)
            ```

            And that is the step.
            """
        var kept: [Int] = []
        for length in 1...whole.count {
            let sofar = String(whole.prefix(length))
            let found = Markup.boundaries(sofar).map { sofar.distance(from: sofar.startIndex, to: $0) }
            XCTAssertEqual(Array(found.prefix(kept.count)), kept, "at \(length) characters")
            kept = found
        }
    }

    func testAPieceKeepsItsPlaceAsTheRestArrives() {
        let whole = String(repeating: "A paragraph of about forty characters.\n\n", count: 120)
        var closed: [Markup.Piece] = []
        for length in stride(from: 200, to: whole.count, by: 137) {
            let sofar = String(whole.prefix(length))
            guard let (settled, _) = Markup.written(sofar) else { continue }
            // Every piece but the last is closed; the last is still taking blocks.
            let done = settled.dropLast()
            for (before, now) in zip(closed, done) {
                XCTAssertEqual(before.id, now.id)
                XCTAssertEqual(before.text, now.text, "a closed piece was rewritten")
            }
            XCTAssertGreaterThanOrEqual(done.count, closed.count, "a closed piece was dropped")
            closed = Array(done)
        }
        XCTAssertGreaterThan(closed.count, 1, "nothing was ever closed")
    }

    func testWhatIsWrittenIsWhatIsThereWithNothingAdded() throws {
        let text = "One.\n\n## Two\n\nThree.\n\n```sh\nls\n```\n\nFour."
        let (settled, open) = try XCTUnwrap(Markup.written(text))

        XCTAssertEqual(settled.map(\.text).joined() + open.text, text)
        XCTAssertEqual(settled.first?.gap, 0, "nothing goes above the first piece")
    }

    func testADocumentWithNowhereSafeToCutIsDrawnWhole() {
        XCTAssertNil(Markup.written("- one\n\n- two\n\n- three"))
        XCTAssertNil(Markup.written("A paragraph that has not ended yet"))
    }

    /// The piece boundaries are this panel's own and not markdown's, so they must not show.
    ///
    /// Hosted rather than run through `ImageRenderer`: Textual carries a block's spacing on a
    /// preference, and a preference is resolved by an update pass a renderer does not make —
    /// measured there, every gap in the document reads as zero and a seam that lost one looks
    /// right.
    func testTheSamePictureIsDrawnInPiecesOrWhole() throws {
        let sections = (0..<8).map { index in
            """
            ## Section \(index)

            The decode step reads every active weight once per token. A `cache` of length *n*
            adds `n × kv` bytes to that read, which is why a long answer decodes slower.

            ```python
            def step(model, cache):
                return model(cache.keys, cache.values)
            ```

            - the weights are read whole
            - the cache grows with the answer
            """
        }
        let text = sections.joined(separator: "\n\n")

        for length in [400, 900, 1500, 2400, text.count] {
            let sofar = String(text.prefix(length))
            let whole = picture(Prose(text: sofar))
            let pieces = picture(Prose(text: sofar, writing: true))

            XCTAssertEqual(whole.height, pieces.height, accuracy: 0.5, "at \(length)")
            XCTAssertEqual(whole.pixels, pieces.pixels, "the seam shows at \(length)")
        }
    }

    /// The view as a window would lay it out, and the pixels it drew.
    private func picture(_ view: some View) -> (height: CGFloat, pixels: Data?) {
        let host = NSHostingView(
            rootView: AnyView(view.frame(width: 392).environment(\.tokens, .dark))
        )
        host.frame = NSRect(x: 0, y: 0, width: 392, height: 20000)
        let window = NSWindow(
            contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false
        )
        window.contentView = host
        // Preferences settle over update passes, and the layout before they have is the layout
        // with no block spacing in it at all.
        for _ in 0..<6 {
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        }
        let height = host.fittingSize.height
        host.frame = NSRect(x: 0, y: 0, width: 392, height: height)
        host.layoutSubtreeIfNeeded()
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            return (height, nil)
        }
        host.cacheDisplay(in: host.bounds, to: rep)
        guard let bytes = rep.bitmapData else { return (height, nil) }
        return (height, Data(bytes: bytes, count: rep.bytesPerRow * rep.pixelsHigh))
    }
}

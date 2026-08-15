import SwiftUI
import XCTest
@testable import OmniaPanel

@MainActor
final class ChatTests: XCTestCase {
    func testTranscriptTracksWhetherItsScrollIsAtTheEnd() {
        let body = String(reflecting: Transcript(app: AppModel(), openHistory: {}).body)

        XCTAssertTrue(body.contains("ScrollGeometry"))
    }

    /// The reader scrolls against the height the scroll view reports, so it has to be the
    /// height there is. A lazy stack reports what it has built and revises it as the last turn
    /// grows, and past the end of that number is a viewport with no turn in it.
    func testTheTranscriptGrowsByWhatTheTurnBeingWrittenGrewBy() {
        let paragraph = "The decode step reads every active weight once per token, which is "
            + "what makes it bandwidth-bound.\n\n"
        let app = AppModel()
        app.chat.turns = (0..<8).map { index in
            var made = Turn(role: index.isMultiple(of: 2) ? "user" : "assistant")
            made.text = "## Turn \(index)\n\n" + String(repeating: paragraph, count: 3)
            return made
        }
        let written = app.chat.turns[7].text
        let host = hosted(Transcript(app: app, openHistory: {}))
        let before = content(host)

        app.chat.turns[7].text = written + paragraph
        settle(host)

        // What the paragraph is worth on its own, so the transcript is held to the thing that
        // grew rather than to a number written down here.
        let grew = hosted(Prose(text: written + paragraph)).fittingSize.height
            - hosted(Prose(text: written)).fittingSize.height

        XCTAssertEqual(content(host) - before, grew, accuracy: 2)
    }

    /// The scroll view's document, which is the height the reader is scrolling against.
    private func content(_ host: NSView) -> CGFloat {
        func scroller(_ view: NSView) -> NSScrollView? {
            if let found = view as? NSScrollView { return found }
            return view.subviews.compactMap(scroller).first
        }
        return scroller(host)?.documentView?.frame.height ?? -1
    }

    private func hosted(_ view: some View) -> NSHostingView<AnyView> {
        let host = NSHostingView(
            rootView: AnyView(view.frame(width: 392).environment(\.tokens, .dark))
        )
        host.frame = NSRect(x: 0, y: 0, width: 392, height: 480)
        let window = NSWindow(
            contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false
        )
        window.contentView = host
        settle(host)
        return host
    }

    private func settle(_ host: NSView) {
        for _ in 0..<8 {
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        }
    }

    func testExpandedThinkingControlsItsOwnScrollPosition() {
        var turn = Turn(role: "assistant")
        turn.reasoning = "Reasoning"

        let body = String(reflecting: Thinking(turn: turn).body)

        XCTAssertTrue(body.contains("ScrollView"))
        XCTAssertTrue(body.contains("ScrollViewReader"))
    }

    func testLoadReceiptBelongsOnlyToPrecedingUserTurn() {
        var answer = Turn(role: "assistant")
        answer.timings = Timings(
            loadSeconds: 2.4,
            ttftSeconds: 2.8,
            prefillTokensPerSecond: nil,
            tokensPerSecond: nil,
            ceilingFraction: nil,
            speculation: nil
        )
        let chat = ChatModel()
        chat.turns = [Turn(role: "user", text: "Hello"), answer]

        XCTAssertEqual(chat.loadReceipt(after: 0, liveLoad: nil), 2.4)
        XCTAssertNil(chat.loadReceipt(after: 1, liveLoad: nil))
    }

    func testLoadReceiptUsesLiveTimingOnlyWhileAnswerIsWriting() {
        var answer = Turn(role: "assistant")
        answer.writing = true
        let chat = ChatModel()
        chat.turns = [Turn(role: "user", text: "Hello"), answer]

        XCTAssertEqual(chat.loadReceipt(after: 0, liveLoad: 1.2), 1.2)
        chat.turns[1].writing = false
        XCTAssertNil(chat.loadReceipt(after: 0, liveLoad: 1.2))
    }

    func testAssistantMetricsDoNotRepeatLoadTiming() {
        var answer = Turn(role: "assistant")
        answer.timings = Timings(
            loadSeconds: 2.4,
            ttftSeconds: 2.8,
            prefillTokensPerSecond: nil,
            tokensPerSecond: 10,
            ceilingFraction: nil,
            speculation: nil
        )

        let line = answer.metricsLine(live: nil)

        XCTAssertNotNil(line)
        XCTAssertFalse(line?.contains("load") == true)
    }

    func testNonResidentNoteAppearsOnlyForReopenedConversation() {
        let chat = ChatModel()
        chat.model = "org/model"
        chat.turns = [Turn(role: "user", text: "Hello")]

        XCTAssertFalse(chat.shouldShowLoadNote(resident: false))
        chat.restored = true
        XCTAssertTrue(chat.shouldShowLoadNote(resident: false))
        chat.clear()
        XCTAssertFalse(chat.shouldShowLoadNote(resident: false))
    }

    func testTheModelsOwnKnobsBeatWhatTheCheckpointDeclares() {
        var declared = Knobs()
        declared.temperature = 0.6
        declared.topK = 20
        var own = Sampling()
        own.temperature = 0.1

        let made = declared.under(own)

        XCTAssertEqual(made.temperature, 0.1)
        XCTAssertEqual(made.topK, 20, "a knob the model never named stays the checkpoint's")
    }
}

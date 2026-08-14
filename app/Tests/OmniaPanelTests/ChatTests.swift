import SwiftUI
import XCTest
@testable import OmniaPanel

@MainActor
final class ChatTests: XCTestCase {
    func testTranscriptTracksWhetherItsScrollIsAtTheEnd() {
        let body = String(reflecting: Transcript(app: AppModel(), openHistory: {}).body)

        XCTAssertTrue(body.contains("ScrollGeometry"))
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
}

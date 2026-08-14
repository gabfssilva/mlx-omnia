import XCTest
@testable import OmniaPanel

final class LayoutTests: XCTestCase {
    func testPanelUsesCompactHeight() {
        XCTAssertEqual(Panel.height, 620)
    }

    func testStatusCapsSummarySoLogCanFillRemainingHeight() {
        XCTAssertEqual(ServerLayout.summaryMaxHeight, 160)
    }
}

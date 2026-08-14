import XCTest
import SwiftUI
@testable import OmniaPanel

final class LayoutTests: XCTestCase {
    func testMenuBarIconLoadsFromAppResources() {
        XCTAssertEqual(MenuBarIcon.image.size, CGSize(width: 24, height: 15))
        XCTAssertTrue(MenuBarIcon.image.isTemplate)
    }

    func testPanelUsesCompactHeight() {
        XCTAssertEqual(Panel.height, 620)
    }

    @MainActor
    func testTuningKnobUsesNumericInput() {
        let knob = Knob(
            name: "temperature",
            value: .constant(nil),
            low: 0,
            high: 2
        )
        let body = String(reflecting: knob.body)

        XCTAssertTrue(body.contains("TextField"))
        XCTAssertFalse(body.contains("Rail"))
    }

}

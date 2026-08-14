import XCTest
@testable import OmniaPanel

final class LayoutTests: XCTestCase {
    func testMenuBarIconLoadsFromAppResources() {
        XCTAssertEqual(MenuBarIcon.image.size, CGSize(width: 24, height: 15))
        XCTAssertTrue(MenuBarIcon.image.isTemplate)
    }

    func testPanelUsesCompactHeight() {
        XCTAssertEqual(Panel.height, 620)
    }

}

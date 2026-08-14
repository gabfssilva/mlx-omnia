import XCTest
@testable import OmniaPanel

final class NavigationTests: XCTestCase {
    func testPrimaryTabsContainOnlyDestinations() {
        XCTAssertEqual(Tab.allCases, [.server, .models, .chat])
    }

    func testServerTabsPutSettingsFirst() {
        XCTAssertEqual(ServerTab.allCases, [.settings, .status])
    }

    func testModelTabsGroupModelTools() {
        XCTAssertEqual(ModelsTab.allCases, [.library, .quantize, .benchmark])
        XCTAssertEqual(ModelsTab.allCases.map(\.label), ["Library", "Quantize", "Benchmark"])
    }
}

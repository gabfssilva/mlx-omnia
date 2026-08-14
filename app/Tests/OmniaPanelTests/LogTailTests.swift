import Foundation
import XCTest
@testable import OmniaPanel

final class LogTailTests: XCTestCase {
    func testReadsWholeFileWithinLimit() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("first\nsecond\n".utf8).write(to: file)

        XCTAssertEqual(try LogTail.read(file, maxBytes: 100), "first\nsecond\n")
    }

    func testDropsPartialFirstLineWhenLimited() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("first line\nsecond line\nthird line\n".utf8).write(to: file)

        XCTAssertEqual(try LogTail.read(file, maxBytes: 24), "second line\nthird line\n")
    }

    func testOpenForWritingTruncatesExistingFile() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("stale output".utf8).write(to: file)

        let handle = try LogTail.openForWriting(file)
        try handle.close()

        XCTAssertEqual(try Data(contentsOf: file), Data())
    }
}

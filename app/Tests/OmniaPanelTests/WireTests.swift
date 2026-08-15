import XCTest
@testable import OmniaPanel

/// What the panel writes when a field is left empty. The daemon resolves four levels —
/// request over profile over model over checkpoint — and an *absent* knob is what makes a
/// level defer to the one under it. A body that spelled every field would be this screen
/// silently pinning the checkpoint's own defaults to whatever the editor happened to show.
final class WireTests: XCTestCase {
    private func written(_ body: any Encodable) throws -> [String: Any] {
        let data = try Client.encoder.encode(AnyEncodable(body))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testAnEmptyEditorWritesNoKnobAtAll() throws {
        XCTAssertTrue(try written(Sampling()).isEmpty)
    }

    func testAKnobTheEditorSetsIsWrittenUnderTheDialectsOwnSpelling() throws {
        var sampling = Sampling()
        sampling.topP = 0.9
        sampling.repetitionPenalty = 1.1
        sampling.reasoningEffort = "high"

        let body = try written(sampling)

        XCTAssertEqual(body["top_p"] as? Double, 0.9)
        XCTAssertEqual(body["repetition_penalty"] as? Double, 1.1)
        XCTAssertEqual(body["reasoning_effort"] as? String, "high")
        XCTAssertNil(body["temperature"])
    }

    func testAProfileWithNoSystemPromptWritesNone() throws {
        let body = try written(ProfileBody(sampling: Sampling(), systemPrompt: nil))

        XCTAssertNil(body["system_prompt"])
        XCTAssertTrue(try XCTUnwrap(body["sampling"] as? [String: Any]).isEmpty)
    }
}

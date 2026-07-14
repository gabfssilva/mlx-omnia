import Foundation
import Testing

@testable import Sideros

struct Case: Decodable, Sendable {
    let text: String
    let ids: [Int32]
}

private let fixtures = Bundle.module.resourceURL!.appendingPathComponent("Fixtures")

private let tokenizer = try! GPT2Tokenizer(
    vocab: fixtures.appendingPathComponent("vocab.json"),
    merges: fixtures.appendingPathComponent("merges.txt")
)

private let cases = try! JSONDecoder().decode(
    [Case].self, from: Data(contentsOf: fixtures.appendingPathComponent("tokenizer.json")))

@Test func fixtureIsLoaded() {
    #expect(cases.count == 12)
}

@Test(arguments: cases)
func encodeMatchesReference(testCase: Case) {
    #expect(tokenizer.encode(testCase.text) == testCase.ids)
}

@Test(arguments: cases)
func decodeRoundTrips(testCase: Case) {
    #expect(tokenizer.decode(testCase.ids) == testCase.text)
}

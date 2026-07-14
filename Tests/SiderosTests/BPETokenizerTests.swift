import Foundation
import Testing

@testable import Sideros

/// Neither tokenizer round-trips to the input: Gemma prepends `<bos>`, and special tokens
/// decode back as their literal content, so the reference decode is stored too.
struct BPECase: Decodable, Sendable {
    let text: String
    let ids: [Int32]
    let decoded: String
}

private func bpeCases(_ name: String) -> [BPECase] {
    let url = Bundle.module.resourceURL!
        .appendingPathComponent("Fixtures")
        .appendingPathComponent(name)
    return try! JSONDecoder().decode([BPECase].self, from: Data(contentsOf: url))
}

private let qwen2 = try! Qwen2Tokenizer(directory: qwen2Directory())
private let qwen2Cases = bpeCases("qwen2_tokenizer.json")

private let gemma3 = try! Gemma3Tokenizer(directory: gemma3Directory())
private let gemma3Cases = bpeCases("gemma3_tokenizer.json")

private let lfm2 = try! Lfm2Tokenizer(directory: lfm2MoeDirectory())
private let lfm2Cases = bpeCases("lfm2_tokenizer.json")

// Qwen3.5 adds \p{M} to Qwen2's pattern on paper, but HF's oniguruma leaves combining
// marks outside the letter runs anyway — Qwen2Tokenizer reproduces it; the Devanagari,
// Arabic and Thai cases pin that.
private let qwen35 = try! Qwen2Tokenizer(directory: qwen35Directory())
private let qwen35Cases = bpeCases("qwen3_5_tokenizer.json")

@Test(arguments: qwen2Cases)
func qwen2EncodeMatchesReference(testCase: BPECase) {
    #expect(qwen2.encode(testCase.text) == testCase.ids)
}

@Test(arguments: qwen2Cases)
func qwen2DecodeMatchesReference(testCase: BPECase) {
    #expect(qwen2.decode(testCase.ids) == testCase.decoded)
}

@Test(arguments: gemma3Cases)
func gemma3EncodeMatchesReference(testCase: BPECase) {
    #expect(gemma3.encode(testCase.text) == testCase.ids)
}

@Test(arguments: gemma3Cases)
func gemma3DecodeMatchesReference(testCase: BPECase) {
    #expect(gemma3.decode(testCase.ids) == testCase.decoded)
}

@Test(arguments: lfm2Cases)
func lfm2EncodeMatchesReference(testCase: BPECase) {
    #expect(lfm2.encode(testCase.text) == testCase.ids)
}

@Test(arguments: lfm2Cases)
func lfm2DecodeMatchesReference(testCase: BPECase) {
    #expect(lfm2.decode(testCase.ids) == testCase.decoded)
}

@Test(arguments: qwen35Cases)
func qwen35EncodeMatchesReference(testCase: BPECase) {
    #expect(qwen35.encode(testCase.text) == testCase.ids)
}

@Test(arguments: qwen35Cases)
func qwen35DecodeMatchesReference(testCase: BPECase) {
    #expect(qwen35.decode(testCase.ids) == testCase.decoded)
}

import Foundation
import Sideros

/// Resolved from the Hugging Face cache; populate it with `make weights`.
func hubSnapshot(_ repo: String) throws -> URL {
    let cache =
        ProcessInfo.processInfo.environment["HF_HOME"].map { URL(filePath: $0) }
        ?? URL(filePath: NSHomeDirectory()).appending(path: ".cache/huggingface")
    let snapshots = cache.appending(path: "hub/models--\(repo.replacingOccurrences(of: "/", with: "--"))/snapshots")
    let entries = try FileManager.default.contentsOfDirectory(
        at: snapshots, includingPropertiesForKeys: nil)
    guard let first = entries.first else {
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: snapshots.path])
    }
    return first
}

func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}

var repo: String?
var quantize = false
var label: String?
var tokens = 128
var runs = 5
var promptPath = "reference/bench_prompt.txt"
var imagePath: String?
var emit = false
var emitIds = false
var draft: String?
var speculated = 5

var iterator = CommandLine.arguments.dropFirst().makeIterator()
while let argument = iterator.next() {
    switch argument {
    case "--quantize": quantize = true
    case "--label": label = iterator.next()
    case "--tokens": tokens = iterator.next().flatMap { Int($0) } ?? tokens
    case "--runs": runs = iterator.next().flatMap { Int($0) } ?? runs
    case "--prompt": promptPath = iterator.next() ?? promptPath
    case "--image": imagePath = iterator.next()
    case "--emit": emit = true
    case "--ids": emitIds = true
    case "--draft": draft = iterator.next()
    case "--speculated": speculated = iterator.next().flatMap { Int($0) } ?? speculated
    default: repo = argument
    }
}

guard let repo else {
    print("usage: sideros-bench <hub-repo> [--quantize] [--label name] [--tokens N] [--runs N] [--prompt path] [--image path]")
    exit(1)
}

let prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
let input: [Segment] = imagePath.map {
    [.image(.file(URL(filePath: $0))), .text(prompt)]
} ?? [.text(prompt)]
let model = try LanguageModel(
    directory: hubSnapshot(repo),
    options: ModelOptions(
        quantization: quantize ? Quantization() : nil,
        draft: draft.map { try hubSnapshot($0) }))

/// One timed generation. TTFT covers prefill plus the first decoded token; the stream
/// runs to exactly `tokens` forward passes because `stop: []` disables the stop set.
@MainActor
func run() async throws -> (ttft: Double, decode: Double) {
    let clock = ContinuousClock()
    let start = clock.now
    var first: ContinuousClock.Instant?
    let options = GenerationOptions(maxTokens: tokens, sampler: .greedy, stop: [], speculated: speculated)
    for try await _ in try model(input, options: options) where first == nil {
        first = clock.now
    }
    let end = clock.now
    let ttft = seconds((first ?? end) - start)
    return (ttft, Double(tokens - 1) / (seconds(end - start) - ttft))
}

// Not a benchmark: prints the greedy continuation for parity checks against mlx-lm.
// --ids prints the token ids instead, for teacher-forced logit comparisons.
if emit {
    let options = GenerationOptions(maxTokens: tokens, sampler: .greedy, stop: [])
    if imagePath != nil {
        print(try await model.generate(input, options: options))
    } else if emitIds {
        print(model.generateIds(prompt, options: options).map(String.init).joined(separator: " "))
    } else {
        print(try await model.generate(prompt, options: options))
    }
    exit(0)
}

_ = try await run()
var samples: [(ttft: Double, decode: Double)] = []
for _ in 0..<runs { samples.append(try await run()) }

let decodes = samples.map(\.decode)
let name = (label ?? repo).padding(toLength: 18, withPad: " ", startingAt: 0)
print(name + String(
    format: "ttft %6.0f ms   decode %6.1f tok/s   (min %.1f, max %.1f, n=%d)",
    median(samples.map(\.ttft)) * 1000,
    median(decodes), decodes.min()!, decodes.max()!, runs))

import Foundation
import Sideros

/// The model behind the HTTP layer: one checkpoint, one generation at a time. There is no
/// batching yet, so concurrent requests queue on `gate` instead of interleaving — which is
/// also what keeps MLX's graph to a single writer. Unchecked because `LanguageModel` holds
/// MLX arrays, which are not Sendable; the gate is what makes that safe.
public final class Engine: @unchecked Sendable {
    public let name: String
    private let model: LanguageModel
    private let template: ChatTemplate
    private let maxTokens: Int
    private let gate = Gate()

    public init(
        directory: URL, name: String, maxTokens: Int = 2048,
        options: ModelOptions = ModelOptions()
    ) throws {
        self.name = name
        self.maxTokens = maxTokens
        self.template = try ChatTemplate(directory: directory)
        self.model = try LanguageModel(directory: directory, options: options)
    }

    public func stream(_ request: ChatRequest) throws -> AsyncThrowingStream<Completion, Error> {
        let prompt = try template(request.messages)
        let options = GenerationOptions(
            maxTokens: request.maxTokens ?? maxTokens,
            sampler: sampler(request),
            stopStrings: request.stop,
            seed: request.seed)

        return AsyncThrowingStream { continuation in
            let task = Task {
                await gate.acquire()
                do {
                    for try await item in model.stream(prompt, options: options) {
                        // A client that hung up stops costing tokens at the next one.
                        guard !Task.isCancelled else { break }
                        continuation.yield(item)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await gate.release()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func sampler(_ request: ChatRequest) -> Sampler {
        guard let temperature = request.temperature, temperature > 0 else { return .greedy }
        guard let topP = request.topP, topP < 1 else { return .temperature(temperature) }
        return .topP(topP, temperature: temperature)
    }
}

/// A FIFO async mutex: the request that arrives while a generation runs waits its turn
/// rather than being rejected.
private actor Gate {
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        guard busy else {
            busy = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    func release() {
        if waiting.isEmpty {
            busy = false
        } else {
            waiting.removeFirst().resume()
        }
    }
}

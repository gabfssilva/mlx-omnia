import Foundation
import MLX

public struct LanguageModel: Model {
    public enum LoadError: Error {
        case unsupported(modelType: String)
    }

    public enum PromptError: Error {
        /// An image against a checkpoint with no tower — or whose export dropped it. Refused,
        /// never quietly dropped: a prompt that says it showed a picture and did not is worse
        /// than one that fails.
        case noVisionTower
        /// The tower produced a different number of rows than the pad run reserved for it.
        case featureCount(rows: Int, pads: Int)
    }

    /// The image half of a VLM: the tower, the processor that feeds it, and the ids that mark
    /// where its rows go.
    private struct Vision {
        let tower: Qwen35Vision
        let processor: Qwen35ImageProcessor
        let pad: Int32
        let start: Int32
        let end: Int32
        let merge: Int
    }

    /// Enough of config.json to pick the architecture; each config decodes the rest.
    private struct Header: Decodable {
        let modelType: String
        let eosTokenId: EOSTokens?
    }

    /// A scalar or a list depending on the family — Qwen3.6 exports declare two.
    private struct EOSTokens: Decodable {
        let ids: Set<Int32>
        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let id = try? container.decode(Int32.self) {
                ids = [id]
            } else {
                ids = Set(try container.decode([Int32].self))
            }
        }
    }

    private let decoder: any CausalLM
    private let draft: (any CausalLM)?
    private let tokenizer: any Tokenizer
    private let eos: Set<Int32>
    private let vision: Vision?

    public init(directory: URL, options: ModelOptions = ModelOptions()) throws {
        var tower: Vision?
        draft = try options.draft.map { try LanguageModel(directory: $0).decoder }
        let jsonDecoder = JSONDecoder()
        jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
        let header = try jsonDecoder.decode(
            Header.self, from: Data(contentsOf: directory.appending(path: "config.json")))

        let precision = options.precision
        let quantization = options.quantization
        // Qwen3.5's 0.8B declares eos only inside text_config; every other family
        // spells it at the top level.
        var eosFallback: Set<Int32> = []
        switch header.modelType {
        case "gpt2":
            let config = try GPT2Config(directory: directory)
            decoder = GPT2(
                config: config,
                parameters: try loadGPT2Parameters(
                    directory: directory, config: config, precision: precision,
                    quantization: quantization),
                quantization: quantization)
            tokenizer = try GPT2Tokenizer(
                vocab: directory.appending(path: "vocab.json"),
                merges: directory.appending(path: "merges.txt"))
        case "qwen2":
            let config = try Qwen2Config(directory: directory)
            decoder = Qwen2(
                config: config,
                parameters: try loadQwen2Parameters(
                    directory: directory, config: config, precision: precision,
                    quantization: quantization),
                quantization: config.quantization ?? quantization,
                skinny: options.draft != nil)
            tokenizer = try Qwen2Tokenizer(directory: directory)
        case "qwen3":
            let config = try Qwen3Config(directory: directory)
            decoder = Qwen3(
                config: config,
                parameters: try loadQwen3Parameters(
                    directory: directory, config: config, precision: precision,
                    quantization: quantization),
                quantization: config.quantization ?? quantization,
                skinny: options.draft != nil)
            tokenizer = try Qwen2Tokenizer(directory: directory)
        case "qwen3_moe":
            let config = try Qwen3MoeConfig(directory: directory)
            decoder = Qwen3Moe(
                config: config,
                parameters: try loadQwen3MoeParameters(
                    directory: directory, config: config, precision: precision,
                    quantization: quantization),
                quantization: config.quantization ?? quantization)
            tokenizer = try Qwen2Tokenizer(directory: directory)
        case "qwen3_5", "qwen3_5_moe":
            let config = try Qwen35Config(directory: directory)
            let parameters = try loadQwen35Parameters(
                directory: directory, config: config, precision: precision)
            decoder = Qwen35(
                config: config, parameters: parameters, quantization: config.quantization)
            // tokenizer.json spells \p{M} into the Qwen2 pattern, but HF's oniguruma
            // leaves marks outside the letter runs anyway — the behavior is Qwen2's,
            // and the combining-mark fixture cases pin it.
            tokenizer = try Qwen2Tokenizer(directory: directory)
            eosFallback = [config.eosTokenId]
            // The tower comes up only if the export kept its weights *and* the processor's
            // config came along. Missing either, the checkpoint loads as text and an image
            // against it throws.
            if let vision = config.visionConfig, let pad = config.imageTokenId,
                let start = config.visionStartTokenId, let end = config.visionEndTokenId,
                let processor = try? Qwen35ImageProcessor(directory: directory)
            {
                let visual = parameters.filter { $0.key.hasPrefix("visual.") }
                tower = Vision(
                    tower: Qwen35Vision(config: vision, parameters: visual),
                    processor: processor, pad: pad, start: start, end: end,
                    merge: vision.spatialMergeSize)
            }
        case "lfm2_moe":
            let config = try Lfm2MoeConfig(directory: directory)
            decoder = Lfm2Moe(
                config: config,
                parameters: try loadLfm2MoeParameters(
                    directory: directory, config: config, precision: precision))
            tokenizer = try Lfm2Tokenizer(directory: directory)
        case "gemma3_text":
            let config = try Gemma3Config(directory: directory)
            decoder = Gemma3(
                config: config,
                parameters: try loadGemma3Parameters(
                    directory: directory, config: config, precision: precision,
                    quantization: quantization),
                quantization: quantization)
            tokenizer = try Gemma3Tokenizer(directory: directory)
        default:
            throw LoadError.unsupported(modelType: header.modelType)
        }
        eos = header.eosTokenId?.ids ?? eosFallback
        vision = tower
    }

    /// Ids, spliced embeddings and 3-D positions. The tower runs here, once per image, before
    /// the generation loop exists — so nothing downstream has to know a picture went by.
    func compose(_ segments: [Segment]) throws -> Prompt {
        var ids: [Int32] = []
        var spliced: [(start: Int, features: MLXArray)] = []
        var temporal: [Int32] = []
        var rows: [Int32] = []
        var columns: [Int32] = []
        var position: Int32 = 0

        /// Text reads the same position in all three sections, and advances by one.
        func text(_ encoded: [Int32]) {
            for id in encoded {
                ids.append(id)
                temporal.append(position)
                rows.append(position)
                columns.append(position)
                position += 1
            }
        }

        for segment in segments {
            switch segment {
            case .text(let string):
                text(tokenizer.encode(string))

            case .image(let image):
                guard let vision else { throw PromptError.noVisionTower }
                let (patches, grid) = try vision.processor(image)
                let features = vision.tower(patches, grid: grid)

                // The two markers are ordinary text; only the pads between them carry a grid.
                text([vision.start])

                let height = grid.h / vision.merge
                let width = grid.w / vision.merge
                let pads = grid.t * height * width
                guard features.dim(0) == pads else {
                    throw PromptError.featureCount(rows: features.dim(0), pads: pads)
                }

                spliced.append((ids.count, features))
                for _ in 0..<grid.t {
                    for row in 0..<height {
                        for column in 0..<width {
                            ids.append(vision.pad)
                            temporal.append(position)
                            rows.append(position + Int32(row))
                            columns.append(position + Int32(column))
                        }
                    }
                }
                // And this is the whole trap: the position resumes past the image's longer
                // side, not past the row count it just spent.
                position += Int32(max(grid.h, grid.w) / vision.merge)

                text([vision.end])
            }
        }

        guard !spliced.isEmpty else { return Prompt(ids: ids) }

        // MLXArray is a reference: the scatter lands in the array the embed just produced, and
        // nobody else holds it.
        let embeddings = decoder.embed(MLXArray(ids, [1, ids.count]), offset: 0)
        for (start, features) in spliced {
            embeddings[0..., start..<(start + features.dim(0)), 0...] =
                expandedDimensions(features, axis: 0)
        }

        let highest = max(temporal.max()!, rows.max()!, columns.max()!)
        return Prompt(
            ids: ids, embeddings: embeddings,
            positions: MLXArray(temporal + rows + columns, [3, ids.count]),
            delta: Int(highest) + 1 - ids.count)
    }

    public func callAsFunction(
        _ prompt: String, options: GenerationOptions = GenerationOptions()
    ) -> AsyncThrowingStream<String, Error> {
        let generation = Generation(
            decoder: decoder, draft: draft, tokenizer: tokenizer,
            prompt: Prompt(ids: tokenizer.encode(prompt)),
            stop: options.stop ?? eos, options: options)
        return AsyncThrowingStream(unfolding: { generation.next() })
    }

    public func callAsFunction(
        _ segments: [Segment], options: GenerationOptions = GenerationOptions()
    ) throws -> AsyncThrowingStream<String, Error> {
        let generation = Generation(
            decoder: decoder, draft: draft, tokenizer: tokenizer, prompt: try compose(segments),
            stop: options.stop ?? eos, options: options)
        return AsyncThrowingStream(unfolding: { generation.next() })
    }

    public func generate(
        _ segments: [Segment], options: GenerationOptions = GenerationOptions()
    ) async throws -> String {
        var text = ""
        for try await chunk in try self(segments, options: options) { text += chunk }
        return text
    }

    /// The same deltas as `callAsFunction`, followed by one `.done`: why it ended and what
    /// it cost. This is the stream an API server reports from — and the only one that
    /// honors `options.stopStrings`.
    public func stream(_ prompt: String, options: GenerationOptions = GenerationOptions())
        -> AsyncThrowingStream<Completion, Error>
    {
        stream(Prompt(ids: tokenizer.encode(prompt)), options: options)
    }

    public func stream(_ segments: [Segment], options: GenerationOptions = GenerationOptions())
        throws -> AsyncThrowingStream<Completion, Error>
    {
        stream(try compose(segments), options: options)
    }

    private func stream(_ prompt: Prompt, options: GenerationOptions)
        -> AsyncThrowingStream<Completion, Error>
    {
        let ids = prompt.ids
        let generation = Generation(
            decoder: decoder, draft: draft, tokenizer: tokenizer, prompt: prompt,
            stop: options.stop ?? eos, options: options)
        var filter = StopFilter(options.stopStrings)
        var halted: String?
        var started = false
        var done = false

        return AsyncThrowingStream(unfolding: {
            if done { return nil }
            if !started {
                started = true
                return .start(promptTokens: ids.count)
            }
            while halted == nil {
                guard let delta = generation.next() else { break }
                let (emit, halt) = filter.push(delta)
                halted = halt
                // A halt still owes the caller the text before the stop string.
                if !emit.isEmpty { return .delta(emit) }
                if halt != nil { break }
            }
            done = true
            return .done(
                Finish(
                    reason: halted != nil || generation.stopped ? .stop : .length,
                    stopString: halted, promptTokens: ids.count,
                    completionTokens: generation.generated.count))
        })
    }

    public func generate(_ prompt: String, options: GenerationOptions = GenerationOptions()) async throws
        -> String
    {
        var text = ""
        for try await chunk in self(prompt, options: options) { text += chunk }
        return text
    }

    /// The generated ids rather than text: what a parity harness feeds to another
    /// runtime to compare logits position by position.
    public func generateIds(_ prompt: String, options: GenerationOptions = GenerationOptions())
        -> [Int32]
    {
        let generation = Generation(
            decoder: decoder, draft: draft, tokenizer: tokenizer,
            prompt: Prompt(ids: tokenizer.encode(prompt)),
            stop: options.stop ?? eos, options: options)
        while generation.next() != nil {}
        return generation.generated
    }
}

/// The generation loop, one token per `next()`. Unchecked because MLX arrays are not
/// Sendable and the stream drives this from a single consumer at a time.
final class Generation: @unchecked Sendable {
    private let decoder: any CausalLM
    /// The same decoder when it can be told a position the cache does not know. Resolved once,
    /// here, rather than by a type query on every step.
    private let positioned: (any PositionedLM)?
    private let draft: (any CausalLM)?
    private let tokenizer: any Tokenizer
    private let stop: Set<Int32>
    private let options: GenerationOptions

    private var cache: [KVCache]
    private var draftCache: [KVCache] = []
    private let prompt: Prompt
    /// Sampled on the GPU and async-evaluated, but not yet read back as an Int32.
    private var upcoming: MLXArray?
    fileprivate var generated: [Int32] = []
    /// A stop token ended it, rather than `maxTokens` running out.
    fileprivate var stopped = false
    private var emitted = ""
    /// Speculative state: the whole sequence both models agree on, and the accepted
    /// tokens not yet handed to the stream.
    private var committed: [Int32]
    private var pending: [Int32] = []

    init(
        decoder: any CausalLM, draft: (any CausalLM)? = nil, tokenizer: any Tokenizer,
        prompt: Prompt, stop: Set<Int32>, options: GenerationOptions
    ) {
        self.decoder = decoder
        self.positioned = decoder as? any PositionedLM
        self.draft = draft
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.committed = prompt.ids
        self.cache = decoder.makeCache()
        self.draftCache = draft?.makeCache() ?? []
        self.stop = stop
        self.options = options
        if let seed = options.seed { MLXRandom.seed(seed) }
    }

    /// One forward over `tokens`, returning the sampled next token, still lazy.
    private func step(_ tokens: MLXArray) -> MLXArray {
        // Read before the forward: the layers grow the cache as they run. The two offsets part
        // company here — `embed` wants the row (only GPT-2's learned table reads it), the
        // rotation wants the position, and after an image those are not the same number.
        let offset = cache[0].offset
        let embeddings = decoder.embed(tokens, offset: offset)
        let (logits, grown) = rotating(embeddings, prompt.rotation(atRow: offset), cache: cache)
        cache = grown
        return options.sampler.pick(logits[0, -1])
    }

    /// The prompt's own forward. Its rows may already hold the tower's output, and its rotation
    /// reads a position per token rather than a count of them.
    private func prefill() -> MLXArray {
        let embeddings =
            prompt.embeddings
            ?? decoder.embed(MLXArray(prompt.ids, [1, prompt.ids.count]), offset: 0)
        let (logits, grown) = rotating(
            embeddings, prompt.positions.map { .explicit($0) } ?? .offset(0), cache: cache)
        cache = grown
        return options.sampler.pick(logits[0, -1])
    }

    /// Six of the seven models take their rotation from the cache and never see this. For the
    /// seventh with a text prompt, `delta` is zero and the positions are the row counts — so it
    /// runs the same expression it always did, and is bit-identical by construction.
    private func rotating(
        _ embeddings: MLXArray, _ positions: RopePositions, cache: [KVCache]
    ) -> (logits: MLXArray, cache: [KVCache]) {
        guard let positioned else { return decoder.logits(embeddings, cache: cache) }
        return positioned.logits(embeddings, positions: positions, cache: cache)
    }

    /// Only the generated ids are decoded, never the prompt: that is what keeps the
    /// prompt's own special tokens (Gemma's `<bos>`) out of the stream.
    func next() -> String? {
        // A speculative round rebuilds the sequence from its ids and re-embeds it as text,
        // which would throw the picture away. Qwen3.6 has no draft to begin with — its DeltaNet
        // cache cannot rewind — so this is a guard, not a path.
        if let draft, options.sampler.isGreedy, prompt.embeddings == nil {
            return speculated(draft)
        }
        return pipelined()
    }

    /// One forward over `tokens`, appending their keys to the model's cache and
    /// returning the full logits row per position.
    private func forward(_ model: any CausalLM, cache: inout [KVCache], tokens: MLXArray)
        -> MLXArray
    {
        let embeddings = model.embed(tokens, offset: cache[0].offset)
        let (logits, grown) = model.logits(embeddings, cache: cache)
        cache = grown
        return logits
    }

    /// One speculation round: the draft proposes `speculated` tokens one by one, the
    /// target checks them all in a single forward — the round costs one read of the big
    /// weights however many survive. A rejected draft token leaves keys in both caches
    /// that describe a sequence that never happened; `trimmed` cuts them back to the
    /// accepted prefix. Greedy only: equality against the target's own argmax is the
    /// whole acceptance rule, so the output is the target's, token for token.
    private func speculate(_ draft: any CausalLM) {
        // The first round is the target's alone: the prompt forward already yields a
        // token (that is TTFT), and the draft has nothing to add to it.
        if committed.count == prompt.ids.count {
            let logits = forward(decoder, cache: &cache, tokens: MLXArray(committed, [1, committed.count]))
            let token = argMax(logits[0, -1], axis: -1).item(Int32.self)
            committed.append(token)
            pending.append(token)
            return
        }

        // Each model is fed whatever committed tokens its cache has not seen — after a
        // full acceptance the draft is one token further behind than the target. The
        // round still has a single sync, on `judged` below.
        let gap = Array(committed[draftCache[0].offset...])
        var input = MLXArray(gap, [1, gap.count])
        var draftTokens: [MLXArray] = []
        for _ in 0..<options.speculated {
            let logits = forward(draft, cache: &draftCache, tokens: input)
            let token = argMax(logits[0, -1], axis: -1)
            // Dispatch each step as it is built: the GPU runs draft step n while the
            // CPU builds step n+1, instead of idling until the round's one sync.
            asyncEval(token)
            draftTokens.append(token)
            input = token.reshaped(1, 1)
        }

        let targetGap = Array(committed[cache[0].offset...])
        let verified = concatenated(
            [MLXArray(targetGap)] + draftTokens.map { $0.reshaped(1) }, axis: 0)
        let logits = forward(decoder, cache: &cache, tokens: verified.reshaped(1, -1))
        // Row j predicts the token after verified[j]; the last drafts.count + 1 rows
        // judge every draft and provide the target's own token past the accepted run.
        let judged = argMax(
            logits[0, (targetGap.count - 1)..., 0...], axis: -1)

        eval(judged)
        let drafts = draftTokens.map { $0.item(Int32.self) }
        let predictions = judged.asArray(Int32.self)

        var accepted: [Int32] = []
        for (i, token) in drafts.enumerated() {
            guard predictions[i] == token else { break }
            accepted.append(token)
        }

        let length = committed.count + accepted.count
        cache = cache.map { $0.trimmed(to: length) }
        draftCache = draftCache.map { $0.trimmed(to: min(length, $0.offset)) }

        let round = accepted + [predictions[accepted.count]]
        committed += round
        pending += round
    }

    private func speculated(_ draft: any CausalLM) -> String? {
        while generated.count < options.maxTokens {
            if pending.isEmpty { speculate(draft) }
            let token = pending.removeFirst()
            guard !stop.contains(token) else {
                stopped = true
                return nil
            }
            generated.append(token)

            let text = tokenizer.decode(generated)
            guard text != emitted, !text.hasSuffix("\u{FFFD}") else { continue }

            defer { emitted = text }
            return String(text.dropFirst(emitted.count))
        }
        return nil
    }

    private func pipelined() -> String? {
        if upcoming == nil, generated.isEmpty {
            let first = prefill()
            asyncEval(first)
            upcoming = first
        }
        while generated.count < options.maxTokens, let current = upcoming {
            // The pipeline: submit the next step's graph before syncing this one's token,
            // so the CPU builds graph n+1 while the GPU still works on step n. The step
            // past a stop token is speculative waste; the stream never reads it.
            if generated.count + 1 < options.maxTokens {
                let next = step(current.reshaped(1, 1))
                asyncEval(next)
                upcoming = next
            } else {
                upcoming = nil
            }

            let token = current.item(Int32.self)
            guard !stop.contains(token) else {
                stopped = true
                return nil
            }
            generated.append(token)

            // A token can be half of a UTF-8 sequence — a byte-level piece, or one of
            // Gemma's byte fallbacks. Hold it back until the next one completes it.
            let text = tokenizer.decode(generated)
            guard text != emitted, !text.hasSuffix("\u{FFFD}") else { continue }

            defer { emitted = text }
            return String(text.dropFirst(emitted.count))
        }
        return nil
    }
}

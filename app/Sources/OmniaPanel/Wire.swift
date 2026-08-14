// The daemon's shapes, typed off what it serves — `mlx_omnia.server`'s catalog, state and
// benchmark responses.
//
// Where a field is optional there it is optional here, because FastAPI serialises null
// rather than dropping the key. Fields this panel does not draw are left out: a decoder
// ignores what it was not asked for, which is what lets the daemon grow a field without
// breaking this.

import Foundation

// ── the machine, and every resource the stream carries ───────────────────

struct SystemInfo: Decodable {
    let chip: String
    let gpuCores: Int
    let memoryBytes: Int
    let bandwidthSustainedGbs: Double
    let diskFreeBytes: Int
    let version: String
    let constants: [String: String]
}

struct Health: Decodable {
    let status: String
    let models: [String]
    let pid: Int
    let uptime: Double
    let version: String
}

struct Slot: Decodable, Identifiable {
    let id: String
    let weightsBytes: Int
    let kvBytes: Int
    let loadedAt: Double
    let lastUsed: Double?
}

struct Queue: Decodable {
    let running: Int
    let waiting: Int
}

struct EngineState: Decodable {
    let models: [Slot]
    let queue: Queue
    let residentBytes: Int
    let kvBytes: Int
    /// The two tiers of the prefix cache, as they stand: what the resident tries hold, and
    /// what the conversations they evicted weigh on disk.
    let prefixMemoryBytes: Int
    let prefixDiskBytes: Int
}

/// `Setting.value` is a number, a string, or null — and which one it is is the setting's.
enum Scalar: Decodable {
    case number(Double)
    case text(String)
    case nothing

    init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if single.decodeNil() { self = .nothing } else if let value = try? single.decode(Double.self) {
            self = .number(value)
        } else if let value = try? single.decode(String.self) {
            self = .text(value)
        } else {
            self = .nothing
        }
    }

    var number: Double? {
        if case let .number(value) = self { return value }
        return nil
    }

    var text: String? {
        if case let .text(value) = self { return value }
        return nil
    }

    var unset: Bool {
        if case .nothing = self { return true }
        return false
    }
}

struct Setting: Decodable {
    let value: Scalar
    let effect: String
    let note: String?
}

/// What one field of the config is set to. `nothing` is a value the daemon means — no key,
/// no TTL — and is why this is written rather than left out.
enum Written: Encodable {
    case whole(Int)
    case text(String)
    case nothing

    func encode(to encoder: Encoder) throws {
        var out = encoder.singleValueContainer()
        switch self {
        case let .whole(value): try out.encode(value)
        case let .text(value): try out.encode(value)
        case .nothing: try out.encodeNil()
        }
    }
}

/// A PATCH of `/admin/config`, spelling the keys it carries.
///
/// The daemon writes exactly what the body named — `model_fields_set` — so a key left out
/// and a key sent as null say different things: the first keeps what the row holds, the
/// second clears it. A struct of optionals could not tell them apart, because the encoder
/// drops a nil rather than writing one.
struct ConfigPatch: Encodable {
    let fields: [String: Written]

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ name: String) { stringValue = name }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    func encode(to encoder: Encoder) throws {
        var out = encoder.container(keyedBy: Key.self)
        for (name, value) in fields { try out.encode(value, forKey: Key(name)) }
    }
}

struct SamplingDefaults: Decodable {
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let minP: Double?
    let repetitionPenalty: Double?
}

struct CatalogEntry: Decodable, Identifiable, Equatable {
    let id: String
    let architecture: String
    let quantization: String?
    let dtype: String?
    let context: Int?
    let bytesOnDisk: Int
    let bytesPerToken: Double?
    let kvBytesPerToken: Double?
    let vocabSize: Int?
    let shape: String?
    let resident: Bool
    let supported: Bool
    /// Whether a turn may carry an image. Per checkpoint and not per family: two of the
    /// ported architectures carry VL in the name with the tower dropped at load, and one
    /// family ships both multimodal and text-only conversions.
    let sees: Bool
    let defaults: SamplingDefaults?

    static func == (lhs: CatalogEntry, rhs: CatalogEntry) -> Bool {
        lhs.id == rhs.id && lhs.resident == rhs.resident && lhs.bytesOnDisk == rhs.bytesOnDisk
    }
}

/// What one image would cost the checkpoint it was priced against: the size its tower
/// actually reads, and the rows that reserves in the prompt. The arithmetic is the family's
/// and lives in the engine — this panel asks and does not compute.
struct ImageCost: Decodable {
    let height: Int
    let width: Int
    let tokens: Int
}

struct Progress: Decodable {
    let message: String
    let completed: Int
    let total: Int?
}

/// `subject` carries a model, or the list of them a bench run was started over.
enum Subject: Decodable {
    case one(String)
    case many([String])

    init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if let value = try? single.decode(String.self) {
            self = .one(value)
        } else if let value = try? single.decode([String].self) {
            self = .many(value)
        } else {
            self = .many([])
        }
    }

    var names: [String] {
        switch self {
        case let .one(value): return [value]
        case let .many(values): return values
        }
    }
}

struct Job: Decodable, Identifiable {
    let id: String
    let kind: String
    let subject: [String: Subject]
    let state: String
    let progress: Progress
    let updatedAt: Double
    let error: String?

    var moving: Bool { state == "pending" || state == "running" }

    var model: String? {
        (subject["model"] ?? subject["models"])?.names.first
    }

    var target: String? {
        if case let .one(value) = subject["target"] { return value }
        return nil
    }

    /// null until the job knows a total; a bar drawn from a guessed one is worse than none.
    var fraction: Double? {
        guard let total = progress.total, total > 0 else { return nil }
        return min(1, Double(progress.completed) / Double(total))
    }
}

struct Speculation: Decodable {
    let rounds: Int
    let proposed: Int
    let accepted: Int
}

struct Sample: Decodable {
    let model: String
    let state: String
    let promptTokens: Int
    let completionTokens: Int
    let startedAt: Double
    let loadSeconds: Double?
    let ttft: Double?
    let tokensPerSecond: Double?
    let prefillTokensPerSecond: Double?
    let ceilingFraction: Double?
    let speculation: Speculation?
}

struct Aggregate: Decodable {
    let model: String
    let requests: Int
    let tokensPerSecond: Double?
    let ceilingFraction: Double?
}

struct Snapshot: Decodable {
    let live: Sample?
    let requests: [Sample]
    let models: [Aggregate]
}

// ── the Hub, settings and profiles ───────────────────────────────────────

struct HubModel: Decodable, Identifiable, Equatable {
    let id: String
    let downloads: Int?
    let likes: Int?
}

struct CheckpointFile: Decodable, Identifiable {
    let name: String
    let size: Int
    var id: String { name }
}

// ── the decode step, as the engine traced it ─────────────────────────────
//
// The daemon records the graph by running one token through the tree it builds and reading
// which module handed its array to which. It reads no weight to do it, so the shape below
// arrives for a checkpoint that has never been loaded.

struct GraphNode: Decodable, Identifiable {
    let id: String
    let label: String
    /// The class that ran. What it tells a reader is where the loader fused something: a
    /// block whose projections arrive as `qkv_proj` is one the checkpoint spells in three.
    let kind: String
    /// `norm`, `attention`, `recurrent`, `dense`, `experts`, `router`, `join`, `port`, …
    let role: String
    /// Which implementations resolved under this node. A model declares an operation and
    /// never a kernel; this is the one that won, and empty means the step took another path.
    let kernels: [String]
}

struct GraphEdge: Decodable, Hashable {
    let source: String
    let target: String
    /// Whether the trace saw the array pass, or only saw the two run in that order. The
    /// second is drawn dashed — a supposition the daemon refuses to pass off as a reading.
    let observed: Bool
}

struct BlockGraph: Decodable, Identifiable {
    let layers: [Int]
    let kind: String
    let nodes: [GraphNode]
    let edges: [GraphEdge]
    /// What the block holds itself and no node inside it accounts for — a fused add-norm
    /// stands for the residual and the norm after it, and belongs to neither mixer.
    let kernels: [String]

    var id: Int { layers.first ?? 0 }
}

struct Blueprint: Decodable {
    let spine: [GraphNode]
    let blocks: [BlockGraph]
    /// The width the trace was taken at. One token is a decode step, which is the shape the
    /// engine is measured at and the one a reader is asking about.
    let tokens: Int
}

struct Sampling: Codable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var minP: Double?
    var repetitionPenalty: Double?
    var seed: Int?
    var reasoningEffort: String?
    var reasoningBudget: Int?
}

/// Naming the technique is what turns it on, and a null `kind` is off. `mtp` drafts with
/// a head inside the model's own checkpoint, so `drafter` stays null.
struct SpeculationSetting: Codable {
    var kind: String?
    var drafter: String?
    var blockSize: Int?
}

struct Features: Codable {
    var speculation: SpeculationSetting?
}

struct ProfileView: Decodable {
    let model: String
    let name: String
    let sampling: Sampling
    let systemPrompt: String?
}

struct SettingsView: Decodable {
    let model: String
    let features: Features
    let available: [String]
    let mtpAvailable: Bool
    let unavailableReason: String?
}

// ── benchmarks ───────────────────────────────────────────────────────────

struct SpeedBody: Decodable {
    let context: Int
    let generate: Int
    let concurrency: Int
    let rounds: Int
    let prefillTps: Double?
    let ttftP50Ms: Double?
    let decodeTps: Double?
    let ceilingTps: Double?
    let ceilingFraction: Double?
}

struct QualityBody: Decodable {
    let dataset: String
    let items: Int
    let accuracy: Double?
    let correct: Int?
}

struct FidelityBody: Decodable {
    let reference: String
    let corpus: String
    let tokens: Int
    let klMean: Double?
    let top1: Double?
    let top5: Double?
}

struct BenchRun: Decodable, Identifiable {
    let id: String
    let kind: String
    let model: String
    let key: String
    let state: String
    let reason: String?
    let createdAt: Double
    let speed: SpeedBody?
    let quality: QualityBody?
    let fidelity: FidelityBody?
}

// ── quantization ─────────────────────────────────────────────────────────

struct PlanLeaf: Decodable {
    let path: String
    let kind: String
    let shape: [Int]
    /// null says the leaf stays dense, which is what the default or an override asked.
    let bits: Int?
    let groupSize: Int?
    let bytes: Int
}

struct PricedPlan: Decodable {
    let leaves: [PlanLeaf]
    let totalBytes: Int
    let weights: Int
    let bitsPerWeight: Double
    /// What the produced `model.safetensors` will weigh.
    let entryBytes: Int
}

/// The leaves collapse into the groups the numbered segments make. Two spellings come out
/// of it: the label a reader gets, where
/// a number is a `*`, and the key an override is taken under, where it is `\d+` and every
/// other segment is matched literally.
struct LeafGroup: Identifiable {
    /// The `fullmatch` pattern that matches exactly these leaves and no others.
    let pattern: String
    let label: String
    let leaves: Int
    let bytes: Int
    /// The widths the plan gave these leaves, as a span — nil when every one of them stays
    /// dense. The allocator spends a group unevenly, so one member's number is not the
    /// group's: `4–8bit` is what is true about it.
    let widths: ClosedRange<Int>?
    /// How many of them stay dense inside a group that is otherwise packed.
    let dense: Int
    /// The group size, where the whole group shares one.
    let groupSize: Int?
    var id: String { pattern }
}

/// `re.escape`'s own set: the daemon compiles the key with Python's `re`, so a literal
/// segment must not arrive there as a metacharacter.
private let metacharacters = Set(#"()[]{}?*+-|^$\.&~# "#)

private func escaped(_ segment: String) -> String {
    String(segment.flatMap { character -> [Character] in
        metacharacters.contains(character) || character.isWhitespace
            ? ["\\", character] : [character]
    })
}

func groupLeaves(_ leaves: [PlanLeaf]) -> [LeafGroup] {
    var held: [String: [PlanLeaf]] = [:]
    var order: [String] = []
    for leaf in leaves {
        let pattern = segments(leaf.path)
            .map { part in numbered(part) ? #"\d+"# : escaped(part) }
            .joined(separator: #"\."#)
        if held[pattern] == nil { order.append(pattern) }
        held[pattern, default: []].append(leaf)
    }
    return order.compactMap { pattern in
        guard let members = held[pattern], let first = members.first else { return nil }
        let packed = members.compactMap(\.bits)
        return LeafGroup(
            pattern: pattern,
            label: segments(first.path)
                .map { part in numbered(part) ? "*" : part }
                .joined(separator: "."),
            leaves: members.count,
            bytes: members.reduce(0) { $0 + $1.bytes },
            widths: packed.isEmpty ? nil : packed.min()!...packed.max()!,
            dense: members.count - packed.count,
            groupSize: members.allSatisfy { $0.groupSize == first.groupSize }
                ? first.groupSize : nil
        )
    }
}

private func segments(_ path: String) -> [String] {
    path.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
}

private func numbered(_ segment: String) -> Bool {
    !segment.isEmpty && segment.allSatisfy(\.isNumber)
}

// ── request bodies ───────────────────────────────────────────────────────

/// One group of leaves against the rest of the plan. `bits` nil is dense, which the wire
/// spells as a null value — the daemon reads the two the way `ByPath` does: a pattern with
/// a format takes it, a pattern with null stays dense, and a pattern absent gets the plan's
/// own. `groupSize` nil is the plan's, because the group size is a decision about the whole
/// checkpoint and a group that departs from it says so.
struct LeafOverride: Encodable, Equatable {
    var bits: Int?
    var groupSize: Int?

    private enum Key: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }

    func encode(to encoder: Encoder) throws {
        guard let bits else {
            var out = encoder.singleValueContainer()
            try out.encodeNil()
            return
        }
        var out = encoder.container(keyedBy: Key.self)
        try out.encode(bits, forKey: .bits)
        try out.encodeIfPresent(groupSize, forKey: .groupSize)
    }
}

/// The selection, with its keys spelled out rather than converted.
///
/// `overrides` is keyed by patterns read off the checkpoint's own leaves, and the
/// snake-casing encoder rewrites every key it writes — including a dictionary's. A tree
/// with an `A_log` or a `Wqkv` in it would leave here lowercased, match no leaf, and be
/// refused for a reason nobody typed.
struct QuantizeRequest: Encodable {
    var source: String
    var mode: String
    var method: String
    var bits: Int?
    var groupSize: Int?
    var targetBpw: Double?
    var hardCapBpw: Double?
    var repo: String?
    var overrides: [String: LeafOverride] = [:]

    enum CodingKeys: String, CodingKey {
        case source, mode, method, bits, repo, overrides
        case groupSize = "group_size"
        case targetBpw = "target_bpw"
        case hardCapBpw = "hard_cap_bpw"
    }
}

struct BenchRequest: Encodable {
    var kind = "speed"
    var models: [String]
    var contexts = [4096]
    var generates = [256]
    var concurrencies = [1]
    var rounds = 3
    var thermalGateC = 50
    var skipIfMeasured = false
}

struct ProfileBody: Encodable {
    var sampling: Sampling
}

struct SettingsBody: Encodable {
    var features: Features
}

struct PullBody: Encodable {
    var repo: String
}

// ── the chat dialect, `/api/openai/v1/chat/completions` ──────────────────
//
// Upstream's shapes, which is what the panel speaks to the daemon here — the same request
// any other client would send, so what this screen exercises is the served path and not a
// private one. `x_mlx_omnia` on the usage frame is the daemon's own, and the one field
// upstream has no room for: what the turn cost.

/// Where the dialect puts an image: a `data:` URI, because a daemon that fetched an
/// `https://` one would be making requests at a client's word from inside the network it
/// was told to serve.
struct ImageUrl: Encodable {
    let url: String
}

enum ContentPart: Encodable {
    case text(String)
    case image(String)

    private enum Key: String, CodingKey { case type, text, imageUrl = "image_url" }

    func encode(to encoder: Encoder) throws {
        var out = encoder.container(keyedBy: Key.self)
        switch self {
        case .text(let body):
            try out.encode("text", forKey: .type)
            try out.encode(body, forKey: .text)
        case .image(let uri):
            try out.encode("image_url", forKey: .type)
            try out.encode(ImageUrl(url: uri), forKey: .imageUrl)
        }
    }
}

/// A turn on the way out. `content` is a string when the turn is only text and the parts
/// themselves when one of them is a picture — both shapes are the dialect's own, and the
/// simpler one is worth keeping for the turns that do not need the other.
enum Said: Encodable {
    case text(String)
    case parts([ContentPart])

    func encode(to encoder: Encoder) throws {
        var out = encoder.singleValueContainer()
        switch self {
        case .text(let body): try out.encode(body)
        case .parts(let parts): try out.encode(parts)
        }
    }
}

struct SentMessage: Encodable {
    let role: String
    let content: Said
}

/// The request, with its keys spelled out rather than converted.
///
/// Every other body in this panel is written by the snake-casing encoder, and this one
/// cannot be: `tools` carries a JSON schema somebody pasted, and a strategy that renamed
/// its keys would hand the model a schema nobody wrote — `unitType` silently becoming
/// `unit_type` in the function the answer is graded against.
struct ChatBody: Encodable {
    var model: String
    var messages: [SentMessage]
    var stream = true
    var streamOptions = StreamOptions()
    var maxTokens = 4096
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var minP: Double?
    var reasoningEffort: String?
    var tools: [AnyJSON]?

    struct StreamOptions: Encodable {
        var includeUsage = true

        enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools
        case streamOptions = "stream_options"
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case reasoningEffort = "reasoning_effort"
    }
}

/// What the daemon says a turn cost, on the usage frame under a key the dialect does not
/// define. Every number here is the engine's own — the load, the queue and the prefill are
/// one wait from out here, with no way to tell them apart.
struct Timings: Decodable {
    let loadSeconds: Double?
    let ttftSeconds: Double?
    let prefillTokensPerSecond: Double?
    let tokensPerSecond: Double?
    let ceilingFraction: Double?
    let speculation: Speculation?
}

struct CalledFunctionDelta: Decodable {
    let name: String?
    let arguments: String?
}

/// A call arrives in pieces like the text does: the index says which one is being written,
/// the name lands once, and the arguments accumulate as JSON text.
struct ToolCallDelta: Decodable {
    let index: Int
    let id: String?
    let function: CalledFunctionDelta?
}

struct Delta: Decodable {
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ToolCallDelta]?
}

struct Choice: Decodable {
    let delta: Delta?
    let finishReason: String?
}

struct Usage: Decodable {
    let promptTokens: Int?
    let completionTokens: Int?
    let totalTokens: Int?
}

struct ChatChunk: Decodable {
    let choices: [Choice]?
    let usage: Usage?
    let xMlxOmnia: Timings?
}

/// A tools array a reader pasted, carried through unread. The dialect hands the schema to
/// the template with `tojson`, so what the model sees is what was typed — parsing it here
/// to type it would only mean writing it back out differently.
struct AnyJSON: Encodable {
    let value: Any

    func encode(to encoder: Encoder) throws {
        var out = encoder.singleValueContainer()
        try out.encode(AnyJSON.wrapped(value))
    }

    private static func wrapped(_ value: Any) -> AnyCodableValue {
        switch value {
        case let text as String: return .text(text)
        case let flag as Bool: return .flag(flag)
        case let number as Int: return .whole(number)
        case let number as Double: return .number(number)
        case let list as [Any]: return .list(list.map(wrapped))
        case let map as [String: Any]: return .map(map.mapValues(wrapped))
        default: return .nothing
        }
    }
}

indirect enum AnyCodableValue: Encodable {
    case text(String)
    case flag(Bool)
    case whole(Int)
    case number(Double)
    case list([AnyCodableValue])
    case map([String: AnyCodableValue])
    case nothing

    func encode(to encoder: Encoder) throws {
        var out = encoder.singleValueContainer()
        switch self {
        case .text(let value): try out.encode(value)
        case .flag(let value): try out.encode(value)
        case .whole(let value): try out.encode(value)
        case .number(let value): try out.encode(value)
        case .list(let value): try out.encode(value)
        case .map(let value): try out.encode(value)
        case .nothing: try out.encodeNil()
        }
    }
}

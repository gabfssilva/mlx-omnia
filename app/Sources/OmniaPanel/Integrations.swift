// The line somebody pastes into a terminal to point a coding agent at this daemon.
//
// Four tiers, because four is the most any of these clients has a slot for, and they are
// named for size rather than for any one client's vocabulary — Claude Code's own names for
// them are haiku, sonnet, opus and fable, and Codex has no names at all.
//
// The shape of each line is the client's and not this app's preference. Claude Code reads
// its models out of the environment; opencode reads a whole config out of
// OPENCODE_CONFIG_CONTENT, merged over the ones on disk; Codex reads neither and takes
// `-c` overrides on the command line; pi and Prime Agent read providers only out of their
// own `models.json`, so those lines write that file — merging one provider into it, never
// replacing what is there.
//
// A tier nobody set is a tier the line leaves out, and the client keeps whatever it had
// for that slot.

import Foundation
import Observation

enum Tier: String, CaseIterable, Identifiable, Sendable {
    case smallest, small, `default`, big

    var id: String { rawValue }
    var label: String { rawValue.capitalized }
}

/// Which checkpoint answers at each size. Kept in `UserDefaults` because the line is copied
/// into a terminal days apart, and choosing four models again after every login is the
/// panel asking for the same answer twice.
@MainActor
@Observable
final class Integrations {
    private static let stored = "integrations.tiers"

    private(set) var models: [String: String]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        models = defaults.dictionary(forKey: Self.stored) as? [String: String] ?? [:]
    }

    private let defaults: UserDefaults

    func model(_ tier: Tier) -> String? { models[tier.rawValue] }

    func set(_ tier: Tier, _ model: String?) {
        models[tier.rawValue] = model
        defaults.set(models, forKey: Self.stored)
    }
}

// ── what a line is written against ───────────────────────────────────────

/// The daemon as a client sees it, plus the tiers. `key` is what goes in the `Authorization`
/// header: the daemon with no key configured answers unauthenticated, but three of these
/// four clients refuse to offer a model whose provider has no credential at all, so the
/// placeholder is sent rather than nothing.
struct Wiring {
    let base: URL
    let key: String?
    let models: [String: String]

    func model(_ tier: Tier) -> String? { models[tier.rawValue] }

    var token: String { key ?? "omnia" }

    var anthropic: String { url("/api/anthropic") }
    var openai: String { url("/api/openai/v1") }

    /// The tiers that name a checkpoint, in size order, deduplicated — what a provider has
    /// to declare for the slots below to resolve.
    func named(_ tiers: [Tier]) -> [String] {
        var seen: [String] = []
        for tier in tiers {
            guard let model = model(tier), !seen.contains(model) else { continue }
            seen.append(model)
        }
        return seen
    }

    private func url(_ path: String) -> String { base.absoluteString + path }
}

/// One client, and the line that starts it against this daemon.
struct Recipe: Identifiable {
    let id: String
    let name: String
    /// The tiers this client has a slot for, and what it calls each one. Drawn under the
    /// row, so what a tier turns into is on screen before the line is copied.
    let slots: [(Tier, String)]
    let line: (Wiring) -> String?

    var mapping: String {
        slots.map { "\($0.0.label.lowercased()) → \($0.1)" }.joined(separator: " · ")
    }

    var tiers: [Tier] { slots.map(\.0) }
}

enum Recipes {
    static let all: [Recipe] = [claude, codex, opencode, pi, prime]

    // ── Claude Code: four slots, all of them environment ──────────────────

    static let claude = Recipe(
        id: "claude",
        name: "Claude Code",
        slots: [
            (.smallest, "haiku"), (.small, "sonnet"), (.default, "opus"), (.big, "fable"),
        ],
        line: { wiring in
            let named: [(Tier, String)] = [
                (.smallest, "HAIKU"), (.small, "SONNET"), (.default, "OPUS"), (.big, "FABLE"),
            ]
            let chosen = named.compactMap { tier, slot in
                wiring.model(tier).map { "ANTHROPIC_DEFAULT_\(slot)_MODEL=\(shell($0))" }
            }
            guard !chosen.isEmpty else { return nil }
            // The base URL stops at the dialect and not at its version: the Anthropic SDK
            // appends `/v1/messages` itself.
            //
            // The other two are fixed, not a tier: a local checkpoint decodes far slower than
            // the Anthropic API, so the SDK's own timeout is too short for it, and there is no
            // Anthropic account behind this token for Claude Code's background traffic to
            // reach.
            let head = [
                "ANTHROPIC_BASE_URL=\(shell(wiring.anthropic))",
                "ANTHROPIC_AUTH_TOKEN=\(shell(wiring.token))",
                "API_TIMEOUT_MS=3000000",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
            ]
            return (head + chosen + ["claude"]).joined(separator: " ")
        }
    )

    // ── Codex: three slots, none of them environment ──────────────────────

    static let codex = Recipe(
        id: "codex",
        name: "Codex",
        slots: [(.small, "subagents"), (.default, "model"), (.big, "review")],
        line: { wiring in
            let named: [(Tier, String)] = [
                (.small, "default_subagent_model"), (.default, "model"), (.big, "review_model"),
            ]
            let chosen = named.compactMap { tier, slot in
                wiring.model(tier).map { "-c \(shell("\(slot)=\($0)"))" }
            }
            guard !chosen.isEmpty else { return nil }
            // No `wire_api`: Codex refuses to load a config that names `chat`, so every
            // provider it talks to is the responses route, which is the daemon's default
            // reading of this base URL too.
            var provider = [
                "-c model_provider=omnia",
                "-c model_providers.omnia.name=Omnia",
                "-c \(shell("model_providers.omnia.base_url=\(wiring.openai)"))",
            ]
            if let key = wiring.key {
                provider.append("-c model_providers.omnia.env_key=OMNIA_API_KEY")
                return (["OMNIA_API_KEY=\(shell(key))", "codex"] + provider + chosen)
                    .joined(separator: " ")
            }
            return (["codex"] + provider + chosen).joined(separator: " ")
        }
    )

    // ── opencode: two slots, the whole config in one variable ─────────────

    static let opencode = Recipe(
        id: "opencode",
        name: "opencode",
        slots: [(.small, "small_model"), (.default, "model")],
        line: { wiring in
            let declared = wiring.named([.default, .small])
            guard !declared.isEmpty else { return nil }
            let config = OpencodeConfig(
                provider: [
                    "omnia": .init(
                        options: .init(baseURL: wiring.openai, apiKey: wiring.token),
                        models: Dictionary(uniqueKeysWithValues: declared.map { ($0, [:]) })
                    ),
                ],
                model: wiring.model(.default).map { "omnia/\($0)" },
                smallModel: wiring.model(.small).map { "omnia/\($0)" }
            )
            guard let json = encoded(config) else { return nil }
            return "OPENCODE_CONFIG_CONTENT=\(shell(json)) opencode"
        }
    )

    // ── pi and Prime Agent: one slot, and a file that has to exist first ──

    static let pi = modelsJSON(id: "pi", name: "pi", command: "pi", file: ".pi/agent/models.json")

    static let prime = modelsJSON(
        id: "prime", name: "Prime Agent", command: "prime-agent",
        file: ".prime/agent/models.json"
    )

    /// The same file read by two clients: pi, and the Prime Intellect fork that renamed the
    /// binary and the directory under it. Neither takes a provider from anywhere else, and
    /// the schema they validate it against is the same one.
    private static func modelsJSON(
        id: String, name: String, command: String, file: String
    ) -> Recipe {
        Recipe(
            id: id,
            name: name,
            slots: [(.default, "--model")],
            line: { wiring in
                let declared = wiring.named(Tier.allCases)
                guard let chosen = wiring.model(.default) ?? declared.first else { return nil }
                let provider = ModelsJSONProvider(
                    baseUrl: wiring.openai,
                    apiKey: wiring.token,
                    models: declared.map { ["id": $0] }
                )
                guard let json = encoded(provider) else { return nil }
                // Merged into the file rather than written over it: `models.json` is where every
                // other local provider this reader has is declared.
                let merge = """
                    python3 -c 'import json,pathlib,sys;p=pathlib.Path.home()/"\(file)";\
                    d=json.loads(p.read_text()) if p.exists() else {};\
                    d.setdefault("providers",{})["omnia"]=json.loads(sys.argv[1]);\
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))' \
                    \(shell(json))
                    """
                var start = [command, "--model", shell("omnia/\(chosen)")]
                // Ctrl+P cycles what `--models` names, which is how the other tiers are reachable
                // from a client with one slot.
                if declared.count > 1 {
                    start += [
                        "--models", shell(declared.map { "omnia/\($0)" }.joined(separator: ",")),
                    ]
                }
                return "\(merge) && \(start.joined(separator: " "))"
            }
        )
    }
}

// ── the two payloads, spelled by the client that reads them ──────────────

private struct OpencodeConfig: Encodable {
    struct Options: Encodable {
        let baseURL: String
        let apiKey: String
    }

    struct Provider: Encodable {
        let npm = "@ai-sdk/openai-compatible"
        let name = "Omnia"
        let options: Options
        let models: [String: [String: String]]
    }

    let provider: [String: Provider]
    let model: String?
    let smallModel: String?

    enum CodingKeys: String, CodingKey {
        case provider, model
        case smallModel = "small_model"
    }
}

private struct ModelsJSONProvider: Encodable {
    /// Every field the daemon's OpenAI dialect does not declare is a 422, so each of these
    /// turns off something the client would otherwise send by default:
    ///
    /// - `developer` is not one of the four roles that route takes, so the system prompt goes
    ///   as `system`;
    /// - `max_completion_tokens` is the default field name for a base URL the client does not
    ///   recognize, and the route reads `max_tokens`;
    /// - `store: false` is sent to any base URL the client considers standard, which a
    ///   localhost one is;
    /// - `strict` rides along in every tool definition unless it is refused here;
    /// - `prompt_cache_key` and `prompt_cache_retention` appear once cache retention is long,
    ///   which is one environment variable away.
    ///
    /// Reasoning effort the route does take, which is why nothing here turns that off.
    struct Compat: Encodable {
        let maxTokensField = "max_tokens"
        let supportsDeveloperRole = false
        let supportsLongCacheRetention = false
        let supportsStore = false
        let supportsStrictMode = false
    }

    let baseUrl: String
    let api = "openai-completions"
    let apiKey: String
    let compat = Compat()
    let models: [[String: String]]
}

private func encoded<Payload: Encodable>(_ payload: Payload) -> String? {
    let encoder = JSONEncoder()
    // Slashes unescaped: every value in these payloads is a URL or a repository path, and
    // `\/` is a line somebody has to read before pasting it.
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(payload) else { return nil }
    return String(data: data, encoding: .utf8)
}

/// A value the shell hands over whole. Model ids are repository paths and the payloads are
/// JSON, so single quotes cover both — and the one character they cannot carry is escaped
/// the only way `sh` allows.
func shell(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

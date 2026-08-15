import XCTest
@testable import OmniaPanel

/// The lines are the product here: a variable named wrong is a client that silently keeps
/// talking to whoever it was talking to before. Each one is checked against the names the
/// client actually reads, and the two that carry JSON are parsed back.
final class IntegrationsTests: XCTestCase {
    private let base = URL(string: "http://127.0.0.1:8642")!

    private func wiring(
        _ models: [Tier: String], key: String? = nil
    ) -> Wiring {
        Wiring(
            base: base,
            key: key,
            models: Dictionary(uniqueKeysWithValues: models.map { ($0.key.rawValue, $0.value) })
        )
    }

    private let four: [Tier: String] = [
        .smallest: "mlx-community/Qwen3-0.6B-4bit",
        .small: "mlx-community/Qwen3-8B-4bit",
        .default: "mlx-community/Qwen3-30B-A3B-4bit",
        .big: "mlx-community/GLM-4.6-4bit",
    ]

    // ── Claude Code ──────────────────────────────────────────────────────

    func testClaudeNamesEveryTierAndStopsTheBaseUrlBeforeTheVersion() {
        let line = Recipes.claude.line(wiring(four))
        XCTAssertEqual(
            line,
            "ANTHROPIC_BASE_URL='http://127.0.0.1:8642/api/anthropic'"
                + " ANTHROPIC_AUTH_TOKEN='omnia'"
                + " ANTHROPIC_DEFAULT_HAIKU_MODEL='mlx-community/Qwen3-0.6B-4bit'"
                + " ANTHROPIC_DEFAULT_SONNET_MODEL='mlx-community/Qwen3-8B-4bit'"
                + " ANTHROPIC_DEFAULT_OPUS_MODEL='mlx-community/Qwen3-30B-A3B-4bit'"
                + " ANTHROPIC_DEFAULT_FABLE_MODEL='mlx-community/GLM-4.6-4bit'"
                + " claude"
        )
    }

    func testClaudeLeavesOutTheTiersNobodySet() {
        let line = Recipes.claude.line(wiring([.default: "a"])) ?? ""
        XCTAssertTrue(line.contains("ANTHROPIC_DEFAULT_OPUS_MODEL='a'"))
        XCTAssertFalse(line.contains("HAIKU"))
        XCTAssertFalse(line.contains("SONNET"))
        XCTAssertFalse(line.contains("FABLE"))
    }

    func testClaudeCarriesTheConfiguredKey() {
        let line = Recipes.claude.line(wiring([.default: "a"], key: "s3cret")) ?? ""
        XCTAssertTrue(line.contains("ANTHROPIC_AUTH_TOKEN='s3cret'"))
    }

    // ── Codex ────────────────────────────────────────────────────────────

    func testCodexDeclaresTheProviderAndItsThreeSlots() {
        let line = Recipes.codex.line(wiring(four)) ?? ""
        XCTAssertTrue(line.hasPrefix("codex -c model_provider=omnia"))
        XCTAssertTrue(
            line.contains("-c 'model_providers.omnia.base_url=http://127.0.0.1:8642/api/openai/v1'")
        )
        // Codex 0.147 refuses to load a config that names `wire_api = "chat"`.
        XCTAssertFalse(line.contains("wire_api"))
        XCTAssertTrue(line.contains("-c 'default_subagent_model=mlx-community/Qwen3-8B-4bit'"))
        XCTAssertTrue(line.contains("-c 'model=mlx-community/Qwen3-30B-A3B-4bit'"))
        XCTAssertTrue(line.contains("-c 'review_model=mlx-community/GLM-4.6-4bit'"))
        // Codex has no slot for the smallest tier, and inventing one would be a fourth
        // model this client never asks for.
        XCTAssertFalse(line.contains("Qwen3-0.6B"))
    }

    func testCodexAsksForAKeyOnlyWhenTheDaemonHasOne() {
        let open = Recipes.codex.line(wiring([.default: "a"])) ?? ""
        XCTAssertFalse(open.contains("env_key"))
        XCTAssertFalse(open.contains("OMNIA_API_KEY"))

        let closed = Recipes.codex.line(wiring([.default: "a"], key: "s3cret")) ?? ""
        XCTAssertTrue(closed.hasPrefix("OMNIA_API_KEY='s3cret' codex "))
        XCTAssertTrue(closed.contains("-c model_providers.omnia.env_key=OMNIA_API_KEY"))
    }

    // ── opencode ─────────────────────────────────────────────────────────

    func testOpencodeCarriesAConfigThatParsesBack() throws {
        let line = try XCTUnwrap(Recipes.opencode.line(wiring(four)))
        let config = try json(after: "OPENCODE_CONFIG_CONTENT=", in: line, ending: " opencode")

        XCTAssertEqual(config["model"] as? String, "omnia/mlx-community/Qwen3-30B-A3B-4bit")
        XCTAssertEqual(config["small_model"] as? String, "omnia/mlx-community/Qwen3-8B-4bit")

        let provider = try XCTUnwrap(
            (config["provider"] as? [String: Any])?["omnia"] as? [String: Any]
        )
        XCTAssertEqual(provider["npm"] as? String, "@ai-sdk/openai-compatible")
        let options = try XCTUnwrap(provider["options"] as? [String: Any])
        XCTAssertEqual(options["baseURL"] as? String, "http://127.0.0.1:8642/api/openai/v1")
        XCTAssertEqual(options["apiKey"] as? String, "omnia")
        // Only the two slots this client has, and only once each.
        XCTAssertEqual(
            Set((provider["models"] as? [String: Any])?.keys ?? [:].keys),
            ["mlx-community/Qwen3-30B-A3B-4bit", "mlx-community/Qwen3-8B-4bit"]
        )
    }

    func testOpencodeDeclaresOneModelWhenBothTiersNameIt() throws {
        let line = try XCTUnwrap(Recipes.opencode.line(wiring([.default: "a", .small: "a"])))
        let config = try json(after: "OPENCODE_CONFIG_CONTENT=", in: line, ending: " opencode")
        let provider = try XCTUnwrap(
            (config["provider"] as? [String: Any])?["omnia"] as? [String: Any]
        )
        XCTAssertEqual((provider["models"] as? [String: Any])?.count, 1)
    }

    // ── pi ───────────────────────────────────────────────────────────────

    func testPiMergesAProviderAndStartsOnTheDefaultTier() throws {
        let line = try XCTUnwrap(Recipes.pi.line(wiring(four)))
        XCTAssertTrue(line.contains("d.setdefault(\"providers\",{})[\"omnia\"]"))
        XCTAssertTrue(line.contains("pi --model 'omnia/mlx-community/Qwen3-30B-A3B-4bit'"))
        // Every tier is declared, smallest first, because Ctrl+P cycles what `--models` names.
        XCTAssertTrue(
            line.contains(
                "--models 'omnia/mlx-community/Qwen3-0.6B-4bit,omnia/mlx-community/Qwen3-8B-4bit,"
                    + "omnia/mlx-community/Qwen3-30B-A3B-4bit,omnia/mlx-community/GLM-4.6-4bit'"
            )
        )

        let provider = try json(after: "))' ", in: line, ending: " && pi")
        XCTAssertEqual(provider["baseUrl"] as? String, "http://127.0.0.1:8642/api/openai/v1")
        XCTAssertEqual(provider["api"] as? String, "openai-completions")
        XCTAssertEqual(provider["apiKey"] as? String, "omnia")
        XCTAssertEqual(
            (provider["compat"] as? [String: Any])?["supportsDeveloperRole"] as? Bool, false
        )
        XCTAssertEqual((provider["models"] as? [[String: String]])?.count, 4)
    }

    func testPiFallsBackToWhateverTierIsSet() throws {
        let line = try XCTUnwrap(Recipes.pi.line(wiring([.big: "a"])))
        XCTAssertTrue(line.contains("pi --model 'omnia/a'"))
        XCTAssertFalse(line.contains("--models"))
    }

    // ── nothing set, and the quoting ─────────────────────────────────────

    func testNoTierMeansNoLine() {
        for recipe in Recipes.all {
            XCTAssertNil(recipe.line(wiring([:])), recipe.name)
        }
    }

    func testEveryLineHandsTheShellWholeValues() {
        let line = Recipes.claude.line(wiring([.default: "it's/here"])) ?? ""
        XCTAssertTrue(line.contains("ANTHROPIC_DEFAULT_OPUS_MODEL='it'\\''s/here'"))
    }

    func testRecipesSayWhatEachTierBecomes() {
        XCTAssertEqual(
            Recipes.claude.mapping, "smallest → haiku · small → sonnet · default → opus · big → fable"
        )
        XCTAssertEqual(Recipes.opencode.tiers, [.small, .default])
    }

    // ── reading a payload back out of a line ─────────────────────────────

    private func json(
        after opening: String, in line: String, ending: String
    ) throws -> [String: Any] {
        let start = try XCTUnwrap(line.range(of: opening))
        let stop = try XCTUnwrap(line.range(of: ending, range: start.upperBound..<line.endIndex))
        var quoted = String(line[start.upperBound..<stop.lowerBound])
        XCTAssertTrue(quoted.hasPrefix("'") && quoted.hasSuffix("'"), quoted)
        quoted = String(quoted.dropFirst().dropLast())
        let data = try XCTUnwrap(quoted.data(using: .utf8))
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

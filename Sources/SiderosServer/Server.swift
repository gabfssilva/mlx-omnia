import Hummingbird

public struct Server: Sendable {
    private let engine: Engine

    public init(engine: Engine) {
        self.engine = engine
    }

    public func run(host: String = "127.0.0.1", port: Int = 8080) async throws {
        let engine = engine
        let router = Router()

        // Every dialect lives under its own prefix. Serving them at their native paths
        // instead would make `/v1/models` ambiguous: OpenAI and Anthropic both define it,
        // with different bodies.
        router.get("/openai/v1/models") { _, _ in try OpenAI.models(engine: engine) }
        router.post("/openai/v1/chat/completions") { request, _ in
            try await OpenAI.chat(request, engine: engine)
        }
        router.post("/openai/v1/responses") { request, _ in
            try await Responses.create(request, engine: engine)
        }
        router.post("/anthropic/v1/messages") { request, _ in
            try await Anthropic.messages(request, engine: engine)
        }
        // The model name and the method share a path segment ("qwen:generateContent"), and
        // the name may itself carry slashes — so the whole tail is the route.
        router.post("/gemini/v1beta/models/**") { request, _ in
            try await Gemini.generate(request, engine: engine)
        }

        let app = Application(
            router: router, configuration: .init(address: .hostname(host, port: port)))
        try await app.runService()
    }
}

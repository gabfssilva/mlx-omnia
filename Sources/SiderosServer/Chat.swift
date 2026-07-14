import Sideros

/// The canonical request. Every dialect — OpenAI, Anthropic, Gemini — translates into this
/// and back; nothing downstream of here knows which one it came from.
public struct ChatRequest: Sendable {
    public var messages: [Message]
    public var maxTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var stop: [String]
    public var seed: UInt64?
    public var stream: Bool

    public init(
        messages: [Message], maxTokens: Int? = nil, temperature: Float? = nil,
        topP: Float? = nil, stop: [String] = [], seed: UInt64? = nil, stream: Bool = false
    ) {
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.stop = stop
        self.seed = seed
        self.stream = stream
    }
}

public struct Message: Sendable {
    public enum Role: String, Sendable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

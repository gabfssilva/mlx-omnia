public protocol Tokenizer: Sendable {
    func encode(_ text: String) -> [Int32]
    func decode(_ ids: [Int32]) -> String
}

extension GPT2Tokenizer: Tokenizer {}
extension Qwen2Tokenizer: Tokenizer {}
extension Gemma3Tokenizer: Tokenizer {}
extension Lfm2Tokenizer: Tokenizer {}

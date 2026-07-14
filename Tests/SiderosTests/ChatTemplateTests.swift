import Foundation
import Testing

@testable import Sideros
@testable import SiderosServer

private struct TemplateCase: Decodable {
    let repo: String
    let messages: [Turn]
    let rendered: String

    struct Turn: Decodable {
        let role: String
        let content: String
    }
}

/// The prompt a chat request turns into is the transformers one, character for character —
/// a template that renders *almost* right produces fluent, wrong text and no test failure.
/// Both shapes are covered: Qwen3 keeps the template in tokenizer_config.json, LFM2.5 in a
/// chat_template.jinja beside it.
@Test func chatTemplateMatchesTransformers() throws {
    let url = Bundle.module.resourceURL!.appending(path: "Fixtures/chat_template.json")
    let cases = try JSONDecoder().decode([TemplateCase].self, from: Data(contentsOf: url))
    #expect(cases.count == 2)

    for golden in cases {
        let template = try ChatTemplate(directory: hubSnapshot(golden.repo))
        let messages = golden.messages.map {
            Message(role: Message.Role(rawValue: $0.role)!, content: $0.content)
        }
        let rendered = try template(messages)
        #expect(rendered == golden.rendered)
    }
}

/// A base checkpoint has no template. Falling back to concatenating the turns would produce
/// fluent nonsense, so loading fails instead.
@Test func chatTemplateRefusesABaseModel() throws {
    #expect(throws: ChatTemplate.LoadError.self) {
        try ChatTemplate(directory: try hubSnapshot("google--gemma-3-270m"))
    }
}

/// The stop string is cut out, and it is found even when the tokens straddle it: "wor" and
/// "ld" arrive separately, so nothing may be emitted while a suffix could still open one.
@Test func stopFilterHoldsBackAmbiguousTails() {
    var filter = StopFilter(["world"])

    #expect(filter.push("hello ").emit == "hello ")
    // "wor" could still become "world" — holding it back is the whole point.
    #expect(filter.push("wor").emit == "")
    let halt = filter.push("ld!")
    #expect(halt.emit == "")
    #expect(halt.halt == "world")
}

/// A tail that turns out not to open a stop string is released, not lost.
@Test func stopFilterReleasesTailsThatDoNotMatch() {
    var filter = StopFilter(["END"])
    #expect(filter.push("the E").emit == "the ")
    #expect(filter.push("agle").emit == "Eagle")
}

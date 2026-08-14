import SwiftUI

struct LogsView: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    @State private var content = ""
    @State private var following = true

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 7) {
                DotView(state: app.store.down ? "bad" : "ok")
                Text("daemon.log").mono(10.5, t.fg2)
                Spacer(minLength: 0)
                if !following {
                    ActionText(label: "Jump to latest") { following = true }
                }
            }

            ScrollViewReader { proxy in
                ScrollView {
                    Text(content.isEmpty ? "No output yet." : content)
                        .mono(10.5, content.isEmpty ? t.fg3 : t.fg2)
                        .lineSpacing(3)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                    Color.clear.frame(height: 1).id("log-end")
                }
                .background(t.field)
                .overlay(Rectangle().strokeBorder(t.hair, lineWidth: 1))
                .onChange(of: content) {
                    if following { proxy.scrollTo("log-end", anchor: .bottom) }
                }
                .onChange(of: following) {
                    if following { proxy.scrollTo("log-end", anchor: .bottom) }
                }
                .onScrollGeometryChange(for: Bool.self) { geometry in
                    geometry.contentOffset.y + geometry.containerSize.height
                        >= geometry.contentSize.height - 8
                } action: { _, atEnd in
                    following = atEnd
                }
            }
        }
        .task {
            let file = Daemon.logs.appendingPathComponent("daemon.log")
            while !Task.isCancelled {
                let next = await Task.detached(priority: .utility) {
                    try? LogTail.read(file, maxBytes: 1_048_576)
                }.value ?? ""
                if next != content { content = next }
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
    }
}

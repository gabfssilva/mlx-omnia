// The panel's shell: the head that never leaves the screen, the three destinations, and
// whichever screen is on.
//
// The window has no room for a 228 pt sidebar here, so what the sidebar carried is split
// in two. The band, the dot and the port move into the head — what is looked at ten times
// a day is already open before a destination is chosen.

import SwiftUI

private enum BrandLogo {
    static let image: NSImage = {
        guard let url = Bundle.module.url(forResource: "logo", withExtension: "svg"),
              let image = NSImage(contentsOf: url)
        else { preconditionFailure("logo.svg is missing from the app resources") }
        image.isTemplate = true
        return image
    }()
}

struct PanelView: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    /// What the model screen is about. The catalog answers first, so a repository becomes a
    /// checkpoint the moment its download lands and the screen changes state without anybody
    /// navigating — as does a model deleted from under the screen, which resolves to nothing
    /// and drops back to the tab.
    private var opened: Opened? {
        switch app.opened {
        case .none: return nil
        case let .model(id): return app.store.entry(id).map(Opened.disk)
        case let .hub(found): return app.store.entry(found.id).map(Opened.disk) ?? .hub(found)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            Head(app: app)
            Segmented(
                options: Tab.allCases.map { ($0, $0.label) },
                chosen: app.tab,
                size: 12,
                status: { tab -> (state: String, text: String)? in
                    guard tab == .server else { return nil }
                    return (app.store.down ? "bad" : "ok", ":\(app.store.port)")
                },
                pick: { app.go($0) }
            )
            .padding(.horizontal, Panel.pad)

            // Chat owns its own scrolling: the transcript scrolls and the composer does
            // not, which one scroll view over the whole tab cannot do.
            if app.tab == .chat, opened == nil {
                ChatView(app: app).frame(maxHeight: .infinity)
            } else if app.tab == .server, opened == nil {
                ServerView(app: app).frame(maxHeight: .infinity)
            } else {
                ModelsRoot(app: app, opened: opened).frame(maxHeight: .infinity)
            }
        }
        .background(t.surface)
    }

}

private struct ModelsRoot: View {
    @Bindable var app: AppModel
    let opened: Opened?

    var body: some View {
        VStack(spacing: 0) {
            Segmented(
                options: ModelsTab.allCases.map { ($0, $0.label) },
                chosen: app.modelsTab,
                size: 11.5,
                pick: {
                    app.modelsTab = $0
                    app.opened = .none
                }
            )
            .padding(.horizontal, Panel.pad)
            .padding(.top, 12)

            ScrollView {
                VStack(alignment: .leading, spacing: 7) {
                    if let opened {
                        ModelScreen(app: app, opened: opened)
                    } else {
                        switch app.modelsTab {
                        case .library: ModelsView(app: app)
                        case .quantize: QuantizeView(app: app)
                        case .benchmark: BenchView(app: app)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, Panel.pad)
                .padding(.top, 14)
                .padding(.bottom, 12)
            }
            .scrollIndicators(.never)
        }
    }
}

struct Head: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    private var track: CGFloat { Panel.width - 2 * Panel.pad - 76 - 22 - 16 }

    var body: some View {
        let store = app.store
        let total = store.memoryBytes
        let used = Double(store.state?.residentBytes ?? 0)
        let ceiling = store.ceiling
        let down = store.down

        HStack(spacing: 8) {
            Image(nsImage: BrandLogo.image)
                .resizable()
                .scaledToFit()
                .frame(width: 76, height: 36)
                .foregroundStyle(t.fg)

            // The band: the residents drawn to its scale — weights solid, KV the same
            // material thinned. Its figures live on the instrument rather than below it.
            ZStack(alignment: .leading) {
                memoryFill(store, total: total, weightsOpacity: 0.78, kvOpacity: 0.34)
                // The tick is the daemon's number. With the daemon gone the band is empty,
                // and a mark left standing on it is the last answer pretending to be the
                // current one.
                if let ceiling, !down {
                    Rectangle().fill(t.fg2)
                        .frame(width: 1.5)
                        .offset(x: CGFloat(ceiling / total) * track)
                }
                memoryReading(used: used, total: total, ceiling: ceiling, down: down,
                              primary: t.fg, secondary: t.fg3)
                memoryReading(used: used, total: total, ceiling: ceiling, down: down,
                              primary: .white, secondary: .white)
                    .mask(memoryFill(store, total: total, weightsOpacity: 1, kvOpacity: 1))
            }
            .frame(maxWidth: .infinity)
            .frame(height: 22)
            .background(t.sel)
            .clipShape(RoundedRectangle(cornerRadius: 7))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(t.hair, lineWidth: 1))

            Overflow(app: app)
        }
        .padding(.horizontal, Panel.pad)
        .padding(.top, 12)
        .padding(.bottom, 11)
    }

    private func memoryFill(
        _ store: EngineStore, total: Double, weightsOpacity: Double, kvOpacity: Double
    ) -> some View {
        HStack(spacing: 1) {
            ForEach(store.models) { slot in
                let material = t.mat(store.materials[slot.id] ?? 0)
                Rectangle().fill(material.opacity(weightsOpacity))
                    .frame(width: CGFloat(Double(slot.weightsBytes) / total) * track)
                if slot.kvBytes > 0 {
                    Rectangle().fill(material.opacity(kvOpacity))
                        .frame(width: CGFloat(Double(slot.kvBytes) / total) * track)
                }
            }
            Spacer(minLength: 0)
        }
    }

    private func memoryReading(
        used: Double, total: Double, ceiling: Double?, down: Bool,
        primary: Color, secondary: Color
    ) -> some View {
        HStack(spacing: 6) {
            Text(down ? "— of 128 GB" : "\(Fmt.gb(used)) of \(Fmt.gb(total, 0)) GB")
                .mono(10.5, primary, weight: .semibold).lineLimit(1)
            if let ceiling, !down {
                Text("· ceiling \(Fmt.gb(ceiling, 0))").mono(10.5, secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
    }
}

/// What is the app's rather than the daemon's: quitting. Restart and Stop are here as well
/// as under Server on purpose — this is the shortcut for when no screen is open, and Server
/// is where whose process it is gets said.
///
/// The theme left this menu when Server grew a settings screen: an appearance control
/// belongs beside the other settings, not in the app's own menu.
struct Overflow: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        Menu {
            Button("Restart engine") { app.restartEngine() }
            Button("Stop engine") { app.stopEngine() }
            Divider()
            Button("Quit MLX Omnia") { NSApplication.shared.terminate(nil) }
        } label: {
            Text("⋯").sans(13, t.fg3)
                .frame(width: 22, height: 22)
                .background(t.sel)
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        // `.button` and not `.borderlessButton`: the borderless style throws the label's
        // own face away and draws the text alone, which loses the square this sits in.
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
    }
}

/// What a refused action left to say, drawn where the screen that started it can see it.
struct Refusal: View {
    @Environment(\.tokens) private var t
    let message: String

    var body: some View {
        Text(message).sans(11, t.bad)
            .lineSpacing(3)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

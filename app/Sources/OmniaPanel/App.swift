// The panel that hangs off the menu bar icon.
//
// `MenuBarExtra(.window)` is the whole reason this part is Swift: anchoring under the
// icon, dismissal on click-away, the notch, every display and Stage Manager are the
// system's here, and none of them is arithmetic this app has to keep.
//
// Three destinations and no window behind them: the daemon, its model tools, and a chat
// whose conversations are kept in the daemon's own store.

import AppKit
import Observation
import SwiftUI

enum Tab: String, CaseIterable, Hashable {
    case server, models, chat

    var label: String {
        switch self {
        case .chat: return "Chat"
        case .server: return "Server"
        case .models: return "Models"
        }
    }
}

enum ServerTab: CaseIterable, Hashable {
    case settings, status
}

enum ModelsTab: CaseIterable, Hashable {
    case library, quantize, benchmark

    var label: String {
        switch self {
        case .library: return "Library"
        case .quantize: return "Quantize"
        case .benchmark: return "Benchmark"
        }
    }
}

enum Mode: String, CaseIterable {
    case light, system, dark

    var label: String { rawValue.capitalized }
}

/// What is pushed over the tab, whole-panel. A repository carries the row it was opened
/// from rather than only its id: what the Hub says about it — how many have taken it, how
/// many liked it — is not on this disk to look up afterwards.
enum Pushed: Equatable {
    case none
    case model(String)
    case hub(HubModel)
}

@MainActor
@Observable
final class AppModel {
    static let shared = AppModel()

    let store = EngineStore()
    let daemon = Daemon()
    let downloads = Downloads()
    let chat = ChatModel()

    var mode: Mode = .system
    var tab: Tab = .server
    var modelsTab: ModelsTab = .library
    /// What is pushed over the list. `.none` is the list itself.
    var opened: Pushed = .none
    /// What the Models shelf is filtered by. Here rather than in the view because the chat
    /// writes it: a conversation whose checkpoint is gone links to that shelf, and a link
    /// that arrives at 78 unfiltered rows has not taken anybody anywhere.
    var modelSearch = ""
    /// What a refused action left to say. Cleared by the next one.
    var refusal = ""

    /// Whether a start is in flight, which is the difference between the down card saying
    /// the engine is being started and the down card offering to start it.
    var starting = false

    private var running = false
    private var following: Task<Void, Never>?

    /// The streams outlive the panel: the icon reports a daemon that went away whether or
    /// not anybody has the panel open, so these are the app's tasks and not a view's.
    func start() {
        guard !running else { return }
        running = true
        // Touched here and not in Settings: constructing it is what puts the scheduled
        // check on the clock, and a panel nobody opens is still a copy to be updated.
        _ = Updates.shared.available
        startEngine()
        following = Task { await store.follow() }
        Task { await store.pulse() }
        Task { await downloads.boot() }
    }

    func startEngine() {
        starting = true
        act { [self] in
            defer { starting = false }
            try await daemon.boot()
            // The daemon needs a moment to bind before the stream finds it; the follow
            // loop redials every second, so this only holds the card's own wording.
            try? await Task.sleep(for: .seconds(3))
        }
    }

    /// Stopping the engine starts by letting go of the stream. The daemon shuts down
    /// gracefully — it waits for open connections — and `/admin/events` is one this panel
    /// holds open forever, so a SIGTERM with the stream still attached leaves the engine
    /// draining and the panel drawing the daemon it has just asked to stop.
    func stopEngine() {
        act { [self] in
            release()
            defer { resume() }
            try await daemon.stop()
        }
    }

    func restartEngine() {
        act { [self] in
            release()
            defer { resume() }
            try await daemon.restart()
        }
    }

    private func release() {
        following?.cancel()
        following = nil
        store.health = nil
    }

    private func resume() {
        following = Task { await store.follow() }
    }

    func go(_ tab: Tab) {
        self.tab = tab
        opened = .none
    }

    /// The region grab, which needs the panel out of the way: it is a window over whatever
    /// is being asked about, and the crosshair cannot select what it covers. Dismissing it
    /// is what `MenuBarExtra` does when the app stops being active, so this asks for that
    /// and reopens nothing — the reader comes back to a panel holding the clip.
    func grab() {
        NSApp.hide(nil)
        chat.grabbing = true
        act { [self] in
            defer { chat.grabbing = false }
            let clip = try await Attach.region()
            NSApp.activate(ignoringOtherApps: true)
            chat.attach(clip)
        }
    }

    /// Run an action a screen started, and keep a refusal from taking the app with it.
    func act(_ work: @escaping () async throws -> Void) {
        Task {
            do {
                refusal = ""
                try await work()
            } catch {
                refusal = error.localizedDescription
            }
        }
    }
}

final class Delegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // `LSUIElement` in the bundle's Info.plist already says this; saying it here too
        // is what makes `swift run` out of the checkout behave like the bundle.
        NSApp.setActivationPolicy(.accessory)
        MainActor.assumeIsolated { AppModel.shared.start() }
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Quit is a stop: the daemon this panel started goes with it.
        MainActor.assumeIsolated { AppModel.shared.daemon.standDown() }
    }
}

@main
struct OmniaPanelApp: App {
    @NSApplicationDelegateAdaptor(Delegate.self) private var delegate
    @State private var app = AppModel.shared

    var body: some Scene {
        MenuBarExtra {
            PanelRoot(app: app)
        } label: {
            // The daemon that is not answering is the state the bar sees more than the
            // window does — the app opens at login, the engine may not be up.
            Image(systemName: app.store.down ? "minus" : "waveform.path.ecg")
                .opacity(app.store.down ? 0.45 : 1)
        }
        .menuBarExtraStyle(.window)
    }
}

struct PanelRoot: View {
    @Environment(\.colorScheme) private var scheme
    @Bindable var app: AppModel

    var body: some View {
        let dark = app.mode == .system ? scheme == .dark : app.mode == .dark
        PanelView(app: app)
            .environment(\.tokens, dark ? Tokens.dark : Tokens.light)
            .preferredColorScheme(app.mode == .system ? nil : (app.mode == .dark ? .dark : .light))
            .frame(width: Panel.width, height: Panel.height)
    }
}

enum Panel {
    /// 504x620, one size for every tab: `MenuBarExtra(.window)` takes any frame, but a
    /// frame that changed with the tab would resize the window under the cursor.
    ///
    /// The narrower height keeps the panel clear of the screen edge while the destinations
    /// retain their own scroll regions.
    static let width: CGFloat = 504
    static let height: CGFloat = 620
    static let pad: CGFloat = 14
}

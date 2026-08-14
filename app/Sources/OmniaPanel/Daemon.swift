// The engine process, and whether this panel is the one that started it.
//
// The panel discovers a daemon on /admin/health, starts one when nobody answers, and owns
// only the one it started: quitting takes that daemon with it, and Stop/Restart refuse on
// any other with the reason. Same rule as the window's `app/api/daemon.py`, and the same
// `--parent-pid` handshake — the daemon watches this pid and shuts itself down when it
// goes, which is the only path that survives a force quit.

import Foundation
import Observation

@MainActor
@Observable
final class Daemon {
    private var process: Process?

    var owned: Bool { process?.isRunning == true }

    static let foreign = "The engine was started outside the panel. Stop it where it was started."

    /// The checkout this panel was built inside, which is where `uv run` has to start and
    /// where the venv's console scripts live. `OMNIA_PROJECT` overrides it, exactly as the
    /// window's module does.
    static let root: URL? = {
        if let named = ProcessInfo.processInfo.environment["OMNIA_PROJECT"] {
            return URL(fileURLWithPath: named)
        }
        var here = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        for _ in 0..<12 {
            here.deleteLastPathComponent()
            if here.path == "/" { break }
            let marker = here.appendingPathComponent("pyproject.toml")
            if FileManager.default.fileExists(atPath: marker.path),
               let text = try? String(contentsOf: marker, encoding: .utf8),
               text.contains("mlx-omnia") {
                return here
            }
        }
        return nil
    }()

    /// The interpreter shipped inside this .app, or nil outside one. `mise run dmg` lays a
    /// CPython with the engine installed into it under Contents/Resources/engine; a panel
    /// built by `build.sh` alone has none, and falls through to the checkout.
    ///
    /// The console scripts are not used even when they are there: their shebang holds the
    /// path they were installed under, which is wrong the moment the .app is dragged into
    /// /Applications. `-m` is addressed relative to the interpreter that runs it.
    static var bundled: URL? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        let python = resources.appendingPathComponent("engine/bin/python3")
        return FileManager.default.fileExists(atPath: python.path) ? python : nil
    }

    /// How to start one of Omnia's console scripts, in the three places the panel runs from:
    /// the interpreter inside the bundle, the checkout's venv, or `uv` when there is no venv
    /// to point at. The module is named for the bundle's sake — a script is a path there,
    /// and `-m` is not.
    static func command(_ script: String, module: String) -> (URL, [String])? {
        if let bundled { return (bundled, ["-m", module]) }
        guard let root else { return nil }
        let beside = root.appendingPathComponent(".venv/bin/\(script)")
        if FileManager.default.fileExists(atPath: beside.path) { return (beside, []) }
        let places = ["/opt/homebrew/bin/uv", "/usr/local/bin/uv",
                      NSHomeDirectory() + "/.local/bin/uv"]
        guard let uv = places.first(where: { FileManager.default.fileExists(atPath: $0) })
        else { return nil }
        return (URL(fileURLWithPath: uv), ["run", "--project", root.path, script])
    }

    static var logs: URL {
        let directory = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Logs/mlx-omnia")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func up() async -> Bool {
        var request = URLRequest(url: URL(string: Client.base.absoluteString + "/admin/health")!)
        request.timeoutInterval = 1.5
        guard let (_, answer) = try? await URLSession.shared.data(for: request),
              let answer = answer as? HTTPURLResponse
        else { return false }
        return (200..<300).contains(answer.statusCode)
    }

    /// Start one if nobody answers. Raises rather than swallowing, because the two callers
    /// — the launch and the button on the down card — both have somewhere to put the
    /// reason.
    func boot() async throws {
        if owned { return }
        if await Self.up() { return }
        try spawn()
    }

    private func spawn() throws {
        guard let (executable, arguments) =
            Self.command("omnia-server", module: "mlx_omnia.server.main")
        else {
            throw Refused(detail: "no engine in this bundle and no checkout to start one from")
        }
        // Truncated at each start, because the run whose lines answer "why is it not up"
        // is the one that has just failed, and uvicorn writes an access line per request.
        let log = Self.logs.appendingPathComponent("daemon.log")
        let handle = try LogTail.openForWriting(log)

        let child = Process()
        child.executableURL = executable
        child.arguments = arguments + ["--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]
        child.currentDirectoryURL = Self.root
        child.standardOutput = handle
        child.standardError = handle
        try child.run()
        process = child
    }

    /// SIGTERM and wait: uvicorn shuts down and the port is free for a respawn. No SIGKILL
    /// fallback — a daemon that ignores SIGTERM is reported, not escalated.
    func stop() async throws {
        guard let child = process, child.isRunning else {
            throw Refused(detail: Self.foreign)
        }
        process = nil
        child.terminate()
        await withCheckedContinuation { (settled: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                child.waitUntilExit()
                settled.resume()
            }
        }
    }

    func restart() async throws {
        if owned {
            try await stop()
        } else if await Self.up() {
            throw Refused(detail: Self.foreign)
        }
        try await boot()
    }

    /// Quitting takes the daemon this panel started with it.
    func standDown() {
        process?.terminate()
    }

}

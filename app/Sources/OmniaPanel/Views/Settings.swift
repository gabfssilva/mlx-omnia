// The daemon's settings, under what it is holding.
//
// Which of them is a control is the daemon's own answer and not this screen's: every field
// of `/admin/config` carries an `effect`. `applied` becomes a live control — the change
// counts from the next request. `restart` is one field, the port, drawn as the value it is
// and dashed. `inert` is two, and those are not broken settings but facts: the daemon keeps
// both values and reads neither, so they sit under Fixed with nothing to turn.

import AppKit
import SwiftUI

struct DaemonSettings: View {
    @Bindable var app: AppModel

    var body: some View {
        Engine(app: app)
        Prefixes(app: app)
        Endpoints(app: app)
        Storage(app: app)
        Appearance(app: app)
        Fixed(app: app)
        ProcessRow(app: app)
        About()
    }
}

/// One PATCH, one field. The daemon announces `config` on the event stream when it lands,
/// so nothing here holds a copy of what it just wrote.
@MainActor
private func patch(_ app: AppModel, _ key: String, _ value: Written) {
    app.act {
        try await Client.send("PATCH", "/admin/config", body: ConfigPatch(fields: [key: value]))
    }
}

private let gib = 1024.0 * 1024 * 1024

// ── engine ───────────────────────────────────────────────────────────────

private struct Engine: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    /// Where the thumb is while it is being dragged. The written value is the store's, and
    /// it only moves once the gesture ends.
    @State private var dragged: Double?

    private static let idleStep = 300.0

    var body: some View {
        let store = app.store
        let ceiling = dragged ?? store.ceiling ?? store.memoryBytes
        let idle = store.config["idle_ttl_seconds"]?.value.number
        let key = store.config["api_key"]?.value.text
        let lower: (() -> Void)? = (idle ?? 0) > Self.idleStep
            ? { patch(app, "idle_ttl_seconds", .whole(Int((idle ?? 0) - Self.idleStep))) } : nil
        let raise: (() -> Void)? = idle == nil
            ? nil : { patch(app, "idle_ttl_seconds", .whole(Int((idle ?? 0) + Self.idleStep))) }

        SectionHead(title: "Engine")
        SettingsGroup {
            SettingRow(label: "Port", first: true) { InertField(text: store.port) }
            SettingNote(text: "Changing the port restarts the engine, which nothing here does yet.")

            SettingRow(label: "Memory ceiling") {
                Rail(
                    fraction: ceiling / store.memoryBytes,
                    settle: { part in
                        dragged = nil
                        patch(app, "memory_limit_bytes", .whole(Int(snapped(part) )))
                    },
                    slide: { part in dragged = snapped(part) }
                )
                .frame(width: 150)
                Text("\(Fmt.gb(ceiling, 0)) GB").mono(11.5, t.fg2)
                    .frame(width: 52, alignment: .trailing)
            }

            SettingRow(label: "Unload after idle") {
                Text(idle.map { "\(Int(($0 / 60).rounded())) min" } ?? "never")
                    .mono(11.5, idle == nil ? t.fg3 : t.fg2)
                Nudge(down: lower, up: raise)
                Latch(on: idle != nil) {
                    patch(app, "idle_ttl_seconds", idle == nil ? .whole(1800) : .nothing)
                }
            }

            SettingRow(label: "Model not resident") {
                Segmented(
                    options: [("load", "Load it"), ("fail", "Refuse")],
                    chosen: store.config["not_resident"]?.value.text ?? "load",
                    size: 11.5,
                    pick: { choice in patch(app, "not_resident", .text(choice)) }
                )
                .fixedSize()
            }

            SettingRow(label: "API key") {
                Text(key ?? "none — the daemon answers unauthenticated")
                    .mono(11.5, key == nil ? t.fg3 : t.fg2)
                    .lineLimit(1).truncationMode(.middle)
            }
        }
    }

    /// The ceiling in gibibyte steps: a slider that lands on 119.6 writes a number nobody
    /// asked for, and 4 GiB is the floor because a ceiling under one checkpoint is a daemon
    /// that never keeps anything.
    private func snapped(_ part: Double) -> Double {
        max(4 * gib, (part * app.store.memoryBytes / gib).rounded() * gib)
    }
}

// ── the prefix cache, in two tiers ───────────────────────────────────────

/// One cache with two floors. The trie in memory is what the resident models keep of the
/// conversations they have already read; the disk tier is what that trie evicted, and it is
/// the one that survives a restart. Both ceilings are shared by every model, so what is
/// drawn is what is used out of what is allowed — never a breakdown, which at thirty models
/// is thirty rows inside a 504 pt panel.
private struct Prefixes: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        let store = app.store
        let ceiling = store.ceiling
        let memoryLimit = store.config["prefix_cache_bytes"]?.value.number ?? 0
        let diskLimit = store.config["prefix_disk_bytes"]?.value.number ?? 0
        let memoryHeld = Double(store.state?.prefixMemoryBytes ?? 0)
        let diskHeld = Double(store.state?.prefixDiskBytes ?? 0)

        SectionHead(title: "Prefix cache")
        SettingsGroup {
            SettingRow(label: "In memory", first: true) {
                Text(pair(memoryHeld, memoryLimit)).mono(11.5, t.fg2)
                Nudge(
                    down: memoryLimit >= gib / 2
                        ? { patch(app, "prefix_cache_bytes", .whole(Int(memoryLimit - gib / 2))) } : nil,
                    up: { patch(app, "prefix_cache_bytes", .whole(Int(memoryLimit + gib / 2))) }
                )
                ActionText(label: "Clear") { clear("memory") }
            }
            TrackBar(fraction: share(memoryHeld, memoryLimit)).padding(.bottom, 6)
            if let ceiling {
                SettingNote(text: "Counted inside the \(Fmt.gb(ceiling, 0)) GB ceiling.")
            }

            SettingRow(label: "On disk") {
                Text(pair(diskHeld, diskLimit)).mono(11.5, t.fg2)
                Nudge(
                    down: diskLimit >= gib
                        ? { patch(app, "prefix_disk_bytes", .whole(Int(diskLimit - gib))) } : nil,
                    up: { patch(app, "prefix_disk_bytes", .whole(Int(diskLimit + gib))) }
                )
                ActionText(label: "Clear") { clear("disk") }
            }
            TrackBar(fraction: share(diskHeld, diskLimit)).padding(.bottom, 8)
        }
    }

    private func clear(_ tier: String) {
        app.act { try await Client.send("DELETE", "/admin/prefixes/\(tier)") }
    }

    private func pair(_ held: Double, _ limit: Double) -> String {
        "\(Fmt.gb(held)) of \(Fmt.gb(limit)) GB"
    }

    private func share(_ held: Double, _ limit: Double) -> Double {
        limit > 0 ? held / limit : 0
    }
}

// ── endpoints, storage, appearance ───────────────────────────────────────

private struct Endpoints: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    /// What each dialect is served at. Gemini is `v1beta` and not `v1` because that is the
    /// version Google's own clients name.
    private static let served = [
        ("OpenAI", "/api/openai/v1"),
        ("Anthropic", "/api/anthropic/v1"),
        ("Gemini", "/api/gemini/v1beta"),
    ]

    var body: some View {
        SectionHead(title: "Endpoints")
        SettingsGroup {
            ForEach(Array(Self.served.enumerated()), id: \.offset) { index, pair in
                SettingRow(label: pair.0, first: index == 0) {
                    Text(pair.1).mono(11.5, t.fg2).lineLimit(1)
                    PushButton(label: "Copy") { copy(pair.1) }
                }
            }
        }
    }

    /// The whole address and not the path: what is copied is pasted into a client, which
    /// needs the host and the port too.
    private func copy(_ path: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(Client.base.absoluteString + path, forType: .string)
    }
}

private struct Storage: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        let store = app.store
        SectionHead(title: "Storage")
        SettingsGroup {
            SettingRow(label: "Checkpoints", first: true) {
                Text("\(store.catalog.count)").mono(11.5, t.fg2)
            }
            SettingRow(label: "Disk free") {
                Text(store.system.map { "\(Fmt.gb(Double($0.diskFreeBytes), 0)) GB" } ?? "—")
                    .mono(11.5, t.fg2)
            }
        }
    }
}

private struct Appearance: View {
    @Bindable var app: AppModel

    var body: some View {
        SectionHead(title: "Appearance")
        SettingsGroup {
            SettingRow(label: "Theme", first: true) {
                Segmented(
                    options: Mode.allCases.map { ($0, $0.label) },
                    chosen: app.mode,
                    size: 11.5,
                    pick: { app.mode = $0 }
                )
                .fixedSize()
            }
        }
    }
}

// ── what the daemon keeps and does not read ──────────────────────────────

/// Two values the daemon stores and acts on with nothing. They are drawn as facts, beside
/// the API key, rather than as controls with an apology under them.
private struct Fixed: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        let store = app.store
        SectionHead(title: "Fixed")
        SettingsGroup {
            SettingRow(label: "Concurrent requests", first: true) {
                Text(store.config["max_concurrent_requests"]?.value.number.map { "\(Int($0))" } ?? "1")
                    .mono(11.5, t.fg2)
            }
            SettingNote(text: "One at a time until continuous batching lands.")
            SettingRow(label: "Catalog directory") {
                Text(store.config["catalog_directory"]?.value.text ?? "—")
                    .mono(11.5, t.fg2)
                    .lineLimit(1).truncationMode(.head)
                    .frame(maxWidth: 230, alignment: .trailing)
            }
            SettingNote(text: "Set by HF_HUB_CACHE when the daemon starts.")
        }
    }
}

private struct ProcessRow: View {
    @Bindable var app: AppModel

    var body: some View {
        SectionHead(title: "Daemon")
        SettingsGroup {
            SettingRow(label: "This panel's process", first: true) {
                PushButton(label: "Restart") { app.restartEngine() }
                PushButton(label: "Stop", kind: "danger") { app.stopEngine() }
            }
        }
    }
}

/// The version, and the check when there is one to make. A build with no updater says so
/// instead of offering a button that can only refuse — see `Updates`.
private struct About: View {
    @Environment(\.tokens) private var t

    var body: some View {
        SectionHead(title: "About")
        SettingsGroup {
            SettingRow(label: "Version", first: true) {
                Text(Updates.version).mono(11.5, t.fg2)
                if Updates.shared.available {
                    PushButton(label: "Check for updates") { Updates.shared.check() }
                }
            }
            if !Updates.shared.available {
                SettingNote(text: "This build carries no updater.")
            }
        }
    }
}

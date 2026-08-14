// Models: the window's shelf at the panel's width.
//
// The window gives a row three lines — name, id, meta — and 86 pt. Here a row is two
// lines and 52: the id goes, because the name and the meta together already say which
// checkpoint this is, and the id is what the screen behind this one is for.

import SwiftUI

struct ModelsView: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    @State private var hub: [HubModel] = []

    var body: some View {
        @Bindable var app = app
        let store = app.store
        let query = app.modelSearch
        let needle = query.lowercased()
        let onDisk = store.catalog.filter { needle.isEmpty || $0.id.lowercased().contains(needle) }
        let loaded = store.residentIds
        let resident = onDisk.filter { loaded.contains($0.id) }
        let idle = onDisk.filter { !loaded.contains($0.id) }
        let pulls = app.downloads.active.values
            .filter { needle.isEmpty || $0.repo.lowercased().contains(needle) }
            .sorted { $0.repo < $1.repo }
        let known = Set(store.catalog.map(\.id)).union(pulls.map(\.repo))
        let found = hub.filter { !known.contains($0.id) }

        SearchBox(placeholder: "Search disk and the Hub…", text: $app.modelSearch)
            // A breath after the keystroke, so the Hub is asked about words and not
            // letters. `task(id:)` cancels the last one, which is the debounce.
            .task(id: query) {
                guard query.count >= 2 else {
                    hub = []
                    return
                }
                try? await Task.sleep(for: .milliseconds(350))
                guard !Task.isCancelled else { return }
                let escaped = query
                    .addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
                // The Hub not answering leaves the section out, as it does in the window.
                hub = (try? await Client.get("/admin/hub/models?query=\(escaped)&limit=20")) ?? []
            }

        if !resident.isEmpty {
            SectionHead(title: "Resident · \(resident.count)")
            ForEach(resident) { entry in CatalogRow(app: app, entry: entry) }
        }

        if !idle.isEmpty || !pulls.isEmpty {
            SectionHead(title: "On this Mac · \(idle.count + pulls.count)")
            ForEach(pulls) { pull in DownloadRow(app: app, pull: pull) }
            ForEach(idle) { entry in CatalogRow(app: app, entry: entry) }
        }

        if !found.isEmpty {
            SectionHead(title: "From the Hub")
            ForEach(found) { one in HubRow(app: app, found: one) }
        }

        if resident.isEmpty && idle.isEmpty && pulls.isEmpty && found.isEmpty {
            EmptyNote(
                message: query.isEmpty
                    ? "Nothing on disk yet — search the Hub above."
                    : "Nothing on disk matches — search the Hub."
            )
        }

        if !app.refusal.isEmpty {
            Refusal(message: app.refusal).padding(.top, 4)
        }
    }

}

/// One shelf row: the material, two lines, and whatever the right edge has to say.
struct ShelfRow<Tail: View, Extra: View>: View {
    @Environment(\.tokens) private var t
    let material: Color
    let name: String
    let meta: String
    var metaColor: Color?
    var open: (() -> Void)?
    @ViewBuilder var tail: Tail
    @ViewBuilder var extra: Extra

    var body: some View {
        Pressable(action: open) {
            HStack(spacing: 9) {
                Material(color: material, height: 32)
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).sans(13.5, t.fg, weight: .semibold)
                        .lineLimit(1).truncationMode(.tail)
                    Text(meta).sans(11.5, metaColor ?? t.fg2)
                        .lineLimit(1).truncationMode(.tail)
                }
                Spacer(minLength: 0)
                VStack(alignment: .trailing, spacing: 4) { tail }
            }
            extra
        }
    }
}

private struct CatalogRow: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let entry: CatalogEntry

    var body: some View {
        let store = app.store
        let resident = store.residentIds.contains(entry.id)
        let material = resident ? t.mat(store.materials[entry.id] ?? 0) : t.sel
        // The PUT answers 202 and the load goes on in a job, so the row asks the daemon
        // whether one is still moving rather than remembering that it started one.
        let loading = store.jobs(kind: ["load"], model: entry.id) != nil

        ShelfRow(
            material: material,
            name: Fmt.displayName(entry.id),
            meta: meta,
            open: { app.opened = .model(entry.id) },
            tail: {
                if resident {
                    PillView(label: "Resident", kind: "resident")
                } else if loading {
                    PushButton(label: "Loading…")
                } else if !entry.supported {
                    PillView(label: "Unsupported", kind: "hub")
                } else if let short {
                    // Where the verb would be, because that is the answer to the same
                    // question: this one is not going into memory as things stand. How much
                    // it is over by is on the hover, not on seventy-odd rows at once.
                    PillView(label: "No VRAM available", kind: "hub")
                        .help("Needs \(Fmt.gb(short, 1)) GB more than the ceiling leaves")
                } else {
                    // The verb on the row rather than one screen in: what the shelf is read
                    // for is which of these to put in memory.
                    PushButton(label: "Load", action: load)
                }
            },
            extra: {
                // Indeterminate, because a load reports no total to divide against — the same
                // bar the engine's own start wears.
                if loading { Waiting().padding(.top, 9).padding(.bottom, 2) }
            }
        )
    }

    private var short: Double? { app.store.shortBy(entry) }

    private func load() {
        app.act {
            try await Client.send("PUT", "/admin/models/\(Client.at(entry.id))/residency")
        }
    }

    private var meta: String {
        let quant = entry.quantization ?? entry.dtype ?? "—"
        let context = entry.context.map(Fmt.tokens) ?? "—"
        return "\(entry.architecture) · \(quant) · \(Fmt.gb(Double(entry.bytesOnDisk))) GB · ctx \(context)"
    }
}

private struct DownloadRow: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let pull: Pull

    var body: some View {
        ShelfRow(
            material: t.sel,
            name: Fmt.displayName(pull.repo),
            meta: pull.error ?? pull.message,
            metaColor: pull.error == nil ? nil : t.bad,
            tail: {
                Text("✕").sans(11, t.fg3)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        app.act { try await Client.send("DELETE", "/admin/jobs/\(pull.job)") }
                    }
            },
            extra: {
                if let fraction = pull.fraction {
                    TrackBar(fraction: fraction).padding(.top, 8).padding(.bottom, 5)
                }
                Text(line).mono(10, t.fg3).lineLimit(1)
            }
        )
    }

    private var line: String {
        var said = [pull.total.map { Fmt.sizePair(Double(pull.completed), Double($0)) } ?? pull.share]
        if let rate = pull.rate { said.append(Fmt.rate(rate)) }
        if let eta = pull.eta { said.append(Fmt.left(eta)) }
        return said.joined(separator: " · ")
    }
}

private struct HubRow: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let found: HubModel

    var body: some View {
        ShelfRow(
            material: t.sel,
            name: Fmt.displayName(found.id),
            meta: counts,
            // The row opens like every other one: what a repository weighs and what is in
            // it are the questions asked before downloading it, and the Hub answers both.
            open: { app.opened = .hub(found) },
            tail: {
                PushButton(label: "Download", kind: "primary") {
                    app.act {
                        let job: Job? = try await Client.send(
                            "POST", "/admin/models", body: PullBody(repo: found.id)
                        )
                        if let job { app.downloads.follow(job) }
                    }
                }
            },
            extra: { EmptyView() }
        )
    }

    private var counts: String {
        var said: [String] = []
        if let downloads = found.downloads { said.append("\(Fmt.grouped(Double(downloads))) downloads") }
        if let likes = found.likes { said.append("\(Fmt.grouped(Double(likes))) likes") }
        return said.isEmpty ? "on the Hub" : said.joined(separator: " · ")
    }
}

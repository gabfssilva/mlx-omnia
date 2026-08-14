// One model, pushed over the shelf — the one on this disk and the one still on the Hub.
//
// Whether the weights are here decides what the screen has, not which screen it is. The
// card and the file listing exist on both sides; what the checkpoint declares and what it
// is served under exist only once the bytes do; and the verb is Download on one side and
// Load on the other. A repository that finishes downloading becomes the checkpoint under
// the reader, with nobody navigating.
//
// The card is where it opens, and the verbs are up in the identity row rather than under
// the face: a README on this disk runs to 79 kB, and a verb at the foot of twenty screens
// of it is a verb nobody can reach.

import AppKit
import SwiftUI

enum Opened: Equatable {
    case disk(CatalogEntry)
    case hub(HubModel)

    var id: String {
        switch self {
        case let .disk(entry): return entry.id
        case let .hub(found): return found.id
        }
    }

    var entry: CatalogEntry? {
        if case let .disk(entry) = self { return entry }
        return nil
    }
}

private enum Face: String, CaseIterable {
    case card, overview, anatomy, configure, files

    var label: String { rawValue.capitalized }
}

struct ModelScreen: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let opened: Opened

    @State private var face: Face?
    @State private var card: String?
    @State private var files: [CheckpointFile]?
    @State private var blueprint: Blueprint?
    @State private var untraceable = ""
    @State private var settled = false
    @State private var trouble = ""

    var body: some View {
        let store = app.store
        let entry = opened.entry
        let resident = entry.map { store.residentIds.contains($0.id) } ?? false
        let pull = app.downloads.active[opened.id]

        Text("‹ Models").sans(12.5, t.accent, weight: .semibold)
            .padding(.horizontal, 2)
            .padding(.bottom, 10)
            .contentShape(Rectangle())
            .onTapGesture { app.opened = .none }

        HStack(alignment: .center, spacing: 10) {
            Material(
                color: resident ? t.mat(store.materials[opened.id] ?? 0) : t.sel, height: 44
            )
            VStack(alignment: .leading, spacing: 3) {
                Text(Fmt.displayName(opened.id)).display(17, t.fg)
                    .lineLimit(1).truncationMode(.tail)
                Text(opened.id).mono(10, t.fg3).lineLimit(1).truncationMode(.middle)
                state(entry, resident: resident)
            }
            Spacer(minLength: 0)
            verbs(entry, resident: resident, pull: pull)
        }

        if let pull {
            // Started from here or from the shelf — the row and this screen watch the same
            // job, so the bar is wherever the reader happens to be.
            VStack(alignment: .leading, spacing: 6) {
                if let fraction = pull.fraction { TrackBar(fraction: fraction) }
                Text(line(pull)).mono(10, t.fg3).lineLimit(1).truncationMode(.middle)
            }
            .padding(.top, 10)
        }

        facts(entry, store: store)

        HStack(spacing: 16) {
            ForEach(faces, id: \.self) { one in
                Text(one.label)
                    .sans(
                        13,
                        chosen == one ? t.fg : t.fg3,
                        weight: chosen == one ? .semibold : .regular
                    )
                    .contentShape(Rectangle())
                    .onTapGesture { face = one }
            }
            Spacer(minLength: 0)
        }
        .padding(.top, 14)
        .padding(.bottom, 9)
        .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }

        switch chosen {
        case .card: reading
        case .overview:
            if let entry { Overview(entry: entry, store: store) }
        case .anatomy:
            // Asked for when the face is opened and not with the rest: tracing a checkpoint
            // costs the daemon half a second of its threadpool, and most readings of a model
            // never come here.
            AnatomyView(blueprint: blueprint, trouble: untraceable)
                .task(id: opened.id) { await outline() }
        case .configure:
            if let entry { Configure(app: app, entry: entry) }
        case .files:
            if trouble.isEmpty {
                FileRows(files: files).padding(.top, 8)
            } else {
                Refusal(message: trouble).padding(.top, 10)
            }
        }

        if !app.refusal.isEmpty {
            Refusal(message: app.refusal).padding(.top, 8)
        }

        Color.clear.frame(height: 0).task(id: opened.id) { await read() }
    }

    // ── what is on, and where it opens ───────────────────────────────────

    private var faces: [Face] {
        opened.entry == nil ? [.card, .files] : Face.allCases
    }

    /// Where the screen opens. The card, when there is one to read — and the state's own
    /// face when there is not, which is 41 of the 78 checkpoints on this disk: what is
    /// quantized here ships no README, and landing on a face that says so is landing
    /// nowhere.
    private var chosen: Face {
        if let face, faces.contains(face) { return face }
        if !settled || card != nil { return .card }
        return opened.entry == nil ? .files : .configure
    }

    @ViewBuilder
    private var reading: some View {
        if let card {
            let (front, prose) = Front.split(card)
            VStack(alignment: .leading, spacing: 0) {
                if let front {
                    FrontChips(front: front).padding(.top, 12).padding(.bottom, 2)
                }
                if prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text("This card is its front matter and nothing else.")
                        .sans(12.5, t.fg3)
                        .padding(.top, 10)
                } else {
                    Clamped { Prose(text: prose) }.padding(.top, 10)
                }
            }
        } else if settled {
            Text(
                opened.entry == nil
                    ? "This repository has no card on the Hub."
                    : "This checkpoint ships no card."
            )
            .sans(12.5, t.fg3)
            .padding(.top, 14)
        } else {
            Text("asking the daemon…").mono(10.5, t.fg3).padding(.vertical, 10)
        }
    }

    // ── the head ─────────────────────────────────────────────────────────

    @ViewBuilder
    private func state(_ entry: CatalogEntry?, resident: Bool) -> some View {
        HStack(spacing: 8) {
            if let entry {
                if resident {
                    PillView(label: "resident", kind: "resident")
                } else if entry.supported {
                    PillView(label: "on disk")
                } else {
                    PillView(label: "no loader for \(entry.architecture)")
                }
            } else {
                PillView(label: "on the Hub", kind: "hub")
                if let counts { Text(counts).sans(11, t.fg3).lineLimit(1) }
            }
        }
    }

    @ViewBuilder
    private func verbs(_ entry: CatalogEntry?, resident: Bool, pull: Pull?) -> some View {
        HStack(spacing: 6) {
            if let entry {
                if resident {
                    PushButton(label: "Unload") {
                        app.act {
                            try await Client.send(
                                "DELETE", "/admin/models/\(Client.at(entry.id))/residency"
                            )
                        }
                    }
                } else if app.store.jobs(kind: ["load"], model: entry.id) != nil {
                    PushButton(label: "Loading…", kind: "primary")
                } else if let short = app.store.shortBy(entry) {
                    // No button where a press would only buy a refusal from the daemon: the
                    // ceiling has already answered.
                    PillView(label: "No VRAM available", kind: "hub")
                        .help("Needs \(Fmt.gb(short, 1)) GB more than the ceiling leaves")
                } else if entry.supported {
                    PushButton(label: "Load", kind: "primary") {
                        app.act {
                            try await Client.send(
                                "PUT", "/admin/models/\(Client.at(entry.id))/residency"
                            )
                        }
                    }
                }
            } else if let pull {
                PushButton(label: "Cancel", kind: "danger") {
                    app.act { try await Client.send("DELETE", "/admin/jobs/\(pull.job)") }
                }
            } else {
                PushButton(label: "Download", kind: "primary") {
                    app.act {
                        let job: Job? = try await Client.send(
                            "POST", "/admin/models", body: PullBody(repo: opened.id)
                        )
                        if let job { app.downloads.follow(job) }
                    }
                }
            }
            overflow(entry, resident: resident)
        }
    }

    /// What is not the verb of the moment: the page this came from, and the one action that
    /// cannot be taken back.
    private func overflow(_ entry: CatalogEntry?, resident: Bool) -> some View {
        Menu {
            if let repo = hubRepo, let link = URL(string: "https://huggingface.co/\(Client.at(repo))") {
                Button("Open on the Hub ↗") { NSWorkspace.shared.open(link) }
            }
            if let entry {
                Divider()
                Button("Delete from disk", role: .destructive) {
                    app.act {
                        try await Client.send("DELETE", "/admin/models/\(Client.at(entry.id))")
                        app.opened = .none
                    }
                }
                .disabled(resident)
                // The daemon refuses this with a 409 while the weights are in memory, and a
                // menu that says why beats a refusal that arrives after the press.
                if resident { Text("Unload it before deleting it") }
            }
        } label: {
            Text("⋯").sans(13, t.fg3)
                .frame(width: 30, height: 30)
                .background(t.elev)
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(t.hair2, lineWidth: 1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
    }

    @ViewBuilder
    private func facts(_ entry: CatalogEntry?, store: EngineStore) -> some View {
        Card {
            if let entry {
                let sample = store.sample(entry.id)
                HStack(spacing: 10) {
                    FactView(
                        label: "Quantization", value: entry.quantization ?? entry.dtype ?? "—"
                    )
                    FactView(label: "Size", value: "\(Fmt.gb(Double(entry.bytesOnDisk))) GB")
                    FactView(label: "Context", value: entry.context.map(Fmt.tokens) ?? "—")
                }
                HStack(spacing: 10) {
                    FactView(
                        label: "Decode",
                        value: store.decode(entry.id)?
                            .replacingOccurrences(of: " tok/s", with: "") ?? "—"
                    )
                    FactView(
                        label: "Prefill",
                        value: sample?.prefillTokensPerSecond.map { Fmt.grouped($0) } ?? "—"
                    )
                    FactView(
                        label: "TTFT",
                        value: sample?.ttft.map { String(format: "%.0f ms", $0 * 1000) } ?? "—"
                    )
                }
                .padding(.top, 11)
            } else {
                HStack(spacing: 10) {
                    // Scaled rather than in GB: a repository on the Hub is anything from a
                    // half-megabyte test model to 237 GB, and `0.0 GB` is not a size.
                    FactView(label: "Download", value: files == nil ? "—" : Fmt.bytes(total))
                    FactView(label: "Files", value: files.map { "\($0.count)" } ?? "—")
                    FactView(label: "Shards", value: shards)
                }
            }
        }
        .padding(.top, 12)
    }

    // ── what it is made of ───────────────────────────────────────────────

    private var counts: String? {
        guard case let .hub(found) = opened else { return nil }
        var said: [String] = []
        if let downloads = found.downloads {
            said.append("\(Fmt.grouped(Double(downloads))) downloads")
        }
        if let likes = found.likes { said.append("\(Fmt.grouped(Double(likes))) likes") }
        return said.isEmpty ? nil : said.joined(separator: " · ")
    }

    private var total: Double {
        Double(files?.reduce(0) { $0 + $1.size } ?? 0)
    }

    /// How the weights arrive: one file or many. It is the one thing about a checkpoint's
    /// shape that is legible before any of it is read.
    private var shards: String {
        guard let files else { return "—" }
        let count = files.filter { $0.name.hasSuffix(".safetensors") }.count
        return count == 0 ? "—" : "\(count)"
    }

    /// Which ids name a page on the Hub. `local/…` is what this daemon writes when it
    /// quantizes, and an id with more than one slash was never a repository — neither has
    /// anything to open.
    private var hubRepo: String? {
        let id = opened.id
        guard id.split(separator: "/").count == 2, !id.hasPrefix("local/") else { return nil }
        return id
    }

    private func line(_ pull: Pull) -> String {
        var said = [
            pull.total.map { Fmt.sizePair(Double(pull.completed), Double($0)) } ?? pull.share
        ]
        if let rate = pull.rate { said.append(Fmt.rate(rate)) }
        if let eta = pull.eta { said.append(Fmt.left(eta)) }
        return said.joined(separator: " · ")
    }

    private var routes: (card: String, files: String) {
        let at = Client.at(opened.id)
        return opened.entry == nil
            ? ("/admin/hub/models/\(at)/card", "/admin/hub/models/\(at)/files")
            : ("/admin/models/\(at)/card", "/admin/models/\(at)/files")
    }

    /// Both at once, because the screen needs both to draw its first frame: the card
    /// decides which face opens, and the listing is what the Hub side prices a download
    /// with. A card that will not load is a card-less model; a listing that will not is
    /// the Hub being unreachable, which is a different thing and says so.
    ///
    /// The face is chosen again here rather than kept: which one a model opens on depends
    /// on whether it has a card, so a face held from the last model would answer for this
    /// one.
    /// The graph of one decode step. The daemon builds the tree to answer it and reads no
    /// weight doing so, so this asks the same thing of a resident model and of one that has
    /// never been loaded — and a checkpoint whose loader raises says so instead of drawing.
    private func outline() async {
        guard blueprint == nil, untraceable.isEmpty else { return }
        do {
            blueprint = try await Client.get(
                "/admin/models/\(Client.at(opened.id))/blueprint"
            )
        } catch {
            untraceable = error.localizedDescription
        }
    }

    private func read() async {
        settled = false
        face = nil
        card = nil
        files = nil
        blueprint = nil
        untraceable = ""
        trouble = ""
        let (cardRoute, filesRoute) = routes
        async let asked = Client.text(cardRoute)
        async let listed: [CheckpointFile] = Client.get(filesRoute)
        var found = (try? await asked) ?? nil
        // A checkpoint downloaded here has no README on this disk: the daemon fetches what
        // the loader reads, and a card is not one of those files. The repository still has
        // one, so the Hub answers for it — a request spent only when the disk had nothing.
        if found == nil, opened.entry != nil, let repo = hubRepo {
            found = (try? await Client.text("/admin/hub/models/\(Client.at(repo))/card")) ?? nil
        }
        card = found
        do {
            files = try await listed
        } catch {
            files = []
            trouble = error.localizedDescription
        }
        settled = true
    }
}

// ── overview: what the checkpoint declares ───────────────────────────────

private struct Overview: View {
    @Environment(\.tokens) private var t
    let entry: CatalogEntry
    let store: EngineStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SettingRow(label: "Architecture", first: true) {
                Text(entry.architecture).mono(11.5, t.fg2).lineLimit(1)
            }
            SettingRow(label: "Vocabulary") {
                Text(entry.vocabSize.map { Fmt.grouped(Double($0)) } ?? "—").mono(11.5, t.fg2)
            }
            SettingRow(label: "Shape") {
                Text(entry.shape ?? "—").mono(11.5, t.fg2).lineLimit(1).truncationMode(.middle)
            }
            SettingRow(label: "Per token") {
                Text(bytesPerToken).mono(11.5, t.fg2).lineLimit(1)
            }
            SettingRow(label: "KV per token") {
                Text(kvPerToken).mono(11.5, t.fg2).lineLimit(1)
            }
            if let defaults = entry.defaults, let said = sampling(defaults) {
                Eyebrow(text: "Checkpoint defaults")
                    .padding(.horizontal, 2)
                    .padding(.top, 12)
                    .padding(.bottom, 4)
                Text(said).mono(11.5, t.fg2)
                    .padding(.horizontal, 2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.top, 4)
    }

    /// What decode reads per token, and what the cache grows by. Both are the daemon's
    /// numbers — the first is the denominator every % of ceiling divides by.
    private var bytesPerToken: String {
        entry.bytesPerToken.map(Fmt.bytes) ?? "—"
    }

    private var kvPerToken: String {
        entry.kvBytesPerToken.map(Fmt.bytes) ?? "—"
    }

    /// What the checkpoint's own `generation_config.json` declares. A knob it does not
    /// declare is absent, which is not the same as one it declares neutral.
    private func sampling(_ defaults: SamplingDefaults) -> String? {
        var said: [String] = []
        if let value = defaults.temperature { said.append(String(format: "temp %.2f", value)) }
        if let value = defaults.topP { said.append(String(format: "top-p %.2f", value)) }
        if let value = defaults.topK { said.append("top-k \(value)") }
        if let value = defaults.minP { said.append(String(format: "min-p %.2f", value)) }
        if let value = defaults.repetitionPenalty {
            said.append(String(format: "rep %.2f", value))
        }
        return said.isEmpty ? nil : said.joined(separator: " · ")
    }
}

// ── configure: speculation and profiles ──────────────────────────────────

private struct Configure: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let entry: CatalogEntry

    @State private var settings: SettingsView?
    @State private var profiles: [(name: String, knobs: String)] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Eyebrow(text: "Features")
                .padding(.horizontal, 2)
                .padding(.top, 12)
                .padding(.bottom, 4)

            if let settings {
                SettingRow(label: "Speculation", first: true) {
                    choices(settings)
                }
                if let reason = settings.unavailableReason {
                    Text(reason).mono(10.5, t.fg3)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                        .padding(.top, 2)
                }
            } else {
                SettingRow(label: "Speculation", dim: true, first: true) {
                    Text("asking the daemon…").mono(10.5, t.fg3)
                }
            }

            if !profiles.isEmpty {
                Eyebrow(text: "Profiles")
                    .padding(.horizontal, 2)
                    .padding(.top, 12)
                    .padding(.bottom, 4)
                ForEach(Array(profiles.enumerated()), id: \.offset) { index, profile in
                    HStack(spacing: 9) {
                        Text(profile.name).mono(12.5, t.fg, weight: .medium).lineLimit(1)
                        Spacer(minLength: 0)
                        Text(profile.knobs).mono(10, t.fg3).lineLimit(1).truncationMode(.tail)
                    }
                    .padding(.horizontal, 2)
                    .padding(.vertical, 7)
                    .overlay(alignment: .top) {
                        if index > 0 { Rectangle().fill(t.hair).frame(height: 1) }
                    }
                }
            }

            Text("+ New profile").sans(12.5, t.accent, weight: .semibold)
                .padding(.horizontal, 2)
                .padding(.top, 10)
                .contentShape(Rectangle())
                .onTapGesture { add() }
        }
        .task(id: entry.id) { await load() }
    }

    @ViewBuilder
    private func choices(_ settings: SettingsView) -> some View {
        let kind = settings.features.speculation?.kind
        let drafter = settings.features.speculation?.drafter
        // Wrapping, because a Mac with three drafters on disk has more names than 200 pt.
        FlowRow(spacing: 6) {
            option("off", on: kind == nil) { choose(nil, nil) }
            if settings.mtpAvailable {
                option("mtp", on: kind == "mtp") { choose("mtp", nil) }
            }
            ForEach(settings.available, id: \.self) { candidate in
                option(
                    Fmt.displayName(candidate),
                    on: kind == "dflash" && drafter == candidate
                ) { choose("dflash", candidate) }
            }
        }
    }

    private func option(_ label: String, on: Bool, pick: @escaping () -> Void) -> some View {
        Text(label).sans(11.5, on ? t.fg : t.fg2)
            .lineLimit(1)
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .background(on ? t.accentSoft : .clear)
            .overlay(Capsule().strokeBorder(on ? t.accent : t.hair, lineWidth: 1))
            .clipShape(Capsule())
            .contentShape(Rectangle())
            .onTapGesture(perform: pick)
    }

    private func choose(_ kind: String?, _ drafter: String?) {
        app.act {
            let body = SettingsBody(
                features: Features(
                    speculation: kind == nil
                        ? nil : SpeculationSetting(kind: kind, drafter: drafter, blockSize: nil)
                )
            )
            settings = try await Client.send(
                "PUT", "/admin/models/\(Client.at(entry.id))/settings", body: body
            )
        }
    }

    private func add() {
        app.act {
            let name = "profile-\(profiles.count + 1)"
            let _: ProfileView? = try await Client.send(
                "PUT", "/admin/models/\(Client.at(entry.id))/profiles/\(Client.at(name))",
                body: ProfileBody(sampling: Sampling())
            )
            await load()
        }
    }

    private func load() async {
        // A daemon that is down settles for nothing, which is what the row already says.
        settings = try? await Client.get("/admin/models/\(Client.at(entry.id))/settings")
        profiles = await Self.profiles(of: entry.id)
    }

    /// There is no route that lists a model's profiles. What does list them is the
    /// dialect's own catalog: a model with a profile `code` is served under `model` and
    /// `model:code`, so the names are the suffixes of the ids under this one.
    private static func profiles(of model: String) async -> [(name: String, knobs: String)] {
        struct Served: Decodable {
            struct Entry: Decodable { let id: String }
            let data: [Entry]
        }
        guard let served: Served = try? await Client.get("/api/openai/v1/models") else { return [] }
        let prefix = "\(model):"
        let names = served.data.map(\.id)
            .filter { $0.hasPrefix(prefix) && $0.count > prefix.count }
            .map { String($0.dropFirst(prefix.count)) }
        var held: [(name: String, knobs: String)] = []
        for name in names {
            let view: ProfileView? = try? await Client.get(
                "/admin/models/\(Client.at(model))/profiles/\(Client.at(name))"
            )
            held.append((name, knobs(view?.sampling)))
        }
        return held
    }

    /// What this profile actually sets, in the order the window's editor asks for it. A
    /// profile that sets nothing is served at the engine's defaults, and says so.
    private static func knobs(_ sampling: Sampling?) -> String {
        guard let sampling else { return "—" }
        var said: [String] = []
        if let value = sampling.temperature { said.append(String(format: "temp %.2f", value)) }
        if let value = sampling.topP { said.append(String(format: "top-p %.2f", value)) }
        if let value = sampling.topK { said.append("top-k \(value)") }
        return said.isEmpty ? "engine defaults" : said.joined(separator: " · ")
    }
}

// ── files: what the checkpoint is made of ────────────────────────────────

/// What a checkpoint is made of, drawn the same on either side of a download: the listing
/// the daemon reads off the disk, and the one it prices off the Hub, are the same shape.
struct FileRows: View {
    @Environment(\.tokens) private var t
    let files: [CheckpointFile]?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let files {
                ForEach(Array(files.enumerated()), id: \.offset) { index, file in
                    HStack(spacing: 9) {
                        Text(file.name).mono(11.5, t.fg2).lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 0)
                        Text(Fmt.bytes(Double(file.size))).mono(10, t.fg3)
                    }
                    .padding(.horizontal, 2)
                    .padding(.vertical, 6)
                    .overlay(alignment: .top) {
                        if index > 0 { Rectangle().fill(t.hair).frame(height: 1) }
                    }
                }
            } else {
                Text("asking the daemon…").mono(10.5, t.fg3).padding(.vertical, 10)
            }
        }
    }
}

/// A row that wraps — the speculation options are as many as the drafters on disk, and a
/// card's front matter is as many chips as it has tags.
struct FlowRow: Layout {
    var spacing: CGFloat = 6
    /// Rows against the right edge, which is where a settings row's value sits. The chips
    /// of a card are a block of their own and start at the margin.
    var trailing = true

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 200
        var x: CGFloat = 0, y: CGFloat = 0, line: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > width, x > 0 {
                x = 0
                y += line + spacing
                line = 0
            }
            x += size.width + spacing
            line = max(line, size.height)
        }
        return CGSize(width: width, height: y + line)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        var x: CGFloat = 0, y: CGFloat = 0, line: CGFloat = 0
        var rows: [[(LayoutSubview, CGSize)]] = [[]]
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > bounds.width, x > 0 {
                x = 0
                y += line + spacing
                line = 0
                rows.append([])
            }
            rows[rows.count - 1].append((view, size))
            x += size.width + spacing
            line = max(line, size.height)
        }
        y = bounds.minY
        for row in rows {
            let width = row.reduce(0) { $0 + $1.1.width } + spacing * CGFloat(max(0, row.count - 1))
            var cursor = trailing ? bounds.maxX - width : bounds.minX
            let tallest = row.reduce(0) { max($0, $1.1.height) }
            for (view, size) in row {
                view.place(
                    at: CGPoint(x: cursor, y: y + (tallest - size.height) / 2),
                    proposal: ProposedViewSize(size)
                )
                cursor += size.width + spacing
            }
            y += tallest + spacing
        }
    }
}

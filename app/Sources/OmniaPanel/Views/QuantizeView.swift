// Quantize: the window's two columns, stacked.
//
// The rail becomes the block that closes the screen, because on a panel the projected
// size is the answer being waited for and not a sidebar to it.
//
// The rules below are the server's, restated because this side of the wire is Swift: an
// exponent-scaled mode fixes method, width and group size; GPTQ packs only 2, 4 and 8;
// oQ and oQe take the budget nobody else may name. Every one of them is confessed before
// the 400, never enforced instead of it — the daemon still decides.

import SwiftUI

private let modes: [(key: String, label: String, shape: (group: Int, bits: Int)?)] = [
    ("affine", "Affine", nil),
    ("mxfp4", "MXFP4", (32, 4)),
    ("mxfp8", "MXFP8", (32, 8)),
    ("nvfp4", "NVFP4", (16, 4)),
]

private let methods: [(String, String)] = [
    ("rtn", "RTN"), ("awq", "AWQ"), ("gptq", "GPTQ"), ("oq", "oQ"), ("oqe", "oQe"),
]

private let widths = [2, 3, 4, 5, 6, 8]
private let groupSizes = [32, 64, 128]
/// GPTQ packs its own codes; only these have a layout verified against `mx.quantize`.
private let gptqWidths: Set<Int> = [2, 4, 8]
private let dtypes = ["bfloat16": "bf16", "float16": "fp16", "float32": "fp32"]

struct QuantizeView: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    @State private var source = ""
    @State private var mode = "affine"
    @State private var method = "rtn"
    @State private var bits = 4
    @State private var groupSize = 64
    @State private var overrides: [String: LeafOverride] = [:]
    @State private var repo = ""
    @State private var named = false
    @State private var target = ""
    @State private var cap = ""
    @State private var plan: PricedPlan?
    @State private var refused = ""
    @State private var pricing = false
    @State private var started = ""
    @State private var job: Job?
    @State private var failure = ""
    @State private var leavesOpen = false

    private var exponent: (key: String, label: String, shape: (group: Int, bits: Int)?)? {
        modes.first { $0.key == mode && $0.shape != nil }
    }

    private var allocated: Bool { method == "oq" || method == "oqe" }

    var body: some View {
        let dense = app.store.catalog.filter { $0.quantization == nil }
        let chosen = dense.first { $0.id == source }

        if dense.isEmpty {
            EmptyNote(message: "Nothing dense on this disk to quantize — fetch a model under Models.")
        } else {
            labelled("Source") { picker(dense, chosen) }

            labelled(
                "Format",
                hint: exponent.flatMap { one in
                    one.shape.map { shape in
                        "\(shape.bits) bits at group \(shape.group). "
                            + "Method, width and group size come with the format."
                    }
                }
            ) {
                Segmented(options: modes.map { ($0.key, $0.label) }, chosen: mode) { pick(mode: $0) }
            }

            if exponent == nil {
                labelled("Method") {
                    Segmented(options: methods, chosen: method) { pick(method: $0) }
                }
                HStack(alignment: .top, spacing: 12) {
                    labelled("Width") {
                        Segmented(
                            options: widths.map { ($0, String($0)) },
                            chosen: bits,
                            dimmed: method == "gptq" ? Set(widths.filter { !gptqWidths.contains($0) }) : [],
                            pick: { pick(bits: $0) }
                        )
                    }
                    labelled("Group size") {
                        Segmented(
                            options: groupSizes.map { ($0, String($0)) }, chosen: groupSize
                        ) { groupSize = $0 }
                    }
                    .frame(width: 132)
                }
            }

            if allocated {
                // Two controls, so two eyebrows: named after the field the way every other
                // row on this screen is, and what each one is worth left empty is the
                // placeholder's to say.
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .top, spacing: 12) {
                        labelled("Target") { MonoField(hint: "RTN + 1.0", text: $target) }
                        labelled("Hard cap") { MonoField(hint: "same as target", text: $cap) }
                    }
                    Text("Effective bits per weight, scales and biases included.")
                        .sans(11, t.fg3)
                        .lineSpacing(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            labelled("Output id") {
                VStack(alignment: .leading, spacing: 4) {
                    MonoField(hint: "org/name", text: Binding(
                        get: { repo },
                        set: { value in
                            named = true
                            repo = value
                        }
                    ))
                    if !repo.isEmpty && !Self.wellNamed(repo) {
                        Refusal(message: "Two segments, and nothing that could name a directory: org/name.")
                    }
                }
            }

            rail(chosen)

            // The header stays through a refusal, because the overrides that may have
            // caused one are cleared from it and the price that failed carries no rows.
            if plan != nil || !overrides.isEmpty {
                leaves(plan)
            }
        }

        // The daemon's presence is a dependency too: a price asked while it was still
        // coming up is retried when it answers, instead of standing as a refusal nobody
        // re-asks.
        Color.clear.frame(height: 0)
            .task(id: Price(source: source, mode: mode, method: method, bits: bits,
                            groupSize: groupSize, overrides: overrides, target: target,
                            cap: cap, up: !app.store.down)) {
                await price()
            }
            .task(id: started) { await watch() }
    }

    // ── the controls ─────────────────────────────────────────────────────

    @ViewBuilder
    private func labelled<Content: View>(
        _ label: String, hint: String? = nil, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Eyebrow(text: label)
            content()
            if let hint {
                Text(hint).sans(11, t.fg3)
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.top, 6)
    }

    private func picker(_ dense: [CatalogEntry], _ chosen: CatalogEntry?) -> some View {
        Menu {
            ForEach(dense) { entry in
                Button("\(Fmt.displayName(entry.id))   \(grain(entry))") { pick(source: entry.id) }
            }
        } label: {
            HStack(spacing: 9) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(app.store.residentIds.contains(source)
                        ? t.mat(app.store.materials[source] ?? 0) : t.sel)
                    .frame(width: 9, height: 9)
                Text(source.isEmpty ? "Pick a checkpoint" : Fmt.displayName(source))
                    .sans(13, source.isEmpty ? t.fg3 : t.fg, weight: .medium)
                    .lineLimit(1).truncationMode(.tail)
                Spacer(minLength: 0)
                if let chosen {
                    Text(grain(chosen)).mono(10, t.fg3).lineLimit(1)
                }
                Text("▾").sans(9, t.fg3)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(t.elev)
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.hair, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private func grain(_ entry: CatalogEntry) -> String {
        let dtype = entry.dtype.map { dtypes[$0] ?? $0 } ?? "?"
        return "\(dtype) · \(Fmt.gb(Double(entry.bytesOnDisk))) GB"
    }

    // ── the projected block ──────────────────────────────────────────────

    @ViewBuilder
    private func rail(_ chosen: CatalogEntry?) -> some View {
        let running = job.map(\.moving) ?? false
        Card {
            Eyebrow(text: "Projected")
            if let plan {
                HStack(alignment: .lastTextBaseline, spacing: 6) {
                    Text(Fmt.gb(Double(plan.entryBytes)))
                        .font(.system(size: 30, design: .monospaced))
                        .tracking(-0.9)
                        .foregroundColor(t.fg)
                    Text("GB").mono(11, t.fg3)
                }
                .padding(.top, 4)

                TrackBar(
                    fraction: chosen.map { entry in
                        entry.bytesOnDisk == 0
                            ? 0 : min(1, Double(plan.entryBytes) / Double(entry.bytesOnDisk))
                    } ?? 0,
                    height: 4
                )
                .padding(.top, 10)

                HStack(spacing: 8) {
                    Text(String(format: "%.2f bits / weight", plan.bitsPerWeight)).mono(10, t.fg3)
                    Spacer(minLength: 0)
                    if let chosen {
                        Text("source \(Fmt.gb(Double(chosen.bytesOnDisk))) GB").mono(10, t.fg3)
                    }
                }
                .padding(.top, 5)

                kv("If loaded now", loaded(plan))
                    .padding(.top, 7)
                kv(
                    "Leaves packed",
                    "\(plan.leaves.filter { $0.bits != nil }.count) / \(plan.leaves.count)"
                )

            } else {
                Text("—")
                    .font(.system(size: 30, design: .monospaced))
                    .foregroundColor(t.fg)
                    .padding(.top, 4)
                if !refused.isEmpty { Refusal(message: refused) }
            }

            if !failure.isEmpty { Refusal(message: failure).padding(.top, 8) }

            if let job, !job.moving {
                Group {
                    if job.state == "error" || job.state == "cancelled" {
                        Refusal(message: job.error ?? "cancelled — nothing was kept")
                    } else {
                        Text("Done — \(job.progress.message) is in the catalog.")
                            .sans(11, t.ok).lineSpacing(3)
                    }
                }
                .padding(.top, 8)
            }

            if running, let job {
                Text((job.fraction.map { "\(Int(($0 * 100).rounded()))% · " } ?? "")
                    + job.progress.message)
                    .mono(10, t.fg2).lineLimit(1).truncationMode(.tail)
                    .padding(.top, 8)
            }

            action(running).padding(.top, 11)
        }
        .opacity(pricing ? 0.55 : 1)
        .padding(.top, 8)
    }

    @ViewBuilder
    private func action(_ running: Bool) -> some View {
        let ready = plan != nil && refused.isEmpty && Self.wellNamed(repo) && !running
        if running {
            PushButton(label: "Cancel", kind: "danger", wide: true) {
                guard let job else { return }
                app.act { try await Client.send("DELETE", "/admin/jobs/\(job.id)") }
            }
        } else {
            PushButton(label: "Quantize", kind: "primary", wide: true, action: ready ? launch : nil)
        }
    }

    private func kv(_ label: String, _ value: String) -> some View {
        HStack(spacing: 10) {
            Eyebrow(text: label)
            Spacer(minLength: 0)
            Text(value).mono(11.5, t.fg, weight: .medium).lineLimit(1)
        }
        .padding(.vertical, 6)
        .overlay(alignment: .bottom) { Rectangle().fill(t.hair).frame(height: 1) }
    }

    private func loaded(_ plan: PricedPlan) -> String {
        guard let state = app.store.state else { return "—" }
        let resident = Double(state.residentBytes + state.kvBytes)
        let total = app.store.memoryBytes
        return "\(Fmt.gb(resident + Double(plan.entryBytes))) / \(Int((total / Fmt.gib).rounded())) GB"
    }

    /// The plan leaf by leaf is six hundred rows; the groups the numbered segments make are
    /// twenty, and each is one decision — the width this part of the tree is packed at, and
    /// the group size it is packed against. What the row shows is the daemon's answer for
    /// those leaves, so a width picked here is read back from the next price rather than
    /// drawn from what was clicked.
    @ViewBuilder
    private func leaves(_ plan: PricedPlan?) -> some View {
        let groups = plan.map { groupLeaves($0.leaves) } ?? []
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text(leavesOpen ? "▾" : "▸").sans(10, t.fg3)
                Text("Per leaf group").sans(12.5, t.fg2)
                if let plan {
                    Text("\(groups.count) groups · \(plan.leaves.count) leaves").mono(10, t.fg3)
                }
                if !overrides.isEmpty {
                    Text("\(overrides.count) overridden").mono(10, t.accent)
                }
                Spacer(minLength: 0)
                if !overrides.isEmpty {
                    ActionText(label: "Clear") { overrides = [:] }
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { leavesOpen.toggle() }

            if leavesOpen {
                ForEach(groups) { group in row(group) }
            }
        }
        .padding(.top, 11)
        .opacity(pricing ? 0.55 : 1)
    }

    private func row(_ group: LeafGroup) -> some View {
        let held = overrides[group.pattern]
        // The group size is offered once a width is pinned here: on its own it would pin
        // the width too, and quietly hold this group at a number the screen no longer says.
        let pinned = held?.bits != nil && exponent == nil
        return HStack(spacing: 8) {
            Text(group.label)
                .mono(10, group.widths == nil ? t.fg3 : t.fg2)
                .lineLimit(1).truncationMode(.head)
                .help(group.pattern)
            Spacer(minLength: 4)
            Text("\(group.leaves)×").mono(10, t.fg3)
            Drop(shown: width(group, pinned), set: held != nil, options: widthChoices) { choice in
                pick(group.pattern, width: choice)
            }
            if pinned, let held {
                Drop(
                    shown: "g\(held.groupSize ?? groupSize)",
                    set: held.groupSize != nil,
                    options: groupChoices
                ) { choice in
                    pick(group.pattern, group: choice)
                }
            }
        }
        .padding(.vertical, 5)
        .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
    }

    /// What the daemon priced these leaves at. The group size rides along only where no
    /// second control is carrying it, and only where it departs from the plan's own.
    private func width(_ group: LeafGroup, _ pinned: Bool) -> String {
        guard let span = group.widths else { return "dense" }
        guard group.dense == 0 else { return "mixed" }
        guard span.lowerBound == span.upperBound else {
            return "\(span.lowerBound)–\(span.upperBound)bit"
        }
        let bits = span.lowerBound
        guard !pinned, let size = group.groupSize, size != groupSize || exponent != nil else {
            return "\(bits)bit"
        }
        return "\(bits)bit g\(size)"
    }

    /// The plan's own first, which is what picking it goes back to. Under an exponent mode
    /// the format is the mode, so the only override left is dense — and under the allocator
    /// every width is on offer, because a pinned leaf is a decision it works around.
    private var widthChoices: [Drop.Item] {
        var made = [Drop.Item(key: "", label: "Plan · \(placeholder)")]
        if exponent == nil {
            for value in allocated ? widths : widths.filter({ $0 != bits }) {
                made.append(Drop.Item(
                    key: String(value),
                    label: "\(value)-bit",
                    off: method == "gptq" && !gptqWidths.contains(value)
                ))
            }
        }
        made.append(Drop.Item(key: "dense", label: "dense"))
        return made
    }

    private var groupChoices: [Drop.Item] {
        [Drop.Item(key: "", label: "Plan · g\(groupSize)")]
            + groupSizes.filter { $0 != groupSize }.map { Drop.Item(key: String($0), label: "g\($0)") }
    }

    private var placeholder: String {
        if let exponent { return exponent.label }
        return allocated ? "auto" : "\(bits)-bit"
    }

    private func pick(_ pattern: String, width choice: String) {
        switch choice {
        case "": overrides.removeValue(forKey: pattern)
        case "dense": overrides[pattern] = LeafOverride()
        default:
            guard let value = Int(choice) else { return }
            // The group size the row already carried survives a change of width: the two
            // are picked apart, and re-picking one is not a way of forgetting the other.
            overrides[pattern] = LeafOverride(
                bits: value, groupSize: overrides[pattern]?.groupSize
            )
        }
    }

    private func pick(_ pattern: String, group choice: String) {
        guard let held = overrides[pattern], held.bits != nil else { return }
        overrides[pattern] = LeafOverride(bits: held.bits, groupSize: Int(choice))
    }

    // ── the daemon's answers ─────────────────────────────────────────────

    private struct Price: Equatable {
        let source: String
        let mode: String
        let method: String
        let bits: Int
        let groupSize: Int
        let overrides: [String: LeafOverride]
        let target: String
        let cap: String
        let up: Bool
    }

    private func request() -> QuantizeRequest {
        QuantizeRequest(
            source: source,
            mode: mode,
            method: method,
            bits: exponent == nil ? bits : nil,
            groupSize: exponent == nil ? groupSize : nil,
            targetBpw: allocated ? Double(target) : nil,
            hardCapBpw: allocated ? Double(cap) : nil,
            repo: nil,
            overrides: overrides
        )
    }

    private func price() async {
        guard !source.isEmpty else {
            plan = nil
            refused = ""
            return
        }
        try? await Task.sleep(for: .milliseconds(180))
        guard !Task.isCancelled else { return }
        pricing = true
        defer { pricing = false }
        do {
            plan = try await Client.send(
                "POST", "/admin/quantizations/plan", body: request(), spelled: true
            )
            refused = ""
        } catch {
            plan = nil
            // The daemon's refusal is the message.
            refused = error.localizedDescription
        }
    }

    private func launch() {
        app.act {
            failure = ""
            var body = request()
            body.repo = repo
            do {
                let accepted: Job? = try await Client.send(
                    "POST", "/admin/quantizations", body: body, spelled: true
                )
                job = accepted
                started = accepted?.id ?? ""
            } catch {
                // A refused job is a message, not a crash.
                failure = error.localizedDescription
            }
        }
    }

    private func watch() async {
        guard !started.isEmpty else { return }
        do {
            for try await frame in Client.events("/admin/jobs/\(started)/events") {
                if let held = try? Client.decoder.decode(Job.self, from: frame) { job = held }
            }
        } catch {
            // The stream dying is not news: the jobs resource still carries the progress.
        }
    }

    // ── the rules confessed before the 400 ───────────────────────────────

    private func pick(source: String) {
        self.source = source
        // The patterns belong to the tree they were read off: carried over, they match no
        // leaf and the next price fails for the wrong reason.
        overrides = [:]
        rename(source, mode, bits)
    }

    private func pick(mode key: String) {
        mode = key
        if modes.first(where: { $0.key == key })?.shape != nil {
            // The exponent grid has no bias to search and no width to pick, so the
            // controls it decides go with it — and an override that names a width is one
            // of them. Dense is still a decision the mode leaves open.
            method = "rtn"
            target = ""
            cap = ""
            overrides = overrides.filter { $0.value.bits == nil }
        }
        rename(source, key, bits)
    }

    private func pick(method key: String) {
        method = key
        if key != "oq" && key != "oqe" {
            target = ""
            cap = ""
        }
    }

    private func pick(bits value: Int) {
        bits = value
        rename(source, mode, value)
    }

    private func rename(_ chosen: String, _ chosenMode: String, _ chosenBits: Int) {
        guard !named else { return }
        repo = Self.suggest(chosen, chosenMode, chosenBits)
    }

    /// `local/` because a load by id asks the Hub first, and a name that also exists up
    /// there resolves to somebody else's weights.
    private static func suggest(_ source: String, _ mode: String, _ bits: Int) -> String {
        guard !source.isEmpty else { return "" }
        let tail = source.split(separator: "/").last.map(String.init) ?? source
        return "local/\(tail)-\(mode == "affine" ? "\(bits)bit" : mode)"
    }

    /// Two segments, and nothing that could name a directory.
    private static func wellNamed(_ repo: String) -> Bool {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        return parts.allSatisfy { part in
            !part.isEmpty && part.unicodeScalars.allSatisfy(allowed.contains)
        }
    }
}

/// A field-sized trigger with a real menu behind it, at the leaf row's scale. The accent is
/// the whole of the "this one was decided here" mark: a group left to the plan wears the
/// same face as the number beside it.
private struct Drop: View {
    @Environment(\.tokens) private var t

    struct Item: Identifiable {
        let key: String
        let label: String
        /// A width the current method cannot pack stays on the list — the rule is worth
        /// seeing — but takes no click.
        var off = false
        var id: String { key }
    }

    let shown: String
    var set = false
    let options: [Item]
    let pick: (String) -> Void

    var body: some View {
        Menu {
            ForEach(options) { option in
                Button(option.label) { pick(option.key) }.disabled(option.off)
            }
        } label: {
            HStack(spacing: 5) {
                Text(shown).mono(10, set ? t.accent : t.fg2, weight: set ? .semibold : .regular)
                    .lineLimit(1)
                Text("▾").sans(8, t.fg3)
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(t.field)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(set ? t.accent : t.hair2, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
    }
}

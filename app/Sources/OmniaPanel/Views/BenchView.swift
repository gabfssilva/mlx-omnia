// Bench: the window's five columns, folded into two lines.
//
// Decode, prefill and TTFT do not fit across a panel's width beside a name, so the numbers
// move under it and the verdict — the share of the ceiling — keeps the right edge, which is
// the column that is actually scanned. Tapping a measured row opens the three axes and the
// key of the run each came from. Nothing here recomputes what the run decided.

import SwiftUI

struct BenchView: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    @State private var speed: [String: BenchRun] = [:]
    @State private var quality: [String: BenchRun] = [:]
    @State private var fidelity: [String: BenchRun] = [:]
    @State private var opened = ""

    var body: some View {
        let store = app.store
        let active = store.jobs.filter { ["bench", "benchmark"].contains($0.kind) && $0.moving }.count

        if store.catalog.isEmpty {
            EmptyNote(message: "Nothing on disk to measure — fetch a model under Models.")
        } else {
            ForEach(ordered) { entry in
                BenchRow(
                    app: app,
                    entry: entry,
                    speed: speed[entry.id],
                    quality: quality[entry.id],
                    fidelity: fidelity[entry.id],
                    on: opened == entry.id,
                    pick: { opened = opened == entry.id ? "" : entry.id }
                )
            }
        }

        if !app.refusal.isEmpty {
            Refusal(message: app.refusal).padding(.top, 4)
        }

        // Refetched when the bench queue drains, so a finished run lands without a reopen.
        Color.clear.frame(height: 0).task(id: active) { await load() }
    }

    /// Measured, then moving, then the rest. A disk with 78 checkpoints buries the four
    /// that have numbers if the catalog's own order is kept.
    private var ordered: [CatalogEntry] {
        app.store.catalog.enumerated().sorted { left, right in
            let a = rank(left.element), b = rank(right.element)
            return a == b ? left.offset < right.offset : a < b
        }.map(\.element)
    }

    private func rank(_ entry: CatalogEntry) -> Int {
        if speed[entry.id]?.speed != nil { return 0 }
        if app.store.jobs(kind: ["bench", "benchmark"], model: entry.id) != nil { return 1 }
        return entry.supported ? 2 : 3
    }

    private func load() async {
        // A daemon that is down has no runs to show.
        async let one: [BenchRun]? = try? Client.get("/admin/benchmarks/runs?kind=speed")
        async let two: [BenchRun]? = try? Client.get("/admin/benchmarks/runs?kind=quality")
        async let three: [BenchRun]? = try? Client.get("/admin/benchmarks/runs?kind=fidelity")
        speed = Self.latest(await one)
        quality = Self.latest(await two)
        fidelity = Self.latest(await three)
    }

    private static func latest(_ rows: [BenchRun]?) -> [String: BenchRun] {
        var newest: [String: BenchRun] = [:]
        for run in rows ?? [] where run.state == "ok" {
            if let held = newest[run.model], held.createdAt >= run.createdAt { continue }
            newest[run.model] = run
        }
        return newest
    }
}

private struct BenchRow: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let entry: CatalogEntry
    let speed: BenchRun?
    let quality: BenchRun?
    let fidelity: BenchRun?
    let on: Bool
    let pick: () -> Void

    var body: some View {
        let store = app.store
        let job = store.jobs(kind: ["bench", "benchmark"], model: entry.id)
        let body = speed?.speed
        let resident = store.residentIds.contains(entry.id)

        Pressable(stroke: on ? t.accent : t.hair, action: body == nil ? nil : pick) {
            HStack(spacing: 9) {
                Material(
                    color: resident ? t.mat(store.materials[entry.id] ?? 0) : t.sel,
                    height: 28, width: 3.5
                )
                Text(Fmt.displayName(entry.id)).sans(13.5, t.fg, weight: .semibold)
                    .lineLimit(1).truncationMode(.tail)
                Spacer(minLength: 0)
                tail(job, body)
            }

            if let job {
                TrackBar(fraction: job.fraction ?? 0).padding(.top, 8).padding(.bottom, 5)
                Text(job.progress.message.isEmpty ? "running…" : job.progress.message)
                    .mono(10, t.fg3).lineLimit(1).truncationMode(.tail)
            } else {
                Text(second(body)).mono(10, body == nil ? t.fg3 : t.fg2)
                    .lineLimit(1).truncationMode(.tail)
                    .padding(.top, 6)
            }

            if on, let speed, speed.speed != nil {
                Axes(speed: speed, quality: quality, fidelity: fidelity)
            }
        }
    }

    @ViewBuilder
    private func tail(_ job: Job?, _ body: SpeedBody?) -> some View {
        if let job {
            Text(job.fraction.map { "\(Int(($0 * 100).rounded()))%" } ?? "running")
                .mono(10, t.fg3)
        } else if let body {
            VerdictView(text: body.ceilingFraction.map { "\(Int(($0 * 100).rounded()))% of ceiling" })
        } else if entry.supported {
            PushButton(label: "Run") {
                app.act {
                    try await Client.send(
                        "POST", "/admin/benchmarks", body: BenchRequest(models: [entry.id])
                    )
                }
            }
        } else {
            PillView(label: "Unsupported", kind: "hub")
        }
    }

    private func second(_ body: SpeedBody?) -> String {
        guard let body else {
            return entry.supported ? "not measured" : "no loader for \(entry.architecture)"
        }
        return [
            body.decodeTps.map { String(format: "%.1f decode", $0) } ?? "—",
            body.prefillTps.map { "\(Fmt.grouped($0)) prefill" } ?? "—",
            body.ttftP50Ms.map { String(format: "%.0f ms TTFT", $0) } ?? "—",
        ].joined(separator: " · ")
    }
}

/// The three axes and the key of the run each came from.
private struct Axes: View {
    @Environment(\.tokens) private var t
    let speed: BenchRun?
    let quality: BenchRun?
    let fidelity: BenchRun?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            axis("Speed", speed, first: true) {
                VerdictView(
                    text: speed?.speed?.ceilingFraction.map { "\(Int(($0 * 100).rounded()))% of ceiling" }
                )
            }
            axis("Quality", quality) {
                VerdictView(text: quality?.quality?.accuracy.map { Fmt.percent($0, 1) })
            }
            axis("Fidelity", fidelity) {
                VerdictView(text: fidelityVerdict)
            }
        }
        .padding(.top, 9)
        .overlay(alignment: .top) { Rectangle().fill(t.hair2).frame(height: 1) }
    }

    private var fidelityVerdict: String? {
        guard let body = fidelity?.fidelity else { return nil }
        if let top1 = body.top1 { return "top1 \(Fmt.percent(top1, 1))" }
        if let kl = body.klMean { return String(format: "KL %.2e", kl) }
        return nil
    }

    private func axis<Verdict: View>(
        _ label: String, _ run: BenchRun?, first: Bool = false, @ViewBuilder verdict: () -> Verdict
    ) -> some View {
        HStack(spacing: 8) {
            Text(label).sans(12.5, t.fg, weight: .semibold)
                .frame(width: 56, alignment: .leading)
            Text(run?.key ?? "not measured").mono(10, t.fg2)
                .lineLimit(1).truncationMode(.tail)
            Spacer(minLength: 0)
            verdict()
        }
        .padding(.horizontal, 2)
        .padding(.vertical, 7)
        .overlay(alignment: .top) {
            if !first { Rectangle().fill(t.hair).frame(height: 1) }
        }
    }
}

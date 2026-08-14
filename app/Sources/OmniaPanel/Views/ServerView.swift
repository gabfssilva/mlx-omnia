// Server: the tab the panel opens on, and the reason the panel exists.
//
// Status, the daemon's output, and its settings share the server destination but not one
// scroll: the first two are read often, while configuration is visited rarely.
//
// A daemon that is not answering has no settings to draw: what a PATCH would reach is not
// there, and rows reading "—" would be a form about nothing.

import SwiftUI

struct ServerView: View {
    @Bindable var app: AppModel
    @State private var tab: ServerTab = .settings

    var body: some View {
        VStack(spacing: 0) {
            Segmented(
                options: [(.settings, "Settings"), (.status, "Status")],
                chosen: tab,
                size: 11.5,
                pick: { tab = $0 }
            )
            .padding(.horizontal, Panel.pad)
            .padding(.top, 12)

            switch tab {
            case .settings:
                ScrollView {
                    VStack(alignment: .leading, spacing: 7) {
                        if app.store.down {
                            ServerUnavailable(port: app.store.port)
                        } else {
                            DaemonSettings(app: app)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, Panel.pad)
                    .padding(.top, 14)
                    .padding(.bottom, 12)
                }
                .scrollIndicators(.never)
            case .status:
                ServerStatus(app: app)
            }
        }
    }
}

private struct ServerStatus: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel

    var body: some View {
        let store = app.store
        let moving = store.jobs.filter(\.moving)
        let recent = store.metrics?.requests.prefix(4) ?? []

        GeometryReader { area in
            ScrollView {
                VStack(alignment: .leading, spacing: 7) {
                    if store.down {
                        Card {
                            Text(app.starting
                                ? "The engine is not answering on :\(store.port) — starting it…"
                                : "The engine is not answering on :\(store.port).")
                                .sans(12.5, t.fg3)
                                .multilineTextAlignment(.center)
                                .frame(maxWidth: .infinity)
                                .padding(.horizontal, 4)
                                .padding(.top, 12)
                                .padding(.bottom, 4)
                            if app.starting {
                                Waiting()
                                    .padding(.horizontal, 40)
                                    .padding(.top, 14)
                                    .padding(.bottom, 6)
                            } else {
                                HStack {
                                    Spacer(minLength: 0)
                                    PushButton(label: "Start engine", kind: "primary") {
                                        app.startEngine()
                                    }
                                    Spacer(minLength: 0)
                                }
                                .padding(.top, 12)
                                .padding(.bottom, 4)
                            }
                        }
                    } else if !store.models.isEmpty {
                        SectionHead(title: "Resident · \(store.models.count)")
                        ForEach(store.models) { slot in
                            Resident(app: app, slot: slot)
                        }
                    }

                    if !moving.isEmpty {
                        SectionHead(title: "Running")
                        ForEach(moving) { job in
                            JobCard(app: app, job: job)
                        }
                    }

                    if !recent.isEmpty {
                        SectionHead(title: "Recent")
                        ForEach(Array(recent.enumerated()), id: \.offset) { _, sample in
                            HStack(spacing: 10) {
                                Text(Fmt.clock(sample.startedAt)).mono(10, t.fg3)
                                    .frame(width: 38, alignment: .leading)
                                Text("\(Fmt.displayName(sample.model)) · \(sample.completionTokens) tokens")
                                    .sans(12.5, t.fg2)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                Spacer(minLength: 0)
                                Text(sample.tokensPerSecond.map {
                                    String(format: "%.1f tok/s", $0)
                                } ?? "—").mono(10, t.fg3)
                            }
                            .padding(.horizontal, 2)
                            .padding(.vertical, 6)
                            .overlay(alignment: .top) {
                                Rectangle().fill(t.hair).frame(height: 1)
                            }
                        }
                    }

                    if !app.refusal.isEmpty {
                        Refusal(message: app.refusal).padding(.top, 4)
                    }

                    SectionHead(title: "Log")
                    LogsView(app: app)
                        .frame(minHeight: 220, maxHeight: .infinity)
                        .layoutPriority(1)
                }
                .frame(
                    maxWidth: .infinity,
                    minHeight: max(0, area.size.height - 26),
                    alignment: .topLeading
                )
                .padding(.horizontal, Panel.pad)
                .padding(.top, 14)
                .padding(.bottom, 12)
            }
            .scrollIndicators(.never)
        }
    }
}

private struct ServerUnavailable: View {
    @Environment(\.tokens) private var t
    let port: String

    var body: some View {
        Card {
            Text("The engine is not answering on :\(port).")
                .sans(12.5, t.fg3)
                .frame(maxWidth: .infinity)
                .multilineTextAlignment(.center)
                .padding(.vertical, 12)
        }
    }
}

private struct Resident: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let slot: Slot

    var body: some View {
        let store = app.store
        let material = t.mat(store.materials[slot.id] ?? 0)
        let sample = store.sample(slot.id)

        Pressable(action: {
            app.tab = .models
            app.modelsTab = .library
            app.opened = .model(slot.id)
        }) {
            HStack(spacing: 9) {
                Material(color: material, height: 30)
                VStack(alignment: .leading, spacing: 2) {
                    Text(Fmt.displayName(slot.id)).sans(13.5, t.fg, weight: .semibold)
                        .lineLimit(1).truncationMode(.tail)
                    Text("\(Fmt.gb(Double(slot.weightsBytes))) GB + \(Fmt.gb(Double(slot.kvBytes))) KV")
                        .mono(10, t.fg3).lineLimit(1)
                }
                Spacer(minLength: 0)
                Sparkline(values: store.points(slot.id))
                Text("⏏").sans(12, t.fg3)
                    .padding(.leading, 6)
                    .padding(.trailing, 2)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        app.act {
                            try await Client.send("DELETE", "/admin/models/\(Client.at(slot.id))/residency")
                        }
                    }
            }

            if let measure = measures(sample) {
                HStack(spacing: 8) {
                    if let decode = store.decode(slot.id) { ChipView(text: decode) }
                    Text(measure).mono(10, t.fg3).lineLimit(1)
                }
                .padding(.top, 7)
            }
        }
    }

    /// The artifact's second line: the share of the ceiling this request reached, then the
    /// prefill and the time to the first token.
    private func measures(_ sample: Sample?) -> String? {
        guard let sample else { return nil }
        var said: [String] = []
        if let share = sample.ceilingFraction { said.append("\(Int((share * 100).rounded()))% ceil") }
        if let prefill = sample.prefillTokensPerSecond { said.append("\(Fmt.grouped(prefill)) prefill") }
        if let ttft = sample.ttft { said.append(String(format: "%.0f ms", ttft * 1000)) }
        return said.isEmpty ? nil : said.joined(separator: " · ")
    }
}

private struct JobCard: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let job: Job

    var body: some View {
        Card {
            HStack(spacing: 8) {
                Text(name).sans(12.5, t.fg2).lineLimit(1).truncationMode(.tail)
                Spacer(minLength: 0)
                Text("✕").sans(10.5, t.fg3)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        app.act { try await Client.send("DELETE", "/admin/jobs/\(job.id)") }
                    }
            }
            TrackBar(fraction: job.fraction ?? 0)
                .padding(.top, 6)
                .padding(.bottom, 5)
            Text(line).mono(10, t.fg3).lineLimit(1).truncationMode(.tail)
        }
    }

    private var name: String {
        let held = job.model.map(Fmt.displayName) ?? "?"
        guard job.kind == "quantize", let target = job.target else { return held }
        let landing = Fmt.displayName(target)
        let trimmed = landing.hasPrefix("\(held)-")
            ? String(landing.dropFirst(held.count + 1)) : landing
        return "\(held) → \(trimmed.isEmpty ? landing : trimmed)"
    }

    private var line: String {
        let message = job.progress.message.isEmpty ? "running…" : job.progress.message
        guard let fraction = job.fraction else { return message }
        return "\(Int((fraction * 100).rounded()))% · \(message)"
    }
}

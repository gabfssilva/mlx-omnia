// Every resource the panel draws, as of the last frame.
//
// The daemon's own contract and nothing beside it — `/admin/system` once, then one
// `/admin/events` stream carrying whole resources — so this app is a client of the daemon
// and not a second opinion about it.

import Foundation
import Observation

private struct Peek: Decodable {
    let resource: String
}

private struct Envelope<T: Decodable>: Decodable {
    let value: T
}

@MainActor
@Observable
final class EngineStore {
    var system: SystemInfo?
    var state: EngineState?
    var health: Health?
    var config: [String: Setting] = [:]
    var catalog: [CatalogEntry] = []
    var jobs: [Job] = []
    var metrics: Snapshot?

    /// The decode trace the sparkline draws. The daemon keeps no history to ask for, so
    /// the panel records one for as long as it runs — the v1 reason, unchanged.
    static let traceLength = 56
    var trace: [String: [Double]] = [:]

    /// Advances once a second so text that only ages — and the pulse — is redrawn.
    var tick = 0

    var down: Bool { health == nil }

    var port: String { system?.constants["port"] ?? Client.port }

    /// Resident, biggest first — the order the band and the roster are drawn in.
    var models: [Slot] {
        (state?.models ?? []).sorted { ($0.weightsBytes + $0.kvBytes) > ($1.weightsBytes + $1.kvBytes) }
    }

    /// One material per resident model, by load order: colour is identity, so a block does
    /// not change colour because another one left.
    var materials: [String: Int] {
        let byAge = (state?.models ?? []).sorted { $0.loadedAt < $1.loadedAt }
        return Dictionary(uniqueKeysWithValues: byAge.enumerated().map { ($1.id, $0) })
    }

    var residentIds: Set<String> { Set((state?.models ?? []).map(\.id)) }

    var ceiling: Double? { config["memory_limit_bytes"]?.value.number }

    /// What the ceiling still has room for, or nil while the daemon has not said what the
    /// ceiling is.
    var headroom: Double? {
        ceiling.map { $0 - Double(state?.residentBytes ?? 0) }
    }

    /// How much more than the headroom this checkpoint would take — nil when it fits, which
    /// includes a daemon that has not answered yet: whether a load is admitted is the
    /// daemon's arithmetic, and a panel that refused on a number it does not have would be
    /// refusing on its own guess.
    func shortBy(_ entry: CatalogEntry) -> Double? {
        guard let headroom, Double(entry.bytesOnDisk) > headroom else { return nil }
        return Double(entry.bytesOnDisk) - headroom
    }

    var memoryBytes: Double { Double(system?.memoryBytes ?? 137_438_953_472) }

    func entry(_ model: String) -> CatalogEntry? { catalog.first { $0.id == model } }

    func aggregate(_ model: String) -> Aggregate? { metrics?.models.first { $0.model == model } }

    func live(_ model: String) -> Sample? {
        guard let sample = metrics?.live, sample.model == model else { return nil }
        return sample
    }

    /// The newest numbers this model has: the live register while it generates, else the
    /// most recent completed request. The wire carries requests newest first.
    func sample(_ model: String) -> Sample? {
        if let live = live(model) { return live }
        return metrics?.requests.first { $0.model == model && $0.state == "completed" }
    }

    func decode(_ model: String) -> String? {
        let speed = sample(model)?.tokensPerSecond ?? aggregate(model)?.tokensPerSecond
        return speed.map { String(format: "%.1f tok/s", $0) }
    }

    func prefillTtft(_ model: String) -> String? {
        guard let sample = sample(model) else { return nil }
        let prefill = sample.prefillTokensPerSecond
        let ttft = sample.ttft
        if prefill == nil && ttft == nil { return nil }
        let left = prefill.map { Fmt.grouped($0) } ?? "—"
        let right = ttft.map { String(format: "%.0f ms", $0 * 1000) } ?? "—"
        return "\(left) prefill · \(right)"
    }

    func points(_ model: String) -> [Double] { trace[model] ?? [] }

    func jobs(kind: [String], model: String) -> Job? {
        jobs.first { job in
            kind.contains(job.kind) && job.moving
                && (job.subject["models"] ?? job.subject["model"])?.names.contains(model) == true
        }
    }

    // ── filling it ───────────────────────────────────────────────────────

    /// A daemon that is down is a daemon somebody is restarting, and a client reconnecting
    /// on a tight loop is the load the restart does not need.
    private static let redial: Duration = .seconds(1)

    func follow() async {
        while !Task.isCancelled {
            do {
                if system == nil {
                    system = try await Client.get("/admin/system")
                }
                for try await frame in Client.events("/admin/events") {
                    apply(frame)
                }
            } catch {
                // The daemon went away, or has not come up yet. Which one it was is the
                // next dial's to find out — and what it was is on stderr, which is where
                // a panel run from the terminal is debugged.
                FileHandle.standardError.write(Data("follow: \(error)\n".utf8))
            }
            health = nil
            try? await Task.sleep(for: Self.redial)
        }
    }

    private func apply(_ frame: Data) {
        guard let named = try? Client.decoder.decode(Peek.self, from: frame) else { return }
        switch named.resource {
        case "health":
            health = try? Client.decoder.decode(Envelope<Health>.self, from: frame).value
        case "config":
            if let held = try? Client.decoder.decode(Envelope<[String: Setting]>.self, from: frame) {
                config = held.value
            }
        case "models":
            if let held = try? Client.decoder.decode(Envelope<[CatalogEntry]>.self, from: frame) {
                catalog = held.value
            }
        case "state":
            state = try? Client.decoder.decode(Envelope<EngineState>.self, from: frame).value
        case "jobs":
            if let held = try? Client.decoder.decode(Envelope<[Job]>.self, from: frame) {
                jobs = held.value
            }
        case "metrics":
            metrics = try? Client.decoder.decode(Envelope<Snapshot>.self, from: frame).value
        default:
            // A daemon newer than this panel. Ignoring the frame is what lets one add a
            // resource without breaking the other.
            break
        }
    }

    /// Sample every model's share of the ceiling, once a second, for as long as the panel
    /// runs — and advance the beat the ageing text is redrawn by.
    func pulse() async {
        while !Task.isCancelled {
            if let snapshot = metrics {
                for aggregate in snapshot.models {
                    let point = (live(aggregate.model)?.ceilingFraction ?? 0) * 100
                    var held = trace[aggregate.model] ?? []
                    held.append(point)
                    if held.count > Self.traceLength { held.removeFirst(held.count - Self.traceLength) }
                    trace[aggregate.model] = held
                }
            }
            tick &+= 1
            try? await Task.sleep(for: .seconds(1))
        }
    }
}

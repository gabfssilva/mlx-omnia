// Every download the daemon is running, keyed by the repository it is fetching.
//
// It lives above the views because a download outlives the screen that started it. The
// boot pass is what makes a restart work — the active jobs say which downloads are
// running, and each one's `subject.model` says which row it belongs under.

import Foundation
import Observation

struct Pull: Identifiable {
    let job: String
    let repo: String
    let state: String
    let message: String
    let completed: Int
    /// Absent until the first response: a bar drawn from a guessed total is worse than none.
    let total: Int?
    let error: String?
    /// Bytes per second, null until a second frame gives it something to divide.
    let rate: Double?
    let at: Double

    var id: String { repo }

    var fraction: Double? {
        guard let total, total > 0 else { return nil }
        return Double(completed) / Double(total)
    }

    var share: String {
        fraction.map { "\(Int(($0 * 100).rounded()))%" } ?? "—"
    }

    var eta: Double? {
        guard let total, let rate, rate > 0 else { return nil }
        return Double(total - completed) / rate
    }
}

@MainActor
@Observable
final class Downloads {
    /// Smoothed, because a shard already in the staging cache completes in a single frame
    /// and the instantaneous reading there is gigabytes per second.
    private static let smoothing = 0.35

    var active: [String: Pull] = [:]
    private var followed: Set<String> = []

    /// Watch one download's stream until it ends. Idempotent per job, so the boot pass and
    /// a fresh `pull` naming the same job start one watcher between them.
    func follow(_ job: Job) {
        guard job.kind == "download", !followed.contains(job.id),
              case let .one(repo)? = job.subject["model"]
        else { return }
        followed.insert(job.id)
        fold(repo, job)
        Task { [weak self] in
            do {
                for try await frame in Client.events("/admin/jobs/\(job.id)/events") {
                    guard let held = try? Client.decoder.decode(Job.self, from: frame) else { continue }
                    self?.fold(repo, held)
                }
            } catch {
                // The stream dying is not itself news: a daemon that went away is already
                // drawn by the head's dot.
            }
            self?.followed.remove(job.id)
        }
    }

    private func fold(_ repo: String, _ frame: Job) {
        // A finished download is the catalog's now, and a cancelled one took its bytes
        // with it; only a failure has something left to offer, so only a failure stays.
        if frame.state == "ok" || frame.state == "cancelled" {
            active[repo] = nil
            return
        }
        let previous = active[repo]
        active[repo] = Pull(
            job: frame.id,
            repo: repo,
            state: frame.state,
            message: frame.progress.message,
            completed: frame.progress.completed,
            total: frame.progress.total,
            error: frame.error,
            rate: Self.rate(previous, frame),
            at: frame.updatedAt
        )
    }

    /// A terminal failure stays on screen until it is acted on; this is the acting.
    func forget(_ repo: String) { active[repo] = nil }

    /// What was already running before this panel opened.
    func boot() async {
        guard let running: [Job] = try? await Client.get("/admin/jobs?active=true") else { return }
        for job in running { follow(job) }
    }

    private static func rate(_ previous: Pull?, _ frame: Job) -> Double? {
        guard let previous else { return nil }
        let seconds = frame.updatedAt - previous.at
        guard seconds > 0 else { return previous.rate }
        let instant = Double(frame.progress.completed - previous.completed) / seconds
        guard let held = previous.rate else { return instant }
        return held + smoothing * (instant - held)
    }
}

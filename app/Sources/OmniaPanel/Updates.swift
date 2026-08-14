// The updater, and the one place that decides whether this build has one.
//
// Sparkle reads its whole configuration off Info.plist — the feed, the schedule, and the
// public key an update has to be signed with. Nothing here names any of them; what this
// decides is only whether to start it at all.
//
// No key, no updater. `SUPublicEDKey` is written by `mise run sparkle:keys` and is absent
// from a fresh checkout on purpose: Sparkle without it refuses every update rather than
// taking an unsigned one on trust, and a "Check for updates" button that can only ever
// report a refusal is worse than no button. The same guard covers `swift run`, where there
// is no bundle to update in the first place.

import Foundation
import Sparkle

@MainActor
final class Updates {
    static let shared = Updates()

    private let controller: SPUStandardUpdaterController?

    /// What the panel says it is: the tag, with the commit count behind it. `build.sh`
    /// stamps both, so a build from a checkout with no tag still has a number that moves.
    static var version: String {
        let info = Bundle.main.infoDictionary ?? [:]
        let short = info["CFBundleShortVersionString"] as? String ?? "0.0.0"
        let build = info["CFBundleVersion"] as? String ?? "0"
        return "\(short) (\(build))"
    }

    private init() {
        let info = Bundle.main.infoDictionary ?? [:]
        guard info["SUFeedURL"] is String, info["SUPublicEDKey"] is String else {
            controller = nil
            return
        }
        // `startingUpdater: true` is what puts the scheduled check on the clock, so this
        // has to be reached at launch and not the first time Settings is drawn.
        controller = SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil
        )
    }

    /// Whether this build can be updated at all. The Settings row draws what it says.
    var available: Bool { controller != nil }

    /// The check a reader asked for, with Sparkle's own window in front of it — including
    /// "you're up to date", which the scheduled check stays quiet about.
    func check() {
        controller?.checkForUpdates(nil)
    }
}

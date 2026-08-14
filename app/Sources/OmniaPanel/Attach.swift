// Where a picture comes from on a machine whose screen is the thing being asked about.
//
// Three doors, one landing place: PNG bytes and the pixel size the daemon prices. The
// region grab is first because it is the one a menu bar app is uniquely placed to offer —
// the panel is already over whatever the question is about.

import AppKit
import CoreGraphics
import Foundation

struct Clip: Identifiable, Equatable {
    let id = UUID()
    /// What the source was, in the reader's words: it is drawn on the chip.
    let origin: String
    let png: Data
    let height: Int
    let width: Int
    /// What the daemon says this costs the model in force. Nil until it has answered, and
    /// nil again whenever the model changes — a count from another checkpoint's arithmetic
    /// is worse than no count.
    var cost: ImageCost?

    var uri: String { "data:image/png;base64,\(png.base64EncodedString())" }

    var thumbnail: NSImage? { NSImage(data: png) }

    static func == (lhs: Clip, rhs: Clip) -> Bool { lhs.id == rhs.id && lhs.cost?.tokens == rhs.cost?.tokens }
}

enum Attach {
    /// `screencapture -i`, which is the same crosshair ⌘⇧4 draws, writing a PNG this reads
    /// back and unlinks. `-x` silences the shutter: the panel is a window over somebody
    /// else's work and does not need to announce itself.
    ///
    /// Nil is the escape key, and it is not an error — the file simply never appears.
    static func region() async throws -> Clip? {
        // Without the permission `screencapture` writes nothing and says nothing — it
        // exits zero and leaves no file, which reads exactly like the escape key. Asking
        // first is what turns a grab that quietly does nothing into a sentence.
        guard CGPreflightScreenCaptureAccess() else {
            _ = CGRequestScreenCaptureAccess()
            throw Refused(detail: """
                Grabbing a region needs Screen Recording. Grant it to MLX Omnia under \
                System Settings → Privacy & Security → Screen Recording, then try again.
                """)
        }
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnia-grab-\(UUID().uuidString).png")
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        task.arguments = ["-i", "-x", path.path]
        // The handler is set before the process is started, not after: a crosshair
        // dismissed as fast as a synthetic one ends the process before the assignment
        // lands, and the continuation is then never resumed — the grab hangs, silently,
        // and only sometimes.
        try await withCheckedThrowingContinuation { (resume: CheckedContinuation<Void, Error>) in
            task.terminationHandler = { _ in resume.resume() }
            do { try task.run() } catch {
                task.terminationHandler = nil
                resume.resume(throwing: error)
            }
        }
        defer { try? FileManager.default.removeItem(at: path) }
        guard let data = try? Data(contentsOf: path) else { return nil }
        return clip("Region grab", data)
    }

    static func file() -> Clip? {
        let picker = NSOpenPanel()
        picker.allowedContentTypes = [.png, .jpeg, .heic, .tiff, .gif, .bmp, .webP]
        picker.allowsMultipleSelection = false
        picker.message = "An image for this turn"
        // The panel is an accessory app: without this the open panel opens behind whatever
        // is in front, and the click that summoned it looks like it did nothing.
        NSApp.activate(ignoringOtherApps: true)
        guard picker.runModal() == .OK, let url = picker.url,
              let data = try? Data(contentsOf: url)
        else { return nil }
        return clip(url.lastPathComponent, data)
    }

    static func pasteboard() -> Clip? {
        let board = NSPasteboard.general
        // The PNG first: what a screenshot on the clipboard already is, and reading it back
        // avoids a re-encode. `NSImage` is the fallback for a copy out of an app that put a
        // TIFF or a file promise there instead.
        if let data = board.data(forType: .png) { return clip("Pasted image", data) }
        guard let image = NSImage(pasteboard: board) else { return nil }
        return encoded("Pasted image", image)
    }

    /// The bytes as they arrived, with the pixel size read off them. The size is the
    /// **pixel** one and not the point one: a retina grab is twice its own frame, and the
    /// tower reads pixels.
    private static func clip(_ origin: String, _ data: Data) -> Clip? {
        guard let source = NSBitmapImageRep(data: data) else { return nil }
        let height = source.pixelsHigh
        let width = source.pixelsWide
        guard height > 0, width > 0 else { return nil }
        // PNG through: only what is not already one is re-encoded, so a grab is sent as
        // the file `screencapture` wrote.
        if data.starts(with: [0x89, 0x50, 0x4E, 0x47]) {
            return Clip(origin: origin, png: data, height: height, width: width)
        }
        guard let png = source.representation(using: .png, properties: [:]) else { return nil }
        return Clip(origin: origin, png: png, height: height, width: width)
    }

    private static func encoded(_ origin: String, _ image: NSImage) -> Clip? {
        guard let tiff = image.tiffRepresentation else { return nil }
        return clip(origin, tiff)
    }
}

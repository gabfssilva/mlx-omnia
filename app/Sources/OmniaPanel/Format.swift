// Numbers and names as they are read.
// Storage and memory are GiB under the label GB; a download is decimal GB, because that
// is what the Hub quotes and what the two sides of a progress bar divide out against.

import Foundation

enum Fmt {
    static let gib = 1024.0 * 1024 * 1024

    static func gb(_ bytes: Double, _ digits: Int = 1) -> String {
        String(format: "%.\(digits)f", bytes / gib)
    }

    static func tokens(_ count: Int) -> String {
        count >= 1000 ? "\(Int((Double(count) / 1000).rounded()))k" : "\(count)"
    }

    /// A quantization suffix in a repository name is not part of the model's name.
    private static let packed = try? NSRegularExpression(
        pattern: "-(?:\\d+bit|mxfp4|nf4|bf16|fp16|fp32|int[48])$", options: [.caseInsensitive]
    )

    static func displayName(_ model: String) -> String {
        let last = model.split(separator: "/").last.map(String.init) ?? model
        guard let packed else { return last }
        return packed.stringByReplacingMatches(
            in: last, range: NSRange(last.startIndex..., in: last), withTemplate: ""
        )
    }

    private static func scale(_ amount: Double) -> (Double, String) {
        amount >= 1e9 ? (1e9, "GB") : (1e6, "MB")
    }

    static func size(_ amount: Double) -> String {
        let (div, unit) = scale(amount)
        return String(format: "%.2f %@", amount / div, unit)
    }

    /// Both sides in the total's unit: "2.31 / 4.48 GB".
    static func sizePair(_ done: Double, _ total: Double) -> String {
        let (div, unit) = scale(total)
        return String(format: "%.2f / %.2f %@", done / div, total / div, unit)
    }

    static func rate(_ bytesPerSecond: Double) -> String {
        "\(Int((bytesPerSecond / 1e6).rounded())) MB/s"
    }

    /// Rounded to the unit above the one it is about to leave: a countdown that reads
    /// "119s" is a number nobody subtracts from.
    static func left(_ seconds: Double) -> String {
        if seconds < 90 { return "\(Int(seconds.rounded()))s left" }
        let minutes = Int((seconds / 60).rounded())
        return minutes < 90 ? "\(minutes)m left" : "\(Int((Double(minutes) / 60).rounded()))h left"
    }

    /// A checkpoint's own file sizes, which run from a 400-byte config to a 30 GB shard —
    /// `size` starts at MB and prints "0.00 MB" for most of a repository.
    static func bytes(_ amount: Double) -> String {
        if amount >= 1e9 { return String(format: "%.2f GB", amount / 1e9) }
        if amount >= 1e6 { return String(format: "%.2f MB", amount / 1e6) }
        if amount >= 1e3 { return String(format: "%.1f kB", amount / 1e3) }
        return "\(Int(amount)) B"
    }

    /// `4,180` and never `4.180`: the panel is written in English and the daemon's numbers
    /// are read beside its own, so the separator is the format's and not the machine's.
    /// `en_US` and not `en_US_POSIX` — POSIX has no grouping separator at all.
    static func grouped(_ value: Double) -> String {
        let shaper = NumberFormatter()
        shaper.locale = Locale(identifier: "en_US")
        shaper.numberStyle = .decimal
        shaper.maximumFractionDigits = 0
        return shaper.string(from: NSNumber(value: value)) ?? String(Int(value))
    }

    /// Milliseconds under a second and seconds above it: a cold 30B's load and a warm one's
    /// time to the first token are two orders of magnitude apart and share one line.
    static func duration(_ seconds: Double) -> String {
        if seconds < 1 { return "\(Int((seconds * 1000).rounded())) ms" }
        return String(format: seconds < 10 ? "%.2f s" : "%.1f s", seconds)
    }

    static func clock(_ epoch: Double) -> String {
        let shaper = DateFormatter()
        shaper.dateFormat = "HH:mm"
        return shaper.string(from: Date(timeIntervalSince1970: epoch))
    }

    /// When a conversation was last touched. The clock while the day it happened on is still
    /// nameable — the list says Today and Yesterday over it — and the date once it is not.
    static func stamp(_ epoch: Double) -> String {
        let moment = Date(timeIntervalSince1970: epoch)
        let calendar = Calendar.current
        if calendar.isDateInToday(moment) || calendar.isDateInYesterday(moment) {
            return clock(epoch)
        }
        let shaper = DateFormatter()
        shaper.locale = Locale(identifier: "en_US")
        shaper.dateFormat = "MMM d"
        return shaper.string(from: moment)
    }

    static func percent(_ fraction: Double, _ digits: Int = 0) -> String {
        String(format: "%.\(digits)f%%", fraction * 100)
    }
}

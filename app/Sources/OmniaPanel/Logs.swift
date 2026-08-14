import Foundation

enum LogTail {
    static func openForWriting(_ file: URL) throws -> FileHandle {
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: 0)
        return handle
    }

    static func read(_ file: URL, maxBytes: UInt64) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }

        let size = try handle.seekToEnd()
        let start = size > maxBytes ? size - maxBytes : 0
        try handle.seek(toOffset: start)
        var data = try handle.readToEnd() ?? Data()

        if start > 0 {
            guard let newline = data.firstIndex(of: 0x0A) else { return "" }
            data.removeSubrange(data.startIndex...newline)
        }
        return String(decoding: data, as: UTF8.self)
    }
}

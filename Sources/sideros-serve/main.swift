import Foundation
import SiderosServer

/// A hub repo id resolves against the local cache; anything that exists on disk is taken
/// as the checkpoint directory itself.
func resolve(_ model: String) throws -> URL {
    let path = URL(filePath: model)
    if FileManager.default.fileExists(atPath: path.appending(path: "config.json").path) {
        return path
    }
    let cache =
        ProcessInfo.processInfo.environment["HF_HOME"].map { URL(filePath: $0) }
        ?? URL(filePath: NSHomeDirectory()).appending(path: ".cache/huggingface")
    let snapshots = cache.appending(
        path: "hub/models--\(model.replacingOccurrences(of: "/", with: "--"))/snapshots")
    guard
        let snapshot = try FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        ).first
    else {
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: snapshots.path])
    }
    return snapshot
}

var model: String?
var name: String?
var host = "127.0.0.1"
var port = 8080
var maxTokens = 2048

var iterator = CommandLine.arguments.dropFirst().makeIterator()
while let argument = iterator.next() {
    switch argument {
    case "--name": name = iterator.next()
    case "--host": host = iterator.next() ?? host
    case "--port": port = iterator.next().flatMap { Int($0) } ?? port
    case "--max-tokens": maxTokens = iterator.next().flatMap { Int($0) } ?? maxTokens
    default: model = argument
    }
}

guard let model else {
    print(
        "usage: sideros-serve <hub-repo|directory> [--name id] [--host h] [--port n] [--max-tokens n]"
    )
    exit(1)
}

let engine = try Engine(directory: try resolve(model), name: name ?? model, maxTokens: maxTokens)
try await Server(engine: engine).run(host: host, port: port)

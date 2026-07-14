import Foundation
import MLX

/// Loads a checkpoint that may be sharded: with a `model.safetensors.index.json` present,
/// every shard the weight map names is read and merged; without one, the single-file
/// `model.safetensors` is read directly. Names never collide across shards — the index
/// maps each tensor to exactly one file.
func loadCheckpoint(directory: URL) throws -> [String: MLXArray] {
    let index = directory.appending(path: "model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: index.path) else {
        return try loadArrays(url: directory.appending(path: "model.safetensors"))
    }

    struct Index: Decodable { let weightMap: [String: String] }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let shards = Set(try decoder.decode(Index.self, from: Data(contentsOf: index)).weightMap.values)

    var parameters: [String: MLXArray] = [:]
    for shard in shards.sorted() {
        try parameters.merge(loadArrays(url: directory.appending(path: shard))) { first, _ in first }
    }
    return parameters
}

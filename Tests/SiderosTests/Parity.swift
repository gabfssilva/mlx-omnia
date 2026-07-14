import Foundation
import MLX

/// Max absolute deviation divided by the reference tensor's scale.
///
/// GPT-2's residual stream carries outlier dimensions in the thousands, so an absolute
/// threshold that is loose at block 0 is meaningless at block 5.
func relativeDiff(_ ours: MLXArray, _ reference: MLXArray) -> Float {
    abs(ours - reference).max().item(Float.self) / abs(reference).max().item(Float.self)
}

func fixture(_ name: String) throws -> [String: MLXArray] {
    try loadArrays(url: Bundle.module.resourceURL!.appending(path: "Fixtures/\(name)"))
}

/// Resolved from the Hugging Face cache; populate it with `make weights`.
func hubSnapshot(_ repo: String) throws -> URL {
    let cache =
        ProcessInfo.processInfo.environment["HF_HOME"].map { URL(filePath: $0) }
        ?? URL(filePath: NSHomeDirectory()).appending(path: ".cache/huggingface")
    let snapshots = cache.appending(path: "hub/models--\(repo)/snapshots")
    let entries = try FileManager.default.contentsOfDirectory(
        at: snapshots, includingPropertiesForKeys: nil)
    guard let first = entries.first else {
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: snapshots.path])
    }
    return first
}

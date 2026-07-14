import MLX

/// Keys and values already projected and rotated, for one layer. A value: appending
/// returns the grown cache rather than writing into this one. A short-conv layer
/// (LFM2) stores its sliding window here instead — same type, so a hybrid model's
/// cache stays one homogeneous array.
public struct KVCache {
    /// How many keys were cached before the current forward. Rotary offset and mask
    /// geometry both read it.
    public let offset: Int

    private let keys: MLXArray?
    private let values: MLXArray?
    /// The last kernel−1 conv inputs, exclusive with keys/values.
    public let window: MLXArray?
    /// A linear-attention layer's recurrent state (Qwen3.5's gated DeltaNet), carried
    /// alongside the conv window and as impossible to rewind.
    public let state: MLXArray?

    public init() {
        offset = 0
        keys = nil
        values = nil
        window = nil
        state = nil
    }

    private init(
        offset: Int, keys: MLXArray?, values: MLXArray?, window: MLXArray?,
        state: MLXArray? = nil
    ) {
        self.offset = offset
        self.keys = keys
        self.values = values
        self.window = window
        self.state = state
    }

    /// The cache cut back to its first `length` positions: how speculative decoding
    /// discards the keys of rejected draft tokens. The buffer is kept and only the offset
    /// moves — the rows past it are stale until the next append writes over them, and
    /// nothing reads past the offset. A conv window cannot rewind — the inputs it would
    /// rewind to were not kept.
    func trimmed(to length: Int) -> KVCache {
        precondition(window == nil && state == nil, "a conv or recurrent cache cannot rewind")
        guard length < offset, let keys, let values else { return self }
        return KVCache(offset: length, keys: keys, values: values, window: nil)
    }

    /// The window slid past `length` new positions. `newWindow` is the tail of the
    /// padded conv input, already cut to kernel−1 rows by the layer; `state` is the
    /// recurrent state past the same positions, when the layer keeps one.
    func slid(window newWindow: MLXArray, state newState: MLXArray? = nil, by length: Int)
        -> KVCache
    {
        KVCache(
            offset: offset + length, keys: nil, values: nil, window: newWindow,
            state: newState)
    }

    /// Rows are written into a buffer that grows in blocks, not concatenated. Concatenating
    /// copies `offset` rows on every step, so its cost is O(context) per token; the write
    /// copies one row.
    ///
    /// The trade is real and reverses with context, measured on 30B-A3B decode against the
    /// concatenating version (interleaved A/B, order flipped):
    ///
    ///     ~430 tokens   152.6 vs 150.0 tok/s   concat wins by 3%
    ///     ~1760 tokens  109   vs 117           write wins by 7%
    ///     ~4000 tokens   77   vs  96           write wins by 25%
    ///
    /// Below ~1k tokens the copy is still cheap and the buffer's fixed costs (the zeroed
    /// allocation, the block-boundary copy) do not pay for themselves. An earlier attempt
    /// measured only the short prompt and concluded preallocation was slower — true there,
    /// and false everywhere this model is actually used.
    ///
    /// `MLXArray` is a class and its subscript setter swaps the array in place, so the
    /// buffer is one object however many `KVCache` values name it. mlx donates it on the
    /// write when nothing else holds it — that donation is what makes the write O(1), and
    /// it is why the returned views must die with the step that uses them. They do:
    /// attention consumes them and drops them.
    func appending(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (keys: MLXArray, values: MLXArray, cache: KVCache)
    {
        let length = newKeys.dim(2)
        let needed = offset + length

        let keyBuffer = reserving(keys, needed, like: newKeys)
        let valueBuffer = reserving(values, needed, like: newValues)
        keyBuffer[0..., 0..., offset..<needed, 0...] = newKeys
        valueBuffer[0..., 0..., offset..<needed, 0...] = newValues

        return (
            keyBuffer[0..., 0..., ..<needed, 0...],
            valueBuffer[0..., 0..., ..<needed, 0...],
            KVCache(offset: needed, keys: keyBuffer, values: valueBuffer, window: nil)
        )
    }
}

/// The block the cache grows by. A decode step needs one row; growing by one would copy
/// the cache every step, which is the concatenation this replaces.
private let block = 256

/// A buffer with room for `needed` rows, keeping what `buffer` already holds. Growth
/// copies, so it happens once per block, not once per token.
private func reserving(_ buffer: MLXArray?, _ needed: Int, like row: MLXArray) -> MLXArray {
    if let buffer, buffer.dim(2) >= needed { return buffer }

    let capacity = (needed + block - 1) / block * block
    var shape = row.shape
    shape[2] = capacity
    let grown = zeros(shape, dtype: row.dtype)
    if let buffer {
        grown[0..., 0..., ..<buffer.dim(2), 0...] = buffer
    }
    return grown
}

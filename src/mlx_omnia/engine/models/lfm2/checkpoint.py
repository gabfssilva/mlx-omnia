import mlx.core as mx


def fuse_dense_mlp(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    for layer in range(layers):
        prefix = f"model.layers.{layer}.feed_forward."
        keys = [f"{prefix}{name}.weight" for name in ("w1", "w3")]
        if not all(key in weights for key in keys):
            continue
        fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
        mx.eval(fused)
        weights[f"{prefix}w13.weight"] = fused
    return weights

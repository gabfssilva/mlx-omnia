import mlx.core as mx


def test_dense_model_runs_rotary_and_nope_layers() -> None:
    from mlx_omnia.engine.core.attention import DenseConfig, DenseModel

    config = DenseConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        rope_scaling=None,
        tie_word_embeddings=True,
        intermediate_size=24,
        eos_token_id=(2,),
    )
    model = DenseModel(config, rotary=(True, False))
    cache = model.make_cache()

    logits = model(mx.array([[1, 2, 3]]), cache)

    assert logits.shape == (1, 3, config.vocab_size)
    assert [layer.offset for layer in cache] == [3, 3]

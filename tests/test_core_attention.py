import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache


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


def test_dense_attention_composes_its_collaborators() -> None:
    from mlx_omnia.engine.core.attention import DenseAttention, Projected

    class Projection(nn.Module):
        def __call__(self, x: mx.array) -> Projected[mx.array]:
            return Projected(x, 2 * x, 3 * x, 4 * x)

    class Transform(nn.Module):
        def __call__(
            self, queries: mx.array, keys: mx.array, position: int | mx.array
        ) -> tuple[mx.array, mx.array]:
            return queries + position, keys + position

    class Context(nn.Module):
        def position(self, cache: LayerCache) -> int | mx.array:
            return cache.offset

        def attend(
            self,
            queries: mx.array,
            keys: mx.array,
            values: mx.array,
            cache: LayerCache,
            mask: mx.array | str | None,
            scale: float,
        ) -> mx.array:
            return (queries + keys + values) * scale

    class Output(nn.Module):
        def __call__(self, attended: mx.array, residual: mx.array, auxiliary: mx.array) -> mx.array:
            return attended + residual + auxiliary

    cache = LayerCache()
    cache.offset = 2
    attention = DenseAttention(
        projection=Projection(),
        transform=Transform(),
        context=Context(),
        output=Output(),
        scale=0.5,
    )

    result = attention(mx.array([[[1.0]]]), "causal", cache)

    assert result.item() == 10.0

"""Mamba2 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. The trunk is all delta, so what is under test is
`BatchedDeltaCache` — the conv window and the recurrent state stacked per row — against the
family's own single-row forward."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.model import Mamba2


def tiny_model() -> Mamba2:
    mx.random.seed(11)
    model = Mamba2(
        Mamba2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_heads=4,
            head_dim=16,
            state_size=32,
            n_groups=1,
            conv_kernel=4,
            expand=2,
            vocab_size=64,
            chunk_size=8,
        )
    )
    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())
    return model


PROMPTS = ([3, 14, 15, 9, 2], [27, 1, 8])


def test_batched_rows_match_solo_rows() -> None:
    model = tiny_model()
    solo = [model.make_cache() for _ in PROMPTS]
    batched = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, solo, batched, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    tokens = [mx.array([p[-1]]) for p in PROMPTS]
    for _ in range(4):
        rows = [
            model(token[None], cache)[:, -1, :]
            for token, cache in zip(tokens, solo, strict=True)
        ]
        together = model(mx.stack(tokens), batch(batched))[:, -1, :]
        mx.eval(*rows, together)
        for index, row in enumerate(rows):
            difference = float(mx.max(mx.abs(together[index : index + 1] - row)).item())
            ceiling = float(mx.max(mx.abs(row)).item())
            assert difference / ceiling < 1e-4, f"row {index} diverged"
        tokens = [mx.argmax(row, axis=-1).reshape(1) for row in rows]


def test_rows_are_isolated() -> None:
    """Corrupting one row's cache must move that row and no other. Both kinds of state the
    mixer carries are poisoned: the recurrent state and the conv window."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    poisoned = batched[0][0]
    assert isinstance(poisoned, DeltaCache)
    assert poisoned.state is not None and poisoned.window is not None
    poisoned.state = poisoned.state + 1.0
    poisoned.window = poisoned.window + 1.0

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

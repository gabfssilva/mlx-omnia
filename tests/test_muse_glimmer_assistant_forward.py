"""Muse-Glimmer DFlash drafter, fp32 parity against transformers.

Every tolerance is `3x` the fixture's own measured fp32-vs-fp64 floor for that tensor.

The drafter's parity surface, all of it peculiar to a drafter and none of it shared with
the trunk it drafts for:
- Q from the block's 16 rows, K/V from `concat(context, block)`, so q and k do not share
  a length and rope lands on them at different offsets
- bidirectional attention among the block's rows (no causal mask at all)
- per-head q/k RMSNorm *with* weights (the trunk's QK norm is shared and scaleless)
- 8 KV heads against the trunk's 2
- `encoder` = `output_norm_enc(fc(context))`, one projection for every layer
- no embedding and no head in the tree: the block comes in as rows and leaves as rows

The rewind path (context cached across rounds, block never cached) is pinned by the
stepwise-vs-block test below: feeding the context in two calls has to land where feeding
it in one does.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from conftest import checkpoint_dir, floor, load_golden, relative_diff, requires_checkpoint

from mlx_omnia.engine.models.muse_glimmer.checkpoint import load_assistant
from mlx_omnia.engine.models.muse_glimmer.dflash import MuseGlimmerAssistant

FIXTURE = Path(__file__).parent / "fixtures" / "muse_glimmer_assistant_forward.safetensors"
REPO = "meta-models/Muse-Glimmer-30B-assistant"
N_LAYER = 5


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> MuseGlimmerAssistant:
    return load_assistant(checkpoint_dir(REPO), dtype=mx.float32)


def blocks(model: MuseGlimmerAssistant, golden: dict[str, mx.array]) -> list[mx.array]:
    """Every block's output plus the encoder's, in one forward — the fixture captured the
    same tensors through hooks on the same modules."""
    context = model.encoder(golden["context_hidden_states"])
    cache = model.make_cache()
    x = golden["noise_embeds"]
    outputs = [context]
    for block, layer_cache in zip(model.layers, cache, strict=True):
        block.absorb(context, layer_cache)
        x = block(x, layer_cache)
        outputs.append(x)
    outputs.append(model.norm(x))
    return outputs


@requires_checkpoint(REPO)
def test_blocks_match_transformers(model: MuseGlimmerAssistant, golden: dict[str, mx.array]):
    encoder, *rest = blocks(model, golden)
    assert relative_diff(encoder, golden["encoder"]) < floor(golden, "encoder")
    for layer in range(N_LAYER):
        name = f"block_{layer}"
        assert relative_diff(rest[layer], golden[name]) < floor(golden, name), name
    assert relative_diff(rest[-1], golden["norm"]) < floor(golden, "norm")


@requires_checkpoint(REPO)
def test_forward_matches_transformers(model: MuseGlimmerAssistant, golden: dict[str, mx.array]):
    hidden = model(golden["noise_embeds"], golden["context_hidden_states"])
    reference = golden["last_hidden_state"]
    assert relative_diff(hidden, reference) < floor(golden, "last_hidden_state")


@requires_checkpoint(REPO)
def test_context_split_across_rounds(model: MuseGlimmerAssistant, golden: dict[str, mx.array]):
    """A round hands over only the positions accepted since the last one, and the cache
    holds the rest. Splitting the same context in two calls has to land where one call
    does — the cached rows must keep their positions, and the block must keep sitting past
    all of them."""
    features = golden["context_hidden_states"]
    noise = golden["noise_embeds"]
    whole = model(noise, features)

    cache = model.make_cache()
    head, tail = features[:, :17, :], features[:, 17:, :]
    model(noise, head, cache)
    split = model(noise, tail, cache)
    assert relative_diff(split, whole) < floor(golden, "last_hidden_state")


@requires_checkpoint(REPO)
def test_block_never_enters_the_cache(model: MuseGlimmerAssistant, golden: dict[str, mx.array]):
    """What persists is one row per context position and nothing else: a round that
    proposed 16 rows leaves the next round's offsets exactly where the accepted prefix
    ends, or every position after it is wrong."""
    cache = model.make_cache()
    model(golden["noise_embeds"], golden["context_hidden_states"], cache)
    assert [layer.offset for layer in cache] == [golden["context_hidden_states"].shape[1]] * N_LAYER

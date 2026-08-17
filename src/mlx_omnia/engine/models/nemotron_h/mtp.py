"""The MTP head Nemotron-H ships inside the target's own shards.

Not a language model and not a second checkpoint: `mtp.*` sits in the same safetensors as the
trunk, and `checkpoint._drop_mtp` throws it away. One step maps the pair *(the target's hidden
at position i, the embedding of the token at i+1)* to a hidden row that the **target's**
`lm_head` turns into logits for position i+2 — so there is no embedding table and no head in
this tree, and none among the `mtp.*` leaves either.

Authoritative semantics: vllm's `model_executor/models/nemotron_h_mtp.py`. There is no other
— transformers ignores `mtp.*` the way it ignores Qwen's, and nothing in MLX implements this
head. The step is

    h = eh_proj(concat[enorm(embedding), hnorm(target_hidden)])
    h = <the family's own blocks, one per pattern entry>(h)
    h = final_layernorm(h)

with the two RMSNorm applied **before** the concat, never over the concatenated vector, and
the embedding first in it.

The only place this parts with vllm is bookkeeping. There a block returns `(mixer_out,
residual)` and the fused add-norm carries the residual forward, so the last layer of the step
has to add it back before `final_layernorm` (`nemotron_h_mtp.py:117-122`). Here
`NemotronHBlock` already returns `x + mixer(norm(x))`, so composing the blocks produces that
same sum and the end norm applies to it directly.
"""

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.models.nemotron_h.config import ATTENTION, MAMBA, BlockKind, NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.layers.block import NemotronHBlock

DEFAULT_BLOCK = 2
"""How many ids a round proposes when nobody says. Measured, not read off the config.

`num_nextn_predict_layers` is 1, which says the head predicts one token and nothing about how
many are worth verifying together. Chaining it to `b - 1` costs one more of the target's rows
per step, and on a sparse trunk a row is not free: every drafted token routes to its own
experts, so a 3-row verification reads about twice the MoE bytes of one row and a 4-row one
about 2.3x. Against that, measured acceptance falls 0.830, 0.621, 0.390 by depth.

The product of the two, end to end on the bf16 30B-A3B over three prompts: 0.82-0.92x at 2,
**0.95-1.07x at 3**, 0.88-1.06x at 4, 0.81-0.95x at 6. Three is where it stops losing.

Quantizing the trunk does not move the choice and does not make the head pay better. On the
nvfp4 entry, interleaved over two prompts: 0.94-0.98x at 2, **0.97-1.09x at 3**, 0.86-1.02x at
4, with the head's own acceptance at 0.809 / 0.564 / 0.327 by depth. A round of width 2 there
costs 2.2-2.3 plain steps, so it pays from about 1.3 accepted of 2 — which prose does not
always reach (1.25) and code does (1.41).

Those block numbers belong to the replay verify. The compiled fixed-shape verify moved the
optimum: a rejection is a slot pick instead of a replay, so depth 2's low acceptance (0.564)
stops paying for its extra row. Interleaved on the nvfp4 entry with the compiled path, blocks
2 / 3 / 4 read 214 / 211 / 183 tok/s cold and 198 / 192 / 171 warm against 176-183 plain —
block 2 (one drafted id per round) wins every round, block 4 loses to no speculation at
all."""


class NemotronHMTPBlock(NemotronHBlock):
    """A block of the trunk with the step's edges bolted on.

    Literally the trunk's block — same norm, same mixer, same names, so `layers.0.norm` and
    `layers.0.mixer.*` land where the checkpoint puts them. What the step adds is at its two
    ends and never in the middle: the fusion on the first block, the end norm on the last.
    With a two-block step those are two different blocks, which is where this parts with the
    Qwen head, where both fall on the same one.
    """

    def __init__(
        self,
        config: NemotronHConfig,
        block_type: BlockKind,
        *,
        fuses: bool,
        ends: bool,
    ) -> None:
        super().__init__(config, block_type)
        if fuses:
            self.enorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        if ends:
            self.final_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def fuse(self, embeddings: mx.array, hidden: mx.array) -> mx.array:
        """The two normed halves side by side, projected back down to one width. Separately
        normed and *then* concatenated: normalizing the concatenation instead would mix the
        two statistics, which is the mutation `66.2` asks for."""
        enorm, hnorm = self.enorm, self.hnorm
        assert isinstance(enorm, nn.RMSNorm) and isinstance(hnorm, nn.RMSNorm)
        # Not narrowed to `nn.Linear`: `nn.quantize` replaces it with a `QuantizedLinear`,
        # which is a `Module` of its own and not a subclass — an entry written with the head
        # packed reaches here holding one. Norms are never quantized, so those still narrow.
        fused = self.eh_proj(mx.concatenate([enorm(embeddings), hnorm(hidden)], axis=-1))
        assert isinstance(fused, mx.array)
        return fused

    def finish(self, x: mx.array) -> mx.array:
        norm = self.final_layernorm
        assert isinstance(norm, nn.RMSNorm)
        return norm(x)


class NemotronHMTP(nn.Module):
    """One MTP step, `pattern` blocks long.

    `pattern` is not read off the config: the `mtp_hybrid_override_pattern` vllm reads
    (`nemotron_h_mtp.py:231`) is absent from the 30B-A3B's `config.json`, and the tensor names
    are what say `("*", "E")`. `checkpoint.mtp_pattern` is where they are read.

    A step and not a stack of them. `num_nextn_predict_layers` is 1 on every published
    checkpoint and vllm asserts the same; drafting more than one token is running this step
    again over the hidden it just produced, which is the proposer's business and not the
    tree's.
    """

    def __init__(self, config: NemotronHConfig, pattern: tuple[BlockKind, ...]) -> None:
        super().__init__()
        if not pattern:
            raise ValueError("an MTP step with no blocks")
        if MAMBA in pattern:
            raise ValueError(
                f"a recurrent block in the MTP step {''.join(pattern)}: the step is re-run per "
                "drafted token and a mamba state would carry across the re-runs"
            )
        self.config = config
        self.pattern = pattern
        self.block = DEFAULT_BLOCK
        self.layers = [
            NemotronHMTPBlock(
                config, kind, fuses=index == 0, ends=index == len(pattern) - 1
            )
            for index, kind in enumerate(pattern)
        ]

    def make_cache(self) -> list[LayerCache]:
        """One per block, as the trunk does. The attention block is the only one with state,
        and a step that drafts a single token from a single position never grows it — but the
        proposer decides that, so the cache exists either way."""
        return [KVCache() if kind == ATTENTION else LayerCache() for kind in self.pattern]

    def __call__(
        self,
        embeddings: mx.array,
        hidden: mx.array,
        cache: Sequence[LayerCache] | None = None,
    ) -> mx.array:
        """`[B, T, hidden]` in for both halves, `[B, T, hidden]` out — rows for the target's
        `lm_head`, not logits. `embeddings` is the *raw* lookup of the token after each of
        `hidden`'s positions (`core.api.Draftable.raw_embed`)."""
        cache = cache if cache is not None else self.make_cache()
        first, *rest = self.layers
        assert isinstance(first, NemotronHMTPBlock)
        x = first(first.fuse(embeddings, hidden), cache[0])
        for block, layer_cache in zip(rest, cache[1:], strict=True):
            assert isinstance(block, NemotronHMTPBlock)
            x = block(x, layer_cache)
        last = self.layers[-1]
        assert isinstance(last, NemotronHMTPBlock)
        return last.finish(x)

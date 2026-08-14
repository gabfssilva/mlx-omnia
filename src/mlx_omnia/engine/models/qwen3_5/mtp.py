"""The MTP head Qwen3.8 ships inside the target's own shards.

Not a language model and not a second checkpoint: `mtp.*` sits in the same safetensors as
the trunk, and `checkpoint._renamed` throws it away. One step maps the pair *(the target's
hidden at position i, the embedding of the token at i+1)* to a hidden row that the
**target's** `lm_head` turns into logits for position i+2 — so there is no embedding table
and no head in this tree, and none among the `mtp.*` leaves either.

Authoritative semantics: vllm's `model_executor/models/qwen3_5_mtp.py` (c794754), corroborated
by SGLang's `srt/models/qwen3_5_mtp.py` (1af761a). There is no other — transformers ignores
`mtp.*` (`_keys_to_ignore_on_load_unexpected`), and nothing in MLX implements this head.
The step is

    h = fc(concat[pre_fc_norm_embedding(embedding), pre_fc_norm_hidden(hidden)])
    h = <one full-attention block of the trunk's own shape>(h)
    h = norm(h)

with the two RMSNorm applied **before** the concat, never over the concatenated vector, and
the embedding first in it (`qwen3_5_mtp.py:152-155`). Every norm in the head — the pre_fc
pair, the block's own, q/k, the end one — is zero-centered (`Qwen3_5RMSNorm` is
`GemmaRMSNorm`, scale `1 + w`); the loader folds the shift in on raw HF, exactly as it does
for the trunk. The `hidden` half is the trunk's **normed** stream — vllm's runner hands the
step what the model's forward returns, which is `norm(hidden)` (`gpu_model_runner.py:5328`)
— and the embedding half is the raw lookup of the target's table
(`mtp_use_dedicated_embeddings` is false and vllm never reads it).

The chain follows vllm here, and it was measured both ways (2026-08-14, nvfp4 27B entry):
`speculative.Persistent` keeps the drafter's KV across rounds and rotates at absolute
positions, and this head rewards the history — acceptance 0.73 to 0.85 at depth one, 0.45
to 0.63 at depth two, against `speculative.Chained`'s fresh cache per round. Either way an
emitted token never moves: what the drafter attends to only moves acceptance, and the
verification is the target's own argmax. `compile_chain` below serves the fresh-cache
variant and stays for whoever measures the other trade.
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import FixedKVCache, KVCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.qwen3_5.config import Qwen35TextConfig
from mlx_omnia.engine.models.qwen3_5.layers.attention import Qwen35Attention

DEFAULT_BLOCK = 4
"""How many ids a round proposes when nobody says. Measured, not read off the config.

Interleaved in one process on the nvfp4 27B entry, 160 tokens, median of 3, prose and
code prompts (2026-08-14), under `speculative.Persistent` and the compiled fixed-shape
verify. Blocks 3 / 4 / 5 read 55.7 / 56.7 / 51.6 tok/s (prose) and 59.2 / 60.9 / 59.8
(code) against plain 30.8-32.5; acceptance by depth at block 4 is 0.88 / 0.75 / 0.49
(prose) and 0.96 / 0.85 / 0.67 (code). Four wins on both contents and five gives its
gain back, so four is the default and nothing adapts it at runtime — the spread it
would chase is about a token a second, inside the interleave's own noise. (Under the
fresh-cache `Chained` the same sweep read 46.0 at block 2, acceptance 0.73 at depth
one: the history is where the extra depth became worth proposing.)"""


class Qwen35MTPBlock(nn.Module):
    """The trunk's full-attention block, minus the trunk: same attention (q‖gate per head,
    256-wide q/k-norm, partial rope), same MLP, same norm names — `layers.0.self_attn.*`
    lands where the checkpoint puts it. What the trunk's `Qwen35Block` adds on top is
    compile machinery for its decode loop, which a per-round tree has no use for."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.self_attn = Qwen35Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, x: mx.array, cache: KVCache | FixedKVCache, mask: mx.array | None = None
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache, None, mask)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen35MTP(nn.Module):
    """One MTP step, one block long.

    A step and not a stack: `mtp_num_hidden_layers` is 1 on the published checkpoint and
    vllm indexes `spec_step_idx % num_mtp_layers`, which is the same layer every time
    (`qwen3_5_mtp.py:162-163`). Drafting more than one token is running this step again
    over the hidden it just produced, which is the proposer's business and not the tree's.
    """

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        if config.num_experts:
            raise ValueError(
                "the MoE variant of this head is not ported; the dense 27B's is"
            )
        self.config = config
        self.block = DEFAULT_BLOCK
        self.pre_fc_norm_embedding = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = [Qwen35MTPBlock(config)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def compile_chain(
        self,
        embed: Callable[[mx.array], mx.array],
        head: Callable[[mx.array], mx.array],
        width: int,
    ) -> tuple[
        Callable[[mx.array, mx.array], tuple[mx.array, mx.array]],
        Callable[[], None],
    ]:
        """One chained draft step, compiled: (token id [1], hidden [1,1,H]) -> (next id
        [1], next hidden). The step's own KV lives in a fixed buffer of `width` rows
        whose position the caller resets to zero each round — the same graph serves
        every link of the chain, because the only thing that moves between links is
        that position.

        `embed` is the target's raw embedding table and `head` its raw logits
        projection (`speculative.Speculable`); both bake into the trace as constants.
        Each link is one row, so `Qwen35Attention`'s single-row path — the same one the
        trunk's compiled decode traces — serves unchanged, mask and array offset
        included."""
        config = self.config
        (block,) = self.layers
        assert isinstance(block, Qwen35MTPBlock)
        dtype = block.self_attn.q_norm.weight.dtype
        zeros = mx.zeros((1, config.num_key_value_heads, width, config.head_dim), dtype=dtype)
        fixed = FixedKVCache(zeros, zeros, 0)
        columns = mx.arange(width)

        def forward(token: mx.array, hidden: mx.array) -> tuple[mx.array, mx.array]:
            mask = (columns <= fixed.position).reshape(1, 1, 1, width)
            x = block(self.fuse(embed(token[None]), hidden), fixed, mask)
            out = self.norm(x)
            assert isinstance(out, mx.array)
            logits = head(out)
            return mx.argmax(logits[:, -1, :], axis=-1), out

        compiled = mx.compile(forward, inputs=fixed.state, outputs=fixed.state)

        def chain(token: mx.array, hidden: mx.array) -> tuple[mx.array, mx.array]:
            return compiled(token, hidden)

        def reset() -> None:
            fixed.state[2] = mx.array([0], dtype=mx.int32)

        return chain, reset

    def fuse(self, embeddings: mx.array, hidden: mx.array) -> mx.array:
        """The two normed halves side by side, projected back down to one width. Separately
        normed and *then* concatenated, the embedding first — `qwen3_5_mtp.py:152-155`."""
        halves = mx.concatenate(
            [self.pre_fc_norm_embedding(embeddings), self.pre_fc_norm_hidden(hidden)], axis=-1
        )
        fused = self.fc(halves)
        assert isinstance(fused, mx.array)
        return fused

    def __call__(
        self, embeddings: mx.array, hidden: mx.array, cache: list[KVCache] | None = None
    ) -> mx.array:
        cache = self.make_cache() if cache is None else cache
        x = self.fuse(embeddings, hidden)
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            assert isinstance(layer, Qwen35MTPBlock)
            x = layer(x, layer_cache)
        out = self.norm(x)
        assert isinstance(out, mx.array)
        return out

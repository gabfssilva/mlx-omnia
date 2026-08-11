"""Ling 3.0 (`bailing_hybrid`, `BailingMoeV3`): five KDA layers for every MLA layer, a
512-expert router, and an MLA whose cache holds the latent.

Authoritative semantics: the `modeling_bailing_moe_v3.py` shipped inside
`inclusionAI/Ling-3.0-flash`. The numerical reference is a reference-port patch, which
is also where the leaf names of the converted checkpoints come from. The recon is
`docs/models/bailing_hybrid.md`.

What is new against the ports it borrows from:

- **Two mixers under one name.** Layer `i` is MLA when `(i + 1) % layer_group_size == 0`
  (or in the trailing partial group); every other layer is KDA. Both live at
  `layers.{i}.attention.*`, so which leaves exist is decided by the layer index, not by a
  name in the checkpoint.
- **KDA is the gated delta rule with a per-channel decay.** `dt_bias` is `[H·D]`, not
  `[H]`, and with `kda_safe_gate` the log decay is `-lower_bound · sigmoid(exp(A_log)·g)`
  — not the `-exp(A_log)·softplus(g)` of the Qwen3-Next family. The `gated_delta` kernel
  takes both shapes; this is the first port to use the per-channel one.
- **Six projections, one matrix.** q, k, v, f, g and b read the same input, so the loader
  concatenates them on the output axis and the step runs one matmul. Row-aligned, so
  bit-exact; worth the plumbing because the one-token read is limited by the *shape* of
  the matrix and not by the dispatch count — the six leaves move their bytes at 235 GB/s
  and the fused matrix moves the same bytes at 405 GB/s.
- **Three convs, one window.** q, k and v each get their own depthwise causal conv
  (kernel 4, silu), and the three have identical shape, so the taps concatenate on the
  channel axis at load and the cache carries one window of `[1, kernel-1, 3·width]`.
- **MLA in two forms over one cache.** The cache holds the *latent* (`kv_lora_rank + 64`
  columns, one head) instead of the expanded per-head keys and values: 1.15KB per layer
  per token against 20.5KB, which is what makes the advertised 262K context a cache of
  8KB/token. The single-token step absorbs the query into that latent space through
  `embed_q` and projects back with `unembed_out` — no expansion at all; prefill expands
  the prefix and runs the fused SDPA. Both are the same arithmetic reassociated.
- **`kv_b_proj` is split at load** into `embed_q` (the no-position key half, transposed)
  and `unembed_out` (the value half), one matrix per head — the reference converter's
  names and layout, so a
  converted checkpoint loads without a second dialect. A checkpoint that already carries
  the split passes through.
- **Head-wise attention gate**: `g_proj` emits one column per head whose `sigmoid`, taken
  in fp32, scales that head's output before `dense`.
- **The router** is DeepSeek-V3's `noaux_tc` as `glm4_moe` describes it, with two
  deltas: the bias leaf is `expert_bias`, and the dropped groups are masked with `-inf`
  rather than zeroed — `sigmoid + expert_bias` is negative wherever the logit is below
  ≈-2.5, which with 512 experts is most of them, and a zeroed group would then outrank a
  surviving one.
- **A clamped SwiGLU on the late layers.** `expert_swiglu_limit_list` and
  `share_expert_swiglu_limit_list` are not inert: `silu(gate)` is capped above and `up`
  two-sided at the layer's limit. Only one of the four reference implementations reads
  them, so the port follows a measurement rather than a majority — see
  `docs/models/bailing_hybrid.md`.

MTP leaves (`model.layers.{num_hidden_layers}.*`) are dropped at load.
"""

from mlx_omnia.models.bailing_hybrid.checkpoint import CHECKPOINT
from mlx_omnia.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.models.bailing_hybrid.model import BailingHybrid, BailingHybridActivations

__all__ = [
    "CHECKPOINT",
    "BailingHybrid",
    "BailingHybridActivations",
    "BailingHybridConfig",
]

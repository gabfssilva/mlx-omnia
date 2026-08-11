"""DeepSeek-V4-Flash (`deepseek_v4`): hybrid attention over a pooled KV, hyper-connection
residuals, `sqrt(softplus)` routing with hash-routed first layers.

Authoritative semantics: DeepSeek's own `inference/model.py` (there is no
`modeling_deepseek_v4.py` in transformers); the numerical reference this port follows op
by op is an unmerged reference-port patch. `docs/models/deepseek_v4.md` carries
the recon and what each engine decided.

Five things no other ported model has:

- **Three layer kinds by `compress_ratios[i]`**: 0 = sliding window only (layers 0, 1 and
  42), 128 = window plus a pooled KV, 4 = window plus a pooled KV plus a **lightning
  indexer** that selects which pooled columns each query may see. The leaf names are the
  same in all three; what differs is whether `compressor` and `indexer` exist.
- **The compressor**, a recurrent layer of the `GatedDelta` kind: every `ratio` tokens one
  window is pooled into a single KV row by a learned softmax gate plus an absolute
  positional embedding, RMS-normed and rotated at the position of the window's *first*
  token. At `ratio == 4` the pooling window overlaps the previous one (each row reads its
  own lane B plus the predecessor's lane A), so the cache carries the last raw window.
- **MQA with `head_dim` 512, K == V** (one buffer), a **block-diagonal `wo_a`** in 8
  groups — literally a `SwitchLinear` with every slot active — and the output
  **de-rotated** by the same rope with the freqs negated.
- **mHC hyper-connections** instead of `residual + rmsnorm`: the trunk carries
  `hc_mult`(4) copies of the hidden state; each junction collapses them with a learned
  gate and re-expands with a doubly-stochastic 4x4 matrix produced by 20 Sinkhorn
  iterations, all in fp32.
- **Routing** is `sqrt(softplus)` in fp32 with an fp32 selection bias, renormalized
  without epsilon and scaled by 1.5; layers 0-2 do not route at all — the 6 experts come
  from `tid2eid[token_id]`, so they are known the instant the token is sampled.

Two deliberate divergences from the reference, both the house's existing convention:

- **The 128-token window is a mask over a full cache**, not a rotating buffer (the same
  choice `gpt_oss` and `gemma3` make): a masked key contributes nothing, and the rows are
  the same rows. It costs memory the reference does not spend.
- **The indexer's selection is a mask, not a gather** (the same choice `glm_moe_dsa`
  makes): the pooled columns stay in one tensor and the non-selected ones are masked out
  of a single SDPA, instead of gathering the top 512 and running a split softmax. Same
  terms in the softmax, different summation order; the gather is what a dedicated kernel
  would want, and PR 1192 has exactly that kernel for the shapes it fits.

Nothing here is measured. The mHC junction runs in ops (~4 dispatches x 20 iterations x
2 junctions x 43 layers) and, by the arithmetic in the recon, dominates the step: the
fused Sinkhorn kernel other engines have written is the first optimization, not an
afterthought. The routed experts are mxfp4 without biases, so neither the affine
gate-up/down-combine kernels nor the mxfp4 ones (which bake swiglu_oai and a per-row
bias) serve them — `GateUp`/`DownCombine` resolve to the default ops strategy as it
stands.

MTP: `compress_ratios` has one entry per layer **plus one** for the `mtp` block. The extra
entry is dropped by `DeepseekV4Config.ratios`; a loader that assumed
`len == num_hidden_layers` would break on the real config. The `mtp.*` leaves themselves
are dropped at load — some mlx checkpoints ship the three blocks (147 tensors in the
2.4bit-mixed), others strip them.
"""

from mlx_omnia.models.deepseek_v4.checkpoint import CHECKPOINT
from mlx_omnia.models.deepseek_v4.config import DeepseekV4Config
from mlx_omnia.models.deepseek_v4.model import DeepseekV4

__all__ = [
    "CHECKPOINT",
    "DeepseekV4",
    "DeepseekV4Config",
]

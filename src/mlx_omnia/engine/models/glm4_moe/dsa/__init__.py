"""GLM-MoE-DSA: DeepSeek-V3.2's sparse attention — MLA plus a lightning indexer that
picks which keys each query may see.

Authoritative semantics: transformers' modeling_glm_moe_dsa.py.

The trunk is `deepseek_v2`'s MLA with `glm4_moe`'s `noaux_tc` router. What is new is the
**indexer**, a second, much cheaper attention that runs first and produces a selection,
not an output:

- `wq_b` reads the *same* low-rank query `qr` the main attention already computed
  (`q_a_layernorm(q_a_proj(x))`), so the indexer costs no extra projection of the hidden
  state on the query side;
- `wk` projects the hidden state to **one** indexer key per token, LayerNorm'd — one
  head's worth, shared by all `index_n_heads`;
- both rotate over the first `qk_rope_head_dim` dimensions (interleaved or not per
  `indexer_rope_interleave`), and the indexer keys get their own cache;
- the score is `relu(q · kᵀ)` weighted per head by `weights_proj(x)` and summed over
  heads, masked, and the top `index_topk` columns win.

That selection becomes a **mask**, not a gather: this port keeps the decompressed MLA
(the cache holds expanded per-head keys and values, as in `deepseek_v2`), so a
non-selected key contributes nothing and the arithmetic is the reference's. The absorbed
form the reference implementation uses — caching the latent and folding `kv_b_proj`
into per-head embed/unembed matrices — is a different memory profile and a different
set of leaves; it is the optimization this port has not done.

Below `index_topk` keys in the cache nothing is selected away, so short contexts run
exactly like dense MLA.
"""

from mlx_omnia.engine.models.glm4_moe.dsa.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from mlx_omnia.engine.models.glm4_moe.dsa.model import GlmMoEDSA

__all__ = [
    "CHECKPOINT",
    "GlmMoEDSA",
    "GlmMoEDSAConfig",
]

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.models.longcat_flash_ngram.config import LongcatFlashNgramConfig
from mlx_omnia.models.longcat_flash_ngram.layers.cache import NgramCache


class NgramEmbedding(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.m = config.ngram_vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        self.eos = config.eos[0]

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        num_embedders = config.num_embedders
        emb_dim = config.hidden_size // num_embedders
        self.embedders = [
            nn.Embedding(int(self.m + i * 2 + 1), emb_dim)
            for i in range(num_embedders)
        ]
        self.post_projs = [
            nn.Linear(emb_dim, config.hidden_size, bias=False)
            for _ in range(num_embedders)
        ]
        self._compute_vocab_mods()

    def __call__(self, ids: mx.array, cache: NgramCache) -> mx.array:
        seq_len = ids.shape[-1]
        ids64 = ids.astype(mx.int64)
        context = cache.fetch_and_update(ids64)

        x = self.word_embeddings(ids)
        shifted_ids: dict[int, mx.array] = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(context, i - 1)
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = self._vocab_mods[(i, j)]
                ngram_ids = context
                for kk in range(2, i + 1):
                    ngram_ids = ngram_ids + shifted_ids[kk] * mods[kk - 2]
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.embedders[index](new_ids)
                x = x + self.post_projs[index](x_ngram)
        return x / (1 + self.k * (self.n - 1))

    def _compute_vocab_mods(self) -> None:
        vocab_mods: dict[tuple[int, int], list[int]] = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods: list[int] = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods = vocab_mods

    def _shift_right_ignore_eos(self, ids: mx.array, shift: int) -> mx.array:
        """Shift right by ``shift`` positions, zeroing the ``shift`` tokens
        after every EOS — matching transformers' ``_shift_right_ignore_eos``,
        not a simpler zero-pad."""
        if shift <= 0:
            return ids
        length = ids.shape[-1]
        if length <= shift:
            return mx.zeros_like(ids)
        batch_shape = ids.shape[:-1]
        pad = mx.zeros((*batch_shape, shift), dtype=ids.dtype)
        shifted = mx.concatenate([pad, ids[..., :-shift]], axis=-1)
        is_eos = mx.equal(ids, self.eos).astype(mx.int32)
        cumsum = mx.cumsum(is_eos, axis=-1)
        pad_before = mx.zeros((*batch_shape, 1), dtype=mx.int32)
        cumsum_before = mx.concatenate([pad_before, cumsum[..., :-1]], axis=-1)
        if shift + 1 <= length:
            pad_window = mx.zeros((*batch_shape, shift + 1), dtype=mx.int32)
            cumsum_window = mx.concatenate(
                [pad_window, cumsum[..., : -(shift + 1)]], axis=-1
            )
        else:
            cumsum_window = mx.zeros_like(cumsum)
        has_eos = (cumsum_before - cumsum_window) > 0
        return shifted * (1 - has_eos.astype(ids.dtype))

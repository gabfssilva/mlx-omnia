"""Mamba2 (SSD): a pure-SSM trunk with no attention.

Semantics follow transformers' `modeling_mamba2.py` (the naive `torch_forward` is
the reference), with mlx-lm's `ssm_kernel`/`ssm_attn` as the MLX mapping. Every
layer is the mixer alone — no MLP, no MoE — so the block is pre-norm + mixer +
residual. The SSD recurrence (`state = dA·state + dt·B·x; y = state·C + D·x`) is
the decode kernel at T=1 and the chunked surrogate-attention scan at prefill.

Property names are the checkpoint's after the loader normalizes the conv1d layout
(`[conv_dim, 1, kernel]` → `[conv_dim, kernel]`). `A_log`, `dt_bias`, `D` stay
float32 across a dtype cast (the decay saturates if they round-trip through bf16).
The gated RMSNorm folds the gate into the norm *input* (`rms_norm(silu(gate)·x)·w`),
not after — the opposite order from qwen3_5.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import checkpoint, drop_tied_head, reject_dtype_cast, stop_tokens
from sideros.core.cache import DeltaCache
from sideros.core.kernels.ssm_step import ssm_step, ssm_step_applies
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

# A/B switches for the bench: set to False (module attribute, read on every call)
# and the ops chain takes over — it is also the parity reference.
SSM_KERNEL = True

# The compiled one-token step: the ~20 elementwise ops around the recurrence
# collapse into the kernel + a handful of ops. Rides on ssm_step, so it only
# engages when that one does.
COMPILED_STEP = True


@dataclass(frozen=True)
class Mamba2Config:
    hidden_size: int
    num_hidden_layers: int
    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    expand: int
    time_step_rank: int
    time_step_limit: tuple[float, float]
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    vocab_size: int
    tie_word_embeddings: bool
    layer_norm_epsilon: float
    residual_in_fp32: bool
    use_bias: bool
    use_conv_bias: bool
    chunk_size: int
    eos_token_id: tuple[int, ...]

    @property
    def intermediate_size(self) -> int:
        return self.expand * self.hidden_size

    @property
    def conv_dim(self) -> int:
        return self.intermediate_size + 2 * self.n_groups * self.state_size


class Conv1dWeight(nn.Module):
    """The depthwise conv's taps, `[conv_dim, kernel]` — the loader squeezes both
    checkpoint dialects into that shape."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.weight = mx.zeros((config.conv_dim, config.conv_kernel))
        if config.use_conv_bias:
            self.bias = mx.zeros((config.conv_dim,))


def _segsum(x: mx.array) -> mx.array:
    """Stable segment sum: cumsum of the strictly-lower-triangular part, with the
    upper triangle (including diagonal) set to -inf so `exp` zeroes it."""
    length = x.shape[-1]
    expanded = mx.repeat(x[..., None], length, axis=-1)
    masked = mx.tril(expanded, -1)
    segsum = mx.cumsum(masked, axis=-2)
    keep = mx.tril(mx.ones((length, length), dtype=mx.bool_), 0)
    return mx.where(keep, segsum, -float("inf"))


def _compute_dt(
    dt: mx.array, dt_bias: mx.array, time_step_limit: tuple[float, float]
) -> mx.array:
    """softplus(dt + dt_bias) in float32, clamped to the time-step limit."""
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def _ssm_step_ref(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: tuple[float, float],
) -> tuple[mx.array, mx.array]:
    """The SSD recurrence, token by token, entirely in float32 — the parity
    reference for the decode kernel.

    `x` is `[B, T, H, Dh]`, `B`/`C` are `[B, T, G, Ds]`, `dt` is `[B, T, H]`
    (pre-softplus), `state` is `[B, H, Dh, Ds]`. Returns the output `[B, T, H, Dh]`
    in `x`'s dtype and the advanced state.
    """
    _batch, length, num_heads, _head_dim = x.shape
    n_groups = B.shape[2]
    repeats = num_heads // n_groups

    A = -mx.exp(A_log.astype(mx.float32))
    dt_processed = _compute_dt(dt, dt_bias, time_step_limit)
    dA = mx.exp(dt_processed[..., None] * A[None, None, :, None])
    x32 = x.astype(mx.float32)

    B_expanded = mx.repeat(B, repeats, axis=2)
    C_expanded = mx.repeat(C, repeats, axis=2)

    out_list: list[mx.array] = []
    for t in range(length):
        dA_t = dA[:, t]
        dB = dt_processed[:, t, :, None, None] * B_expanded[:, t, :, None, :].astype(
            mx.float32
        )
        dBx = dB * x32[:, t, :, :, None]
        state = state * dA_t[:, :, :, None] + dBx
        y_t = (state * C_expanded[:, t, :, None, :].astype(mx.float32)).sum(axis=-1)
        y_t = y_t + x32[:, t] * D[None, :, None].astype(mx.float32)
        out_list.append(y_t)

    out = mx.stack(out_list, axis=1) if out_list else mx.zeros_like(x32)
    return out.astype(x.dtype), state


def _ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array | None,
    time_step_limit: tuple[float, float],
    step: int = 256,
) -> tuple[mx.array, mx.array]:
    """The chunked SSD (surrogate-attention) scan, ported from mlx-lm's `ssm_attn`.

    `x` is `[B, L, H, Dh]`, `B`/`C` are `[B, L, G, Ds]`, `dt` is `[B, L, H]`
    (pre-softplus). Returns `[B, L, H, Dh]` and the final state `[B, H, Dh, Ds]`.
    """
    batch, length, num_heads, head_dim = x.shape
    n_groups = B.shape[2]
    state_size = B.shape[3]
    repeats = num_heads // n_groups

    dt_processed = _compute_dt(dt, dt_bias, time_step_limit)
    A = -mx.exp(A_log).astype(dt_processed.dtype)
    dtA = dt_processed * A.reshape(1, 1, -1)
    dtx = dt_processed.reshape(batch, length, num_heads, 1) * x.astype(dt_processed.dtype)

    def _chunk(
        dtx: mx.array,
        dtA: mx.array,
        B: mx.array,
        C: mx.array,
        state: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        s = dtx.shape[1]

        B_t = mx.transpose(B, (0, 2, 3, 1))
        cb = mx.swapaxes(C, 1, 2) @ B_t
        cb = mx.repeat(cb, repeats, axis=1)

        decay = mx.exp(_segsum(dtA.swapaxes(1, 2)))
        surrogate = mx.tril(cb * decay, 0)

        y = surrogate @ dtx.swapaxes(1, 2)
        y = mx.swapaxes(y, 1, 2)

        decay_last = decay[:, :, -1:, :].transpose(0, 3, 1, 2)
        B_rep = mx.repeat(B_t, repeats, axis=1).swapaxes(2, 3)
        dtx_decay = dtx * decay_last
        dtx_decay = dtx_decay.swapaxes(1, 2).swapaxes(2, 3)
        next_state = dtx_decay @ B_rep

        if state is not None:
            exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
            next_state = next_state + exp_dtA_cumsum[:, -1, :, None, None] * state
            C_r = C.reshape(batch, s, n_groups, 1, state_size, 1)
            y_prev = (
                state.reshape((batch, 1, n_groups, repeats, head_dim, state_size))
                @ C_r
            ).squeeze(-1).flatten(2, 3)
            y = y + exp_dtA_cumsum[..., None] * y_prev

        return y.astype(x.dtype), next_state

    ys: list[mx.array] = []
    for i in range(0, length, step):
        y, state = _chunk(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
        )
        ys.append(y)

    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, num_heads, 1)
    assert state is not None
    return y, state


_StepReturn = tuple[mx.array, mx.array, mx.array]


class Mamba2Mixer(nn.Module):
    """Selective state space (SSD): in_proj → causal short conv → SSM scan →
    gated RMSNorm → out_proj."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size + config.conv_dim + config.num_heads,
            bias=config.use_bias,
        )
        self.conv1d = Conv1dWeight(config)
        self.A_log = mx.zeros((config.num_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((config.num_heads,), dtype=mx.float32)
        self.D = mx.zeros((config.num_heads,), dtype=mx.float32)
        self.norm = nn.RMSNorm(config.intermediate_size, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.use_bias
        )

    def _conv(self, x: mx.array, cache: DeltaCache) -> mx.array:
        """Depthwise causal conv unrolled into taps, then silu. `[B, T, conv_dim]`
        in and out; the window carries `kernel - 1` rows of history."""
        config = self.config
        kernel = config.conv_kernel
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, config.conv_dim), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for j in range(1, kernel):
            mixed = mixed + padded[:, j : j + length] * taps[:, j]
        if config.use_conv_bias:
            mixed = mixed + self.conv1d.bias
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        groups_state = config.n_groups * config.state_size

        projected = self.in_proj(x)
        gate, conv_input, dt = mx.split(
            projected,
            [config.intermediate_size, config.intermediate_size + config.conv_dim],
            axis=-1,
        )

        conv_output = self._conv(conv_input, cache)
        hidden, b, c = mx.split(
            conv_output,
            [config.intermediate_size, config.intermediate_size + groups_state],
            axis=-1,
        )

        hidden = hidden.reshape(1, length, config.num_heads, config.head_dim)
        b = b.reshape(1, length, config.n_groups, config.state_size)
        c = c.reshape(1, length, config.n_groups, config.state_size)

        state = cache.state
        if SSM_KERNEL and ssm_step_applies(
            config.state_size, config.num_heads, config.n_groups
        ):
            if length == 1:
                if state is None:
                    state = mx.zeros(
                        (1, config.num_heads, config.head_dim, config.state_size),
                        dtype=mx.float32,
                    )
                out, state = ssm_step(
                    hidden,
                    self.A_log,
                    b,
                    c,
                    self.D,
                    dt,
                    self.dt_bias,
                    state,
                    config.time_step_limit,
                )
                out = out.astype(x.dtype)
            else:
                if state is None:
                    state = mx.zeros(
                        (1, config.num_heads, config.head_dim, config.state_size),
                        dtype=mx.float32,
                    )
                out, state = _ssm_attn(
                    hidden,
                    self.A_log,
                    b,
                    c,
                    self.D,
                    dt,
                    self.dt_bias,
                    state,
                    config.time_step_limit,
                    config.chunk_size,
                )
                out = out.astype(x.dtype)
        else:
            if state is None:
                state = mx.zeros(
                    (1, config.num_heads, config.head_dim, config.state_size),
                    dtype=mx.float32,
                )
            out, state = _ssm_step_ref(
                hidden,
                self.A_log,
                b,
                c,
                self.D,
                dt,
                self.dt_bias,
                state,
                config.time_step_limit,
            )

        cache.state = state
        cache.offset += length
        out = out.reshape(1, length, config.intermediate_size)

        # Gated RMSNorm: the gate folds into the norm *input* (silu(gate)·x),
        # then rms_norm in fp32, then round to model dtype and multiply by weight
        # — transformers' order, not qwen3_5's (gate after, not before).
        gate32 = gate.astype(mx.float32)
        hidden = out.astype(mx.float32) * (gate32 * mx.sigmoid(gate32))
        variance = (hidden * hidden).mean(axis=-1, keepdims=True)
        normed = hidden * mx.rsqrt(variance + config.layer_norm_epsilon)
        return self.out_proj((self.norm.weight * normed.astype(out.dtype)).astype(x.dtype))


class Mamba2Block(nn.Module):
    """Pre-norm + mixer + residual. The residual stays in fp32 when
    `residual_in_fp32` (default true)."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config)
        self._traced = self._trace_key()
        self._step: Callable[[mx.array, mx.array, mx.array], _StepReturn] | None = None
        if COMPILED_STEP:
            self._step = mx.compile(self._build_step(), inputs=self.state)

    @staticmethod
    def _trace_key() -> tuple[object, ...]:
        """Everything the compiled step bakes in that tests or benches swap at
        runtime: the kernel binding and the A/B flags — the traced branches read
        those at trace time, so the key carries their value, not the function."""
        return (ssm_step, SSM_KERNEL, COMPILED_STEP)

    def _rebuild(self) -> None:
        key = self._trace_key()
        if self._traced != key:
            self._traced = key
            self._step = mx.compile(self._build_step(), inputs=self.state)

    def _build_step(self) -> Callable[[mx.array, mx.array, mx.array], _StepReturn]:
        """The one-token step as a single trace: norm → in_proj → unrolled conv →
        ssm_step kernel → gated norm → out_proj. The conv window and the recurrent
        state ride in and out as arrays."""
        mixer = self.mixer
        config = mixer.config
        kernel = config.conv_kernel
        groups_state = config.n_groups * config.state_size
        norm = self.norm

        def step(
            x: mx.array, window: mx.array, state: mx.array
        ) -> _StepReturn:
            normed = norm(x)
            projected = mixer.in_proj(normed)
            gate, conv_input, dt = mx.split(
                projected,
                [config.intermediate_size, config.intermediate_size + config.conv_dim],
                axis=-1,
            )
            padded = mx.concatenate([window, conv_input], axis=1)
            taps = mixer.conv1d.weight
            mixed = padded[:, :1] * taps[:, 0]
            for j in range(1, kernel):
                mixed = mixed + padded[:, j : j + 1] * taps[:, j]
            if config.use_conv_bias:
                mixed = mixed + mixer.conv1d.bias
            mixed = mixed * mx.sigmoid(mixed)

            hidden, b, c = mx.split(
                mixed,
                [config.intermediate_size, config.intermediate_size + groups_state],
                axis=-1,
            )
            hidden = hidden.reshape(1, 1, config.num_heads, config.head_dim)
            b = b.reshape(1, 1, config.n_groups, config.state_size)
            c = c.reshape(1, 1, config.n_groups, config.state_size)

            out, state_out = ssm_step(
                hidden,
                mixer.A_log,
                b,
                c,
                mixer.D,
                dt,
                mixer.dt_bias,
                state,
                config.time_step_limit,
            )
            out = out.reshape(1, 1, config.intermediate_size).astype(x.dtype)

            gate32 = gate.astype(mx.float32)
            hidden = out.astype(mx.float32) * (gate32 * mx.sigmoid(gate32))
            variance = (hidden * hidden).mean(axis=-1, keepdims=True)
            normed_out = hidden * mx.rsqrt(variance + config.layer_norm_epsilon)
            projected_out = mixer.out_proj(
                (mixer.norm.weight * normed_out.astype(out.dtype)).astype(x.dtype)
            )
            return projected_out, padded[:, 1:], state_out

        return step

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        if x.shape[1] == 1 and COMPILED_STEP and SSM_KERNEL and self._step is not None:
            self._rebuild()
            window = cache.window
            if window is None:
                window = mx.zeros(
                    (1, config.conv_kernel - 1, config.conv_dim), dtype=x.dtype
                )
            state = cache.state
            if state is None:
                state = mx.zeros(
                    (1, config.num_heads, config.head_dim, config.state_size),
                    dtype=mx.float32,
                )
            out, cache.window, cache.state = self._step(x, window, state)
            cache.offset += 1
            residual = x.astype(mx.float32) if config.residual_in_fp32 else x
            return (residual + out).astype(x.dtype)
        projected = self.mixer(self.norm(x), cache)
        residual = x.astype(mx.float32) if config.residual_in_fp32 else x
        return (residual + projected).astype(x.dtype)


class Mamba2Trunk(nn.Module):
    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Mamba2Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


class Mamba2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Mamba2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[DeltaCache]:
        return [DeltaCache() for _ in range(self.config.num_hidden_layers)]

    def activations(
        self,
        ids: mx.array,
        cache: list[DeltaCache] | None = None,
    ) -> Mamba2Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Mamba2Activations(embedded, blocks, normed, logits)

    def __call__(
        self,
        ids: mx.array,
        cache: list[DeltaCache] | None = None,
    ) -> mx.array:
        return self.activations(ids, cache).logits


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    expand: int
    time_step_rank: NotRequired[int | str]
    time_step_limit: NotRequired[list[float]]
    time_step_min: NotRequired[float]
    time_step_max: NotRequired[float]
    time_step_floor: NotRequired[float]
    vocab_size: int
    tie_word_embeddings: NotRequired[bool]
    layer_norm_epsilon: NotRequired[float]
    residual_in_fp32: NotRequired[bool]
    use_bias: NotRequired[bool]
    use_conv_bias: NotRequired[bool]
    chunk_size: NotRequired[int]
    eos_token_id: NotRequired[int | list[int] | None]


def _config(path: Path) -> Mamba2Config:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "mamba2":
        raise ValueError(f"expected model_type mamba2, got {raw['model_type']!r}")
    time_step_rank = raw.get("time_step_rank", "auto")
    if time_step_rank == "auto":
        time_step_rank = math.ceil(raw["hidden_size"] / 16)
    assert isinstance(time_step_rank, int)
    limit = raw.get("time_step_limit", [0.0, float("inf")])
    eos = raw.get("eos_token_id", 2)
    if eos is None:
        eos = 2
    return Mamba2Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_heads=raw["num_heads"],
        head_dim=raw["head_dim"],
        state_size=raw["state_size"],
        n_groups=raw["n_groups"],
        conv_kernel=raw["conv_kernel"],
        expand=raw["expand"],
        time_step_rank=time_step_rank,
        time_step_limit=(limit[0], limit[1]),
        time_step_min=raw.get("time_step_min", 0.001),
        time_step_max=raw.get("time_step_max", 0.1),
        time_step_floor=raw.get("time_step_floor", 1e-4),
        vocab_size=raw["vocab_size"],
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
        residual_in_fp32=raw.get("residual_in_fp32", True),
        use_bias=raw.get("use_bias", False),
        use_conv_bias=raw.get("use_conv_bias", True),
        chunk_size=raw.get("chunk_size", 256),
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _renamed(name: str) -> str | None:
    """Two dialects: raw HF `backbone.*` vs mlx `model.*`. The HF names
    `embeddings`/`norm_f` map to the tree's `embed_tokens`/`norm`."""
    if name.startswith("backbone."):
        rest = name.removeprefix("backbone.")
        rest = rest.replace("embeddings.", "embed_tokens.", 1)
        rest = rest.replace("norm_f.", "norm.", 1)
        return "model." + rest
    if name.startswith("model."):
        rest = name.removeprefix("model.")
        rest = rest.replace("embeddings.", "embed_tokens.", 1)
        rest = rest.replace("norm_f.", "norm.", 1)
        return "model." + rest
    return name


def weights(
    directory: Path,
    config: Mamba2Config,
    dtype: mx.Dtype | None,
) -> dict[str, mx.array]:
    """The checkpoint's tensors in the tree's names and layout, up to (not
    including) the tree itself."""
    loaded: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        reject_dtype_cast(dtype, part)
        for name, array in part.items():
            renamed = _renamed(name)
            if renamed is None:
                continue
            # A_log, dt_bias, D stay float32 at every precision: the decay
            # saturates if they round-trip through bf16.
            keep_fp32 = (
                renamed.endswith("A_log")
                or renamed.endswith("dt_bias")
                or renamed.endswith(".D")
            )
            if dtype is not None and not keep_fp32:
                loaded[renamed] = array.astype(dtype)
            else:
                loaded[renamed] = array

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    # The torch conv layout `[conv_dim, 1, kernel]` marks a raw HF checkpoint;
    # an mlx conversion arrives as `[conv_dim, kernel, 1]`. Both squeeze to
    # `[conv_dim, kernel]`.
    for name, array in loaded.items():
        if name.endswith("conv1d.weight"):
            if array.ndim == 3 and array.shape[1] == 1:
                loaded[name] = array.squeeze(1)
            elif array.ndim == 3 and array.shape[2] == 1:
                loaded[name] = array.squeeze(2)

    return loaded


def _composite(directory: Path, model: Mamba2) -> LanguageModel[ModelInput]:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    return CompositeModel(
        TextLanguageModel(
            model,
            tokenizer,
            stop=stop_tokens(directory, _eos_tokens(model)),
        ),
        chat_capabilities(directory),
    )


def _eos_tokens(model: Mamba2) -> tuple[int, ...]:
    return model.config.eos_token_id


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    _config,
    Mamba2,
    weights,
    _composite,
)

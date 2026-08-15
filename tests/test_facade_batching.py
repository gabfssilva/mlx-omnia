"""The three vision facades are the chassis, and a text prompt through them is one
generation whichever path answers it.

Each family used to carry its own copy of `language.TextLanguageModel` — the trie, the
drafter, the stream — which is why the server's scheduler never saw a `ContinuousLanguageModel`
under them. They are subclasses now, so the protocol holds; what these tests pin is that
gaining the batched path did not move a segment of the text generation.
"""

from collections.abc import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map

from mlx_omnia.engine.language import ContinuousLanguageModel, GenerationOptions, Text
from mlx_omnia.engine.models.muse_glimmer.config import (
    MuseGlimmerConfig,
    MuseGlimmerRoPE,
    MuseGlimmerTextConfig,
)
from mlx_omnia.engine.models.muse_glimmer.model import MuseGlimmer, MuseGlimmerLanguageModel
from mlx_omnia.engine.models.qwen3_5.config import (
    Qwen35Config,
    Qwen35RoPEParameters,
    Qwen35TextConfig,
)
from mlx_omnia.engine.models.qwen3_5.model import Qwen35, Qwen35LanguageModel
from mlx_omnia.engine.models.step3p7.config import (
    Step3p7AttentionOtherSetting,
    Step3p7Config,
    Step3p7TextConfig,
)
from mlx_omnia.engine.models.step3p7.model import Step3p7, Step3p7LanguageModel
from mlx_omnia.engine.parsers import Segment

VOCAB = 64


class AsciiTokenizer:
    """One id per byte, folded into the tiny checkpoints' vocabulary."""

    def encode(self, text: str) -> Sequence[int]:
        return tuple(byte % VOCAB for byte in text.encode())

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)


def _randomize(model: nn.Module) -> None:
    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())


def _build_qwen3_5() -> Qwen35:
    mx.random.seed(11)
    model = Qwen35(
        Qwen35Config(
            text_config=Qwen35TextConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=VOCAB,
                rms_norm_eps=1e-6,
                layer_types=("full_attention", "linear_attention"),
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                rope_parameters=Qwen35RoPEParameters(
                    rope_theta=10000.0,
                    partial_rotary_factor=0.25,
                    mrope_section=(1, 1, 1),
                ),
                eos_token_id=0,
                tie_word_embeddings=True,
                intermediate_size=64,
            ),
            tie_word_embeddings=True,
        )
    )
    _randomize(model)
    return model


def _build_muse_glimmer() -> MuseGlimmer:
    mx.random.seed(11)
    model = MuseGlimmer(
        MuseGlimmerConfig(
            text_config=MuseGlimmerTextConfig(
                hidden_size=64,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=VOCAB,
                rms_norm_eps=1e-6,
                post_norm_eps=1e-6,
                sliding_window=4,
                qk_scale_factor=1.0,
                output_multiplier=1.0,
                final_logit_softcapping=30.0,
                layer_types=("full_attention", "sliding_attention"),
                layer_rope_theta=(10000.0, 10000.0),
                rope_parameters=MuseGlimmerRoPE(rope_theta=10000.0),
                eos_token_id=0,
            )
        )
    )
    _randomize(model)
    return model


def _build_step3p7() -> Step3p7:
    mx.random.seed(11)
    model = Step3p7(
        Step3p7Config(
            text_config=Step3p7TextConfig(
                hidden_size=64,
                intermediate_size=32,
                num_hidden_layers=2,
                vocab_size=VOCAB,
                rope_theta=(10000.0, 10000.0),
                partial_rotary_factors=(1.0, 1.0),
                layer_types=("full_attention", "full_attention"),
                num_attention_heads=4,
                num_attention_groups=2,
                head_dim=16,
                sliding_window=8,
                use_head_wise_attn_gate=True,
                use_moe=False,
                moe_num_experts=4,
                moe_top_k=2,
                moe_intermediate_size=32,
                share_expert_dim=32,
                moe_router_activation="sigmoid",
                moe_router_scaling_factor=1.0,
                use_moe_router_bias=False,
                need_fp32_gate=False,
                norm_expert_weight=False,
                moe_layers_enum="",
                swiglu_limits=(0.0, 0.0),
                swiglu_limits_shared=(0.0, 0.0),
                attention_other_setting=Step3p7AttentionOtherSetting(
                    attention_type="other_attention",
                    num_attention_heads=4,
                    num_attention_groups=2,
                    head_dim=16,
                    true_head_dim=16,
                ),
                eos_token_id=0,
                bos_token_id=1,
                tie_word_embeddings=True,
            )
        )
    )
    _randomize(model)
    return model


class UnbatchedQwen35(Qwen35LanguageModel):
    """The same facade with the batched path refused, which is the `stream_ids` body."""

    def can_batch(self, options: GenerationOptions) -> bool:
        return False


class UnbatchedMuseGlimmer(MuseGlimmerLanguageModel):
    def can_batch(self, options: GenerationOptions) -> bool:
        return False


class UnbatchedStep3p7(Step3p7LanguageModel):
    def can_batch(self, options: GenerationOptions) -> bool:
        return False


def _qwen3_5() -> tuple[Qwen35LanguageModel, Qwen35LanguageModel]:
    model = _build_qwen3_5()
    tokenizer = AsciiTokenizer()
    return (
        Qwen35LanguageModel(model, tokenizer, None),
        UnbatchedQwen35(model, tokenizer, None),
    )


def _muse_glimmer() -> tuple[MuseGlimmerLanguageModel, MuseGlimmerLanguageModel]:
    model = _build_muse_glimmer()
    tokenizer = AsciiTokenizer()
    return (
        MuseGlimmerLanguageModel(model, tokenizer, None),
        UnbatchedMuseGlimmer(model, tokenizer, None),
    )


def _step3p7() -> tuple[Step3p7LanguageModel, Step3p7LanguageModel]:
    model = _build_step3p7()
    tokenizer = AsciiTokenizer()
    return (
        Step3p7LanguageModel(model, tokenizer, None),
        UnbatchedStep3p7(model, tokenizer, None),
    )


type Facade = Qwen35LanguageModel | MuseGlimmerLanguageModel | Step3p7LanguageModel
type Build = Callable[[], tuple[Facade, Facade]]

FACADES = [
    pytest.param(_qwen3_5, id="qwen3_5"),
    pytest.param(_muse_glimmer, id="muse_glimmer"),
    pytest.param(_step3p7, id="step3p7"),
]

# The generic compiled bucket and a sliding window do not meet: under `mx.compile` a slot's
# offset is a traced array, and `core.attention.ragged_mask` sizes the band by reading it on
# the host. Muse-Glimmer's second layer is windowed, so its batched decode raises before it
# produces a token — until `ragged_mask` learned to take its span from the store's static
# capacity (`Spanned`), which is what keeps this case green under the compiled bucket.
PARITY = [
    pytest.param(_qwen3_5, id="qwen3_5"),
    pytest.param(_muse_glimmer, id="muse_glimmer"),
    pytest.param(_step3p7, id="step3p7"),
]


@pytest.mark.parametrize("build", FACADES)
def test_the_facade_is_a_continuous_language_model(build: Build) -> None:
    facade, _ = build()
    assert isinstance(facade, ContinuousLanguageModel)
    assert facade.can_batch(GenerationOptions(max_tokens=3))


@pytest.mark.parametrize("build", PARITY)
def test_text_batched_matches_the_unbatched_path_segment_for_segment(build: Build) -> None:
    options = GenerationOptions(max_tokens=3)
    batched, plain = build()
    assert isinstance(batched, ContinuousLanguageModel)
    assert batched.can_batch(options)
    assert not plain.can_batch(options)

    produced: list[Segment] = list(batched.stream(Text("ab"), options))
    expected: list[Segment] = list(plain.stream(Text("ab"), options))
    assert produced == expected
    assert produced

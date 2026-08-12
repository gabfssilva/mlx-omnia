"""`generation_config.json` read as sampling defaults.

No checkpoint is loaded: the file is beside the weights, not in them, and what is under
test is the reading — which knobs are taken, which are dropped, and what `do_sample: false`
means for a file that also carries a temperature.
"""

import json
from pathlib import Path

from mlx_omnia.engine.checkpoint import SamplingDefaults, sampling_defaults


def written(directory: Path, **fields: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "generation_config.json").write_text(json.dumps(fields))
    return directory


def test_a_checkpoint_without_the_file_declares_nothing(tmp_path: Path) -> None:
    """Which is not the same as declaring the neutral values: every knob stays `None` so
    the dialect's own default is what fills it."""
    assert sampling_defaults(tmp_path) == SamplingDefaults()


def test_the_knobs_are_read_as_the_file_spells_them(tmp_path: Path) -> None:
    declared = sampling_defaults(
        written(
            tmp_path,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.05,
            repetition_penalty=1.05,
            eos_token_id=[151645, 151643],
        )
    )
    assert declared == SamplingDefaults(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.05, repetition_penalty=1.05
    )


def test_do_sample_false_is_the_whole_answer(tmp_path: Path) -> None:
    """transformers takes the argmax under it and never reads the temperature beside it, so
    a file carrying both means greedy. Reading the 0.6 instead would draw where the
    checkpoint says not to, and `temperature: 0.0` is how the dialects spell the argmax."""
    declared = sampling_defaults(
        written(tmp_path, do_sample=False, temperature=0.6, top_p=0.95, top_k=20)
    )
    assert declared == SamplingDefaults(temperature=0.0)


def test_a_disabled_cut_is_not_a_cut(tmp_path: Path) -> None:
    """`top_k: 0` is how transformers spells *no cut*. Carried across as a value it would be
    a cut that keeps nothing, and the generation would have no token left to draw."""
    declared = sampling_defaults(written(tmp_path, temperature=1.0, top_k=0, top_p=1.0))
    assert declared == SamplingDefaults(temperature=1.0, top_p=1.0)


def test_a_null_knob_is_a_knob_the_file_does_not_declare(tmp_path: Path) -> None:
    """The key written with `null` is what a conversion leaves behind for a knob its source
    did not set — the same silence as the key being absent."""
    assert sampling_defaults(written(tmp_path, temperature=None, top_k=None)) == SamplingDefaults()

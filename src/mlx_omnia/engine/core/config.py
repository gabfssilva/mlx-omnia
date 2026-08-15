"""`load_config(Cls, path_or_dict)`: the dataclass mirrors config.json, extra keys are
dropped, nested dataclasses recurse, lists become tuples so the tree stays frozen and
hashable. No validation beyond what `cls(**...)` itself enforces — the annotations are
documentation of the checkpoint's shape, as the TypedDicts they replace were."""

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_args, get_type_hints


def load_config[T](
    cls: type[T],
    source: dict[str, object] | Path,
    *,
    allowed_model_types: tuple[str, ...] = (),
) -> T:
    if isinstance(source, Path):
        parsed = json.loads(source.read_text())
        assert isinstance(parsed, dict)
        source = parsed
    if allowed_model_types and source.get("model_type") not in allowed_model_types:
        raise ValueError(
            f"expected model_type in {allowed_model_types}, got {source.get('model_type')!r}"
        )
    return _hydrate(cls, source)


def _hydrate[T](cls: type[T], data: dict[str, object]) -> T:
    assert is_dataclass(cls)
    hints = get_type_hints(cls)

    def convert(annotation: object, value: object) -> object:
        target = next((arm for arm in get_args(annotation) if is_dataclass(arm)), annotation)
        if is_dataclass(target) and isinstance(target, type) and isinstance(value, dict):
            return _hydrate(target, value)
        if isinstance(value, list):
            return tuple(value)
        return value

    allowed = {field.name for field in fields(cls)}
    return cls(**{k: convert(hints[k], v) for k, v in data.items() if k in allowed})

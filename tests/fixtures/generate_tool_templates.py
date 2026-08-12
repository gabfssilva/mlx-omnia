# pyright: basic
"""One chat template per family that declares tools, harvested from the local Hub cache.

The round-trip test needs the **template text** and nothing else: a template renders the
assistant's own call when it replays a history, so the characters it writes are the
characters the model writes, and the test that reads them back needs no checkpoint loaded
and no GPU. Storing the text is what keeps it running on a clean machine.

One entry per `model_type`, not per repo: a quantized copy of a checkpoint carries the same
template, and the test measures dialects rather than downloads. Which repo each entry came
from is recorded so a disagreement can be traced back.

Unlike `generate_chat_template.py` this reads the cache directly instead of going through
transformers — nothing here needs a tokenizer, and `AutoTokenizer.from_pretrained` would
reach the network for a repo the cache has only partially.

Run: uv run python packages/engine/tests/fixtures/generate_tool_templates.py
After regenerating, update SHA256SUMS.
"""

import json
from pathlib import Path

HUB = Path.home() / ".cache/huggingface/hub"

SKIP = {"?"}
"""A snapshot with no `config.json` cannot name its family, and an entry keyed by `?` would
collide with the next one like it."""


def _special_tokens(config: dict) -> dict[str, str]:
    """The `*_token` entries, in either shape a tokenizer_config writes them: the literal
    string, or the `{"content": ...}` a slow tokenizer produces. Anything else under a
    matching name (`add_bos_token: true`) is not a token."""
    tokens: dict[str, str] = {}
    for name, value in config.items():
        if not name.endswith("_token"):
            continue
        if isinstance(value, str):
            tokens[name] = value
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            tokens[name] = value["content"]
    for name, value in config.get("model_specific_special_tokens", {}).items():
        if isinstance(value, str):
            tokens[name] = value
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            tokens[name] = value["content"]
    return tokens


def harvested(repo: Path) -> dict | None:
    """The newest snapshot of `repo` that carries a template, or `None`."""
    snapshots = repo / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in sorted(snapshots.iterdir(), reverse=True):
        config_file = snapshot / "tokenizer_config.json"
        config = json.loads(config_file.read_text(errors="replace")) if config_file.exists() else {}
        jinja = snapshot / "chat_template.jinja"
        source = (
            jinja.read_text(errors="replace") if jinja.exists() else config.get("chat_template")
        )
        if isinstance(source, list) and source and isinstance(source[0], dict):
            source = source[0].get("template")
        if not isinstance(source, str):
            continue
        model_file = snapshot / "config.json"
        model_type = "?"
        if model_file.exists():
            model_type = json.loads(model_file.read_text(errors="replace")).get("model_type", "?")
        return {
            "model_type": model_type,
            "repo": repo.name.removeprefix("models--").replace("--", "/"),
            "template": source,
            "special_tokens": _special_tokens(config),
        }
    return None


def main() -> None:
    families: dict[str, dict] = {}
    for repo in sorted(HUB.glob("models--*")):
        entry = harvested(repo)
        if entry is None or "tools" not in entry["template"]:
            continue
        family = entry["model_type"]
        if family in SKIP or family in families:
            continue
        families[family] = entry

    out = Path(__file__).parent / "tool_templates.json"
    payload = {"families": [families[name] for name in sorted(families)]}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in sorted(families):
        print(f"{name:<18} {families[name]['repo']}")
    print(f"wrote {out} ({len(families)} families)")


if __name__ == "__main__":
    main()

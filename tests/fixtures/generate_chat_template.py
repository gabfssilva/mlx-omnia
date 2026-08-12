# pyright: basic
"""Prompts rendered by transformers' `apply_chat_template`, once per template shape we
load.

The fixture stores the **template text** next to the goldens: the render test then needs
no local checkpoint (and is not skipped on a clean machine), while the ids stay gated per
repo, because encoding them takes the `tokenizer.json`.

gpt-oss's harmony prints the date through `strftime_now`; without freezing the clock the
fixture rots by the day, so the generator swaps the `datetime` transformers closes over
(`chat_template_utils.py:22`) and the test renders at the same instant.

Run: uv run python packages/engine/tests/fixtures/generate_chat_template.py
After regenerating, update SHA256SUMS.
"""

import json
from datetime import datetime
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer
from transformers.utils import chat_template_utils

CLOCK = datetime(2026, 1, 2, 3, 4, 5)

REPOS = [
    "mlx-community/Qwen3-0.6B-4bit",  # template inside tokenizer_config.json
    "mlx-community/Qwen3.6-35B-A3B-6bit",  # chat_template.jinja, <think> and vision
    "LiquidAI/LFM2.5-8B-A1B",  # chat_template.jinja
    "openai/gpt-oss-20b",  # harmony: strftime_now
]

VISION = "mlx-community/Qwen3.6-35B-A3B-6bit"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# The checkpoints' templates control whitespace by hand (`{%- ... -%}`), so none of them
# exercises the environment's `trim_blocks`/`lstrip_blocks` — turning both off changes no
# render. This synthetic template, rendered by transformers like any other, is what pins
# the setting.
WHITESPACE = """{% for message in messages %}
    {% if message['role'] == 'user' %}
USER: {{ message['content'] }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
ASSISTANT:
{% endif %}"""

CASES = [
    {"name": "user", "messages": [{"role": "user", "content": "Capital of France?"}]},
    {
        "name": "whitespace",
        "repos": ["mlx-community/Qwen3-0.6B-4bit"],
        "template": WHITESPACE,
        "messages": [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "Capital of France?"},
        ],
    },
    {
        "name": "system+multiturn",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "And of Japan?"},
        ],
    },
    {
        "name": "no-thinking",
        "messages": [{"role": "user", "content": "Hi"}],
        "kwargs": {"enable_thinking": False},
    },
    {
        "name": "tools",
        "messages": [{"role": "user", "content": "Weather in Rio?"}],
        "kwargs": {"tools": TOOLS},
    },
    {
        "name": "image",
        "repos": [VISION],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "<PIXELS>"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
    },
    {
        "name": "two-images",
        "repos": [VISION],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare "},
                    {"type": "image", "image": "<PIXELS>"},
                    {"type": "text", "text": " with "},
                    {"type": "image", "image": "<PIXELS>"},
                ],
            }
        ],
    },
]


class _FrozenDatetime:
    @staticmethod
    def now() -> datetime:
        return CLOCK


def source_of(repo: str) -> str:
    """Two shapes in circulation: the template in a file of its own (Qwen3.6, LFM2.5,
    gpt-oss) or inside tokenizer_config.json (Qwen3)."""
    cached = try_to_load_from_cache(repo, "chat_template.jinja")
    return "chat_template.jinja" if isinstance(cached, str) else "tokenizer_config"


def main() -> None:
    chat_template_utils.datetime = _FrozenDatetime
    fixture: dict[str, object] = {"clock": CLOCK.isoformat(), "repos": {}, "cases": []}
    repos: dict[str, object] = fixture["repos"]
    cases: list[object] = fixture["cases"]

    for repo in REPOS:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        assert isinstance(tokenizer.chat_template, str), repo
        # What transformers puts in the template's scope besides the arguments:
        # `template_kwargs = {**self.special_tokens_map, **kwargs}`
        # (tokenization_utils_base.py:3120).
        repos[repo] = {
            "source": source_of(repo),
            "template": tokenizer.chat_template,
            "special_tokens": tokenizer.special_tokens_map,
        }
        for case in CASES:
            if repo not in case.get("repos", REPOS):
                continue
            kwargs = dict(case.get("kwargs", {}))
            override = case.get("template")
            if override is not None:
                kwargs["chat_template"] = override
            rendered = tokenizer.apply_chat_template(
                case["messages"], tokenize=False, add_generation_prompt=True, **kwargs
            )
            ids = tokenizer.apply_chat_template(
                case["messages"],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **kwargs,
            )
            kwargs.pop("chat_template", None)
            cases.append(
                {
                    "repo": repo,
                    "name": case["name"],
                    "template": override,
                    "messages": case["messages"],
                    "kwargs": kwargs,
                    "rendered": rendered,
                    "ids": list(ids),
                }
            )
            print(f"{repo} {case['name']}: {len(rendered)} chars, {len(ids)} ids")

    out = Path(__file__).parent / "chat_template.json"
    out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

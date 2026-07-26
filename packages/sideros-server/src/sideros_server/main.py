from pathlib import Path

import mlx.core as mx
import uvicorn
from huggingface_hub import hf_hub_download, snapshot_download

from sideros import GPT2Tokenizer, load_gpt2
from sideros_server.app import create_app
from sideros_server.engine import Engine


def main() -> None:
    directory = Path(
        snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"])
    )
    model = load_gpt2(directory, dtype=mx.float16)
    tokenizer = GPT2Tokenizer.from_files(
        Path(hf_hub_download("gpt2", "vocab.json")),
        Path(hf_hub_download("gpt2", "merges.txt")),
    )
    app = create_app(Engine(model, tokenizer, "gpt2"))
    uvicorn.run(app, host="127.0.0.1", port=8642)


if __name__ == "__main__":
    main()

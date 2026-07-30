import numpy as np

from sideros import (
    GenerationOptions,
    Image,
    LanguagePrompt,
    Text,
    load,
)


def main() -> None:
    model = load("Qwen/Qwen3.5-0.8B", local_files_only=True)

    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[:, :128, 0] = 255
    pixels[:, 128:, 2] = 255
    prompt = LanguagePrompt(
        (Image(pixels), Text("Describe only the colors and their positions in one sentence."))
    )

    for piece in model.stream(prompt, GenerationOptions(max_tokens=50)):
        print(piece.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()

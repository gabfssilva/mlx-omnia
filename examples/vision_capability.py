import numpy as np

from sideros import Chat, ChatMessage, GenerationOptions, Image, composite, load


def main() -> None:
    model = composite(
        load("mlx-community/Qwen3-0.6B-4bit", local_files_only=True),
        load("Qwen/Qwen3.5-0.8B", local_files_only=True),
    )

    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[:, :128, 0] = 255
    pixels[:, 128:, 2] = 255
    question: ChatMessage = {
        "role": "user",
        "content": [
            {"type": "image", "image": Image(pixels)},
            {"type": "text", "text": "Which half is red, the left or the right?"},
        ],
    }

    ask = Chat((question,), reasoning_effort="off")
    for piece in model.stream(ask, GenerationOptions(max_tokens=256)):
        print(piece.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()

from mlx_omnia import Chat, ChatMessage, GenerationOptions, load


def main() -> None:
    model = load("local/Qwen3.6-35B-A3B-mlx-oQ3e", local_files_only=True)

    question: ChatMessage = {
        "role": "user",
        "content": (
            "A function representing an autoencoder in Python, using Keras 3, backend agnostic."
        ),
    }

    for piece in model.stream(Chat((question,)), GenerationOptions(max_tokens=8192)):
        print(piece.text, end="", flush=True)

    print()


if __name__ == "__main__":
    main()

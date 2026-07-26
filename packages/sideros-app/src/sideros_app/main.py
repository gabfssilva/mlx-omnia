import os

import flet as ft

from sideros_app.client import Message, list_models, stream_chat

BASE_URL = os.environ.get("SIDEROS_URL", "http://127.0.0.1:8642")


def main(page: ft.Page) -> None:
    page.title = "Sideros"
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH

    history: list[Message] = []
    transcript = ft.ListView(expand=True, spacing=12, auto_scroll=True)
    prompt = ft.TextField(hint_text="Escreva uma mensagem…", expand=True, shift_enter=True)
    model_id = list_models(BASE_URL)[0]
    page.appbar = ft.AppBar(title=ft.Text(f"Sideros — {model_id}"))

    def send() -> None:
        text = prompt.value.strip() if prompt.value else ""
        if not text:
            return
        prompt.value = ""
        history.append(Message("user", text))
        transcript.controls.append(ft.Text(f"você: {text}", selectable=True))
        reply = ft.Text("…", selectable=True)
        transcript.controls.append(reply)
        page.update()

        pieces: list[str] = []
        for piece in stream_chat(BASE_URL, model_id, history):
            pieces.append(piece)
            reply.value = "".join(pieces)
            page.update()
        history.append(Message("assistant", "".join(pieces)))

    prompt.on_submit = send
    composer: list[ft.Control] = [prompt, ft.IconButton(icon=ft.Icons.SEND, on_click=send)]
    page.add(transcript, ft.Row(composer))


def run() -> None:
    ft.run(main)


if __name__ == "__main__":
    run()

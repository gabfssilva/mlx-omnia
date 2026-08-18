import base64
import binascii
import zlib

import numpy as np
import numpy.typing as npt

from mlx_omnia import Image, ImagePart


class UnreadableImage(ValueError):
    """An attachment that cannot become pixels. The message is the client's — it says what was
    wrong with what arrived, and each dialect puts it in its own envelope."""


_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
"""Bytes per pixel by png colour type: grey, rgb, palette index, grey+alpha, rgba."""

_STANDARD = str.maketrans("-_", "+/")
"""The url-safe base64 alphabet back to the standard one: the Gemini SDK encodes bytes with
`-_`, the other two with `+/`, and the same image has to reach the model either way."""


def _bytes(payload: str) -> bytes:
    """`validate=True`, and therefore the translation above first: the permissive default
    *drops* a character outside the alphabet instead of failing, and one `_` silently gone
    shifts every byte after it — an image that decodes to noise rather than to an error."""
    try:
        return base64.b64decode("".join(payload.split()).translate(_STANDARD), validate=True)
    except binascii.Error as bad:
        raise UnreadableImage(f"the image is not base64: {bad}") from bad


def _unfiltered(raw: bytes, height: int, stride: int, step: int) -> bytes:
    """The five row filters of the format, undone. A row is predicted from the row above it and
    a byte from the pixel to its left, so this is sequential by construction."""
    out = bytearray(height * stride)
    previous = bytes(stride)
    at = 0
    for row in range(height):
        kind = raw[at]
        line = bytearray(raw[at + 1 : at + 1 + stride])
        at += 1 + stride
        if kind == 1:
            for i in range(step, stride):
                line[i] = (line[i] + line[i - step]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                up = previous[i]
                corner = previous[i - step] if i >= step else 0
                guess = left + up - corner
                by_left, by_up, by_corner = (
                    abs(guess - left),
                    abs(guess - up),
                    abs(guess - corner),
                )
                if by_left <= by_up and by_left <= by_corner:
                    near = left
                elif by_up <= by_corner:
                    near = up
                else:
                    near = corner
                line[i] = (line[i] + near) & 0xFF
        elif kind != 0:
            raise UnreadableImage(f"row {row} of the png names filter {kind} and there are five")
        out[row * stride : (row + 1) * stride] = line
        previous = bytes(line)
    return bytes(out)


def _pixels(png: bytes) -> npt.NDArray[np.uint8]:
    """An 8-bit png as the `[h, w, 3]` bytes `process_image` takes.

    Png only, and no guessing: a jpeg is refused by name rather than read as noise, and 16-bit
    and interlaced are refused because no client that can send a png needs them. Alpha is
    dropped rather than composited, which is what `PIL.Image.convert("RGB")` does — the
    reference every vision fixture in this repo was made against.
    """
    if not png.startswith(_SIGNATURE):
        raise UnreadableImage("only png is read here, and these bytes carry no png header")
    header: tuple[int, int, int, int, int] | None = None
    palette = b""
    body: list[bytes] = []
    at = len(_SIGNATURE)
    while at + 8 <= len(png):
        length = int.from_bytes(png[at : at + 4])
        kind = png[at + 4 : at + 8]
        payload = png[at + 8 : at + 8 + length]
        at += 12 + length
        if kind == b"IHDR" and len(payload) == 13:
            header = (
                int.from_bytes(payload[:4]),
                int.from_bytes(payload[4:8]),
                payload[8],
                payload[9],
                payload[12],
            )
        elif kind == b"PLTE":
            palette = payload
        elif kind == b"IDAT":
            body.append(payload)
        elif kind == b"IEND":
            break
    if header is None:
        raise UnreadableImage("the png carries no header")
    width, height, depth, colour, interlace = header
    if depth != 8 or interlace != 0 or colour not in _CHANNELS:
        raise UnreadableImage(
            f"this png is not read here: {depth}-bit, colour type {colour}"
            f"{', interlaced' if interlace else ''}"
        )
    if not width or not height:
        raise UnreadableImage("the png has no pixels")
    channels = _CHANNELS[colour]
    stride = width * channels
    try:
        raw = zlib.decompress(b"".join(body))
    except zlib.error as bad:
        raise UnreadableImage(f"the png's pixels do not decompress: {bad}") from bad
    if len(raw) != height * (stride + 1):
        raise UnreadableImage("the png ends before its last row")
    rows = np.frombuffer(_unfiltered(raw, height, stride, channels), dtype=np.uint8)
    grid = rows.reshape(height, width, channels)
    if colour == 3:
        table = np.frombuffer(palette[: len(palette) - len(palette) % 3], dtype=np.uint8)
        entries = table.reshape(-1, 3)
        if int(grid.max()) >= len(entries):
            raise UnreadableImage("the png's palette is shorter than the indices in it")
        return entries[grid[:, :, 0]]
    return np.repeat(grid[:, :, :1], 3, axis=2) if channels < 3 else grid[:, :, :3]


def image_part(payload: str, media_type: str) -> ImagePart:
    """One image on its way into a conversation, out of the base64 a dialect carried it in."""
    if media_type != "image/png":
        raise UnreadableImage(f"{media_type} is not read here: attach the image as image/png")
    return {"type": "image", "image": Image(_pixels(_bytes(payload)))}


def inline_image(url: str) -> ImagePart:
    """OpenAI's two routes spell an image as a URL, and the only one read here is the `data:`
    one that carries the bytes: fetching an `https://` one would have the daemon making
    requests of its own, at a client's word, from inside the network it was told to serve."""
    head, _, payload = url.partition(",")
    if not head.startswith("data:") or not head.endswith(";base64") or not payload:
        raise UnreadableImage(
            "an image travels here as a data URL with the bytes in it "
            "(`data:image/png;base64,…`): nothing fetches a remote image"
        )
    return image_part(payload, head.removeprefix("data:").removesuffix(";base64"))

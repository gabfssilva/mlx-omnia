"""The png encoder the image suites are written against, and the fixture it produces.

The reader under test is new numerical code: the round trip in `test_images_reader.py` writes
every row of this fixture with a different filter, and the goldens beside it are derived from
the format's own definition of Paeth rather than from this encoder.
"""

import base64
import hashlib
import zlib

import numpy as np
import numpy.typing as npt

SIGNATURE = b"\x89PNG\r\n\x1a\n"

PIXELS: npt.NDArray[np.uint8] = (np.arange(6 * 4 * 3).reshape(6, 4, 3) * 37 % 251).astype(np.uint8)
"""Six rows, so the encoder below writes every one of the five filters at least once, and no
two rows alike — a reader that mixed up two rows would still produce the same digest over a
flat field."""


def paeth(left: int, up: int, corner: int) -> int:
    """The format's own predictor, written out here from its definition: the round trip below
    would agree with a reader that got this wrong in the same way, and this one is what the
    golden test pins."""
    guess = left + up - corner
    by_left, by_up, by_corner = abs(guess - left), abs(guess - up), abs(guess - corner)
    if by_left <= by_up and by_left <= by_corner:
        return left
    return up if by_up <= by_corner else corner


def predicted(kind: int, line: bytes, previous: bytes, step: int) -> bytes:
    encoded = bytearray(len(line))
    for at in range(len(line)):
        left = line[at - step] if at >= step else 0
        up = previous[at]
        corner = previous[at - step] if at >= step else 0
        near = (0, left, up, (left + up) >> 1, paeth(left, up, corner))[kind]
        encoded[at] = (line[at] - near) & 0xFF
    return bytes(encoded)


def filtered(grid: npt.NDArray[np.uint8], step: int) -> bytes:
    """The stream an encoder puts inside IDAT: one filter byte per row, cycling through the
    five. Real encoders choose per row exactly like this, and the fixture in `.legacy` that
    this reader was measured against uses four of the five."""
    stream = bytearray()
    previous = bytes(grid.shape[1] * step)
    for index, row in enumerate(grid):
        line = row.tobytes()
        stream.append(index % 5)
        stream += predicted(index % 5, line, previous, step)
        previous = line
    return bytes(stream)


def chunk(kind: bytes, payload: bytes) -> bytes:
    return len(payload).to_bytes(4) + kind + payload + zlib.crc32(kind + payload).to_bytes(4)


def png(
    stream: bytes,
    width: int,
    height: int,
    colour: int,
    *,
    depth: int = 8,
    interlace: int = 0,
    palette: bytes = b"",
) -> bytes:
    """A png around an already-filtered stream, so a test decides what the reader has to undo."""
    header = width.to_bytes(4) + height.to_bytes(4) + bytes([depth, colour, 0, 0, interlace])
    body = chunk(b"IHDR", header)
    if palette:
        body += chunk(b"PLTE", palette)
    return SIGNATURE + body + chunk(b"IDAT", zlib.compress(stream)) + chunk(b"IEND", b"")


PNG = png(filtered(PIXELS, 3), PIXELS.shape[1], PIXELS.shape[0], 2)

BASE64 = base64.b64encode(PNG).decode()
DATA_URL = f"data:image/png;base64,{BASE64}"


def spelling(pixels: npt.NDArray[np.generic]) -> str:
    """The array as `Image` declares it, and not narrowed to `uint8`: what this says is that
    two paths produced the same bytes, and the dtype is part of what it compares."""
    height, width, _ = pixels.shape
    return f"<image {width}x{height} {hashlib.sha256(pixels.tobytes()).hexdigest()[:12]}>"

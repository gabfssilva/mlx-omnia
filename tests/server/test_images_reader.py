"""39, the png reader: what the four dialects hand `image_part`, read back as pixels.

The reader is here rather than beside the routes because it is the same reader for all four,
and because it is new numerical code: the round trip below writes every row of its fixture with
a different filter, and the golden beside it is one derived from the format's own definition of
Paeth rather than from this file's encoder. Split off `test_images.py` for size; the encoder is
in `png_fixtures.py`.
"""

import base64

import numpy as np
import pytest

from mlx_omnia.server.api.responses import UnreadableImage, image_part
from tests.server.png_fixtures import BASE64, PIXELS, PNG, SIGNATURE, chunk, filtered, png


def test_the_reader_undoes_every_filter_of_the_format() -> None:
    """Six rows, one filter each, cycling: `None`, `Sub`, `Up`, `Average` and `Paeth`. An
    encoder in the wild picks per row, so a reader green on `None` alone is green on nothing."""
    read = image_part(BASE64, "image/png")["image"].pixels

    assert read.shape == PIXELS.shape
    assert read.dtype == np.uint8
    assert np.array_equal(read, PIXELS)


PAETH_STREAM = bytes([1, 10, 20, 30, 30, 30, 30, 4, 60, 60, 60, 30, 30, 30])
"""Two rows of a 2x2 rgb image, filtered by hand from the format's definition rather than by
the encoder above — the round trip cannot tell a predictor that is wrong in both directions.

Row 0 is `Sub`: [10,20,30] then [40,50,60] less the pixel to its left. Row 1 is `Paeth` over
[70,80,90],[100,110,120]: for the first pixel there is no left and no corner, so `p = b` and
the predictor is the byte above (10, 20, 30); for the second, `p = a + b - c` is 100, 110, 120
against a = 70, 80, 90, and `pa` is the smallest of the three, so the predictor is the pixel to
the left.
"""

PAETH_PIXELS = np.array(
    [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8
)

TIED_STREAM = bytes([0, 10, 9, 4, 2, 88])
"""Two rows of a 2x1 grey image whose second byte is a tie the format breaks by order. Row 0
is unfiltered: 10 then 9. Row 1 is `Paeth`, and for its second byte a = 12 (the pixel to the
left, once row 1's first byte is read back), b = 9 (above) and c = 10 (above left), so
`p = 11` and the three distances are `pa = 1`, `pb = 2`, `pc = 1`. The format's rule is `pa`
first when it ties, which gives 88 + 12 = 100; breaking the tie the other way gives 98 and
this fixture is the only thing in the suite that can tell them apart."""

TIED_PIXELS = np.array([[[10, 10, 10], [9, 9, 9]], [[12, 12, 12], [100, 100, 100]]], dtype=np.uint8)


def test_the_paeth_predictor_is_the_formats_own() -> None:
    read = image_part(base64.b64encode(png(PAETH_STREAM, 2, 2, 2)).decode(), "image/png")
    assert np.array_equal(read["image"].pixels, PAETH_PIXELS)

    tied = image_part(base64.b64encode(png(TIED_STREAM, 2, 2, 0)).decode(), "image/png")
    assert np.array_equal(tied["image"].pixels, TIED_PIXELS)


def test_grey_and_palette_pngs_come_out_as_rgb() -> None:
    """Three bytes per pixel is what the processor takes, and neither of these two carries
    them: grey repeats its one channel, palette looks its index up."""
    grey = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    read = image_part(
        base64.b64encode(png(filtered(grey[:, :, None], 1), 2, 2, 0)).decode(), "image/png"
    )
    assert np.array_equal(read["image"].pixels, np.repeat(grey[:, :, None], 3, axis=2))

    table = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
    indices = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    coloured = image_part(
        base64.b64encode(png(filtered(indices[:, :, None], 1), 2, 2, 3, palette=table)).decode(),
        "image/png",
    )
    assert np.array_equal(
        coloured["image"].pixels,
        np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 0, 0]]], dtype=np.uint8),
    )


def test_the_alpha_channel_is_dropped_and_not_composited() -> None:
    """`PIL.Image.convert("RGB")` keeps the colour under a transparent pixel; a Core Graphics
    draw composites it against an empty context and gets black. The fixtures this repo measures
    the vision tower against were made with the first one."""
    rgba = np.array([[[10, 20, 30, 0], [40, 50, 60, 255]]], dtype=np.uint8)
    read = image_part(base64.b64encode(png(filtered(rgba, 4), 2, 1, 6)).decode(), "image/png")

    assert np.array_equal(read["image"].pixels, np.array([[[10, 20, 30], [40, 50, 60]]]))


def test_both_base64_alphabets_read_the_same_image() -> None:
    """The Gemini SDK encodes with `-_` and the other two with `+/`, and the same bytes have to
    come out of either. `validate=True` is what makes the difference visible: the permissive
    default drops the characters it does not know, and every byte after the first one shifts."""
    urlsafe = base64.urlsafe_b64encode(PNG).decode()
    assert urlsafe != BASE64, "this fixture does not exercise the two alphabets"

    assert np.array_equal(
        image_part(urlsafe, "image/png")["image"].pixels,
        image_part(BASE64, "image/png")["image"].pixels,
    )


SCRAMBLED = (
    SIGNATURE
    + chunk(b"IHDR", (2).to_bytes(4) + (2).to_bytes(4) + bytes([8, 2, 0, 0, 0]))
    + chunk(b"IDAT", b"not a deflate stream")
    + chunk(b"IEND", b"")
)
"""A png whose pixels never decompress — the one failure that comes out of zlib rather than
out of this reader."""


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        (base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(), "header"),
        (base64.b64encode(SCRAMBLED).decode(), "decompress"),
        (base64.b64encode(b"GIF89a" + PNG[6:]).decode(), "png header"),
        (base64.b64encode(png(filtered(PIXELS, 3), 4, 6, 2, depth=16)).decode(), "16-bit"),
        (base64.b64encode(png(filtered(PIXELS, 3), 4, 6, 2, interlace=1)).decode(), "interlaced"),
        (base64.b64encode(png(filtered(PIXELS, 3)[:-9], 4, 6, 2)).decode(), "last row"),
        (base64.b64encode(png(bytes([5]) + bytes(12), 4, 1, 2)).decode(), "filter 5"),
        ("not base64 at all!!", "base64"),
    ],
)
def test_what_the_reader_does_not_read_is_named(payload: str, named: str) -> None:
    """Each refusal says what was wrong with the bytes: the message is the client's, and "could
    not read the image" leaves it with nothing to change."""
    with pytest.raises(UnreadableImage) as raised:
        image_part(payload, "image/png")

    assert named in str(raised.value)


def test_the_reader_refuses_a_png_whose_palette_is_too_short() -> None:
    """The one index that is not a byte the reader can trust: a table shorter than the indices
    in it would otherwise read past the end of the palette."""
    indices = np.array([[0, 7]], dtype=np.uint8)
    short = png(filtered(indices[:, :, None], 1), 2, 1, 3, palette=bytes([1, 2, 3]))

    with pytest.raises(UnreadableImage) as raised:
        image_part(base64.b64encode(short).decode(), "image/png")

    assert "palette" in str(raised.value)

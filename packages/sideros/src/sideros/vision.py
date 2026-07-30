from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sideros.model import ContentType, Modality

RGB_IMAGE = ContentType(Modality.IMAGE, "image/rgb")


@dataclass(frozen=True)
class Image:
    pixels: npt.NDArray[np.generic]

    @property
    def content_type(self) -> ContentType:
        return RGB_IMAGE

"""What the disk refuses to answer, by name."""

from __future__ import annotations


class UnknownModel(Exception):
    """An id the disk does not answer for."""


class NoModelCard(Exception):
    """The checkpoint ships no README."""


class NoSuchAsset(Exception):
    """The card references a file the checkpoint does not have."""


class TakesNoImage(Exception):
    """The checkpoint has no vision tower, so an image has no cost to quote."""


class ImageSizeInvalid(Exception):
    """An image has a positive height and width."""


class NotTraceable(Exception):
    """No loader for the architecture, or a tree that does not build."""


class ModelResident(Exception):
    """A checkpoint cannot be deleted from under the engine holding it."""

"""A class to represent a single HexBug."""

from enum import auto, Enum


class Color(Enum):
    """
    Color tracker for the bugs.

    TODO: Is using a string fine?
    """

    red = auto()


class Bug:
    """A class to represent a single HexBug."""

    def __init__(self, pose, color):
        """Initialize the Bug."""
        pass

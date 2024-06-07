from abc import ABC, abstractmethod
from typing import Self

from matplotlib.axes import Axes


class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """

    @abstractmethod
    def __init__(self: Self) -> None:
        pass


class Plottable(Base):
    """Class for objects that can be plotted."""

    @abstractmethod
    def plot(self: Self, ax: Axes) -> Axes:
        """Adds plot of object in-place to
        'ax' object. Returns ax.
        """
        pass

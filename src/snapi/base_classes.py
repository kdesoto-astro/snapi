from abc import ABC, abstractmethod
from typing import Optional, Self

from matplotlib.axes import Axes


class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """

    @abstractmethod
    def __init__(self: Self) -> None:
        self._id: Optional[str] = None

    @property
    def id(self) -> Optional[str]:
        """Object identifier."""
        return self._id

    @id.setter
    def id(self, iid: object) -> None:
        try:
            hash(iid)
        except Exception as exc:
            raise ValueError(f"Input {iid} is not hashable!") from exc
        try:
            self._id = str(iid)
        except Exception as exc:
            raise ValueError(f"Input {iid} could not be casted to a string!") from exc


class Plottable(Base):
    """Class for objects that can be plotted."""

    @abstractmethod
    def plot(self: Self, ax: Axes) -> Axes:
        """Adds plot of object in-place to
        'ax' object. Returns ax.
        """
        pass


class MeasurementSet(Base):
    """Base class for storing collection
    of measurements, potentially from different
    instruments and taken at different times.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def filter_by_instrument(self, instrument: str) -> Self:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'
        """
        pass

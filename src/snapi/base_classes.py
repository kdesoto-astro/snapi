from abc import ABC, abstractmethod
from typing import TypeVar

from matplotlib.axes import Axes

MeasT = TypeVar("MeasT", bound="MeasurementSet")


class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """

    @abstractmethod
    def __init__(self) -> None:
        self._id: str = ""

    @property
    def id(self) -> str:
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
    def plot(self, ax: Axes) -> Axes:
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
    def filter_by_instrument(self: MeasT, instrument: str) -> MeasT:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'
        """
        pass

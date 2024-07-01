from typing import Set, TypeVar

from matplotlib.axes import Axes

from .base_classes import MeasurementSet, Plottable
from .lightcurve import LightCurve

PhotT = TypeVar("PhotT", bound="Photometry")


class Photometry(MeasurementSet, Plottable):
    """Contains collection of LightCurve objects."""

    def __init__(self) -> None:
        self._lightcurves: Set[LightCurve] = set()

    def filter_by_instrument(self: PhotT, instrument: str) -> PhotT:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'
        """
        return self  # TODO

    def plot(self, ax: Axes) -> Axes:
        """Plots the collection of light curves."""
        return ax

    def add_lightcurve(self, lc: LightCurve) -> None:
        """Add a light curve to the set of photometry."""
        self._lightcurves.add(lc)

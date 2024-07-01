from typing import Set, TypeVar

from matplotlib.axes import Axes

from .base_classes import MeasurementSet, Plottable
from .spectrum import Spectrum

SpecT = TypeVar("SpecT", bound="Spectroscopy")


class Spectroscopy(MeasurementSet, Plottable):
    """Class for Spectroscopy, which includes a collection
    of individual Spectrum objects. Usually associated with
    a Transient or HostGalaxy object.
    """

    def __init__(self) -> None:
        self._spectra: Set[Spectrum] = set()

    def filter_by_instrument(self: SpecT, instrument: str) -> SpecT:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'
        """
        return self  # TODO

    def plot(self, ax: Axes) -> Axes:
        """Plots all spectra in the collection."""
        return ax

    def add_spectrum(self, spec: Spectrum) -> None:
        """Add spectrum to the collection of spectra."""
        self._spectra.add(spec)

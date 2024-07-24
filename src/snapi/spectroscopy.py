from typing import Optional, Set, TypeVar

import numpy as np
from matplotlib.axes import Axes

from .base_classes import MeasurementSet, Plottable
from .formatter import Formatter
from .spectrum import Spectrum

SpecT = TypeVar("SpecT", bound="Spectroscopy")


class Spectroscopy(MeasurementSet, Plottable):
    """Class for Spectroscopy, which includes a collection
    of individual Spectrum objects. Usually associated with
    a Transient or HostGalaxy object.
    """

    def __init__(
        self,
        spectra: Optional[Set[Spectrum]] = None,
    ) -> None:
        if spectra is None:
            spectra = set()
        self._spectra = spectra

    def filter_by_instrument(self: SpecT, instrument: str) -> SpecT:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'
        """
        return self  # TODO

    def plot(
        self,
        ax: Axes,
        formatter: Optional[Formatter] = None,
        normalize: bool = True,
        vertical_offsets: bool = False,
        overlay_lines: bool = False,
    ) -> Axes:
        """Plots all spectra in the collection."""
        if formatter is None:
            formatter = Formatter()  # make default formatter
        if vertical_offsets:
            if normalize:
                all_fluxes = np.vstack([spec.normalized_fluxes for spec in self._spectra])[::-1]
            else:
                all_fluxes = np.vstack([spec.fluxes for spec in self._spectra])[::-1]
            max_diff = np.max(np.diff(all_fluxes, axis=0))
            offset = max_diff * 1.2

        for spec in self._spectra:
            if vertical_offsets:
                spec.plot(
                    ax, formatter=formatter, normalize=normalize, overlay_lines=overlay_lines, offset=offset
                )
            else:
                spec.plot(ax, formatter=formatter, normalize=normalize, overlay_lines=overlay_lines)
            formatter.rotate_colors()
            formatter.rotate_markers()
        return ax

    def add_spectrum(self, spec: Spectrum) -> None:
        """Add spectrum to the collection of spectra."""
        self._spectra.add(spec)

    def remove_spectrum(self, spec: Spectrum) -> None:
        """Remove spectrum from the collection of spectra."""
        self._spectra.remove(spec)

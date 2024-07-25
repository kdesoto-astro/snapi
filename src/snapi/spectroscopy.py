import copy
from typing import Iterable, Optional, TypeVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

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
        spectra: Optional[Iterable[Spectrum]] = None,
    ) -> None:
        if spectra is None:
            spectra = []
        self._spectra = list(spectra)

        # order spectra
        self._spectra = sorted(self._spectra, key=lambda x: x.time if x.time is not None else np.inf)

    @property
    def spectra(self) -> list[Spectrum]:
        """Return list of spectra."""
        return copy.deepcopy(self._spectra)

    def grid(self, normalize: bool = False) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return common wavelength grid and flux grid for all spectra.
        If normalize is True, normalize fluxes for each spectrum between 0 and 1."""
        wv_arr = [spec.wavelengths for spec in self._spectra]
        all_wvs = np.unique(np.concatenate(wv_arr))
        interp_fluxes = np.zeros((len(self._spectra), len(all_wvs)), dtype=np.float32)
        for i, spec in enumerate(self._spectra):
            interp_fluxes[i] = np.interp(all_wvs, spec.wavelengths, spec.fluxes)
            if normalize:
                interp_fluxes[i] = (interp_fluxes[i] - np.min(interp_fluxes[i])) / np.ptp(interp_fluxes[i])
        return all_wvs, interp_fluxes

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
        overlay_lines: Optional[Iterable[str]] = None,
    ) -> Axes:
        """Plots all spectra in the collection."""
        if formatter is None:
            formatter = Formatter()  # make default formatter

        formatter.reset_colors()
        formatter.reset_markers()

        if vertical_offsets:
            if normalize:
                all_fluxes = self.grid(normalize=True)[1]
            else:
                all_fluxes = self.grid(normalize=False)[1]
            max_diff = np.max(np.diff(all_fluxes, axis=0), axis=1)[
                ::-1
            ]  # t(n) - t(n-1), t(n-1) - t(n-2), ... t(1) - t(0)
            max_diff = np.insert(max_diff, 0, 0)
            cumul_offset = 1.2 * np.cumsum(max_diff)

        for i, spec in enumerate(self._spectra[::-1]):
            if vertical_offsets:
                spec.plot(
                    ax,
                    formatter=formatter,
                    normalize=normalize,
                    overlay_lines=overlay_lines,
                    offset=cumul_offset[i],
                )
            else:
                spec.plot(ax, formatter=formatter, normalize=normalize, overlay_lines=overlay_lines)
            formatter.rotate_colors()
            formatter.rotate_markers()

        formatter.reset_colors()
        formatter.reset_markers()

        for i, spec in enumerate(self._spectra[::-1]):
            if vertical_offsets and (spec.time is not None):
                ax.annotate(
                    rf"$t={round(spec.time, 2)}$",
                    xy=(ax.get_xlim()[1], all_fluxes[len(self._spectra) - i - 1, -1] + cumul_offset[i]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=formatter.edge_color,
                )
            formatter.rotate_colors()
            formatter.rotate_markers()
        return ax

    def add_spectrum(self, spec: Spectrum) -> None:
        """Add spectrum to the collection of spectra."""
        # Find the index where the spectrum should be inserted based on its time
        index = len(self._spectra)
        if spec.time is None:
            self._spectra.append(spec)
            return None

        for i, s in enumerate(self._spectra):
            if s.time is not None and spec.time < s.time:
                index = i
                break
            if s.time is None:
                index = i
                break
        # Insert the spectrum at the determined index
        self._spectra.insert(index, spec)
        return None

    def remove_spectrum(self, spec: Spectrum) -> None:
        """Remove spectrum from the collection of spectra."""
        self._spectra.remove(spec)

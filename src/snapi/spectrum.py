"""Class for a single spectrum."""
import copy
from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .base_classes import Observer, Plottable
from .formatter import Formatter

SequenceNumpy = Union[NDArray[np.float32], Sequence[float]]


class Spectrometer(Observer):
    """A spectrometer object, which can be used to take
    a spectrum of an object.
    """

    def __init__(
        self,
        instrument: str,
        wavelength_start: float,
        wavelength_delta: float,
    ) -> None:
        """Initialize a Spectrometer object."""
        super().__init__(instrument)
        self._wv_start = wavelength_start
        self._wv_delta = wavelength_delta

    @property
    def wavelengths(self) -> NDArray[np.float32]:
        """Return wavelengths."""
        return np.arange(
            self._wv_start,
            self._wv_start + self._wv_delta,
            self._wv_delta,
        ).astype(np.float32)


class Spectrum(Plottable):
    """A single spectrum, associated with
    one timestamp and instrument.
    """

    def __init__(
        self,
        time: Optional[float] = None,
        fluxes: Optional[SequenceNumpy] = None,
        errors: Optional[SequenceNumpy] = None,
        spectrometer: Optional[Spectrometer] = None,
    ) -> None:
        """Initialize a Spectrum object."""

        self._spectrometer = copy.deepcopy(spectrometer)
        self._time = time

        if fluxes is None:
            fluxes = np.array([])
        if errors is None:
            errors = np.array([])

        if spectrometer is not None:
            max_len = max(len(spectrometer.wavelengths), len(fluxes), len(errors))
            if len(spectrometer.wavelengths) < max_len:
                raise ValueError("Too many flux values were provided for given spectrometer.")
            self._wavelengths = spectrometer.wavelengths
        else:
            max_len = max(len(fluxes), len(errors))
            self._wavelengths = np.arange(max_len, dtype=np.float32)

        self._fluxes = np.pad(fluxes, (0, max_len - len(fluxes)), constant_values=np.nan).astype(np.float32)
        self._errors = np.pad(errors, (0, max_len - len(errors)), constant_values=np.nan).astype(np.float32)

    @property
    def time(self) -> Optional[float]:
        """Return time of observation."""
        return self._time

    @property
    def fluxes(self) -> NDArray[np.float32]:
        """Return fluxes."""
        return self._fluxes.copy()

    @property
    def errors(self) -> NDArray[np.float32]:
        """Return errors."""
        return self._errors.copy()

    @property
    def wavelengths(self) -> NDArray[np.float32]:
        """Return wavelengths."""
        return self._wavelengths.copy()

    @property
    def spectrometer(self) -> Optional[Spectrometer]:
        """Return spectrometer."""
        return copy.deepcopy(self._spectrometer)

    @property
    def normalized_fluxes(self) -> NDArray[np.float32]:
        """Return normalized fluxes."""
        return self._fluxes / np.percentile(self._fluxes, 25)

    @property
    def normalized_errors(self) -> NDArray[np.float32]:
        """Return normalized errors."""
        return self._errors / np.percentile(self._fluxes, 25)

    def plot(
        self,
        ax: Axes,
        formatter: Optional[Formatter] = None,
        normalize: bool = True,
        overlay_lines: bool = False,
        offset: float = 0.0,
    ) -> Axes:
        """Plot a single spectrum."""
        if formatter is None:
            formatter = Formatter()
        ax.set_xlabel("Wavelength")

        if normalize:
            ax.set_ylabel("Normalized Flux")
            ax.plot(self._wavelengths, self.normalized_fluxes + offset)
            ax.fill_between(
                self._wavelengths,
                self.normalized_fluxes - self.normalized_errors + offset,
                self.normalized_fluxes + self.normalized_errors + offset,
                alpha=0.5,
            )
        else:
            ax.set_ylabel("Flux")
            ax.plot(self._wavelengths, self._fluxes + offset)
            ax.fill_between(
                self._wavelengths,
                self._fluxes - self._errors + offset,
                self._fluxes + self._errors + offset,
                alpha=0.5,
            )
        if overlay_lines:
            pass
        return ax

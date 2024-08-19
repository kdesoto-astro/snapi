"""Class for a single spectrum."""
import copy
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from astropy.units import Quantity
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .base_classes import Observer, Plottable
from .constants import ION_LINES
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
        num_channels: int,
    ) -> None:
        """Initialize a Spectrometer object."""
        super().__init__(instrument)
        self._wv_start = wavelength_start
        self._wv_delta = wavelength_delta
        self._wv_num = num_channels

    @property
    def wavelengths(self) -> NDArray[np.float32]:
        """Return wavelengths."""
        return np.arange(
            self._wv_start,
            self._wv_start + self._wv_num * self._wv_delta,
            self._wv_delta,
        ).astype(np.float32)


class Spectrum(Plottable):
    """A single spectrum, associated with
    one timestamp and instrument.
    """

    def __init__(
        self,
        time: Optional[Quantity] = None,
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
        return self._time.mjd if self._time is not None else None

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
        return (self._fluxes - np.min(self._fluxes)) / np.ptp(self._fluxes)

    @property
    def normalized_errors(self) -> NDArray[np.float32]:
        """Return normalized errors."""
        return self._errors / np.ptp(self._fluxes)

    def plot(
        self,
        ax: Axes,
        formatter: Optional[Formatter] = None,
        normalize: bool = True,
        overlay_lines: Optional[Iterable[str]] = None,
        annotate: bool = False,
        offset: float = 0.0,
    ) -> Axes:
        """Plot a single spectrum."""
        if formatter is None:
            formatter = Formatter()
        ax.set_xlabel("Wavelength")

        if normalize:
            ax.set_ylabel("Normalized Flux")
            if self._time is not None:
                ax.plot(
                    self._wavelengths,
                    self.normalized_fluxes + offset,
                    color=formatter.edge_color,
                    linewidth=formatter.line_width,
                    label=rf"$t={round(self._time.mjd,2)}$",
                )
            else:
                ax.plot(
                    self._wavelengths,
                    self.normalized_fluxes + offset,
                    color=formatter.edge_color,
                    linewidth=formatter.line_width,
                )
            if self._time is not None and annotate:
                ax.annotate(
                    rf"$t={round(self._time.mjd, 2)}$",
                    xy=(ax.get_xlim()[1], self.normalized_fluxes[-1] + offset),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=formatter.edge_color,
                )
            ax.fill_between(
                self._wavelengths,
                self.normalized_fluxes - self.normalized_errors + offset,
                self.normalized_fluxes + self.normalized_errors + offset,
                alpha=0.5,
                color=formatter.face_color,
            )
        else:
            ax.set_ylabel("Flux")
            if self._time is not None:
                ax.plot(
                    self._wavelengths,
                    self._fluxes + offset,
                    label=rf"$t={round(self._time.mjd,2)}$",
                    color=formatter.edge_color,
                    linewidth=formatter.line_width,
                )
            else:
                ax.plot(
                    self._wavelengths,
                    self._fluxes + offset,
                    color=formatter.edge_color,
                    linewidth=formatter.line_width,
                )
            if self._time is not None and annotate:
                ax.annotate(
                    rf"$t={round(self._time.mjd,2)}$",
                    xy=(ax.get_xlim()[1], self._fluxes[-1] + offset),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color=formatter.edge_color,
                )
            ax.fill_between(
                self._wavelengths,
                self._fluxes - self._errors + offset,
                self._fluxes + self._errors + offset,
                alpha=0.5,
                c=formatter.face_color,
            )
        if not overlay_lines:
            return ax

        for line in overlay_lines:
            if line in ION_LINES:
                for ion_line in ION_LINES[line]:
                    if (ion_line > ax.get_xlim()[1]) or (ion_line < ax.get_xlim()[0]):
                        continue
                    ax.axvline(
                        ion_line,
                        color="gray",
                        linestyle="--",
                        linewidth=0.5 * formatter.line_width,
                    )
            else:
                raise ValueError(f"Unrecognized line type {line}.")
        return ax

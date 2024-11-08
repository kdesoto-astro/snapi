"""Class for a single spectrum."""
import copy
from typing import Any, Iterable, Optional, Sequence, Union, TypeVar, Type

import astropy.units as u
import numpy as np
from astropy.time import Time
from astropy.units import Quantity
from matplotlib.axes import Axes
from numpy.typing import NDArray
import pandas as pd

from .base_classes import Observer, Plottable, Measurement
from .constants import ION_LINES
from .formatter import Formatter

SequenceNumpy = Union[NDArray[np.float64], Sequence[float]]
SpecT = TypeVar("SpecT", bound="Spectrum")

class Spectrometer(Observer):
    """A spectrometer object, which can be used to take
    a spectrum of an object.
    """

    def __init__(
        self,
        instrument: str,
        wavelength_start: Quantity,
        wavelength_delta: Quantity,
        num_channels: int,
    ) -> None:
        """Initialize a Spectrometer object."""
        super().__init__(instrument)
        self._wv_start = wavelength_start
        self._wv_delta = wavelength_delta
        self._wv_num = num_channels

    def __len__(self) -> int:
        """Return the number of channels."""
        return self._wv_num
    
    def __eq__(self, other: object) -> bool:
        """True if all attributes are equal."""
        if not isinstance(other, self.__class__):
            return False
        
        return (
            self.instrument == other.instrument
        ) & np.all(self.wavelengths == other.wavelengths)

    @property
    def wavelengths(self) -> NDArray[np.float64]:
        """Return wavelengths."""
        return np.arange(
            self._wv_start.to(u.AA).value,  # pylint: disable=no-member
            self._wv_start.to(u.AA).value  # pylint: disable=no-member
            + self._wv_num * self._wv_delta.to(u.AA).value,  # pylint: disable=no-member
            self._wv_delta.to(u.AA).value,  # pylint: disable=no-member
        )


class Spectrum(Measurement, Plottable):
    """A single spectrum, associated with
    one timestamp and instrument.
    """

    def __init__(
        self,
        time: Optional[Any] = None,
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
            self._wavelengths = np.arange(max_len, dtype=np.float64)

        self._fluxes = np.pad(fluxes, (0, max_len - len(fluxes)), constant_values=np.nan).astype(np.float64)
        self._errors = np.pad(errors, (0, max_len - len(errors)), constant_values=np.nan).astype(np.float64)


    def __len__(self) -> int:
        """Number of wavelength bins for a spectrum."""
        return len(self._wavelengths)

    def __eq__(self, other: object) -> bool:
        """True if wavelengths, fluxes, errors, and time are equal."""
        if not isinstance(other, self.__class__):
            return False
        
        return np.all(
            self._wavelengths == other.wavelengths
        ) & np.all(
            self._fluxes == other.fluxes
        ) & np.all(
            self._errors == other.errors
        ) & (
            self.time == other.time
        )

    @property
    def time(self) -> Optional[float]:
        """Return time of observation."""
        if self._time is None:
            return None
        if isinstance(self._time, Time):
            return self._time.mjd
        if isinstance(self._time, Quantity):
            return self._time.to(u.d).value  # pylint: disable=no-member
        return self._time

    @property
    def fluxes(self) -> NDArray[np.float64]:
        """Return fluxes."""
        return self._fluxes.copy()

    @property
    def errors(self) -> NDArray[np.float64]:
        """Return errors."""
        return self._errors.copy()

    @property
    def wavelengths(self) -> NDArray[np.float64]:
        """Return wavelengths."""
        return self._wavelengths.copy()

    @property
    def spectrometer(self) -> Optional[Spectrometer]:
        """Return spectrometer."""
        return copy.deepcopy(self._spectrometer)

    @property
    def normalized_fluxes(self) -> NDArray[np.float64]:
        """Return normalized fluxes."""
        return (self._fluxes - np.min(self._fluxes)) / np.ptp(self._fluxes)

    @property
    def normalized_errors(self) -> NDArray[np.float64]:
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
                    label=rf"$t={round(self.time,2)}$" if self.time else r"$t=$None",
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
                    rf"$t={round(self.time, 2)}$" if self.time else r"$t=$None",
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
                    label=rf"$t={round(self.time,2)}$" if self.time else r"$t=$None",
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
                    rf"$t={round(self.time,2)}$" if self.time else r"$t=$None",
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
                color=formatter.face_color,
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
    
    def save(self, file_name: str, path: Optional[str] = None, append: bool = False) -> None:
        """Save LightCurve object as an HDF5 file.

        Parameters
        ----------
        file_name : str
            Name of file to save.
        path : str
            HDF5 path to save LightCurve.
        append : bool
            Whether to append to existing file.
        """
        if path is None:
            path = "/" + str(self.filter)
        mode = "a" if append else "w"

        # Save DataFrame and attributes to HDF5
        with pd.HDFStore(file_name, mode=mode) as store:  # type: ignore
            store.put(path, self._ts)
            # Manually store attributes in the root group
            if self._filter is not None:
                store.get_storer(path).attrs.instrument = str(self._filter.instrument)  # type: ignore
                store.get_storer(path).attrs.band = str(self._filter.band)  # type: ignore
                store.get_storer(path).attrs.center = self._filter.center.value  # type: ignore
                if self._filter.width is not None:
                    store.get_storer(path).attrs.width = self._filter.width.value  # type: ignore

    @classmethod
    def load(
        cls: Type[SpecT],
        file_name: str,
        path: Optional[str] = None,
        archival: bool = False,
    ) -> SpecT:
        """Load LightCurve from saved HDF5 table. Automatically
        extracts feature information.
        """
        if path is None:
            paths = list_datasets(file_name, archival)
            if len(paths) > 1:
                raise ValueError("Multiple datasets found in file. Please specify path.")
            path = paths[0]

        if archival:
            raise NotImplementedError("archival spectrum loading did not exist!")

        with pd.HDFStore(file_name) as store:
            time_series = store[path]  # Load the DataFrame
            # Retrieve attributes
            attrs = store.get_storer(path).attrs  # type: ignore
            if "instrument" in attrs.__dict__:
                if "width" in attrs.__dict__:
                    extracted_filter = Filter(
                        attrs.instrument,
                        attrs.band,
                        attrs.center * u.AA,  # pylint: disable=no-member
                        attrs.width * u.AA,  # pylint: disable=no-member,
                    )
                else:
                    extracted_filter = Filter(
                        attrs.instrument,
                        attrs.band,
                        attrs.center * u.AA,  # pylint: disable=no-member
                    )
                return cls(time_series, filt=extracted_filter)
            return cls(time_series)


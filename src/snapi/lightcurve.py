"""Contains classes for light curves and filters."""
import copy
from typing import Any, Mapping, Optional, Sequence, Type, TypeVar

import astropy.units as u
import numba
import numpy as np
from astropy.timeseries import TimeSeries
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from .base_classes import Base, Plottable
from .formatter import Formatter
from .image import Image

T = TypeVar("T", int, float)
LightT = TypeVar("LightT", bound="LightCurve")


class Filter(Base):
    """Contains instrument and filter information."""

    def __init__(
        self,
        instrument: str,
        band: str,
        center: u.Quantity,
        width: Optional[u.Quantity],
        lim_mag: Optional[float] = None,
    ) -> None:
        self._instrument = instrument
        self._band = band

        if center.unit.physical_type == "frequency":  # convert to wavelength
            self._center = (
                center.to(u.Hz, equivalencies=u.spectral()) ** -1 * u.AA  # pylint: disable=no-member
            )
        elif center.unit.physical_type == "length":  # convert to wavelength
            self._center = center.to(u.AA)  # pylint: disable=no-member
        else:
            raise TypeError("center must be a wavelength or frequency quantity!")

        if width is None:
            self._width = width
        elif width.unit.physical_type == "frequency":  # convert to wavelength
            self._width = width.to(u.Hz, equivalencies=u.spectral()) ** -1 * u.AA  # pylint: disable=no-member
        elif width.unit.physical_type == "length":  # convert to wavelength
            self._width = width.to(u.AA)  # pylint: disable=no-member
        else:
            raise TypeError("width must be a wavelength or frequency quantity!")

        self._lim_mag = lim_mag

    def __str__(self) -> str:
        """Return string representation of filter.
        Format: instrument_band.
        """
        return f"{self._instrument}_{self._band}"

    @property
    def instrument(self) -> str:
        """Return instrument of filter."""
        return self._instrument

    @property
    def band(self) -> str:
        """Return band of filter."""
        return self._band

    @property
    def center(self) -> u.Quantity:
        """Return center of filter,
        in Angstroms.
        """
        return self._center.to(u.AA)  # pylint: disable=no-member

    @property
    def width(self) -> u.Quantity:
        """Return width of filter,
        in Angstroms.
        """
        if self._width is None:
            return self._width
        return self._width.to(u.AA)  # pylint: disable=no-member

    @property
    def lim_mag(self) -> Optional[float]:
        """Return limiting magnitude of filter."""
        return self._lim_mag


@numba.njit(parallel=True)  # type: ignore
def resample_helper(cen: Sequence[T], unc: Sequence[T], num: int) -> NDArray[np.float32]:
    """numba-enhanced helper to generate many resampled LCs."""
    rng = np.random.default_rng()
    sampled_vals = np.zeros((num, len(cen)), dtype=np.float32)
    for i in numba.prange(num):  # pylint: disable=not-an-iterable
        sampled_vals[i] += rng.normal(loc=cen, scale=unc)
    return sampled_vals


@numba.njit(parallel=True)  # type: ignore
def update_merged_fluxes(keep_idxs, flux, flux_unc):
    """Update merged fluxes with numba."""
    new_f = []
    new_ferr = []
    for i in numba.prange(len(keep_idxs)):  # pylint: disable=not-an-iterable
        if i == len(keep_idxs) - 1:
            repeat_idx_subset = np.arange(keep_idxs[i], keep_idxs[i + 1])
        else:
            repeat_idx_subset = np.arange(keep_idxs[i], len(flux))

        weights = 1.0 / flux_unc[repeat_idx_subset] ** 2
        new_f.append(np.average(flux[repeat_idx_subset], weights=weights))
        new_var = np.var(flux[repeat_idx_subset])
        new_var += 1.0 / np.sum(weights)
        new_ferr.append(np.sqrt(new_var))

    return new_f, new_ferr


@numba.njit(parallel=True)  # type: ignore
def calc_all_deltas(series: NDArray[np.float32], use_sum: bool = False) -> NDArray[np.float32]:
    """Calculate all pairwise distances between values in each set,
    assuming the first axis delineates different sets.
    """
    deltas = []
    for i in numba.prange(series.shape[1]):  # pylint: disable=not-an-iterable
        for j in range(i + 1, series.shape[1]):
            if use_sum:
                deltas.append(series[:, j] + series[:, i])
            else:
                deltas.append(series[:, j] - series[:, i])
    return np.array(deltas).T


class LightCurve(Plottable):
    """Class that contains all information for a
    single light curve. Associated with a single instrument
    and filter.

    LightCurve should always be (a) as complete as possible and
    (b) sorted by time.
    """

    def __init__(
        self,
        times: Any,  # TODO: somehow enforce time-like
        fluxes: Optional[Sequence[T]] = None,
        flux_errs: Optional[Sequence[T]] = None,
        mags: Optional[Sequence[T]] = None,
        mag_errs: Optional[Sequence[T]] = None,
        zpts: Optional[Sequence[T]] = None,
        filt: Optional[Filter] = None,
    ) -> None:
        self._filter = filt

        if isinstance(times, TimeSeries):
            self._ts = times
        else:
            self._ts = TimeSeries(
                {
                    "time": times,
                    "flux": fluxes,
                    "flux_unc": flux_errs,
                    "mag": mags,
                    "mag_unc": mag_errs,
                    "zpt": zpts,
                }
            )

        self._rng = np.random.default_rng()
        self._sort()  # sort by time
        self._complete()  # fills in missing info

    def _complete(self) -> None:
        """Given zeropoints, fills in missing apparent
        magnitudes from fluxes and vice versa.
        """
        # first convert fluxes to missing apparent mags
        missing_mag = np.isnan(self._ts["mag"])
        sub_table = self._ts[missing_mag]
        self._ts["mag"][missing_mag] = -2.5 * np.log10(sub_table["flux"]) + sub_table["zpt"]

        # uncertainties
        missing_magunc = np.isnan(self._ts["mag_unc"])
        sub_table = self._ts[missing_magunc]
        self._ts["mag_unc"][missing_magunc] = 2.5 / np.log(10.0) * (sub_table["flux_unc"] / sub_table["flux"])

        # then convert mags to missing fluxes
        missing_flux = np.isnan(self._ts["flux"])
        sub_table = self._ts[missing_flux]
        self._ts["flux"][missing_flux] = 10.0 ** (-1.0 * (sub_table["mag"] - sub_table["zpt"]) / 2.5)

        # uncertainties
        missing_fluxunc = np.isnan(self._ts["flux_unc"])
        sub_table = self._ts[missing_fluxunc]
        self._ts["flux_unc"][missing_fluxunc] = (
            np.log(10.0) / 2.5 * (sub_table["flux"] * sub_table["mag_unc"])
        )

    def _sort(self) -> None:
        """Sort light curve by time."""
        self._ts.sort()

    @property
    def times(self) -> Any:
        """Return sorted light curve times as
        numpy array, in mean julian date.
        """
        return self._ts["time"].mjd.astype(np.float32)  # pylint: disable=no-member; type: ignore

    @times.setter
    def times(self, new_times: Sequence[u.Quantity]) -> None:
        """Replace time values."""
        self._ts.replace_column("time", new_times)
        self._sort()

    @property
    def fluxes(self) -> Any:
        """Return fluxes of sorted LC as
        numpy array.
        """
        return self._ts["flux"].value  # pylint: disable=no-member; type: ignore

    @fluxes.setter
    def fluxes(self, new_fluxes: NDArray[np.float32]) -> None:
        """Replace flux values."""
        self._ts.replace_column("flux", new_fluxes)
        self._complete()

    @property
    def flux_errors(self) -> Any:
        """Return flux uncertainties of sorted LC as
        numpy array.
        """
        return self._ts["flux_unc"].value  # pylint: disable=no-member; type: ignore[no-any-return]

    @flux_errors.setter
    def flux_errors(self, ferr: NDArray[np.float32]) -> None:
        """Replace flux uncertainty values."""
        self._ts.replace_column("flux_unc", ferr)
        self._complete()

    @property
    def mags(self) -> Any:
        """Return magnitudes of sorted LC as
        numpy array.
        """
        return self._ts["mag"].value  # pylint: disable=no-member; type: ignore

    @mags.setter
    def mags(self, new_mags: NDArray[np.float32]) -> None:
        """Replace magnitude values."""
        self._ts.replace_column("mag", new_mags)
        self._complete()

    @property
    def mag_errors(self) -> Any:
        """Return magnitude uncertainties of sorted LC as
        numpy array.
        """
        return self._ts["mag_unc"].value  # pylint: disable=no-member; type: ignore[no-any-return]

    @mag_errors.setter
    def mag_errors(self, merr: NDArray[np.float32]) -> None:
        """Replace magnitude uncertainty values."""
        self._ts.replace_column("mag_unc", merr)
        self._complete()

    @property
    def filter(self) -> Optional[Filter]:
        """Return filter object associated with LightCurve."""
        return self._filter

    @property
    def peak(self) -> dict[str, Any]:
        """The brightest observation in light curve.
        Return as dictionary.
        """
        return self._ts[np.argmax(self._ts["flux"])].as_dict()  # type: ignore[no-any-return]

    def phase(self, phase_0: Optional[float] = None) -> None:
        """
        Phases light curve by t0, which is assumed
        to be in days.
        """
        if phase_0 is None:
            phase_0 = self.peak["time"]
        self._ts["time"] -= phase_0 * u.d  # pylint: disable=no-member

    def truncate(self, max_t: Optional[float] = np.inf, min_t: Optional[float] = -np.inf) -> None:
        """Truncate light curve between min_t and max_t days."""
        gind = (self._ts["time"] > max_t) | (self._ts["time"] < min_t)
        self._ts.remove_rows(np.argwhere(gind))

    def subsample(self, n_points: int) -> None:
        """Subsamples n_points in light curve."""
        if n_points < len(self._ts):
            remove_ind = self._rng.choice(
                np.arange(len(self._ts)), size=len(self._ts) - n_points, replace=False
            )
            self._ts.remove_rows(remove_ind)

    def plot(self, ax: Axes, formatter: Optional[Formatter] = None) -> Axes:
        """Plot a single light curve.
        Face and edge colors determined by formatter.
        """
        if formatter is None:
            ax.errorbar(
                self.times,
                self.fluxes,
                yerr=self.flux_errors,
                c="k",
                fmt="none",
                zorder=500,
            )
            ax.scatter(
                self.times,
                self.fluxes,
                c="b",
                edgecolor="k",
                marker="o",
                zorder=1000,
            )
            return ax

        ax.errorbar(
            self.times,
            self.fluxes,
            yerr=self.flux_errors,
            c=formatter.edge_color,
            fmt="none",
            zorder=500,
        )
        ax.scatter(
            self.times,
            self.fluxes,
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.marker_style,
            zorder=1000,
        )
        return ax

    def add_observations(self: LightT, rows: list[dict[str, Any]]) -> LightT:
        """Add rows to existing timeseries."""
        # TODO: accomodate different formats
        for row in rows:
            self._ts.add_row(row)
        self._sort()
        self._complete()
        return self

    def merge_close_times(self, eps: float = 4e-2) -> None:
        """Merge times that are close enough together
        (e.g. in the same night). Only in-place at the moment.
        """
        t_diffs = np.diff(self._ts["time"])
        repeat_idxs = np.abs(t_diffs) < eps * u.d  # pylint: disable=no-member
        repeat_idxs = np.insert(repeat_idxs, 0, False)
        keep_idxs = np.argwhere(~repeat_idxs)

        new_f, new_ferr = update_merged_fluxes(keep_idxs, self._ts["flux"], self._ts["flux_unc"])
        self._ts.remove_rows(np.argwhere(repeat_idxs))
        self._ts.replace_column("flux", new_f)
        self._ts.replace_column("flux_unc", new_ferr)
        self._complete()

    def pad(self: LightT, n_times: int, inplace: bool = False) -> LightT:
        """Extends light curve by padding.
        Currently, pads at 1000 days past last observation,
        and pads according to limiting values.
        """
        empty_rows = [
            {"time": np.max(self._ts["time"]) + 1000.0 * u.d},  # pylint: disable=no-member
        ] * n_times

        if inplace:
            return self.add_observations(empty_rows)

        lc_copy = copy.deepcopy(self)
        return lc_copy.add_observations(empty_rows)

    def resample(self, mags: bool = False, num: int = 100) -> NDArray[np.float32]:
        """Returns set of num augmented light curves where new
        fluxes are sampled from distribution defined by flux
        uncertainties.
        """
        if mags:
            centers = self._ts["mag"]
            uncs = self._ts["mag_unc"]
        else:
            centers = self._ts["flux"]
            uncs = self._ts["flux_unc"]

        return resample_helper(centers, uncs, num)  # type: ignore

    def convert_to_images(
        self, method: str = "dff-ft", augment: bool = True, **kwargs: Mapping[str, Any]
    ) -> list[Image]:
        """Convert light curve to set of images for ML applications.
        Will return 200 images for augmented LC and 2 for unaugmented.
        """
        if augment:  # augment LC to get many copies before transforming
            series = self.resample(mags=False, num=100)  # TODO: de-hardcode this
            series_t = np.repeat(np.atleast_2d(self._ts["time"]), 100, 0)
        else:
            series = np.atleast_2d(self._ts["flux"])
            series_t = np.atleast_2d(self._ts["time"])

        if method == "dff-dt":  # dff vs dt plots
            delta_times = calc_all_deltas(self._ts["time"])
            delta_fluxes = calc_all_deltas(series)
            avg_fluxes = calc_all_deltas(series, use_sum=True) / 2.0
            dff = delta_fluxes / avg_fluxes
            vals_concat, _, _ = np.histogram2d(delta_times, dff)  # TODO: set bins in future
        elif method in ["gaf", "mtf", "recurrence"]:
            if method == "gaf":  # Gramian angular field
                transformer = GramianAngularField(**kwargs)
            elif method == "mtf":  # Markov transition field
                transformer = MarkovTransitionField(**kwargs)
            else:  # recurrence plot
                transformer = RecurrencePlot(**kwargs)
            vals = transformer.transform(series)
            t_vals = transformer.transform(series_t)
            vals_concat = np.concatenate((vals, t_vals), axis=-1)
        else:
            raise NotImplementedError("Imaging method must be one of: 'gaf', 'mtf', 'recurrence'")

        return [Image(vc) for vc in vals_concat]

    def save(self, file_name: str, save_format: str = "ascii.csv") -> None:
        """Save LightCurve object as an
        astropy table.
        """
        table = self._ts.copy()
        if self._filter is not None:
            table.meta["instrument"] = self._filter.instrument
            table.meta["band"] = self._filter.band
            table.meta["center"] = self._filter.center.value  # in Angstroms
            table.meta["width"] = self._filter.width.value  # in Angstroms
            table.meta["lim_mag"] = self._filter.lim_mag
        table.write(file_name, format=save_format)

    @classmethod
    def load(cls: Type[LightT], file_name: str, save_format: str = "ascii.csv") -> LightT:
        """Load LightCurve from saved table. Automatically
        extracts feature information.
        """
        time_series = TimeSeries.read(file_name, format=save_format, time_column="mjd")
        if "instrument" in time_series.meta:
            extracted_filter = Filter(
                time_series.meta["instrument"],
                time_series.meta["band"],
                time_series.meta["center"] * u.AA,  # pylint: disable=no-member
                time_series.meta["width"] * u.AA,  # pylint: disable=no-member,
                time_series.meta["lim_mag"],
            )
            return cls(time_series, filt=extracted_filter)
        return cls(time_series)

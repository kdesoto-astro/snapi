"""Contains classes for light curves and filters."""
import copy
from typing import Any, Iterable, Mapping, Optional, Sequence, Type, TypeVar, cast

import astropy.constants as const
import astropy.cosmology.units as cu
import astropy.units as u
import extinction
import numba
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15  # pylint: disable=no-name-in-module
from astropy.io.misc import hdf5
from astropy.time import Time
from astropy.timeseries import LombScargle
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from .base_classes import Observer, Plottable
from .formatter import Formatter
from .image import Image
from .utils import calc_mwebv, list_datasets

T = TypeVar("T", int, float, np.float64)
LightT = TypeVar("LightT", bound="LightCurve")


class Filter(Observer):
    """Contains instrument and filter information."""

    def __init__(
        self,
        instrument: str,
        band: str,
        center: u.Quantity,
        width: Optional[u.Quantity] = None,
    ) -> None:
        super().__init__(instrument)
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

    def __str__(self) -> str:
        """Return string representation of filter.
        Format: instrument_band.
        """
        return f"{self._instrument}_{self._band}"

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Filter):
            return False

        return (self.instrument == other.instrument) and (self.band == other.band)


# @numba.njit(parallel=True)  # type: ignore
def resample_helper(cen: Sequence[T], unc: Sequence[T], num: int) -> NDArray[np.float64]:
    """numba-enhanced helper to generate many resampled LCs."""
    rng = np.random.default_rng()
    sampled_vals = np.zeros((num, len(cen)), dtype=np.float64)
    for i in range(num):  # pylint: disable=not-an-iterable
        sampled_vals[i] += rng.normal(loc=cen, scale=unc)
    return sampled_vals


# @numba.njit(parallel=True)  # type: ignore
def update_merged_fluxes(
    keep_idxs: NDArray[np.int64], flux: NDArray[np.float64], flux_unc: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Update merged fluxes with numba."""
    new_f = []
    new_ferr = []
    new_nondet = []
    for i, keep_idx in enumerate(keep_idxs):  # pylint: disable=not-an-iterable
        if i == len(keep_idxs) - 1:
            repeat_idx_subset = np.arange(keep_idx, len(flux))
        else:
            repeat_idx_subset = np.arange(keep_idx, keep_idxs[i + 1])

        nondetect_subset = repeat_idx_subset[~np.isfinite(flux_unc.iloc[repeat_idx_subset])]
        detect_subset = repeat_idx_subset[np.isfinite(flux_unc.iloc[repeat_idx_subset])]

        if len(detect_subset) == 0:
            new_f.append(np.mean(flux.iloc[nondetect_subset]))
            new_ferr.append(np.nan)
            new_nondet.append(True)
            continue

        weights = 1.0 / flux_unc.iloc[detect_subset] ** 2
        new_f.append(np.average(flux.iloc[detect_subset], weights=weights))
        new_var = np.var(flux.iloc[detect_subset])
        new_var += 1.0 / np.sum(weights)
        new_ferr.append(np.sqrt(new_var))
        new_nondet.append(False)

    return np.array(new_f), np.array(new_ferr), np.array(new_nondet)


# @numba.njit(parallel=True)
def calc_all_deltas(series: NDArray[np.float64], use_sum: bool = False) -> NDArray[np.float64]:
    """Calculate all pairwise distances between values in each set,
    assuming the first axis delineates different sets.
    """
    deltas = []
    # for i in range(series.shape[1]):
    for i in numba.prange(series.shape[1]):  # pylint: disable=not-an-iterable
        for j in range(i + 1, series.shape[1]):
            if use_sum:
                deltas.append((series[:, j] + series[:, i]).astype(np.float64))
            else:
                deltas.append((series[:, j] - series[:, i]).astype(np.float64))
    return np.vstack(deltas).T


class LightCurve(Plottable):  # pylint: disable=too-many-public-methods
    """Class that contains all information for a
    single light curve. Associated with a single instrument
    and filter.

    LightCurve should always be (a) as complete as possible and
    (b) sorted by time.
    """

    def __init__(
        self,
        times: Any,  # TODO: somehow enforce time-like
        fluxes: Optional[Iterable[T]] = None,
        flux_errs: Optional[Iterable[T]] = None,
        mags: Optional[Iterable[T]] = None,
        mag_errs: Optional[Iterable[T]] = None,
        zpts: Optional[Iterable[T]] = None,
        upper_limits: Optional[Iterable[bool]] = None,
        filt: Optional[Filter] = None,
        phased: bool = False,
    ) -> None:
        self._phased = phased
        if (filt is not None) and (not isinstance(filt, Filter)):
            raise TypeError("filt must be None or a Filter object!")
        self._filter = filt
        self._ts_cols = ["flux", "flux_unc", "mag", "mag_unc", "zpt"]
        if isinstance(times, pd.DataFrame):
            df = times
            if "time" in df.columns:
                astropy_dt = Time(df["time"], format="mjd").to_datetime()
                # convert to DateTimeIndex
                df = df.set_index(pd.DatetimeIndex(astropy_dt))
            elif "phase" in df.columns:
                # convert to DateTimeIndex
                df = df.set_index(pd.to_timedelta(df["phase"], "D"))
                self._phased = True
            elif isinstance(df.index, pd.TimedeltaIndex):
                self._phased = True

            for k in self._ts_cols:
                if k not in df.columns:
                    df[k] = np.nan * np.ones(len(df.index))

            if "non_detections" not in df.columns:
                df["non_detections"] = np.zeros(len(df.index), dtype=bool)
            else:
                df["non_detections"] = df["non_detections"].astype(bool)

            self._ts = df.loc[:, [*self._ts_cols, "non_detections"]].copy()

        else:
            if fluxes is None:
                fluxes = np.nan * np.ones(len(times))
            if flux_errs is None:
                flux_errs = np.nan * np.ones_like(fluxes)
            if mags is None:
                mags = np.nan * np.ones_like(fluxes)
            if mag_errs is None:
                mag_errs = np.nan * np.ones_like(fluxes)
            if zpts is None:
                zpts = np.nan * np.ones_like(fluxes)
            if upper_limits is None:
                upper_limits = np.zeros_like(fluxes, dtype=bool)

            if self._phased:
                index_col = pd.to_timedelta(times, "D")
            else:
                astropy_times = Time(times, format="mjd")
                astropy_dt = astropy_times.to_datetime()
                index_col = pd.DatetimeIndex(astropy_dt)

            self._ts = pd.DataFrame(
                {
                    "flux": fluxes,
                    "flux_unc": flux_errs,
                    "mag": mags,
                    "mag_unc": mag_errs,
                    "zpt": zpts,
                    "non_detections": upper_limits,
                },
                index=index_col,
            )

        for col in self._ts_cols:
            self._ts[col] = self._ts[col].astype(np.float64)

        self._ts.index.name = "time"
        self._rng = np.random.default_rng()
        self._sort()  # sort by time
        self._complete()  # fills in missing or inconsistent values

        self._image_time_bins = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 10_000])
        self._image_flux_bins = np.array([-1000, -2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 1000])

    def _complete(self, force_update_mags: bool = False, force_update_fluxes: bool = False) -> None:
        """Given zeropoints, fills in missing apparent
        magnitudes from fluxes and vice versa.
        """
        missing_mag = (~np.isnan(self._ts["zpt"])) & (~np.isnan(self._ts["flux"]))
        if not force_update_mags:
            missing_mag = missing_mag & np.isnan(self._ts["mag"])

        sub_table = self._ts[missing_mag]
        if len(sub_table) > 0:
            self._ts.loc[missing_mag, "mag"] = -2.5 * np.log10(sub_table["flux"]) + sub_table["zpt"]

        missing_magunc = ~np.isnan(self._ts["flux"]) & (~np.isnan(self._ts["flux_unc"]))
        if not force_update_mags:
            missing_magunc = missing_magunc & np.isnan(self._ts["mag_unc"])
        sub_table = self._ts[missing_magunc]
        if len(sub_table) > 0:
            self._ts.loc[missing_magunc, "mag_unc"] = (
                2.5 / np.log(10.0) * (sub_table["flux_unc"] / sub_table["flux"])
            )

        missing_flux = (~np.isnan(self._ts["zpt"])) & (~np.isnan(self._ts["mag"]))
        if not force_update_fluxes:
            missing_flux = missing_flux & np.isnan(self._ts["flux"])
        sub_table = self._ts[missing_flux]
        if len(sub_table) > 0:
            self._ts.loc[missing_flux, "flux"] = 10.0 ** (-1.0 * (sub_table["mag"] - sub_table["zpt"]) / 2.5)

        missing_fluxunc = ~np.isnan(self._ts["mag_unc"]) & (~np.isnan(self._ts["flux"]))
        if not force_update_fluxes:
            missing_fluxunc = missing_fluxunc & np.isnan(self._ts["flux_unc"])
        sub_table = self._ts[missing_fluxunc]
        if len(sub_table) > 0:
            self._ts.loc[missing_fluxunc, "flux_unc"] = (np.log(10.0) / 2.5) * (
                sub_table["flux"] * sub_table["mag_unc"]
            )

    def _sort(self) -> None:
        """Sort light curve by time."""
        self._ts.sort_index(inplace=True)

    @property
    def is_phased(self) -> bool:
        """Whether LC is phased or unphased."""
        return self._phased

    @property
    def _mjd(self) -> NDArray[np.float64]:
        """Convert time (index) column to MJDs, in float."""
        if self._phased:
            return self._ts.index.total_seconds().to_numpy() / (24 * 3600)  # type: ignore
        astropy_time = Time(self._ts.index)  # Convert to astropy Time
        return astropy_time.mjd  # type: ignore

    def _convert_to_datetime(self, mjds: NDArray[np.float64]) -> Any:
        """Convert MJD values to datetimes."""
        astropy_time = Time(mjds, format="mjd")
        datetime_values = astropy_time.to_datetime()
        return datetime_values

    def __len__(self) -> int:
        return len(self._ts.index)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LightCurve):
            return False

        return (
            self.detections.equals(other.detections)
            & (self.non_detections.equals(other.non_detections))
            & (self.filter == other.filter)
        )

    def copy(self: LightT) -> LightT:
        """Return a copy of the LightCurve."""
        return copy.deepcopy(self)

    @property
    def times(self) -> NDArray[np.float64]:
        """Return sorted light curve times as
        numpy array, in mean julian date.
        """
        return self._mjd

    @times.setter
    def times(self, new_times: Any) -> None:
        """Replace time values."""
        if self._phased:
            self._ts.set_index(pd.to_timedelta(new_times, "D"), inplace=True)
        else:
            astropy_times = Time(new_times, format="mjd")
            astropy_dt = astropy_times.to_datetime()
            self._ts.set_index(pd.DatetimeIndex(astropy_dt), inplace=True)
        self._sort()

    @property
    def fluxes(self) -> NDArray[np.float64]:
        """Return fluxes of sorted LC as
        numpy array.
        """
        return self._ts["flux"].to_numpy()

    @fluxes.setter
    def fluxes(self, new_fluxes: NDArray[np.float64]) -> None:
        """Replace flux values. Adjusts mags accordingly."""
        try:
            flux_cast = np.array(new_fluxes, dtype=np.float64)
        except ValueError as exc:
            raise TypeError("new_fluxes must be castable to 64-bit float.") from exc
        self._ts["flux"] = flux_cast
        self._ts["mags"] = np.nan
        self._ts["mag_unc"] = np.nan
        self._complete(force_update_mags=True)  # update mags

    @property
    def flux_errors(self) -> NDArray[np.float64]:
        """Return flux uncertainties of sorted LC as
        numpy array.
        """
        return self._ts["flux_unc"].to_numpy()

    @flux_errors.setter
    def flux_errors(self, ferr: NDArray[np.float64]) -> None:
        """Replace flux uncertainty values. Adjusts mags accordingly."""
        try:
            ferr_cast = np.array(ferr, dtype=np.float64)
        except ValueError as exc:
            raise TypeError("ferr must be castable to 64-bit float.") from exc
        self._ts["flux_unc"] = ferr_cast
        self._ts["mags"] = np.nan
        self._ts["mag_unc"] = np.nan
        self._complete(force_update_mags=True)  # update mags

    @property
    def mags(self) -> NDArray[np.float64]:
        """Return magnitudes of sorted LC as
        numpy array.
        """
        return self._ts["mag"].to_numpy()

    @mags.setter
    def mags(self, new_mags: NDArray[np.float64]) -> None:
        """Replace magnitude values. Adjusts fluxes accordingly."""
        try:
            mag_cast = np.array(new_mags, dtype=np.float64)
        except ValueError as exc:
            raise TypeError("new_mags must be castable to 64-bit float.") from exc
        self._ts["mag"] = mag_cast
        self._ts["flux"] = np.nan
        self._ts["flux_unc"] = np.nan
        self._complete(force_update_fluxes=True)  # update fluxes

    @property
    def mag_errors(self) -> NDArray[np.float64]:
        """Return magnitude uncertainties of sorted LC as
        numpy array.
        """
        return self._ts["mag_unc"].to_numpy()

    @mag_errors.setter
    def mag_errors(self, merr: NDArray[np.float64]) -> None:
        """Replace magnitude uncertainty values. Adjusts fluxes accordingly."""
        try:
            merr_cast = np.array(merr, dtype=np.float64)
        except ValueError as exc:
            raise TypeError("merr must be castable to 64-bit float.") from exc
        self._ts["mag_unc"] = merr_cast
        self._ts["flux"] = np.nan
        self._ts["flux_unc"] = np.nan
        self._complete(force_update_fluxes=True)  # first update fluxes

    @property
    def zeropoints(self) -> NDArray[np.float64]:
        """Return zeropoints of sorted LC as
        numpy array.
        """
        return self._ts["zpt"].to_numpy()

    @zeropoints.setter
    def zeropoints(self, new_zpts: NDArray[np.float64], adjust_mags: bool = False) -> None:
        """Replace zeropoint values.
        If adjust_mags is True, recalibrate magnitudes accordingly.
        By default, recalibrates fluxes.
        """
        try:
            zpt_cast = np.array(new_zpts, dtype=np.float64)
        except ValueError as exc:
            raise TypeError("new_zpts must be castable to 64-bit float.") from exc
        self._ts["zpt"] = zpt_cast
        if adjust_mags:
            self._complete(force_update_mags=True)  # adjusts mags
        else:
            self._complete(force_update_fluxes=True)  # adjust fluxes

    @property
    def filter(self) -> Optional[Filter]:
        """Return filter object associated with LightCurve."""
        return self._filter

    @filter.setter
    def filter(self, filt: Filter) -> None:
        """Replace filter associated with LightCurve."""
        if not isinstance(filt, Filter):
            raise TypeError("Input must be Filter object!")
        self._filter = filt

    @property
    def upper_limit_mask(self) -> NDArray[np.bool_]:
        """Return boolean array of upper limits."""
        return self._ts["non_detections"].to_numpy()

    @property
    def non_detections(self) -> pd.DataFrame:
        """Return non-detection observations."""
        return self._ts[self.upper_limit_mask].copy()  # pylint: disable=invalid-unary-operand-type

    @property
    def detections(self) -> pd.DataFrame:
        """Return detection observations."""
        return self._ts[~self.upper_limit_mask].copy()  # pylint: disable=invalid-unary-operand-type

    @property
    def full_time_series(self) -> pd.DataFrame:
        """Return all observations (detections + nondetections) with extra columns
        for filter information. Used in photometry dataframe generation."""
        ts_copy = self._ts.copy()
        ts_copy["filters"] = str(self._filter)
        ts_copy["filt_centers"] = self._filter.center.value if self._filter else None
        ts_copy["filt_widths"] = self._filter.width.value if (self._filter and (self._filter.width is not None)) else None
        return ts_copy

    @property
    def peak(self) -> Any:
        """The brightest observation in light curve.
        Return as dictionary.
        """
        if pd.isna(self._ts["flux"]).all():
            idx = (self.detections["mag"] + self.detections["mag_unc"]).idxmin()
        else:
            idx = (self.detections["flux"] - self.detections["flux_unc"]).idxmax()
        peak_dict = self.detections.loc[idx].to_dict()

        if self._phased:
            peak_dict["time"] = idx.days  # type: ignore
        else:
            astropy_time = Time(idx)  # Convert to astropy Time
            peak_dict["time"] = astropy_time.mjd
        return peak_dict

    def phase(
        self, t0: Optional[float] = None, periodic: bool = False, period: Optional[float] = None
    ) -> None:
        """
        Phases light curve by t0, which is assumed
        to be in days.
        """
        if t0 is None:
            t0 = self.peak["time"]
        
        if self._phased:
            t0_datetime = np.timedelta64(int(t0 * 24 * 60 * 60 * 1e9), "ns")
        else:
            t0_datetime = Time(t0, format="mjd").to_datetime()
        self._ts.set_index(
            self._ts.index - t0_datetime, inplace=True
        )
        if periodic:
            if period is None:
                period = self.calculate_period()
            self._ts.set_index(self._ts.index % period, inplace=True)
        self._phased = True

    def calculate_period(self) -> float:
        """Calculate period of light curve.
        Uses LombScargle periodogram.
        """
        ls = LombScargle(
            self.detections.index,
            self.detections["mag"],
            self.detections["mag_unc"],
        )
        frequency, power = ls.autopower()
        best_freq: float = frequency[np.nanargmax(power)]
        return 1.0 / best_freq

    def truncate(self, min_t: float = -np.inf, max_t: float = np.inf) -> None:
        """Truncate light curve between min_t and max_t days."""
        if max_t < min_t:
            raise ValueError("max_t must be greater than min_t.")
        gind = (self._mjd > max_t) | (self._mjd < min_t)
        self._ts = self._ts[~gind]

    def subsample(self, n_points: int) -> None:
        """Subsamples n_points in light curve."""
        if n_points < 0:
            raise ValueError("n_points must be a nonnegative integer.")
        if n_points < len(self._ts.index):
            keep_ind = self._rng.choice(np.arange(len(self._ts.index)), size=n_points, replace=False)
            self._ts = self._ts.iloc[keep_ind]

    def calibrate_fluxes(self, new_zpt: float) -> None:
        """Calibrate fluxes using zeropoints.
        TODO: GET RID OF THIS
        """
        self._ts["flux"] *= 10.0 ** ((new_zpt - self._ts["zpt"]) / 2.5)
        self._ts["flux_unc"] *= 10.0 ** ((new_zpt - self._ts["zpt"]) / 2.5)
        self._ts["zpt"] = new_zpt
        # shouldn't need any recompletes or sorts

    def plot(
        self,
        ax: Axes,
        formatter: Optional[Formatter] = None,
        mags: bool = True,
    ) -> Axes:
        """Plot a single light curve.
        Face and edge colors determined by formatter.

        If mags is True, plot magnitudes. If not, plot fluxes.
        """
        if formatter is None:
            formatter = Formatter()

        times = self._mjd

        if self._phased:
            ax.set_xlabel("Phase (days)")
        else:
            ax.set_xlabel("Time (MJD)")
        if mags:
            vals = self._ts["mag"].to_numpy()
            val_errs = self._ts["mag_unc"].to_numpy()
            ax.set_ylabel("Magnitude")
            ax.invert_yaxis()
        else:
            vals = self._ts["flux"].to_numpy()
            val_errs = self._ts["flux_unc"].to_numpy()
            ax.set_ylabel("Flux")

        ax.errorbar(
            times[~self.upper_limit_mask],
            vals[~self.upper_limit_mask],
            yerr=val_errs[~self.upper_limit_mask],
            c=formatter.edge_color,
            fmt="none",
            zorder=-1,
        )
        ax.scatter(
            times[~self.upper_limit_mask],
            vals[~self.upper_limit_mask],
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.marker_style,
            s=formatter.marker_size,
            label=str(self._filter),
        )
        # plot non-detections
        ax.scatter(
            times[self.upper_limit_mask],
            vals[self.upper_limit_mask],
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.nondetect_marker_style,
            alpha=formatter.nondetect_alpha,
            s=formatter.nondetect_size,
            zorder=-2,
        )

        return ax

    def add_observations(self: LightT, rows: list[dict[str, Any]]) -> None:
        """Add rows to existing timeseries."""
        if self._phased:
            new_times = [row["time"] for row in rows]
            new_index_td = pd.to_timedelta(new_times, "D")
            new_df = pd.DataFrame.from_records(rows, index=new_index_td)
        else:
            new_times = [self._convert_to_datetime(row["time"]) for row in rows]
            new_index_dt = pd.DatetimeIndex(new_times)
            new_df = pd.DataFrame.from_records(rows, index=new_index_dt)
        new_df.drop(columns="time", inplace=True)
        new_df.index.name = "time"
        self._ts = pd.concat([self._ts, new_df])
        self._sort()
        self._complete()

    def merge_close_times(self, eps: float = 4e-2) -> None:  # TODO: fix for LCs without flux uncertainties
        """Merge times that are close enough together
        (e.g. in the same night). Only in-place at the moment.
        """
        t_diffs = np.diff(self._mjd)
        repeat_idxs = np.abs(t_diffs) < eps  # pylint: disable=no-member

        if np.all(~repeat_idxs):  # no repeats
            return

        repeat_idxs = np.insert(repeat_idxs, 0, False)
        keep_idxs = np.argwhere(~repeat_idxs)

        new_f, new_ferr, new_nondet = update_merged_fluxes(
            keep_idxs, self._ts["flux"], self._ts["flux_unc"]
        )
        self._ts = self._ts[~repeat_idxs]

        self.fluxes = new_f
        self.flux_errors = new_ferr
        self._ts["non_detections"] = new_nondet
        self._complete(force_update_mags=True)
        return

    def merge(self, other: LightT) -> None:
        """Merge other light curve into this one, assuming they are
        from the same instrument and filter.
        """
        if self._filter != other.filter:
            raise ValueError("Filters must be the same to merge light curves!")

        if len(other.detections) > 0:
            # among detections, fill in missing errors/zpts
            nd_mask = self._ts["non_detections"]
            self._ts.loc[~nd_mask, :] = self._ts[~nd_mask].combine_first(other.detections)

            # now for times that exist in both, we want to replace non-detections
            # with detections, where possible
            override_mask1 = other.detections.index.isin(self._ts[nd_mask].index)
            override_mask2 = nd_mask & self._ts.index.isin(other.detections.index)
            self._ts.loc[override_mask2, :] = other.detections[override_mask1]

            # finally, add new times
            nonrepeat_idxs = other.detections.index.difference(self._ts.index)
            if not nonrepeat_idxs.empty:
                non_na_columns_ts = self._ts.columns[~self._ts.isna().all()]
                non_na_columns = other.detections.columns[~other.detections.isna().all()]
                self._ts = pd.concat(
                    [
                        self._ts[non_na_columns_ts],
                        other.detections.loc[nonrepeat_idxs, non_na_columns],  # type: ignore
                    ],
                    ignore_index=False,
                )

        if len(other.non_detections) > 0:
            # update non-detections similarly
            self._ts.loc[nd_mask, :] = self._ts[nd_mask].combine_first(other.non_detections)
            nonrepeat_idxs_2 = other.non_detections.index.difference(self._ts.index)
            if not nonrepeat_idxs_2.empty:
                non_na_columns_ts = self._ts.columns[~self._ts.isna().all()]
                non_na_columns = other.non_detections.columns[~other.non_detections.isna().all()]
                self._ts = pd.concat(
                    [
                        self._ts[non_na_columns_ts],
                        other.non_detections.loc[nonrepeat_idxs_2, non_na_columns],  # type: ignore
                    ],
                    ignore_index=False,
                )

        for col in self._ts_cols:  # ensure all original columns exist
            if col not in self._ts.columns:
                self._ts[col] = np.nan
        self._sort()
        self._complete()

    def pad(self: LightT, fill: dict[str, Any], n_times: int, inplace: bool = False) -> LightT:
        """Extends light curve by padding.
        Currently, pads based on 'fill' dictionary.
        """
        if "time" not in fill:
            raise KeyError("time must be a key in fill")
        if n_times < 0:
            raise ValueError("n_times must be a non-negative integer.")
        if n_times == 0:
            return self

        new_rows = [
            fill,
        ] * n_times

        if inplace:
            self.add_observations(new_rows)
            return self

        lc_copy = self.copy()
        lc_copy.add_observations(new_rows)
        return lc_copy

    def resample(self, num: int = 100, mags: bool = True) -> NDArray[np.float64]:
        """Returns set of num augmented light curves where new
        mags are sampled from distribution defined by mag
        uncertainties.
        """
        if num < 0:
            raise ValueError("num must be a non-negative integer.")
        if mags:
            centers = self._ts["mag"]
            uncs = self._ts["mag_unc"]
        else:
            centers = self._ts["flux"]
            uncs = self._ts["flux_unc"]

        return resample_helper(centers, uncs, num)  # type: ignore

    def absolute(self: LightT, redshift: float) -> LightT:
        """Returns LightCurve with absolute magnitudes.
        Adjusts magnitudes and zeropoints by distance modulus
        and k-corrections.
        """
        #z = redshift * cu.redshift
        #d = z.to(u.Mpc, cu.redshift_distance("Planck15", kind="luminosity"))  # pylint: disable=no-member

        if not self._phased:
            #print("PHASING BY DEFAULT")
            self.phase()

        new_times = self.times / (1.0 + redshift)
        shift_timedelta = pd.to_timedelta(new_times, "D")
        """
        # Calculate the time shift (large value in days)
        shift_days = (redshift * d / const.c).to(u.d).value  # pylint: disable=no-member
        new_times_jd1 = 2_400_000.5 / (1.0 + redshift) - shift_days
        new_times_jd2 = self.times / (1.0 + redshift)

        shift_timedelta = pd.to_timedelta(new_times_jd1 + new_times_jd2, "D")
        """
        # Reconstruct the new Time object using jd1 and jd2
        new_ts = self._ts.copy()
        new_ts.set_index(shift_timedelta, inplace=True)

        k_corr = 2.5 * np.log10(1.0 + redshift)
        distmod = Planck15.distmod(redshift).value
        new_ts["mag"] += (-distmod + k_corr)
        new_ts["zpt"] += (-distmod + k_corr)

        lc_copy = LightCurve(
            times=new_ts,
            filt=self._filter,
            phased=self._phased,
        )
        return cast(LightT, lc_copy)

    def correct_extinction(
        self: LightT, mwebv: Optional[float] = None, coordinates: Optional[Any] = None
    ) -> LightT:
        """Corrects extinction based on LC's filter wavelength. Returns new object."""
        if coordinates is not None:
            mwebv_calced = calc_mwebv(coordinates)

            if (mwebv is not None) and (mwebv_calced != mwebv):
                raise ValueError("Coordinate-calculated MW E(B-V) does not agree with provided MW E(B-V).")

            mwebv = mwebv_calced

        else:
            if mwebv is None:
                raise ValueError("Either coordinates or mwebv must be provided.")

        av_sfd = 2.742 * mwebv
        lc_copy = self.copy()

        if self._filter is None:
            return lc_copy

        # Now figure out how much the magnitude is affected by this dust
        ext_val = extinction.fm07(
            np.array([self._filter.center.to(u.AA).value]), av_sfd, unit="aa"  # pylint: disable=no-member
        )[
            0
        ]  # in magnitudes
        lc_copy.mags -= ext_val
        lc_copy.zeropoints -= ext_val
        return lc_copy

    def convert_to_images(
        self, method: str = "dff-dt", augment: bool = True, **kwargs: Mapping[str, Any]
    ) -> list[Image]:
        """Convert light curve to set of images for ML applications.
        Will return 100 images for augmented LC and 1 for unaugmented.
        """
        if augment:  # augment LC to get many copies before transforming
            series = self.resample(mags=False, num=100)  # TODO: de-hardcode this
            series_t = np.repeat(np.atleast_2d(self._mjd), 100, 0)
        else:
            series = np.atleast_2d(self._ts["flux"])
            series_t = np.atleast_2d(self._mjd)

        if method == "dff-dt":  # dff vs dt plots
            delta_times = calc_all_deltas(series_t)
            delta_fluxes = calc_all_deltas(series)
            avg_fluxes = calc_all_deltas(series, use_sum=True) / 2.0
            dff = delta_fluxes / avg_fluxes
            vals_concat = [
                np.histogram2d(dff[i], delta_times[i], bins=[self._image_flux_bins, self._image_time_bins])[0]
                for i in range(len(dff))
            ]  # TODO: set bins in future
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

        return [Image(vc) for vc in vals_concat]  # type: ignore

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
        cls: Type[LightT],
        file_name: str,
        path: Optional[str] = None,
        archival: bool = False,
    ) -> LightT:
        """Load LightCurve from saved HDF5 table. Automatically
        extracts feature information.
        """
        if path is None:
            paths = list_datasets(file_name, archival)
            if len(paths) > 1:
                raise ValueError("Multiple datasets found in file. Please specify path.")
            path = paths[0]

        if archival:
            ts_astropy = hdf5.read_table_hdf5(file_name, path=path)
            time_series = ts_astropy.to_pandas()
            time_series = time_series.set_index(time_series["time"])
            time_series = time_series.drop(columns="time")
            if "instrument" in ts_astropy.meta:
                if "width" in ts_astropy.meta:
                    extracted_filter = Filter(
                        ts_astropy.meta["instrument"],
                        ts_astropy.meta["band"],
                        ts_astropy.meta["center"] * u.AA,  # pylint: disable=no-member
                        ts_astropy.meta["width"] * u.AA,  # pylint: disable=no-member,
                    )
                else:
                    extracted_filter = Filter(
                        ts_astropy.meta["instrument"],
                        ts_astropy.meta["band"],
                        ts_astropy.meta["center"] * u.AA,  # pylint: disable=no-member
                    )
                return cls(time_series, filt=extracted_filter)
            return cls(time_series)

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

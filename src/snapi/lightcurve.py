"""Contains classes for light curves and filters."""
import copy
from typing import Any, Iterable, Mapping, Optional, Sequence, Type, TypeVar

import astropy.units as u
import numba
import numpy as np
import pandas as pd
from astropy.table.table import QTable
from astropy.timeseries import LombScargle, TimeSeries
from matplotlib.axes import Axes
from numpy.typing import NDArray
#from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from .base_classes import Observer, Plottable
from .formatter import Formatter
from .image import Image
from .utils import list_datasets

T = TypeVar("T", int, float, np.float32)
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


@numba.njit(parallel=True)  # type: ignore
def resample_helper(cen: Sequence[T], unc: Sequence[T], num: int) -> NDArray[np.float32]:
    """numba-enhanced helper to generate many resampled LCs."""
    rng = np.random.default_rng()
    sampled_vals = np.zeros((num, len(cen)), dtype=np.float32)
    for i in numba.prange(num):  # pylint: disable=not-an-iterable
        sampled_vals[i] += rng.normal(loc=cen, scale=unc)
    return sampled_vals


#@numba.njit(parallel=True)  # type: ignore
def update_merged_fluxes(keep_idxs, flux, flux_unc):
    """Update merged fluxes with numba."""
    new_f = []
    new_ferr = []
    for i in range(len(keep_idxs)):  # pylint: disable=not-an-iterable
        if i == len(keep_idxs) - 1:
            repeat_idx_subset = np.arange(keep_idxs[i], len(flux))
        else:
            repeat_idx_subset = np.arange(keep_idxs[i], keep_idxs[i + 1])

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
    ) -> None:
        self._phased = False
        self._filter = filt
        self._ts_cols = ["flux", "flux_unc", "mag", "mag_unc", "zpt"]
        if isinstance(times, QTable):
            if "time" not in times.colnames:
                raise ValueError("TimeSeries must have a 'time' column!")
            for k in self._ts_cols:
                if k not in times.colnames:
                    times[k] = np.nan * np.ones(len(times))
                # convert all columns in self._ts_col to np.float32
                times[k] = times[k].astype(np.float32)
            if "non_detections" not in times.colnames:
                times["non_detections"] = np.zeros(len(times), dtype=bool)

            self._ts = TimeSeries(times[["time", *self._ts_cols, "non_detections"]])  # make copy

        else:
            if fluxes is None:
                fluxes = np.nan * np.ones(len(times))
            else:
                fluxes = np.array(fluxes, dtype=np.float32)
            if flux_errs is None:
                flux_errs = np.nan * np.ones_like(fluxes)
            else:
                flux_errs = np.array(flux_errs, dtype=np.float32)
            if mags is None:
                mags = np.nan * np.ones_like(fluxes)
            else:
                mags = np.array(mags, dtype=np.float32)
            if mag_errs is None:
                mag_errs = np.nan * np.ones_like(fluxes)
            else:
                mag_errs = np.array(mag_errs, dtype=np.float32)
            if zpts is None:
                zpts = np.nan * np.ones_like(fluxes)
            else:
                zpts = np.array(zpts, dtype=np.float32)
            if upper_limits is None:
                upper_limits = np.zeros_like(fluxes, dtype=bool)

            self._ts = TimeSeries(
                {
                    "time": times,
                    "flux": fluxes,
                    "flux_unc": flux_errs,
                    "mag": mags,
                    "mag_unc": mag_errs,
                    "zpt": zpts,
                    "non_detections": upper_limits,
                }
            )
        self._rng = np.random.default_rng()
        self._sort()  # sort by time
        self._complete()  # fills in missing or inconsistent values


    def _complete(self, update_mags: bool=True, update_fluxes: bool=True) -> None:
        """Given zeropoints, fills in missing apparent
        magnitudes from fluxes and vice versa.
        """
        if update_mags:
            # first convert fluxes to missing apparent mags
            missing_mag = (~np.isnan(self._ts["zpt"])) & (~np.isnan(self._ts["flux"]))
            sub_table = self._ts[missing_mag]
            if len(sub_table) > 0:
                self._ts['mag'][missing_mag] = -2.5 * np.log10(sub_table["flux"]) + sub_table["zpt"]

            # uncertainties
            missing_magunc = (
                ~np.isnan(self._ts["flux"]) & (~np.isnan(self._ts["flux_unc"]))
            )
            sub_table = self._ts[missing_magunc]
            if len(sub_table) > 0:
                self._ts['mag_unc'][missing_magunc] = (
                    2.5 / np.log(10.0) * (sub_table["flux_unc"] / sub_table["flux"])
                )

        if update_fluxes:
            # then convert mags to missing fluxes
            missing_flux = (~np.isnan(self._ts["zpt"])) & (~np.isnan(self._ts["mag"]))
            sub_table = self._ts[missing_flux]
            if len(sub_table) > 0:
                self._ts["flux"][missing_flux] = 10.0 ** (-1.0 * (sub_table["mag"] - sub_table["zpt"]) / 2.5)

            # uncertainties
            missing_fluxunc = (
                (~np.isnan(self._ts["mag_unc"]) & (~np.isnan(self._ts["flux"])))
            )
            sub_table = self._ts[missing_fluxunc]
            if len(sub_table) > 0:
                self._ts["flux_unc"][missing_fluxunc] = (
                    (np.log(10.0) / 2.5) * (sub_table["flux"] * sub_table["mag_unc"])
                )

    def _sort(self) -> None:
        """Sort light curve by time."""
        self._ts.sort()

    def __len__(self) -> int:
        return len(self._ts)

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
        """Replace flux values. Adjusts mags accordingly."""
        self._ts.replace_column("flux", new_fluxes)
        self._ts['mags'] = np.nan
        self._ts['mag_unc'] = np.nan
        self._complete(update_fluxes=False) # update mags

    @property
    def flux_errors(self) -> Any:
        """Return flux uncertainties of sorted LC as
        numpy array.
        """
        return self._ts["flux_unc"].value  # pylint: disable=no-member; type: ignore[no-any-return]

    @flux_errors.setter
    def flux_errors(self, ferr: NDArray[np.float32]) -> None:
        """Replace flux uncertainty values. Adjusts mags accordingly."""
        self._ts.replace_column("flux_unc", ferr)
        self._ts['mags'] = np.nan
        self._ts['mag_unc'] = np.nan
        self._complete(update_fluxes=False) # update mags

    @property
    def mags(self) -> Any:
        """Return magnitudes of sorted LC as
        numpy array.
        """
        return self._ts["mag"].value  # pylint: disable=no-member; type: ignore

    @mags.setter
    def mags(self, new_mags: NDArray[np.float32]) -> None:
        """Replace magnitude values. Adjusts fluxes accordingly."""
        self._ts.replace_column("mag", new_mags)
        self._ts['flux'] = np.nan
        self._ts['flux_unc'] = np.nan
        self._complete(update_mags=False) # update fluxes

    @property
    def mag_errors(self) -> Any:
        """Return magnitude uncertainties of sorted LC as
        numpy array.
        """
        return self._ts["mag_unc"].value  # pylint: disable=no-member; type: ignore[no-any-return]

    @mag_errors.setter
    def mag_errors(self, merr: NDArray[np.float32]) -> None:
        """Replace magnitude uncertainty values. Adjusts fluxes accordingly."""
        self._ts.replace_column("mag_unc", merr)
        self._ts['flux'] = np.nan
        self._ts['flux_unc'] = np.nan
        self._complete(update_mags=False) # first update fluxes

    @property
    def zeropoints(self) -> Any:
        """Return zeropoints of sorted LC as
        numpy array.
        """
        return self._ts["zpt"].value  # pylint: disable=no-member; type: ignore

    @zeropoints.setter
    def zeropoints(self, new_zpts: NDArray[np.float32], adjust_mags: bool=False) -> None:
        """Replace zeropoint values.
        If adjust_mags is True, recalibrate magnitudes accordingly.
        By default, recalibrates fluxes.
        """
        self._ts.replace_column("zpt", new_zpts)
        if adjust_mags:
            self._ts['mag'] = np.nan
            self._ts['mag_unc'] = np.nan
            self._complete(update_fluxes=False) # adjusts mags
        else:
            self._ts['flux'] = np.nan
            self._ts['flux_unc'] = np.nan
            self._complete(update_mags=False) # adjust fluxes

    @property
    def filter(self) -> Optional[Filter]:
        """Return filter object associated with LightCurve."""
        return self._filter

    @property
    def upper_limit_mask(self) -> Any:
        """Return boolean array of upper limits."""
        return self._ts["non_detections"].value  # pylint: disable=no-member; type: ignore

    @property
    def non_detections(self) -> TimeSeries:
        """Return non-detection observations."""
        return self._ts[self.upper_limit_mask].copy()  # pylint: disable=invalid-unary-operand-type

    @property
    def detections(self) -> TimeSeries:
        """Return detection observations."""
        return self._ts[~self.upper_limit_mask].copy()  # pylint: disable=invalid-unary-operand-type

    @property
    def peak(self) -> Any:
        """The brightest observation in light curve.
        Return as dictionary.
        """
        return self._ts[np.nanargmax(self._ts["flux"] - self._ts["flux_unc"])]  # type: ignore[no-any-return]

    def phase(
        self, t0: Optional[float] = None, periodic: bool = False, period: Optional[float] = None
    ) -> None:
        """
        Phases light curve by t0, which is assumed
        to be in days.
        """
        if t0 is None:
            t0 = self.peak["time"]
        self._ts["time"] -= t0 * u.d  # pylint: disable=no-member

        if periodic:
            if period is None:
                period = self.calculate_period()
            self._ts["time"] %= period
        self._phased = True

    def calculate_period(self) -> float:
        """Calculate period of light curve.
        Uses LombScargle periodogram.
        """
        ls = LombScargle(
            self.detections["time"],
            self.detections["mag"],
            self.detections["mag_unc"],
        )
        frequency, power = ls.autopower()
        best_freq: float = frequency[np.nanargmax(power)]
        return 1.0 / best_freq

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

    def calibrate_fluxes(self, new_zpt: float) -> None:
        """Calibrate fluxes using zeropoints."""
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
        if self._phased:
            ax.set_xlabel("Phase (days)")
        else:
            ax.set_xlabel("Time (MJD)")
        if mags:
            vals = self.mags
            val_errs = self.mag_errors
            ax.set_ylabel("Magnitude")
            ax.invert_yaxis()
        else:
            vals = self.fluxes
            val_errs = self.flux_errors
            ax.set_ylabel("Flux")

        ax.errorbar(
            self.times[~self.upper_limit_mask],
            vals[~self.upper_limit_mask],
            yerr=val_errs[~self.upper_limit_mask],
            c=formatter.edge_color,
            fmt="none",
        )
        ax.scatter(
            self.times[~self.upper_limit_mask],
            vals[~self.upper_limit_mask],
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.marker_style,
            s=formatter.marker_size,
            label=str(self._filter),
            zorder=10,  # Ensure scatter plot is on top of errorbars
        )
        # plot non-detections
        ax.scatter(
            self.times[self.upper_limit_mask],
            vals[self.upper_limit_mask],
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.nondetect_marker_style,
            alpha=formatter.nondetect_alpha,
            s=formatter.nondetect_size,
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
        t_diffs = np.diff(self._ts["time"].mjd)
        repeat_idxs = np.abs(t_diffs) < eps # pylint: disable=no-member

        if np.all(~repeat_idxs): # no repeats
            return
        
        repeat_idxs = np.insert(repeat_idxs, 0, False)
        keep_idxs = np.argwhere(~repeat_idxs)

        new_f, new_ferr = update_merged_fluxes(keep_idxs, self._ts["flux"], self._ts["flux_unc"])
        self._ts.remove_rows(np.argwhere(repeat_idxs).ravel())
        self.fluxes = new_f
        self.flux_errors = new_ferr
        self._complete()
        return

    def merge(self, other: LightT) -> None:
        """Merge other light curve into this one, assuming they are
        from the same instrument and filter.
        """
        if self._filter != other.filter:
            raise ValueError("Filters must be the same to merge light curves!")

        ts_df = self._ts.to_pandas()
        
        if len(other.detections) > 0:
            other_det_df = other.detections.to_pandas()

            # among detections, fill in missing errors/zpts
            ts_df.loc[~ts_df['non_detections'], :] = ts_df[~ts_df["non_detections"]].combine_first(other_det_df)

            # now for times that exist in both, we want to replace non-detections
            # with detections, where possible
            override_mask1 = other_det_df.index.isin(ts_df[ts_df["non_detections"]].index)
            override_mask2 = ts_df["non_detections"] & ts_df.index.isin(other_det_df.index)
            ts_df.loc[override_mask2, :] = other_det_df[override_mask1]

            # finally, add new times
            nonrepeat_idxs = other_det_df.index.difference(ts_df.index)
            if not nonrepeat_idxs.empty:
                non_na_columns_ts = ts_df.columns[~ts_df.isna().all()]
                non_na_columns = other_det_df.columns[~other_det_df.isna().all()]
                ts_df = pd.concat([ts_df[non_na_columns_ts], other_det_df.loc[nonrepeat_idxs, non_na_columns]], ignore_index=False)

        if len(other.non_detections) > 0:
            other_nondet_df = other.non_detections.to_pandas()

            # update non-detections similarly
            ts_df.loc[ts_df['non_detections'],:] = ts_df[ts_df["non_detections"]].combine_first(other_nondet_df)
            nonrepeat_idxs_2 = other_nondet_df.index.difference(ts_df.index)
            if not nonrepeat_idxs_2.empty:
                non_na_columns_ts = ts_df.columns[~ts_df.isna().all()]
                non_na_columns = other_nondet_df.columns[~other_nondet_df.isna().all()]
                ts_df = pd.concat([ts_df[non_na_columns_ts], other_nondet_df.loc[nonrepeat_idxs_2, non_na_columns]], ignore_index=False)
        
        for col in self._ts.columns: # ensure all original columns exist
            if (col != 'time') and (col not in ts_df.columns):
                ts_df[col] = np.nan
        self._ts = TimeSeries.from_pandas(ts_df).filled(np.nan)
        self._sort()
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
        table = self._ts.copy()
        if self._filter is not None:
            table.meta["instrument"] = self._filter.instrument
            table.meta["band"] = self._filter.band
            table.meta["center"] = self._filter.center.value  # in Angstroms
            if self._filter.width is not None:
                table.meta["width"] = self._filter.width.value
        if append:
            table.write(file_name, format="hdf5", path=path, append=True, serialize_meta=True)
        else:
            table.write(file_name, format="hdf5", path=path, overwrite=True, serialize_meta=True)

    @classmethod
    def load(cls: Type[LightT], file_name: str, path: Optional[str] = None) -> LightT:
        """Load LightCurve from saved HDF5 table. Automatically
        extracts feature information.
        """
        if path is None:
            paths = list_datasets(file_name)
            if len(paths) > 1:
                raise ValueError("Multiple datasets found in file. Please specify path.")
            path = paths[0]

        time_series = TimeSeries.read(
            file_name, format="hdf5", path=path, time_column="time", time_format="mjd"
        )
        if "instrument" in time_series.meta:
            try:
                extracted_filter = Filter(
                    time_series.meta["instrument"],
                    time_series.meta["band"],
                    time_series.meta["center"] * u.AA,  # pylint: disable=no-member
                    time_series.meta["width"] * u.AA,  # pylint: disable=no-member,
                )
            except KeyError:
                extracted_filter = Filter(
                    time_series.meta["instrument"],
                    time_series.meta["band"],
                    time_series.meta["center"] * u.AA,  # pylint: disable=no-member
                )
            return cls(time_series, filt=extracted_filter)
        return cls(time_series)

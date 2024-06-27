import copy
from typing import Any, Mapping, Optional, Self, Sequence, TypeVar

import astropy.units as u
import numba
import numpy as np
from astropy.timeseries import TimeSeries
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from .base_classes import Base, Plottable
from .image import Image

T = TypeVar("T", int, float)


class Filter(Base):
    """Contains instrument and filter information."""

    def __init__(self, instrument: str, center: u.Quantity, width: Optional[u.Quantity]):
        self._instrument = instrument
        self._center = center
        self._width = width


@numba.njit(parallel=True)  # type: ignore
def resample_helper(cen: Sequence[T], unc: Sequence[T], num: int) -> NDArray[np.float32]:
    """numba-enhanced helper to generate many resampled LCs."""
    rng = np.random.default_rng()
    sampled_vals = np.zeros((num, len(cen)), dtype=np.float32)
    for i in numba.prange(num):  # pylint: disable=not-an-iterable
        sampled_vals[i] += rng.normal(loc=cen, scale=unc)
    return sampled_vals


@numba.njit(parallel=True)  # type: ignore
def update_merged_fluxes(keep_idxs, f, f_unc):
    """Update merged fluxes with numba."""
    new_f = []
    new_ferr = []
    for i in numba.prange(len(keep_idxs)):  # pylint: disable=not-an-iterable
        if i == len(keep_idxs) - 1:
            repeat_idx_subset = np.arange(keep_idxs[i], keep_idxs[i + 1])
        else:
            repeat_idx_subset = np.arange(keep_idxs[i], len(f))

        weights = 1.0 / f_unc[repeat_idx_subset] ** 2
        new_f.append(np.average(f[repeat_idx_subset], weights=weights))
        new_var = np.var(f[repeat_idx_subset])
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
    """

    def __init__(
        self,
        times: Sequence[u.Quantity],  # TODO: somehow enforce time-like
        fluxes: Optional[Sequence[T]],
        flux_errs: Optional[Sequence[T]],
        mags: Optional[Sequence[T]],
        mag_errs: Optional[Sequence[T]],
        abs_mags: Optional[Sequence[T]],
        abs_mag_errs: Optional[Sequence[T]],
        zpts: Optional[Sequence[T]],
        filt: Optional[Filter],
    ) -> None:
        self._filter = filt

        self._ts: TimeSeries = TimeSeries(
            {
                "time": times,
                "flux": fluxes,
                "flux_unc": flux_errs,
                "app_mag": mags,
                "app_mag_unc": mag_errs,
                "abs_mag": abs_mags,
                "abs_mag_unc": abs_mag_errs,
                "zpt": zpts,
            }
        )

        self._rng = np.random.default_rng()

    @property
    def times(self) -> Any:
        """Return sorted light curve times as
        numpy array, in mean julian date.
        """
        return self._ts["time"].mjd.astype(np.float32)  # pylint: disable=no-member; type: ignore

    @times.setter
    def times(self, t: NDArray[u.Quantity]) -> None:
        """Replace time values."""
        self._ts.replace_column("time", t)

    @property
    def fluxes(self) -> Any:
        """Return fluxes of sorted LC as
        numpy array.
        """
        return self._ts["flux"].value  # pylint: disable=no-member; type: ignore

    @fluxes.setter
    def fluxes(self, f: NDArray[np.float32]) -> None:
        """Replace flux values."""
        self._ts.replace_column("flux", f)

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

    @property
    def peak(self) -> dict[str, Any]:
        """The brightest observation in light curve.
        Return as dictionary.
        """
        return self._ts[np.argmax(self._ts["flux"])].as_dict()  # type: ignore[no-any-return]

    def phase(self, t0: Optional[float] = None) -> None:
        """
        Phases light curve by t0, which is assumed
        to be in days.
        """
        if t0 is None:
            t0 = self.peak["time"]
        self._ts["time"] -= t0 * u.d  # pylint: disable=no-member

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

    def plot(self, ax: Axes) -> Axes:
        """Plot a single light curve."""
        return ax

    def add_observations(self, rows: list[dict[str, Any]]) -> Self:
        """Add rows to existing timeseries."""
        # TODO: accomodate different formats
        for row in rows:
            self._ts.add_row(row)
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

    def pad(self, n_times: int, inplace: bool = False) -> Self:
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

    def resample(self, mags: bool = False, n: int = 100) -> NDArray[np.float32]:
        """Returns set of n augmented light curves where new
        fluxes are sampled from distribution defined by flux
        uncertainties.
        """
        if mags:
            centers = self._ts["abs_mag"]  # TODO: abs or app mags?
            uncs = self._ts["abs_mag_unc"]
        else:
            centers = self._ts["flux"]
            uncs = self._ts["flux_unc"]

        return resample_helper(centers, uncs, n)  # type: ignore

    def convert_to_images(
        self, method: str = "dff-ft", augment: bool = True, **kwargs: Mapping[str, Any]
    ) -> list[Image]:
        """Convert light curve to set of images for ML applications.
        Will return 200 images for augmented LC and 2 for unaugmented.
        """
        if augment:  # augment LC to get many copies before transforming
            series = self.resample(mags=False, n=100)  # TODO: de-hardcode this
            series_t = np.repeat(np.atleast_2d(self._ts["time"]), 100, 0)
        else:
            series = np.atleast_2d(self._ts["flux"])
            series_t = np.atleast_2d(self._ts["time"])

        if method == "dff-dt":  # dff vs dt plots
            dt = calc_all_deltas(self._ts["time"])
            df = calc_all_deltas(series)
            f = calc_all_deltas(series, use_sum=True)
            dff = df / f
            vals_concat, _, _ = np.histogram2d(dt, dff)  # TODO: set bins in future
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

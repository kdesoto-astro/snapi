"""Stores Photometry class and helper functions."""
import copy
from typing import List, Optional, Set, Tuple, Type, TypeVar

import astropy.units as u
import george
import numpy as np
import scipy
from astropy.time import Time
from astropy.timeseries import TimeSeries
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .base_classes import MeasurementSet, Plottable
from .lightcurve import LightCurve
from .utils import list_datasets

PhotT = TypeVar("PhotT", bound="Photometry")


def generate_gp(
    gp_vals: NDArray[np.float32], gp_errs: NDArray[np.float32], stacked_data: NDArray[np.float32]
) -> george.GP:
    """Generate a Gaussian Process object.

    Parameters
    ----------
    gp_vals: NDArray[np.float32]
        the values of the Gaussian Process
    gp_errs: NDArray[np.float32]
        the errors of the Gaussian Process
    stacked_data: NDArray[np.float32]
        the stacked data

    Returns
    -------
    george.GP
        the Gaussian Process object
    """
    kernel = np.var(gp_vals) * george.kernels.ExpSquaredKernel([100, 1], ndim=2)
    gaussian_process = george.GP(kernel)
    gaussian_process.compute(stacked_data, gp_errs)

    def neg_ln_like(params: NDArray[np.float32]) -> float:
        """Return negative log likelihood of GP."""
        gaussian_process.set_parameter_vector(params)
        return -gaussian_process.log_likelihood(gp_vals)  # type: ignore[no-any-return]

    def grad_neg_ln_like(params: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return gradient of negative log likelihood of GP."""
        gaussian_process.set_parameter_vector(params)
        return -gaussian_process.grad_log_likelihood(gp_vals)  # type: ignore[no-any-return]

    result = scipy.optimize.minimize(
        neg_ln_like, gaussian_process.get_parameter_vector(), jac=grad_neg_ln_like
    )
    gaussian_process.set_parameter_vector(result.x)

    return gaussian_process


class Photometry(MeasurementSet, Plottable):
    """Contains collection of LightCurve objects."""

    def __init__(self, lcs: Optional[set[LightCurve]] = None) -> None:
        if lcs is None:
            self._lightcurves: Set[LightCurve] = set()
        else:
            self._lightcurves = copy.deepcopy(lcs)
        self._rng: np.random.Generator = np.random.default_rng()
        self._ts: TimeSeries = None

        self._generate_time_series()

    def _time_series_helper(
        self,
    ) -> Tuple[List[str], List[float], List[float]]:
        """Return arrays of limiting magnitudes, filter centers, and filter widths."""
        filt_centers = []
        filt_widths = []
        filters = []
        lc_list = list(self._lightcurves)
        for light_curve in lc_list:
            if light_curve.filter is None:
                filters.extend(
                    [
                        "none",
                    ]
                    * len(light_curve.times)
                )
            else:
                filters.extend(
                    [
                        str(light_curve.filter),
                    ]
                    * len(light_curve.times)
                )
            if light_curve.filter is None or light_curve.filter.center is None:
                filt_centers.extend(
                    [
                        np.nan,
                    ]
                    * len(light_curve.times)
                )
            else:
                filt_centers.extend(
                    [
                        light_curve.filter.center.value,
                    ]
                    * len(light_curve.times)
                )
            if light_curve.filter is None or light_curve.filter.width is None:
                filt_widths.extend(
                    [
                        np.nan,
                    ]
                    * len(light_curve.times)
                )
            else:
                filt_widths.extend(
                    [
                        light_curve.filter.width.value,
                    ]
                    * len(light_curve.times)
                )

        return (filters, filt_centers, filt_widths)

    def _generate_time_series(self) -> None:
        """Generate time series from set of light curves."""
        if len(self._lightcurves) == 0:
            return None
        times = Time(
            np.concatenate([lc.times for lc in self._lightcurves]),
            format="mjd",
            scale="utc",
        )
        mags = np.concatenate([lc.mags for lc in self._lightcurves])
        mag_errors = np.concatenate([lc.mag_errors for lc in self._lightcurves])
        fluxes = np.concatenate([lc.fluxes for lc in self._lightcurves])
        flux_errors = np.concatenate([lc.flux_errors for lc in self._lightcurves])
        non_detections = np.concatenate([lc.upper_limit_mask for lc in self._lightcurves])

        filters, filt_centers, filt_widths = self._time_series_helper()

        self._ts = TimeSeries(
            {
                "time": times,
                "flux": fluxes,
                "flux_err": flux_errors,
                "mag": mags,
                "mag_err": mag_errors,
                "non_detections": non_detections,
                "filters": filters,
                "filt_centers": filt_centers,
                "filt_widths": filt_widths,
            }
        )
        self._ts.sort("time")
        return None

    @property
    def times(self) -> NDArray[np.float32]:
        """Return times of observations.

        Returns
        -------
        NDArray[np.float32]
        the times of observations
        """
        return self._ts["time"].to("mjd").value  # type: ignore[no-any-return]

    @property
    def mags(self) -> NDArray[np.float32]:
        """Return magnitudes of observations.

        Returns
        -------
        NDArray[np.float32]
        the magnitudes of observations
        """
        return self._ts["mag"].value  # type: ignore[no-any-return]

    @property
    def fluxes(self) -> NDArray[np.float32]:
        """Return fluxes of observations.

        Returns
        -------
        NDArray[np.float32]
        the fluxes of observations
        """
        return self._ts["flux"].value  # type: ignore[no-any-return]

    @property
    def mag_errors(self) -> NDArray[np.float32]:
        """Return magnitude errors of observations.

        Returns
        -------
        NDArray[np.float32]
        the magnitude errors of observations
        """
        return self._ts["mag_err"].value  # type: ignore[no-any-return]

    @property
    def flux_errors(self) -> NDArray[np.float32]:
        """Return flux errors of observations.

        Returns
        -------
        NDArray[np.float32]
        the flux errors of observations
        """
        return self._ts["flux_err"].value  # type: ignore[no-any-return]

    @property
    def filters(self) -> NDArray[str]:  # type: ignore
        """Return filters of observations.

        Returns
        -------
        List[str]
        the filters of observations
        """
        return self._ts["filters"]  # type: ignore[no-any-return]

    @property
    def lim_mags(self) -> NDArray[np.float32]:
        """Return limiting magnitudes of observations.

        Returns
        -------
        NDArray[np.float32]
        the limiting magnitudes of observations
        """
        return self._ts["lim_mags"].value  # type: ignore[no-any-return]

    @property
    def time_series(self) -> TimeSeries:
        """Return the time series object.

        Returns
        -------
        TimeSeries
        the time series object
        """
        return self._ts.copy()

    @property
    def light_curves(self) -> Set[LightCurve]:
        """Return copy of set of light curves, as
        to not impact the underlying LCs.

        Returns
        -------
        Set[LightCurve]
        the set of light curves
        """
        return copy.deepcopy(self._lightcurves)

    def filter_by_instrument(self: PhotT, instrument: str) -> PhotT:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'

        Parameters
        ----------
        instrument: str
        the name of the instrument to filter by

        Returns
        -------
        Photometry
        the Photometry object with only the
        desired instrument's measurements
        """
        return self  # TODO

    def plot(self, ax: Axes) -> Axes:
        """Plots the collection of light curves.

        Parameters
        ----------
        ax: Axes
        the axes to plot on

        Returns
        -------
        Axes
        the axes with the light curves plotted
        """
        return ax  # TODO

    def add_lightcurve(self, light_curve: LightCurve) -> None:
        """Add a light curve to the set of photometry.

        Parameters
        ----------
        light_curve: LightCurve
            the light curve to add to the set of photometry
        """
        for lc in self._lightcurves:
            if lc.filter == light_curve.filter:
                lc.merge(light_curve)
                self._generate_time_series()
                return None
        self._lightcurves.add(copy.deepcopy(light_curve))
        self._generate_time_series()  # update time series
        return None

    def remove_lightcurve(self, light_curve: LightCurve) -> None:
        """Remove a light curve from the set of photometry.

        Parameters
        ----------
        lc: LightCurve
            the light curve to remove from the set of photometry
        """
        self._lightcurves.remove(light_curve)
        self._generate_time_series()  # update time series

    def tile(self, n_lightcurves: int) -> None:
        """Duplicate light curves until desired number
        is reached. Randomly selects from the set
        without replacement.

        If there are more than n_lightcurves already in set,
        raise ValueError.

        Parameters
        ----------
        n_lightcurves: int
            the number of light curves to have at end
        """
        if len(self._lightcurves) > n_lightcurves:
            raise ValueError("Number of light curves exceeds the desired limit.")
        while len(self._lightcurves) < n_lightcurves:
            num_new = min(len(self._lightcurves), n_lightcurves - len(self._lightcurves))
            sampled_idxs = np.random.choice(len(self._lightcurves), num_new, replace=False)
            new_lcs = copy.deepcopy(np.asarray(self._lightcurves))
            self._lightcurves.update(new_lcs[sampled_idxs])
        self._generate_time_series()  # update time series

    def __len__(self) -> int:
        """
        Length of the set of light curves.

        Returns
        -------
        int
        the number of light curves in the set
        """
        return len(self._lightcurves)

    def _dense_lc_helper(
        self,
        gp_vals: NDArray[np.float32],
        gp_errs: NDArray[np.float32],
        stacked_data: NDArray[np.float32],
        max_spacing: float,
        filt_to_int: dict[str, int],
        nfilts: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Helper function for dense_array method."""
        gaussian_process = generate_gp(gp_vals, gp_errs, stacked_data)

        keep_idxs = np.where(np.abs(np.diff(self.times)) > max_spacing * u.d)[0]  # pylint: disable=no-member
        dense_times = self.times[keep_idxs]

        # map reduced indices to original indices
        idx_map = np.zeros(len(self.times), dtype=bool)
        idx_map[keep_idxs] = True
        idx_map = np.cumsum(idx_map) - 1

        x_pred = np.zeros((len(dense_times) * nfilts, 2))

        for j in np.arange(nfilts):
            x_pred[j::nfilts, 0] = dense_times
            x_pred[j::nfilts, 1] = j

        pred, pred_var = gaussian_process.predict(gp_vals, x_pred, return_var=True)

        dense_arr = np.zeros((len(dense_times), nfilts * 3 + 4), dtype=np.float32)
        dense_arr[:, 0] = dense_times
        dense_arr[:, 1 + 2 * nfilts] = 1

        for filt in np.unique(self.filters):
            filt_int = filt_to_int[filt]
            dense_arr[:, 1 + filt_int] = pred[x_pred[:, 1] == filt_int]
            dense_arr[:, 1 + nfilts + filt_int] = np.sqrt(pred_var[x_pred[:, 1] == filt_int])

        return dense_arr, idx_map

    def _generate_gp_vals_and_errs(self) -> Tuple[NDArray[np.float32], NDArray[np.float32], bool, List[int]]:
        """Generate GP values and errors for use in dense array."""
        # choose between mags or fluxes based on which is more complete
        use_fluxes = len(self.fluxes[~np.isnan(self.fluxes)]) > len(self.mags[~np.isnan(self.mags)])

        # first, remove all rows in time series with nans in ANY field
        nan_mask = np.zeros(len(self._ts), dtype=bool)

        # Iterate over each column and update the mask
        for col in ["flux", "flux_err", "mag", "mag_err"]:
            nan_mask |= ~np.isfinite(self._ts[col])
        if not use_fluxes:
            nan_mask |= ~np.isfinite(self._ts["lim_mags"])

        masked_rows = self._ts[nan_mask]
        self._ts.remove_rows(masked_rows)

        if use_fluxes:
            gp_vals = self.fluxes
            gp_errs = self.flux_errors
        else:
            gp_vals = self.mags - self.lim_mags
            gp_errs = self.mag_errors

        return gp_vals, gp_errs, use_fluxes, masked_rows

    def dense_array(self, max_spacing: float = 4e-2, error_mask: float = 1.0) -> NDArray[np.float32]:
        """Return photometry as dense array for use
        in machine learning models.

        Parameters
        ----------
        max_spacing: np.float32
            the maximum time spacing between observations
        error_mask: np.float32
            the value to replace zero or negative errors with

        Returns
        -------
        NDArray[np.float32]
        the photometry as a dense array
        """
        gp_vals, gp_errs, use_fluxes, masked_rows = self._generate_gp_vals_and_errs()

        # map unique filts to integers
        filt_to_int = {filt: i for i, filt in enumerate(np.unique(self.filters))}
        filts_as_ints = np.array([filt_to_int[filt] for filt in self.filters])  # TODO: more efficient?

        nfilts = len(self._lightcurves)
        stacked_data = np.vstack([self.times, filts_as_ints]).T

        dense_arr, idx_map = self._dense_lc_helper(
            gp_vals, gp_errs, stacked_data, max_spacing, filt_to_int, nfilts
        )

        for filt in np.unique(self.filters):
            filt_int = filt_to_int[filt]
            filt_mask = self.filters == filt
            sub_series = self._ts[filt_mask]
            sub_idx_map = idx_map[filt_mask]

            if not use_fluxes:
                dense_arr[:, 1 + filt_int] += sub_series["lim_mags"][0]

            # fill in true values
            dense_arr[sub_idx_map, 1 + filt_int] = gp_vals[filt_mask]
            dense_arr[sub_idx_map, 1 + nfilts + filt_int] = gp_errs[filt_mask]
            dense_arr[sub_idx_map, 1 + 2 * nfilts + filt_int] = 0
            dense_arr[sub_idx_map, 1 + 3 * nfilts] = sub_series["lim_mags"][0]
            dense_arr[sub_idx_map, 2 + 3 * nfilts] = sub_series["filt_centers"][0]
            dense_arr[sub_idx_map, 2 + 3 * nfilts] = sub_series["filt_widths"][0]

            # fix broken errors - usually if no points in that band
            mask = dense_arr[:, 1 + nfilts + filt_int] <= 0.0
            dense_arr[mask, 1 + nfilts + filt_int] = error_mask
            dense_arr[mask, 1 + 2 * nfilts + filt_int] = 1

        # readd nan rows at end back to time series
        for row in masked_rows:
            self._ts.add_row(row)
        self._ts.sort("time")

        return dense_arr

    def save(self, filename: str, path: str = "photometry") -> None:
        """Save the Photometry object to and HDF5 file.

        Parameters
        ----------
        filename: str
            the filename to save the Photometry object to
        """
        append = False
        for lc in self._lightcurves:
            if not append:
                lc.save(filename, path=f"{path}/{str(lc.filter)}")
                append = True
            else:
                lc.save(filename, path=f"{path}/{str(lc.filter)}", append=True)

    @classmethod
    def load(cls: Type[PhotT], filename: str, path: str = "photometry") -> PhotT:
        """Load the Photometry object from an HDF5 file.

        Parameters
        ----------
        filename: str
            the filename to load the Photometry object from
        """
        lcs = set()
        # get all paths that start with path
        lc_path_list = [lc_path for lc_path in list_datasets(filename) if lc_path.startswith(path)]

        for lc_path in lc_path_list:
            lc = LightCurve.load(filename, path=lc_path)
            lcs.add(lc)

        phot: PhotT = cls(lcs)
        return phot

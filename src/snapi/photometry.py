"""Stores Photometry class and helper functions."""
import copy
from typing import Any, Optional, Tuple, Type, TypeVar

import george
import numpy as np
import pandas as pd
import scipy
from astropy.time import Time
from astropy.timeseries import LombScargleMultiband
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .base_classes import MeasurementSet, Plottable
from .formatter import Formatter
from .lightcurve import Filter, LightCurve
from .utils import list_datasets

PhotT = TypeVar("PhotT", bound="Photometry")


def generate_gp(
    gp_vals: NDArray[np.float64], gp_errs: NDArray[np.float64], stacked_data: NDArray[np.float64]
) -> george.GP:
    """Generate a Gaussian Process object.

    Parameters
    ----------
    gp_vals: NDArray[np.float64]
        the values of the Gaussian Process
    gp_errs: NDArray[np.float64]
        the errors of the Gaussian Process
    stacked_data: NDArray[np.float64]
        the stacked data

    Returns
    -------
    george.GP
        the Gaussian Process object
    """
    kernel = np.var(gp_vals) * george.kernels.ExpSquaredKernel([100, 1], ndim=2)
    gaussian_process = george.GP(kernel)
    gaussian_process.compute(stacked_data, gp_errs)

    def neg_ln_like(params: NDArray[np.float64]) -> float:
        """Return negative log likelihood of GP."""
        gaussian_process.set_parameter_vector(params)
        return -gaussian_process.log_likelihood(gp_vals)  # type: ignore[no-any-return]

    def grad_neg_ln_like(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return gradient of negative log likelihood of GP."""
        gaussian_process.set_parameter_vector(params)
        return -gaussian_process.grad_log_likelihood(gp_vals)  # type: ignore[no-any-return]

    result = scipy.optimize.minimize(
        neg_ln_like, gaussian_process.get_parameter_vector(), jac=grad_neg_ln_like
    )
    gaussian_process.set_parameter_vector(result.x)

    return gaussian_process


class Photometry(MeasurementSet, Plottable):  # pylint: disable=too-many-public-methods
    """Contains collection of LightCurve objects."""

    def __init__(self, lcs: Optional[list[LightCurve]] = None) -> None:
        super().__init__()
        self._lightcurves: list[str] = []
        if lcs is None:
            self._phased = None
        else:
            phot_phased = None
            for lc in lcs:
                if not isinstance(lc, LightCurve):
                    raise TypeError("All elements of 'lcs' must be a LightCurve!")
                if phot_phased is None:
                    phot_phased = lc.is_phased
                # ensure all are either phased or unphased
                if (phot_phased and not lc.is_phased) or (not phot_phased and lc.is_phased):
                    raise ValueError("Light curves must be all phased or unphased!")
                # save as associated object
                self.associated_objects[str(lc.filter)] = LightCurve.__name__
                setattr(self, str(lc.filter), lc.copy())
                self._lightcurves.append(str(lc.filter))
                
            self._phased = phot_phased
            
        self.meta_attrs.extend(["_phased", "_lightcurves"])
        self._rng: np.random.Generator = np.random.default_rng()
        self._ts = pd.DataFrame(
            {
                "flux": [],
                "flux_unc": [],
                "mag": [],
                "mag_unc": [],
                "zpt": [],
                "non_detections": [],
                "filters": [],
                "filt_centers": [],
                "filt_widths": [],
            },
            index=pd.DatetimeIndex([]),
        )
        self.update()
        
    def update(self) -> None:
        """Update steps needed upon modifying child attributes."""
        self._generate_time_series()

    def _generate_time_series(self) -> None:
        """Generate time series from set of light curves."""
        if len(self._lightcurves) == 0:
            return None

        self._ts = pd.concat([getattr(self, lc).full_time_series for lc in self._lightcurves])
        self._ts.sort_values(by=["time", "filters"], inplace=True)

        return None

    @property
    def _mjd(self) -> NDArray[np.float64]:
        """Convert time (index) column to MJDs, in float."""
        if self._phased is None:
            return np.array([])  # empty
        if self._phased:
            return self._ts.index.total_seconds().to_numpy() / (24 * 3600)  # type: ignore
        astropy_time = Time(self._ts.index)  # Convert to astropy Time
        return astropy_time.mjd  # type: ignore

    @property
    def times(self) -> NDArray[np.float64]:
        """Return times of observations.

        Returns
        -------
        NDArray[np.float64]
        the times of observations
        """
        return self._mjd

    @property
    def mags(self) -> NDArray[np.float64]:
        """Return magnitudes of observations.

        Returns
        -------
        NDArray[np.float64]
        the magnitudes of observations
        """
        return self._ts["mag"].to_numpy()

    @property
    def fluxes(self) -> NDArray[np.float64]:
        """Return fluxes of observations.

        Returns
        -------
        NDArray[np.float64]
        the fluxes of observations
        """
        return self._ts["flux"].to_numpy()

    @property
    def mag_errors(self) -> NDArray[np.float64]:
        """Return magnitude errors of observations.

        Returns
        -------
        NDArray[np.float64]
        the magnitude errors of observations
        """
        return self._ts["mag_unc"].to_numpy()

    @property
    def flux_errors(self) -> NDArray[np.float64]:
        """Return flux errors of observations.

        Returns
        -------
        NDArray[np.float64]
        the flux errors of observations
        """
        return self._ts["flux_err"].to_numpy()

    @property
    def filters(self) -> NDArray[str]:  # type: ignore
        """Return filters of observations.

        Returns
        -------
        List[str]
        the filters of observations
        """
        return self._ts["filters"].to_numpy()

    @property
    def zeropoints(self) -> NDArray[np.float64]:
        """Return zeropoints of observations.

        Returns
        -------
        NDArray[np.float64]
        the zeropoints of observations
        """
        return self._ts["zpt"].to_numpy()

    @property
    def upper_limit_mask(self) -> NDArray[np.bool_]:
        """Return mask of upper limits.

        Returns
        -------
        NDArray[bool]
        the mask of upper limits
        """
        return self._ts["non_detections"].to_numpy()

    @property
    def detections(self) -> pd.DataFrame:
        """Return the detections in photometry.

        Returns
        -------
        TimeSeries
        the time series object
        """
        return self._ts[~self._ts["non_detections"]].copy()

    @property
    def non_detections(self) -> pd.DataFrame:
        """Return the non-detections in photometry.

        Returns
        -------
        TimeSeries
        the time series object
        """
        return self._ts[self._ts["non_detections"]].copy()

    @property
    def light_curves(self) -> list[LightCurve]:
        """Return copy of set of light curves, as
        to not impact the underlying LCs.

        Returns
        -------
        List[LightCurve]
        the set of light curves
        """
        return [getattr(self, lc).copy() for lc in self._lightcurves]

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
        filtered_lcs = [getattr(self, lc) for lc in self._lightcurves if lc.split("_")[0] == instrument]
        return self.__class__(filtered_lcs)

    def filter(self: PhotT, filts: Any) -> PhotT:
        """Return new Photometry with only light curves
        from filters in 'filts.'
        """
        filts = np.atleast_1d(filts)
        filts = [str(f) for f in filts]
        filtered_lcs = [getattr(self, filt) for filt in filts if hasattr(self, filt)]
        return self.__class__(filtered_lcs)

    def phase(
        self: PhotT, t0: Optional[float] = None,
        periodic: bool = False,
        period: Optional[float] = None,
        inplace: bool = True
    ) -> None:
        """Return new Photometry with light curves phased.
        By default phases max around 0. If periodic,
        will either use provided period or attempt to
        estimate one.
        """
        if periodic and period is None:
            period = self.calculate_period()
        if t0 is None:
            t0 = np.nanmedian([getattr(self, l).peak["time"] for l in self._lightcurves])
        
        if inplace:
            for lc in self._lightcurves:
                getattr(self, lc).phase(t0=t0, periodic=periodic, period=period)
            self.update()
        else:
            new_lcs = []
            for lc in self._lightcurves:
                new_lcs.append(
                    getattr(self, lc).phase(t0=t0, periodic=periodic, period=period, inplace=False)
                )
            return self.__class__(new_lcs)
                
        

    def calculate_period(self) -> float:
        """Estimate multi-band period of light curves in set."""
        detections = self.detections
        frequency, power = LombScargleMultiband(
            detections["time"].mjd,
            detections["mag"],
            detections["filters"],
            detections["mag_unc"],
        ).autopower()
        best_freq: float = frequency[np.nanargmax(power)]
        return 1.0 / best_freq

    def normalize(self: PhotT) -> None:
        """Normalize the light curves in the set."""
        peak_fluxes = [getattr(self, lc) .peak["flux"] for lc in self._lightcurves]
        for lc_name in self._lightcurves:
            lc = getattr(self, lc_name)
            lc.fluxes /= np.nanmax(peak_fluxes)
            lc.flux_errors /= np.nanmax(peak_fluxes)
        self.update()

    def plot(self, ax: Axes, formatter: Optional[Formatter] = None, mags: bool = True) -> Axes:
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
        if formatter is None:
            formatter = Formatter()  # make default formatter
        for lc in self._lightcurves:
            getattr(self, lc).plot(ax, formatter=formatter, mags=mags)
            formatter.rotate_colors()
            formatter.rotate_markers()
            if mags:
                ax.invert_yaxis()  # prevent double-inversion
        if mags:
            ax.invert_yaxis()
        return ax

    def add_lightcurve(self, light_curve: LightCurve) -> None:
        """Add a light curve to the set of photometry.

        Parameters
        ----------
        light_curve: LightCurve
            the light curve to add to the set of photometry
        """
        if self._phased is None:
            self._phased = light_curve.is_phased
        if self._phased and not light_curve.is_phased:
            raise ValueError("light_curve must be phased before adding to phased photometry.")
        if not self._phased and light_curve.is_phased:
            raise ValueError("cannot add phased light_curve to unphased photometry.")

        light_curve.merge_close_times(eps=1e-5)  # pylint: disable=no-member

        for lc_filt in self._lightcurves:
            # remove any duplicate times
            if lc_filt == str(light_curve.filter):
                getattr(self, lc_filt).merge(light_curve)
                self.update()
                return None
            
        self._lightcurves.append(str(light_curve.filter))
        setattr(self, str(light_curve.filter), light_curve)
        self.associated_objects[str(light_curve.filter)] = LightCurve.__name__
        self.update()
        return None

    def remove_lightcurve(self, filt_name: str) -> None:
        """Remove a light curve from the set of photometry.

        Parameters
        ----------
        lc: LightCurve
            the light curve to remove from the set of photometry
        """
        self._lightcurves.remove(filt_name)
        self.associated_objects.pop(filt_name)
        attr = getattr(self, filt_name)
        del attr
        self.update()

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
        it = 0
        lcs_shuffled = copy.deepcopy(self._lightcurves)
        while len(self._lightcurves) < n_lightcurves:
            it += 1
            np.random.shuffle(lcs_shuffled)
            for lc_name in lcs_shuffled:
                lc = getattr(self, lc_name)
                new_lc = lc.copy()
                if lc.filter is not None:
                    new_lc.filter = Filter(
                        instrument=lc.filter.instrument,
                        band=lc_name + str(it),
                        center=lc.filter.center,
                        width=lc.filter.width,
                    )
                self._lightcurves.append(lc_name + str(it))
                setattr(self, lc_name + str(it), new_lc)
                self.associated_objects[lc_name + str(it)] = LightCurve.__name__
                if len(self._lightcurves) == n_lightcurves:
                    break
        self.update()

    def __len__(self) -> int:
        """
        Length of the set of light curves.

        Returns
        -------
        int
        the number of light curves in the set
        """
        return len(self._lightcurves)

    def __eq__(self, other: object) -> bool:
        """Test photometry equality."""
        if not isinstance(other, self.__class__):
            return False
        
        return self.detections.equals(other.detections) & (
            self.non_detections.equals(other.non_detections)
        )  # order of LCs irrelevant

    def _dense_lc_helper(
        self,
        gp_vals: NDArray[np.float64],
        gp_errs: NDArray[np.float64],
        stacked_data: NDArray[np.float64],
        max_spacing: float,
        filt_to_int: dict[str, int],
        nfilts: int,
        max_n: int=64
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Helper function for dense_array method."""
        keep_idxs = np.where(np.abs(np.diff(self.times)) > max_spacing)[0]  # [1,]
        keep_idxs = np.insert(keep_idxs + 1, 0, 0)  # [0,2]
        # simple case: ((time1, filt1), (time1, filt2), (time2, filt1))
        # here, we want keep_idxs to be [0,2]
        # idx_map is [0, 0, 1]
        # map reduced indices to original indices
        idx_map = np.zeros(len(self.times), dtype=bool)
        idx_map[keep_idxs] = True  # [True, False, True]
        idx_map = np.cumsum(idx_map) - 1  # [0, 0, 1]
        
        gp_vals_keep = gp_vals[idx_map < max_n]
        gp_errs_keep = gp_errs[idx_map < max_n]
        stacked_data_keep = stacked_data[idx_map < max_n]
        dense_times = self.times[keep_idxs][:max_n]
        
        gaussian_process = generate_gp(gp_vals_keep, gp_errs_keep, stacked_data_keep)

        x_pred = np.zeros((len(dense_times) * nfilts, 2))

        for j in np.arange(nfilts):
            x_pred[j::nfilts, 0] = dense_times
            x_pred[j::nfilts, 1] = j

        pred, pred_var = gaussian_process.predict(gp_vals_keep, x_pred, return_var=True)

        dense_arr = np.zeros((nfilts, len(dense_times), 6), dtype=np.float64)
        dense_arr[:, :, 0] = dense_times
        dense_arr[:, :, 3] = 1  # interpolated mask

        for filt in np.unique(self.filters):
            filt_int = filt_to_int[filt]
            dense_arr[filt_int, :, 1] = pred[x_pred[:, 1] == filt_int]
            dense_arr[filt_int, :, 2] = np.sqrt(pred_var[x_pred[:, 1] == filt_int])

        return dense_arr, idx_map, idx_map >= max_n

    def _generate_gp_vals_and_errs(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], bool, pd.DataFrame]:
        """Generate GP values and errors for use in dense array."""
        # choose between mags or fluxes based on which is more complete
        use_fluxes = len(self.fluxes[~np.isnan(self.fluxes)]) > len(self.mags[~np.isnan(self.mags)])

        # first, remove all rows in time series with nans in ANY field
        nan_mask = np.zeros(len(self._ts), dtype=bool)

        # Iterate over each column and update the mask
        if use_fluxes:
            for col in ["flux", "flux_err"]:
                nan_mask |= ~np.isfinite(self._ts[col])
        else:
            for col in [
                "mag",
                "mag_unc",
            ]:
                nan_mask |= ~np.isfinite(self._ts[col])

        masked_rows = self._ts.loc[nan_mask]
        self._ts = self._ts.loc[~nan_mask]

        if use_fluxes:
            gp_vals = self.fluxes
            gp_errs = self.flux_errors
        else:
            gp_vals = (
                self.mags
                - np.nanmax(self.mags[~self.upper_limit_mask])
                - 3.0 * np.nanmax(self.mag_errors[~self.upper_limit_mask])
            )
            gp_errs = self.mag_errors

        return gp_vals, gp_errs, use_fluxes, masked_rows

    def dense_array(self, max_spacing: float = 4e-2, error_mask: float = 1.0) -> NDArray[np.float64]:
        """Return photometry as dense array for use
        in machine learning models.

        Parameters
        ----------
        max_spacing: np.float64
            the maximum time spacing between observations
        error_mask: np.float64
            the value to replace zero or negative errors with

        Returns
        -------
        NDArray[np.float64]
        the photometry as a dense array
        """
        gp_vals, gp_errs, use_fluxes, masked_rows = self._generate_gp_vals_and_errs()

        # map unique filts to integers
        filt_to_int = {filt: i for i, filt in enumerate(np.unique(self.filters))}
        filts_as_ints = np.array([filt_to_int[filt] for filt in self.filters])  # TODO: more efficient?

        nfilts = len(self._lightcurves)
        stacked_data = np.vstack([self.times, filts_as_ints]).T

        dense_arr, idx_map, ignore_m = self._dense_lc_helper(
            gp_vals, gp_errs, stacked_data, max_spacing, filt_to_int, nfilts
        )

        for filt in np.unique(self.filters[~ignore_m]):
            filt_int = filt_to_int[filt]
            filt_mask = (self.filters == filt) & (~self.upper_limit_mask)
            sub_series = self._ts[filt_mask & ~ignore_m]
            sub_idx_map = idx_map[filt_mask & ~ignore_m]

            # fill in true values
            dense_arr[filt_int, sub_idx_map, 1] = gp_vals[filt_mask & ~ignore_m]
            dense_arr[filt_int, sub_idx_map, 2] = gp_errs[filt_mask & ~ignore_m]
            dense_arr[filt_int, sub_idx_map, 3] = 0
            dense_arr[filt_int, :, 4] = sub_series["filt_centers"][0]
            dense_arr[filt_int, :, 5] = sub_series["filt_widths"][0]
            
            # fix broken errors - usually if no points in that band
            mask = dense_arr[filt_int, :, 2] <= 0.0
            dense_arr[filt_int, mask, 2] = error_mask
            dense_arr[filt_int, mask, 3] = 1

        if not use_fluxes: # TODO: why is this here again?
            dense_arr[:, :, 1] += np.nanmax(self._ts["mag"][~self.upper_limit_mask])
            dense_arr[:, :, 1] += 3.0 * np.nanmax(self._ts["mag_unc"][~self.upper_limit_mask])

        

        # readd nan rows at end back to time series
        self._ts = pd.concat([self._ts, masked_rows])
        self._ts.sort_values(by=["time", "filters"], inplace=True)

        return dense_arr

    def absolute(self: PhotT, redshift: float, inplace: bool=False) -> PhotT:
        """Return new Photometry with absolute magnitudes.
        Shifts zeropoints accordingly.
        """
        if inplace:
            if not self._phased:
                self.phase() # first have to phase
            for lc in self._lightcurves:
                getattr(self, lc).absolute(redshift, inplace=True)
            return self
        
        if not self._phased:
            lc_phased = [getattr(self, x).phase(inplace=False) for x in self._lightcurves]
        else:
            lc_phased = [getattr(self, x).copy() for x in self._lightcurves]
        for lc in lc_phased:
            lc.absolute(redshift, inplace=True)
            
        return self.__class__(lc_phased)

    def correct_extinction(
        self: PhotT, mwebv: Optional[float] = None, coordinates: Optional[Any] = None
    ) -> PhotT:
        """Return new Photometry with extinction corrected magnitudes."""
        new_lcs = []
        for lc in self._lightcurves:
            new_lcs.append(getattr(self, lc).correct_extinction(mwebv=mwebv, coordinates=coordinates))
        return self.__class__(new_lcs)
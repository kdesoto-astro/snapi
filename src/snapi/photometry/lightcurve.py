"""Contains classes for light curves and filters."""
from typing import Any, Iterable, Mapping, Optional, Sequence, Type, TypeVar, cast
import time

import astropy.constants as const
import astropy.cosmology.units as cu
import astropy.units as u
import extinction
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15  # pylint: disable=no-name-in-module
from astropy.timeseries import LombScargle
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from ..formatter import Formatter
from ..image import Image
from ..utils import calc_mwebv
from ..base_classes import Measurement, Plottable
from .timeseries import TimeSeries
from .filter import Filter
from .utils import resample_helper, calc_all_deltas, update_merged_fluxes

LightT = TypeVar("LightT", bound="LightCurve")
    
class LightCurve(Measurement, TimeSeries, Plottable):  # pylint: disable=too-many-public-methods
    """Class that contains all information for a
    single light curve. Associated with a single instrument
    and filter.

    LightCurve should always be (a) as complete as possible and
    (b) sorted by time.
    """
    _ts_cols = {
        "flux": np.float64,
        "flux_error": np.float64,
        "mag": np.float64,
        "mag_error": np.float64,
        "zeropoint": np.float64,
        "upper_limit": bool
    }
    
    _name_map = {
        "flux": ["flux", "fluxes"],
        "flux_error": [
            "flux_error",
            "flux_errors",
            "flux_err",
            "flux_errs",
            "flux_unc",
            "flux_uncertainties",
            "sigma_flux",
            "sigma_f"
        ],
        "mag": ["mag", "magnitude", "magnitudes", "mags"],
        "mag_error": [
            "mag_error",
            "mag_errors",
            "mag_err",
            "mag_errs",
            "mag_unc",
            "mag_uncertainties",
            "sigma_mag",
            "sigma_m",
            "magnitude_error",
            "magnitude_errors",
            "magnitude_unc",
            "magnitude_uncertainties",
            "sigma_magnitude",
        ],
        "zeropoint": ["zpt", "zpts", "zeropoint", "zeropoints"],
        "upper_limit": ["non_detection", "non_detections", "upper_limit", "upper_limits", "upper_lim", "upper_lims"]
    }

    def __init__(
        self,
        time_series: Optional[pd.DataFrame] = None,
        filt: Optional[Filter] = None,
        phased: bool = False,
        validate: bool = True,
    ) -> None:
        super().__init__(time_series=time_series, phased=phased, validate=validate)
        self._validate_observer(filt)  

        self._image_time_bins = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 10_000])
        self._image_flux_bins = np.array([-1000, -2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 1000])
        
    @classmethod
    def from_arrays(
        cls,
        **kwargs
    ):
        """Initialize TimeSeries from arrays."""
        for k in ["filt", "filter"]:
            if k in kwargs:
                filt = kwargs.pop(k)
        phased = False
        if "phased" in kwargs:
            phased = kwargs.pop("phased")
        ts = pd.DataFrame(kwargs)
        return cls(ts, phased=phased, filt=filt)
        
    def update(self) -> None:
        """Update steps needed upon modifying child attributes."""
        super().update()
        self._complete()
        
    def _complete(self, force_update_mags: bool = False, force_update_fluxes: bool = False) -> None:
        """Given zeropoints, fills in missing apparent
        magnitudes from fluxes and vice versa.
        """
        missing_mag = (~np.isnan(self._ts["zeropoint"])) & (~np.isnan(self._ts["flux"]))
        if not force_update_mags:
            missing_mag = missing_mag & np.isnan(self._ts["mag"])

        sub_table = self._ts[missing_mag]
        if len(sub_table) > 0:
            self._ts.loc[missing_mag, "mag"] = -2.5 * np.log10(sub_table["flux"]) + sub_table["zeropoint"]

        missing_magunc = ~np.isnan(self._ts["flux"]) & (~np.isnan(self._ts["flux_error"]))
        if not force_update_mags:
            missing_magunc = missing_magunc & np.isnan(self._ts["mag_error"])
        sub_table = self._ts[missing_magunc]
        if len(sub_table) > 0:
            self._ts.loc[missing_magunc, "mag_error"] = (
                2.5 / np.log(10.0) * (sub_table["flux_error"] / sub_table["flux"])
            )

        missing_flux = (~np.isnan(self._ts["zeropoint"])) & (~np.isnan(self._ts["mag"]))
        if not force_update_fluxes:
            missing_flux = missing_flux & np.isnan(self._ts["flux"])
        sub_table = self._ts[missing_flux]
        if len(sub_table) > 0:
            self._ts.loc[missing_flux, "flux"] = 10.0 ** (-1.0 * (sub_table["mag"] - sub_table["zeropoint"]) / 2.5)

        missing_fluxunc = ~np.isnan(self._ts["mag_error"]) & (~np.isnan(self._ts["flux"]))
        if not force_update_fluxes:
            missing_fluxunc = missing_fluxunc & np.isnan(self._ts["flux_error"])
        sub_table = self._ts[missing_fluxunc]
        if len(sub_table) > 0:
            self._ts.loc[missing_fluxunc, "flux_error"] = (np.log(10.0) / 2.5) * (
                sub_table["flux"] * sub_table["mag_error"]
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        
        return (
            self.detections.equals(other.detections)
            & (self.non_detections.equals(other.non_detections))
            & (self.filter == other.filter)
        )

    @property
    def filter(self) -> Optional[Filter]:
        """Return filter object associated with LightCurve."""
        return self._observer

    @filter.setter
    def filter(self, filt: Filter) -> None:
        """Replace filter associated with LightCurve."""
        if not isinstance(filt, Filter):
            raise TypeError("Input must be Filter object!")
        self._observer = filt

    @property
    def non_detections(self) -> pd.DataFrame:
        """Return non-detection observations."""
        return self._ts.loc[self._ts['upper_limit']] # pylint: disable=invalid-unary-operand-type

    @property
    def detections(self) -> pd.DataFrame:
        """Return detection observations."""
        return self._ts.loc[~self._ts['upper_limit']]  # pylint: disable=invalid-unary-operand-type

    @property
    def full_time_series(self) -> pd.DataFrame:
        """Return all observations (detections + nondetections) with extra columns
        for filter information. Used in photometry dataframe generation."""
        ts_copy = self._ts.copy()
        ts_copy["filter"] = str(self._observer)
        ts_copy["filt_center"] = self._observer.center.value if self._observer else np.nan
        ts_copy["filt_width"] = self._observer.width.value if (self._observer and (self._observer.width is not None)) else np.nan
        return ts_copy

    @property
    def _peak_idx(self) -> Any:
        """Get idx of peak mag or flux.
        """
        if pd.isna(self._ts["flux"]).all():
            idx = (self.detections["mag"] + self.detections["mag_error"]).idxmin()
        else:
            idx = (self.detections["flux"] - self.detections["flux_error"]).idxmax()
        return idx

    def calculate_period(self) -> float:
        """Calculate period of light curve.
        Uses LombScargle periodogram.
        """
        ls = LombScargle(
            self.detections.index,
            self.detections["mag"],
            self.detections["mag_error"],
        )
        frequency, power = ls.autopower()
        best_freq: float = frequency[np.nanargmax(power)]
        return 1.0 / best_freq

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
            val_errs = self._ts["mag_error"].to_numpy()
            ax.set_ylabel("Magnitude")
            ax.invert_yaxis()
        else:
            vals = self._ts["flux"].to_numpy()
            val_errs = self._ts["flux_error"].to_numpy()
            ax.set_ylabel("Flux")
            
        ax.errorbar(
            times[~self._ts['upper_limit']],
            vals[~self._ts['upper_limit']],
            yerr=val_errs[~self._ts['upper_limit']],
            c=formatter.edge_color,
            fmt="none",
            zorder=-1,
        )
        ax.scatter(
            times[~self._ts['upper_limit']],
            vals[~self._ts['upper_limit']],
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.marker_style,
            s=formatter.marker_size,
            label=str(self._observer),
        )
        # plot non-detections
        ax.scatter(
            times[self._ts['upper_limit']],
            vals[self._ts['upper_limit']],
            c=formatter.face_color,
            edgecolor=formatter.edge_color,
            marker=formatter.nondetect_marker_style,
            alpha=formatter.nondetect_alpha,
            s=formatter.nondetect_size,
            zorder=-2,
        )

        return ax

    def merge_close_times(self, eps: float = 4e-2, inplace: bool = False) -> None:  # TODO: fix for LCs without flux uncertainties
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
            keep_idxs, self._ts["flux"], self._ts["flux_error"]
        )
        
        if inplace:
            self._ts = self._ts.iloc[~repeat_idxs]
            self.fluxes = new_f
            self.flux_errors = new_ferr
            self._ts["upper_limit"] = new_nondet
            self._complete(force_update_mags=True)
            return self
        
        ts = self._ts.iloc[~repeat_idxs]
        ts['upper_limit'] = new_nondet
        new_lc = self.__class__(ts, phased = self._phased, filt = self._observer, validate=False)
        new_lc.fluxes = new_f
        new_lc.flux_errors = new_ferr
        return new_lc

    def merge(self, other: LightT, inplace: bool = False) -> None:
        """Merge other light curve into this one, assuming they are
        from the same instrument and filter.
        """
        if self._observer != other.filter:
            raise ValueError("Filters must be the same to merge light curves!")
            
        if inplace:
            ts = self._ts
        else:
            ts = self._ts.copy()
            
        if len(other.detections) > 0:
            # among detections, fill in missing errors/zeropoints
            nd_mask = ts["upper_limit"]
            ts.loc[~nd_mask, :] = ts[~nd_mask].combine_first(other.detections)

            # now for times that exist in both, we want to replace non-detections
            # with detections, where possible
            override_mask1 = other.detections.index.isin(ts[nd_mask].index)
            override_mask2 = nd_mask & ts.index.isin(other.detections.index)
            ts.loc[override_mask2, :] = other.detections[override_mask1]

            # finally, add new times
            nonrepeat_idxs = other.detections.index.difference(ts.index)
            if not nonrepeat_idxs.empty:
                non_na_columns_ts = ts.columns[~ts.isna().all()]
                non_na_columns = other.detections.columns[~other.detections.isna().all()]
                ts = pd.concat(
                    [
                        ts[non_na_columns_ts],
                        other.detections.loc[nonrepeat_idxs, non_na_columns],  # type: ignore
                    ],
                    ignore_index=False,
                )

        if len(other.non_detections) > 0:
            nd_mask = ts["upper_limit"]
            # update non-detections similarly
            ts.loc[nd_mask, :] = ts[nd_mask].combine_first(other.non_detections)
            nonrepeat_idxs_2 = other.non_detections.index.difference(ts.index)
            if not nonrepeat_idxs_2.empty:
                non_na_columns_ts = self._ts.columns[~ts.isna().all()]
                non_na_columns = other.non_detections.columns[~other.non_detections.isna().all()]
                ts = pd.concat(
                    [
                        ts[non_na_columns_ts],
                        other.non_detections.loc[nonrepeat_idxs_2, non_na_columns],  # type: ignore
                    ],
                    ignore_index=False,
                )

        for col in self._ts_cols:  # ensure all original columns exist
            if col not in ts.columns:
                ts[col] = np.nan
                
        if inplace:
            self._ts = ts
            self.update()
            return self
        
        new_lc = self.__class__(
            ts, phased = self._phased, filt=self._observer, validate=False
        )
        return new_lc

    def copy(self):
        """Return a copy of the TimeSeries."""
        return self.__class__(
            self._ts,
            phased=self._phased,
            filt=self._observer,
            validate=False
        )
    
    def pad(self: LightT, fill: dict[str, Any], n_times: int, inplace: bool = False) -> LightT:
        """Extends light curve by padding.
        Currently, pads based on 'fill' dictionary.
        """
        if (not self._phased) and ("mjd" not in fill):
            raise KeyError("mjd must be a key in fill")
        elif (self._phased) and ("phase" not in fill):
            raise KeyError("phase must be a key in fill")
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
            centers = self.detections["mag"]
            uncs = self.detections["mag_error"]
        else:
            centers = self.detections["flux"]
            uncs = self.detections["flux_error"]

        return resample_helper(centers, uncs, num)  # type: ignore

    def absolute(self: LightT, redshift: float, inplace=False) -> LightT:
        """Returns LightCurve with absolute magnitudes.
        Adjusts magnitudes and zeropoints by distance modulus
        and k-corrections.
        """
        if not self._phased:
            times = self.phase(inplace=False).times
        else:
            times = self._mjd
            
        new_times = times / (1.0 + redshift)
        shift_timedelta = pd.to_timedelta(new_times, "D")
        
        k_corr = 2.5 * np.log10(1.0 + redshift)
        distmod = Planck15.distmod(redshift).value
        
        if inplace:
            self._ts.set_index(shift_timedelta, inplace=True)
            self._phased = True
            self._ts.index.name = "phase"
            self._ts["mag"] += (-distmod + k_corr)
            self._ts["zeropoint"] += (-distmod + k_corr)
            return self
        
        # Reconstruct the new Time object using jd1 and jd2
        new_ts = self._ts.copy()
        new_ts.set_index(shift_timedelta, inplace=True)
        new_ts["mag"] += (-distmod + k_corr)
        new_ts["zeropoint"] += (-distmod + k_corr)

        return self.__class__(
            new_ts,
            filt=self._observer,
            phased=self._phased,
        )

    def correct_extinction(
        self: LightT,
        mwebv: Optional[float] = None,
        coordinates: Optional[Any] = None,
        inplace: bool = False,
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

        if self._observer is None:
            ext_val = 0.0
        else:
            # Now figure out how much the magnitude is affected by this dust
            ext_val = extinction.fm07(
                np.array([self._observer.center.to(u.AA).value]), av_sfd, unit="aa"  # pylint: disable=no-member
            )[
                0
            ]  # in magnitudes

        if inplace:
            self.mags -= ext_val
            return self
        
        lc_copy = self.copy()
        lc_copy.mags -= ext_val
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
    
    def __setattr__(self, name, value):
        """Custom attribute setters to preserve flux/magnitude balance.
        """
        if (name == "_alias_map") or (name not in self._alias_map):  # Protected attributes
            object.__setattr__(self, name, value)
            return None
            
        df_key = self._alias_map[name]
        self._ts[df_key] = value
        
        if df_key in ('flux', 'flux_error', 'zeropoint'):
            self._complete(force_update_mags=True)
        elif df_key in ('mag', 'mag_error'):
            self._complete(force_update_fluxes=True)
            
        return None
    
    def phase(
        self,
        t0: Optional[float] = None,
        periodic: bool = False,
        period: Optional[float] = None,
        inplace: bool = True
    ) -> None:
        """
        Phases light curve by t0, which is assumed
        to be in days.
        """
        if t0 is None:
            t0_units = self._peak_idx
        else:
            t0_units = pd.to_timedelta(t0, "D")
        if periodic and (period is None):
            period = self.calculate_period()
            
        if inplace:
            if periodic:
                self._ts.set_index(
                    (self._ts.index - t0_units) % period, inplace=True
                )
            else:
                self._ts.set_index(
                    self._ts.index - t0_units, inplace=True
                )
            self._ts.index.name = "phase"
            self._phased = True
        else:
            new_ts = self._ts.copy()
            if periodic:
                new_ts.set_index(
                    (self._ts.index - t0_units) % period, inplace=True
                )
            else:
                new_ts.set_index(
                    self._ts.index - t0_units, inplace=True
                )
            return self.__class__(
                new_ts,
                phased=True,
                filt=self._observer,
                validate=False
            )

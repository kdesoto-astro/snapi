"""Stores Photometry class and helper functions."""
import extinction
from typing import Any, Optional, Tuple, Type, TypeVar
import time
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.timeseries import LombScargleMultiband
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..formatter import Formatter
from .lightcurve import LightCurve
from .filter import Filter
from .utils import generate_gp

PhotT = TypeVar("PhotT", bound="Photometry")

class Photometry(LightCurve):  # pylint: disable=too-many-public-methods
    """Extends LightCurve object to handle multiple filters."""

    _ts_cols = {
        **LightCurve._ts_cols,
        "filter": str,
        "filt_center": np.float64,
        "filt_width": np.float64
    }
    
    _name_map = {
        **LightCurve._name_map,
        "filter": ["filter", "filt", "filters", "filts"],
        "filt_center": ["filt_center", "filt_centers", "centers", "center"],
        "filt_width": ["filt_width", "filt_widths", "widths", "width"]
    }
    
    def __init__(
        self,
        time_series: Optional[pd.DataFrame] = None,
        phased: bool = False,
        validate: bool = True,
        filt: Any = None
    ) -> None:
        super().__init__(time_series, phased=phased, filt=None, validate=validate)
        self.update()
        
    def copy(self):
        """Return a copy of the TimeSeries."""
        return self.__class__(
            self._ts,
            phased=self._phased,
            validate=False
        )
    
    def _sort(self):
        """Sort light curve by time."""
        self._ts.sort_values(by=[self._ts.index.name, "filter"], inplace=True)
        self._ts.sort_index(axis=1, inplace=True)
    
    @classmethod
    def from_light_curves(cls, lcs: list[LightCurve], phased: Optional[bool] = None):
        """Initialize Photometry from list of LightCurve objects."""
        phot_phased = phased
        full_ts = []
        for lc in lcs:
            if not isinstance(lc, LightCurve):
                raise TypeError("All elements of 'lcs' must be a LightCurve!")
            if phot_phased is None:
                phot_phased = lc.is_phased
            # ensure all are either phased or unphased
            if (phot_phased and not lc.is_phased) or (not phot_phased and lc.is_phased):
                raise ValueError("Light curves must be all phased or unphased!")
            # save as associated object
            full_ts.append(
                lc.full_time_series
            )
        return cls(pd.concat(full_ts, copy=False), phased=phot_phased, validate=False)
        
    def update(self) -> None:
        """Update steps needed upon modifying child attributes."""
        self._unique_filters = self._ts["filter"].unique()
        self._sort()
    
    def _filter_single(self, filter_name: str) -> pd.DataFrame:
        """Return photometry for a single filter."""
        return self._ts.loc[self._ts['filter'] == filter_name]
    
    def _construct_lightcurve_single(self, filter_name: str) -> LightCurve:
        """From filter name, reconstruct the light curve for that filter."""
        df_filter = self._filter_single(filter_name)
        filt_name = df_filter['filter'].iloc[0]
        filt_instrument, filt_band = filt_name.split("_")
        filt = Filter(
            instrument=filt_instrument,
            band=filt_band,
            center=df_filter['filt_center'].iloc[0],
            width=df_filter['filt_width'].iloc[0]
        )
        return LightCurve(
            df_filter,
            phased=self._phased,
            filt=filt,
            validate=False
        )

    @property
    def light_curves(self) -> list[LightCurve]:
        """Return copy of set of light curves, as
        to not impact the underlying LCs.

        Returns
        -------
        List[LightCurve]
        the set of light curves
        """
        return [self._construct_lightcurve_single(filt) for filt in self._unique_filters]

    def filter_by_instrument(self: PhotT, instrument: str, inplace: bool=False) -> PhotT:
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
        matched_filts = [x for x in self._unique_filters if x.split("_")[0] == instrument]
        if inplace:
            self._ts = self._ts[self._ts['filter'].isin(matched_filts)]
            self._unique_filters = matched_filts
            return self
        
        return self.__class__(
            self._ts.loc[self._ts['filter'].isin(matched_filts)],
            phased=self._phased,
            validate=False
        )

    def filter_subset(self: PhotT, filts: Any, inplace: bool=False) -> PhotT:
        """Return new Photometry with only light curves
        from filters in 'filts.'
        """
        filts = np.atleast_1d(filts)
        filts = [str(f) for f in filts]
        
        if inplace:
            self._ts = self._ts.loc[self._ts['filter'].isin(filts)]
            self._unique_filters = self._ts['filter'].unique()
        
        filtered_df = self._ts.loc[self._ts['filter'].isin(filts)]
        return self.__class__(
            filtered_df,
            phased = self._phased,
            validate=False
        )
    
    @property
    def _peak_idx(self):
        """Return index associated with peak. Take peak as median of each filter's peak.
        """
        #if pd.isna(self.detections['flux']).all():
        peaks = self.detections.groupby("filter", group_keys=False).apply(
            lambda x: (x["mag"] + x["mag_error"]).idxmin()
        )
        #peaks = self.detections.groupby("filter", group_keys=False).apply(
        #    lambda x: (x["flux"] - x["flux_error"]).idxmax()
        #)
        return peaks.median()

    def calculate_period(self) -> float:
        """Estimate multi-band period of light curves in set."""
        detections = self.detections
        frequency, power = LombScargleMultiband(
            detections["time"].mjd,
            detections["mag"],
            detections["filter"],
            detections["mag_error"],
        ).autopower()
        best_freq: float = frequency[np.nanargmax(power)]
        return 1.0 / best_freq

    def normalize(self: PhotT, inplace: bool=False) -> None:
        """Normalize the light curves in the set."""
        peak_flux = self.detections['flux'].dropna().max()
        
        if inplace:
            self.fluxes /= peak_flux
            self.flux_errors /= peak_flux
            return self
        
        new_phot = self.__class__(
            self._ts,
            phased = self._phased,
            validate=False
        )
        new_phot.fluxes /= peak_flux
        new_phot.flux_errors /= peak_flux
        return new_phot

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
        
        for lc in self.light_curves:
            lc.plot(ax, formatter=formatter, mags=mags)
            formatter.rotate_colors()
            formatter.rotate_markers()
            if mags:
                ax.invert_yaxis()  # prevent double-inversion
        if mags:
            ax.invert_yaxis()
        return ax

    def add_lightcurve(self, light_curve: LightCurve, inplace: bool=False) -> None:
        """Add a light curve to the set of photometry.

        Parameters
        ----------
        light_curve: LightCurve
            the light curve to add to the set of photometry
        """
        if len(self._ts) == 0:
            self._phased = light_curve.is_phased
        if self._phased and not light_curve.is_phased:
            raise ValueError("light_curve must be phased before adding to phased photometry.")
        if not self._phased and light_curve.is_phased:
            raise ValueError("cannot add phased light_curve to unphased photometry.")

        light_curve.merge_close_times(eps=1e-5)  # pylint: disable=no-member
        
        for lc_filt in self._unique_filters:
            # remove any duplicate times
            if lc_filt == str(light_curve.filter):
                lc = self._construct_lightcurve_single(lc_filt)
                lc.merge(light_curve, inplace=True)
                if inplace:
                    self._ts = pd.concat(
                        [self._ts.loc[self._ts['filter'] != lc_filt], lc.full_time_series],
                        copy=False
                    )
                    self.update()
                    return self
                else:
                    new_phot = self.__class__(
                        pd.concat(
                            [self._ts.loc[self._ts['filter'] != lc_filt], lc.full_time_series],
                            copy=False
                        ),
                        phased = self._phased,
                        validate = False
                    )
                    new_phot.update()
                    return new_phot
        if len(self._ts) == 0:
            new_ts = light_curve.full_time_series
        else:
            new_ts = pd.concat(
                [self._ts, light_curve.full_time_series],
                copy=False
            )
        new_ts.index.name = "phase" if self._phased else "mjd"
        
        if inplace:
            self._ts = new_ts
            self.update()
            return None
        
        phot = self.__class__(
            new_ts,
            phased = self._phased,
            validate = False
        )
        phot.update()
        return phot

    def remove_lightcurve(self, filt_name: str, inplace: bool = False):
        """Remove a light curve from the set of photometry.

        Parameters
        ----------
        lc: LightCurve
            the light curve to remove from the set of photometry
        """
        if inplace:
            self._ts = self._ts.loc[self._ts['filter'] != filt_name]
            self.update()
            return self
        
        return self.__class__(
            self._ts.loc[self._ts['filter'] != filt_name],
            phased = self._phased,
            validate = False
        )
        

    def tile(self, n_lightcurves: int, inplace: bool=False) -> None:
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
        if len(self._unique_filters) > n_lightcurves:
            raise ValueError("Number of light curves exceeds the desired limit.")
        it = 0
        n_filts = len(self._unique_filters)
        new_tss = []
        while n_filts < n_lightcurves:
            it += 1
            np.random.shuffle(self._unique_filters)
            for lc_name in self._unique_filters:
                copy_ts = self._filter_single(lc_name)
                copy_ts['filter'] = copy_ts['filter'][0]+f"_{it}"
                new_tss.append(copy_ts)
                n_filts += 1
                    
                if n_filts == n_lightcurves:
                    break
                    
        if inplace:
            self._ts = pd.concat([self._ts, *new_tss], copy=False)
            self.update()
            return self
        phot = self.__class__(
            pd.concat([self._ts, *new_tss], copy=False),
            phased = self._phased,
            validate = False
        )
        phot.update()
        return phot

    def __len__(self) -> int:
        """
        Length of the set of light curves.

        Returns
        -------
        int
        the number of light curves in the set
        """
        return len(self._unique_filters)
    
    def _dense_lc_helper(
        self,
        gp_df: pd.DataFrame,
        stacked_data: NDArray[np.float64],
        max_spacing: float,
        max_n: int=64
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Helper function for dense_array method."""
        keep_idxs = np.where(np.abs(np.diff(gp_df['time'])) > max_spacing)[0]  # [1,]
        keep_idxs = np.insert(keep_idxs + 1, 0, 0)  # [0,2]
        # simple case: ((time1, filt1), (time1, filt2), (time2, filt1))
        # here, we want keep_idxs to be [0,2]
        # idx_map is [0, 0, 1]
        # map reduced indices to original indices
        idx_map = np.zeros(len(gp_df), dtype=bool)
        idx_map[keep_idxs] = True  # [True, False, True]
        gp_df['idx_map'] = np.cumsum(idx_map) - 1  # [0, 0, 1]
        
        gp_df_keep = gp_df.loc[gp_df['idx_map'] < max_n]
        dense_times = gp_df['time'].iloc[keep_idxs[:max_n]]

        stacked_data_keep = stacked_data[idx_map < max_n]
        
        gaussian_process = generate_gp(gp_df_keep['val'].to_numpy(), gp_df_keep['err'].to_numpy(), stacked_data_keep)
        
        nfilts = len(np.unique(gp_df_keep['filter']))
        x_pred = np.zeros((len(dense_times) * nfilts, 2))

        for j in np.arange(nfilts):
            x_pred[j::nfilts, 0] = dense_times
            x_pred[j::nfilts, 1] = j

        pred, pred_var = gaussian_process.predict(gp_df_keep['val'].to_numpy(), x_pred, return_var=True)

        dense_arr = np.zeros((nfilts, len(dense_times), 6), dtype=np.float64)
        dense_arr[:, :, 0] = dense_times
        dense_arr[:, :, 3] = 1  # interpolated mask

        for filt_int in np.unique(gp_df_keep['filter']):
            dense_arr[filt_int, :, 1] = pred[x_pred[:, 1] == filt_int]
            dense_arr[filt_int, :, 2] = np.sqrt(pred_var[x_pred[:, 1] == filt_int])

        return dense_arr, gp_df_keep

    def _generate_gp_vals_and_errs(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], bool, pd.DataFrame]:
        """Generate GP values and errors for use in dense array."""
        # choose between mags or fluxes based on which is more complete
        use_fluxes = len(self.fluxes[~np.isnan(self.fluxes)]) > len(self.mags[~np.isnan(self.mags)])

        # Iterate over each column and update the mask
        if use_fluxes:
            nan_mask = self.detections.loc[:,["flux", "flux_error"]].notna().all(axis=1)
            gp_ts = self.detections.loc[nan_mask, ['flux', 'flux_error', 'filter', 'filt_center', 'filt_width']]
            gp_ts.rename(columns={'flux': 'val', 'flux_error': 'err'}, inplace=True)
            corr = 0.
        else:
            nan_mask = self.detections.loc[:,["mag", "mag_error"]].notna().all(axis=1)
            gp_ts = self.detections.loc[nan_mask, ['mag', 'mag_error', 'filter', 'filt_center', 'filt_width']]
            gp_ts.rename(columns={'mag': 'val', 'mag_error': 'err'}, inplace=True)
            corr = gp_ts['val'].max() + 3.0 * gp_ts['err'].max()
            gp_ts['val'] -= corr
            
        if self._phased:
            gp_ts['time'] = gp_ts.index.total_seconds().to_numpy() / (24 * 3600)  # type: ignore
        else:
            gp_ts['time'] = Time(gp_ts.index).mjd  # Convert to astropy Time
            
        filt_to_int = {filt: i for i, filt in enumerate(np.unique(gp_ts['filter']))}
        filts_as_ints = np.array([filt_to_int[filt] for filt in gp_ts['filter']])  # TODO: more efficient?
        gp_ts['filter'] = filts_as_ints
            
        return gp_ts, corr

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
        gp_df, corr = self._generate_gp_vals_and_errs()

        # map unique filts to integer

        stacked_data = np.vstack([gp_df['time'], gp_df['filter']]).T

        dense_arr, gp_df_keep = self._dense_lc_helper(
            gp_df, stacked_data, max_spacing
        )

        for filt_int in np.unique(gp_df_keep['filter']):
            sub_series = gp_df_keep.loc[gp_df_keep['filter'] == filt_int]

            # fill in true values
            dense_arr[filt_int, sub_series['idx_map'], 1:3] = sub_series[['val', 'err']]
            dense_arr[filt_int, sub_series['idx_map'], 3] = 0
            dense_arr[filt_int, :, 4] = sub_series["filt_center"].iloc[0]
            dense_arr[filt_int, :, 5] = sub_series["filt_width"].iloc[0]
            
            # fix broken errors - usually if no points in that band
            mask = dense_arr[filt_int, :, 2] <= 0.0
            dense_arr[filt_int, mask, 2] = error_mask
            dense_arr[filt_int, mask, 3] = 1

        dense_arr[:,:,1] += corr
        return dense_arr

    def correct_extinction(
        self: PhotT, mwebv: Optional[float] = None, coordinates: Optional[Any] = None,
        inplace: bool = False
    ) -> PhotT:
        """Return new Photometry with extinction corrected magnitudes."""
        if coordinates is not None:
            mwebv_calced = calc_mwebv(coordinates)

            if (mwebv is not None) and (mwebv_calced != mwebv):
                raise ValueError("Coordinate-calculated MW E(B-V) does not agree with provided MW E(B-V).")

            mwebv = mwebv_calced

        else:
            if mwebv is None:
                raise ValueError("Either coordinates or mwebv must be provided.")

        av_sfd = 2.742 * mwebv
        
        ext_vals = extinction.fm07(
            self.filt_centers, av_sfd, unit="aa"  # pylint: disable=no-member
        )
        
        if inplace:
            self.mags -= ext_vals
            return self
        
        lc_copy = self.__class__(
            self._ts, phased=self._phased, validate=False
        )
        lc_copy.mags -= ext_vals
        lc_copy.update()
        return lc_copy
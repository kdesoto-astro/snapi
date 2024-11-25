from typing import Optional, Any
import abc
import time
import copy

import pandas as pd
import numpy as np
from astropy.time import Time
from numpy.typing import NDArray

from ..base_classes import Base

class TimeSeries(Base):
    """Base class encompassing anything timelike.
    Both LightCurve and Photometry inherit from this.
    """
    
    _ts_cols = {}
    _name_map = {}
    _alias_map = {}
    
    def __init__(
        self,
        time_series: Optional[pd.DataFrame] = None,
        phased: bool = False,
        validate: bool = True
    ) -> None:
        """If not validate, do not validate the columns/index of the provided
        DataFrame. Really only set to False when loading saved objects or making
        copies.
        """
        super().__init__()
        self._flip_name_map()
        self._phased = phased
        if time_series is None: # initialize empty
            self._ts = None
        elif isinstance(time_series, pd.DataFrame):
            if validate:
                valid_keys = np.intersect1d(time_series.columns, list(self._alias_map.keys()), assume_unique=True)
                self._ts = time_series[valid_keys].rename(columns={k: self._alias_map[k] for k in valid_keys})

                if ("time" in time_series.columns) and (not self._phased):
                    astropy_dt = Time(time_series["time"], format="mjd").to_datetime()
                    self._ts.index = pd.DatetimeIndex(astropy_dt)
                elif ("time" in time_series.columns) and self._phased:
                    self._ts.index = pd.to_timedelta(time_series["phase"], "D")
                elif "mjd" in time_series.columns:
                    astropy_dt = Time(time_series["mjd"], format="mjd").to_datetime()
                    self._ts.index = pd.DatetimeIndex(astropy_dt)
                    self._phased = False
                elif "phase" in time_series.columns:
                    # convert to DateTimeIndex
                    self._ts.index = pd.to_timedelta(time_series["phase"], "D")
                    self._phased = True
                elif isinstance(time_series.index, pd.TimedeltaIndex):
                    self._phased = True
                elif isinstance(time_series.index, pd.DatetimeIndex):
                    self._phased = False
                else:
                    raise TypeError(
                        "index must be of DatetimeIndex or TimedeltaIndex, or one of ['time','phase','mjd'] must be columns."
                    )
                for (k, v) in self._ts_cols.items():
                    if k not in self._ts.columns:
                        self._ts[k] = False if self._ts_cols[k] == bool else np.nan
                    self._ts[k] = self._ts[k].astype(v)
                    
                self._ts.index.name = "phase" if self._phased else "mjd"
                self.update()

            else:
                self._ts = time_series.copy()
        else:
            raise TypeError("Invalid data type for time_series. Must be None or pd.DataFrame")
        
        
        self._rng = np.random.default_rng()

        self.arr_attrs.append("_ts")
        self.meta_attrs.extend(["_phased", "_rng"])
        
    def _flip_name_map(self):
        """Flip name map to map aliases to intrinsic attributes."""
        self._alias_map = {}
        for (k, vals) in self._name_map.items():
            for v in vals:
                self._alias_map[v] = k
            
    @classmethod
    def from_arrays(
        cls,
        **kwargs
    ):
        """Initialize TimeSeries from arrays."""
        phased = False
        if "phased" in kwargs:
            phased = kwargs.pop("phased")
        ts = pd.DataFrame(kwargs)
        return cls(ts, phased=phased)
    
    def update(self):
        """Sort timeseries."""
        self._sort()
        
    def _sort(self):
        """Sort light curve by time and set column order."""
        if self._ts is None:
            return
        self._ts.sort_index(inplace=True)
        self._ts.sort_index(axis=1, inplace=True)
        return
        
    @property
    def is_phased(self) -> bool:
        """Whether LC is phased or unphased."""
        return self._phased

    @property
    def _mjd(self) -> NDArray[np.float64]:
        """Convert time (index) column to MJDs, in float."""
        if len(self._ts) == 0:
            return np.array([])
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
    
    def __deepcopy__(self, memo):
        """Deepcopy override."""
        if id(self) in memo:
            return memo[id(self)]
    
        # Create new copy
        new_copy = self.copy()
        # Store in memo to prevent re-copying
        memo[id(self)] = new_copy
        return new_copy

    def copy(self):
        """Return a copy of the TimeSeries."""
        return self.__class__(
            self._ts,
            self._phased,
            validate=False
        )
    
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
            self._ts.index.name = "phase"
        else:
            astropy_times = Time(new_times, format="mjd")
            astropy_dt = astropy_times.to_datetime()
            self._ts.set_index(pd.DatetimeIndex(astropy_dt), inplace=True)
            self._ts.index.name = "mjd"
        self._sort()
        
    def add_observations(self, rows: list[dict[str, Any]]) -> None:
        """Add rows to existing timeseries."""
        if (self._ts is None and "phase" in row):
            index_name = "phase"
            self._phased = True
        elif (self._ts is None and "mjd" in row):
            index_name = "mjd"
            self._phased = False
        elif self._phased:
            index_name = "phase"
        else:
            index_name = "mjd"
            
        if index_name == "phase":
            new_times = [row["phase"] for row in rows]
            new_index_td = pd.to_timedelta(new_times, "D")
            new_df = pd.DataFrame.from_records(rows, index=new_index_td)
            new_df.drop(columns="phase", inplace=True)
            new_df.index.name = "phase"
            
        else:
            new_times = [self._convert_to_datetime(row["mjd"]) for row in rows]
            new_index_dt = pd.DatetimeIndex(new_times)
            new_df = pd.DataFrame.from_records(rows, index=new_index_dt)
            new_df.drop(columns="mjd", inplace=True)
            new_df.index.name = "mjd"
        if self._ts is None:
            self._ts = new_df
        else:
            self._ts = pd.concat([self._ts, new_df], copy=False)
        self.update()
        
    @property
    def peak(self) -> Any:
        """The brightest observation in light curve.
        Return as dictionary.
        """
        idx = self._peak_idx
        peak_dict = self.detections.loc[idx].to_dict()

        if self._phased:
            peak_dict["phase"] = idx.total_seconds() / (24 * 3600)  # type: ignore
        else:
            astropy_time = Time(idx)  # Convert to astropy Time
            peak_dict["mjd"] = astropy_time.mjd
        return peak_dict

    @abc.abstractmethod
    def _peak_idx(self):
        """Return the index associated with peak of the time series."""
        pass
        
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
            
    def __getattr__(self, name):
        """Handle attribute logic."""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if (name == "_alias_map") or (name not in self._alias_map):
                raise
            return self._ts[self._alias_map[name]].to_numpy()
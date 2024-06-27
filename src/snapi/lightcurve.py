import copy
from typing import Optional, Self, Sequence, TypeVar

import astropy.units as u
import numpy as np
from astropy.timeseries import TimeSeries
from matplotlib.axes import Axes

from .base_classes import Base, Plottable

T = TypeVar("T", int, float)


class Filter(Base):
    """Contains instrument and filter information."""

    def __init__(self, instrument: str, center: u.Quantity, width: Optional[u.Quantity]):
        self._instrument = instrument
        self._center = center
        self._width = width


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

        self._ts = TimeSeries(
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

    def plot(self, ax: Axes) -> Axes:
        """Plot a single light curve."""
        return ax

    def add_observations(self, rows: list[dict[str, object]]) -> Self:
        """Add rows to existing timeseries."""
        # TODO: accomodate different formats
        for row in rows:
            self._ts.add_row(row)
        return self

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

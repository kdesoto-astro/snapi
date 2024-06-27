import copy
from typing import Any, Mapping, Optional, Self, Sequence, TypeVar

import astropy.units as u
import numpy as np
from astropy.timeseries import TimeSeries
from matplotlib.axes import Axes
from numba import njit
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


@njit  # type: ignore
def augment_helper(cen: Sequence[T], unc: Sequence[T], num: int) -> NDArray[np.float32]:
    """numba-enhanced helper to generate many augmented LCs."""
    rng = np.random.default_rng()
    sampled_vals = np.zeros((num, len(cen)), dtype=np.float32)
    for i in range(num):
        sampled_vals[i] += rng.normal(loc=cen, scale=unc)
    return sampled_vals


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

    def add_observations(self, rows: list[dict[str, Any]]) -> Self:
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

    def augment(self, mags: bool = False, n: int = 100) -> NDArray[np.float32]:
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

        return augment_helper(centers, uncs, n)  # type: ignore

    def convert_to_image(
        self, method: str = "gaf", augment: bool = True, **kwargs: Mapping[str, Any]
    ) -> Image:
        """Convert light curve to an image for ML applications.

        TODO: use fluxes or mags?
        """
        if augment:  # augment LC to get many copies before transforming
            series = self.augment(mags=False, n=100)  # TODO: de-hardcode this
        else:
            series = np.atleast_2d(self._ts["flux"])

        if method == "gaf":  # Gramian angular field
            transformer = GramianAngularField(**kwargs)
        elif method == "mtf":  # Markov transition field
            transformer = MarkovTransitionField(**kwargs)
        elif method == "recurrence":  # recurrence plot
            transformer = RecurrencePlot(**kwargs)
        else:
            raise NotImplementedError("Imaging method must be one of: 'gaf', 'mtf', 'recurrence'")

        vals = transformer.transform(series)
        return Image(vals)

from matplotlib.axes import Axes

from .base_classes import Plottable


class LightCurve(Plottable):
    """Class that contains all information for a
    single light curve. Associated with a single instrument
    and filter.
    """

    def __init__(self) -> None:
        pass

    def plot(self, ax: Axes) -> Axes:
        """Plot a single light curve."""
        return ax

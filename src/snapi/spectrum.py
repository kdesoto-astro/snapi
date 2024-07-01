from matplotlib.axes import Axes

from .base_classes import Plottable


class Spectrum(Plottable):
    """A single spectrum, associated with
    one timestamp and instrument.
    """

    def __init__(self) -> None:
        pass

    def plot(self, ax: Axes) -> Axes:
        """Plot a single spectrum."""
        return ax

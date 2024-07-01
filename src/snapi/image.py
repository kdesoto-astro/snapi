from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .base_classes import Plottable


class Image(Plottable):
    """Class for any image that has pixels and
    is two-dimensional.
    """

    def __init__(
        self, pixels: Sequence[Union[int, float]], extent: Optional[Sequence[Union[int, float]]] = None
    ) -> None:
        self._extent: Optional[tuple[float, float, float, float]] = None

        p = np.asarray(pixels).astype(np.float32)
        if p.ndim != 2:
            raise ValueError("Pixels must be a 2D array!")
        self._values: NDArray[np.float32] = p
        if extent is not None:
            e = np.asarray(extent).astype(np.float32)
            if e.ndim != 1 or len(e) != 4:
                raise ValueError("Extent must be a 1d, 4-element sequence!")
            self._extent = tuple(e)

    def plot(self, ax: Axes) -> Axes:
        if self._extent is None:
            ax.imshow(self._values, cmap="inferno")  # UNHARD CODE
        else:
            ax.imshow(self._values, cmap="inferno", extent=self._extent)  # TODO: unhard-code
        return ax

import re
from typing import Union

import colorcet as cc  # pylint: disable=import-error
import matplotlib.markers as mmarkers
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import numpy as np
from numpy.typing import NDArray


def darken_colormap(colormap: Union[list[str], NDArray[np.str_]]) -> NDArray[np.str_]:
    """Darken the colors in a given colormap."""
    colormap = np.array(colormap)
    dimmed_colormap = []
    for color in colormap:
        # Split the color string after the pound sign into three sets of two letters
        r, g, b = re.findall(r"\w\w", color.strip("#"))
        rgb = np.array([int(r, 16), int(g, 16), int(b, 16)])
        dimmed_vals = np.clip(rgb * 0.7, 0, 255).astype(int)
        dimmed_color = f"#{dimmed_vals[0]:02x}{dimmed_vals[1]:02x}{dimmed_vals[2]:02x}"
        dimmed_colormap.append(dimmed_color)
    return np.array(dimmed_colormap)


class Formatter:
    """Contains all plotting settings, including
    choice of color schemes, label conventions, and
    figure dimensions.
    """

    def __init__(self) -> None:
        self._marker_styles = ["o", "^", "*"]  # rotation
        self._face_colors: Union[list[str], NDArray[np.str_]] = cc.glasbey_dark
        self._edge_colors: NDArray[np.str_] = darken_colormap(self._face_colors)

        self._face_color_index = 0
        self._edge_color_index = 0

        self._marker_index = 0
        self._marker_size = 36.0  # default marker size

        self.nondetect_alpha = 0.3
        self.nondetect_marker_style = ""
        self.nondetect_size = 0.0

        # Define the downward arrow marker path
        arrow_edge = 0.3
        middle_edge = np.sqrt(3.0) * arrow_edge
        arrow_path = mpath.Path(
            vertices=[(-arrow_edge, 1.2), (arrow_edge, 1.2), (0, 1.2 - middle_edge), (-arrow_edge, 1.2)],
            codes=[mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY],
        )
        self.arrow_vertices = arrow_path.vertices.tolist()  # type: ignore
        self.arrow_codes = arrow_path.codes.tolist()  # type: ignore
        self._update_nondetection_properties()

    def _update_nondetection_properties(self) -> None:
        """Add downward arrow to a given marker.
        Returns combined marker.
        """
        marker_style = mmarkers.MarkerStyle(marker=self.marker_style)
        marker_path = marker_style.get_path().transformed(marker_style.get_transform())

        # Combine both paths
        vert = marker_path.vertices.tolist()  # type: ignore
        codes = marker_path.codes.tolist()  # type: ignore
        combined_vertices = vert + self.arrow_vertices
        combined_codes = codes + self.arrow_codes

        # Define the custom combined marker path
        combined_marker = mpath.Path(combined_vertices, combined_codes)

        # Register the custom marker with alpha value
        scale_transform = mtransforms.Affine2D().scale(
            4.0 * self.marker_size
        )  # Adjust the scale factor as needed
        scaled_combined_marker = combined_marker.transformed(scale_transform)
        bounds = scaled_combined_marker.get_extents()
        self.nondetect_size = max(bounds.width, bounds.height)
        self.nondetect_marker_style = mmarkers.MarkerStyle(marker=scaled_combined_marker)  # type: ignore

    def rotate_colors(self) -> None:
        """Rotate colors for next plot."""
        self._face_color_index = (self._face_color_index + 1) % len(self._face_colors)
        self._edge_color_index = (self._edge_color_index + 1) % len(self._edge_colors)

    def rotate_markers(self) -> None:
        """Rotate marker styles for next plot."""
        self._marker_index = (self._marker_index + 1) % len(self._marker_styles)
        self._update_nondetection_properties()

    @property
    def face_color(self) -> str:
        """Returns the current color for point faces."""
        return self._face_colors[self._face_color_index]

    @property
    def edge_color(self) -> str:
        """Returns the current color for point edges."""
        edge_color: str = self._edge_colors[self._edge_color_index]
        return edge_color

    @property
    def marker_style(self) -> str:
        """Returns the current marker style."""
        return self._marker_styles[self._marker_index]

    @property
    def marker_size(self) -> float:
        """Returns the current marker size."""
        return self._marker_size

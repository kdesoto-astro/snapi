"""Stores all formatting information in an object for consistent
formatting across plots."""
import re
from typing import Optional, Union

import colorcet as cc  # pylint: disable=import-error
import matplotlib.markers as mmarkers
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Annotation
from matplotlib.ticker import AutoMinorLocator
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

    def __init__(
        self,
        linewidth: float = 2.0,
        markersize: float = 36.0,
        fontname: str = "Verdana",
        face_colors: Union[list[str], NDArray[np.str_]] = cc.glasbey_dark,
        edge_colors: Optional[NDArray[np.str_]] = None,
        marker_styles: Optional[list[str]] = None,
        nondetect_alpha: float = 0.3,
        nondetect_marker_style: str = "",
        nondetect_size: float = 0.0,
    ) -> None:
        if edge_colors is None:
            edge_colors = darken_colormap(face_colors)
        if marker_styles is None:
            marker_styles = ["o", "s", "D", "v", "^", "<", ">", "p", "P", "*", "h", "H", "X", "d"]

        self._marker_styles = marker_styles
        self._face_colors = face_colors
        self._edge_colors = edge_colors
        self._fontname = fontname

        self._face_color_index = 0
        self._edge_color_index = 0

        self._marker_index = 0
        self._marker_size = markersize
        self._line_width = linewidth

        self.nondetect_alpha = nondetect_alpha
        self.nondetect_marker_style = nondetect_marker_style
        self.nondetect_size = nondetect_size

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

    def reset_colors(self) -> None:
        """Reset color rotation."""
        self._face_color_index = 0
        self._edge_color_index = 0

    def reset_markers(self) -> None:
        """Reset marker rotation."""
        self._marker_index = 0
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

    @property
    def line_width(self) -> float:
        """Returns the current line width."""
        return self._line_width

    def make_plot_pretty(self, ax: Axes) -> None:
        """Makes the plot pretty.
        Code taken from Karthik's plotting utils.
        """
        annotations = [child for child in ax.get_children() if isinstance(child, Annotation)]
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig = ax.get_figure()
        ax.tick_params(
            which="major",
            bottom="on",
            top="on",
            left="on",
            right="on",
            direction="in",
        )
        ax.tick_params(
            which="minor",
            bottom="on",
            top="on",
            left="on",
            right="on",
            direction="in",
        )
        if fig is None:
            ax.tick_params(
                which="major",
                length=20,
            )
            ax.tick_params(
                which="minor",
                length=7.5,
            )
            ax.set_xlabel(ax.get_xlabel(), fontsize=16)
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        else:
            ax.tick_params(which="major", length=2.0 * fig.get_figwidth(), labelsize=2.0 * fig.get_figwidth())
            ax.tick_params(
                which="minor",
                length=0.75 * fig.get_figwidth(),
            )
            ax.set_xlabel(ax.get_xlabel(), fontsize=2.5 * fig.get_figwidth())
            ax.set_ylabel(ax.get_ylabel(), fontsize=2.5 * fig.get_figwidth())

            # check if plot has legend
            if ax.get_legend() is not None:
                # check if plot has legend
                legend = ax.get_legend()
                for text in legend.get_texts():
                    text.set_fontsize(2.0 * fig.get_figwidth())
            if ax.get_title() is not None:
                ax.set_title(ax.get_title(), fontsize=3.0 * fig.get_figwidth())

            for annotation in annotations:
                annotation.set_fontsize(2.0 * fig.get_figwidth())

        # edit all fonts to Verdana
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontname(self._fontname)

        for annotation in annotations:
            annotation.set_fontname(self._fontname)

    def add_legend(self, ax: Axes, ncols: int = 4, pretty: bool = True) -> None:
        """Add a legend to the plot.
        If pretty is True, the legend will be formatted according
        to the make_plot_pretty method.
        """
        handles, _ = ax.get_legend_handles_labels()
        num_handles = len(handles)
        num_rows = (num_handles + ncols - 1) // ncols
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25 - 0.05 * num_rows),
            ncol=ncols,
        )
        if not pretty:
            return None
        for item in ax.get_legend().get_texts():
            item.set_fontname(self._fontname)
            fig = ax.get_figure()
            if fig is not None:
                item.set_fontsize(2.0 * fig.get_figwidth())
        return None
    
    
    def set_aspect_ratio(self, ax: Axes, ratio: float = 1.0):
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

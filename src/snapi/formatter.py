DEFAULT_CMAP_FACE = ["#EE6677", "#4477AA"]
DEFAULT_CMAP_EDGE = ["#BB5566", "#004488"]


class Formatter:
    """Contains all plotting settings, including
    choice of color schemes, label conventions, and
    figure dimensions.
    """

    def __init__(self) -> None:
        self._marker_styles = ["o", "^", "*"]  # rotation
        self._face_colors = DEFAULT_CMAP_FACE
        self._edge_colors = DEFAULT_CMAP_EDGE

        self._color_index = 0
        self._marker_index = 0

    def rotate_colors(self) -> None:
        """Rotate colors for next plot."""
        self._color_index += 1

        if self._color_index >= len(self._face_colors):  # check overflow
            self._color_index = 0

    @property
    def face_color(self) -> str:
        """Returns the current color for point faces."""
        return self._face_colors[self._color_index]

    @property
    def edge_color(self) -> str:
        """Returns the current color for point edges."""
        return self._edge_colors[self._color_index]

    @property
    def marker_style(self) -> str:
        """Returns the current marker style."""
        return self._edge_colors[self._marker_index]

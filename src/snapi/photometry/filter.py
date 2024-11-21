from typing import Any, Optional

import astropy.units as u
import numpy as np

from ..base_classes import Observer

class Filter(Observer):
    """Contains instrument and filter information."""

    def __init__(
        self,
        instrument: str = "",
        band: str = "",
        center: Any = np.nan,
        width: Optional[Any] = None,
    ) -> None:
        super().__init__(instrument)
        self._band = band
        
        self._set_center(center)
        self._set_width(width)
        self.update()
        self.meta_attrs.extend(['_band', '_center', '_width']) # stored as floats assuming u.AA

    
    def _set_center(self, center: Any) -> None:
        """Checks possible formatting for center and saves accordingly."""
        if isinstance(center, u.Quantity):
            if center.unit.physical_type == "length":
                self._center = center.to(u.AA).value
            elif center.unit.physical_type == "frequency":  # convert to wavelength
                self._center = (
                    center.to(u.Hz, equivalencies=u.spectral()) ** -1 * u.AA  # pylint: disable=no-member
                ).value
            else:
                raise ValueError("center's units must be of type length or frequency")
        else:
            try:
                self._center = float(center)
            except:
                raise ValueError("center must be convertible to a float if without units.")
            
    def _set_width(self, width: Any) -> None:
        """Checks possible formatting for width and saves accordingly."""
        if width is None:
            self._width = width
        elif isinstance(width, u.Quantity):
            if width.unit.physical_type == "frequency":  # convert to wavelength
                self._width = (width.to(u.Hz, equivalencies=u.spectral()) ** -1 * u.AA).value  # pylint: disable=no-member
            elif width.unit.physical_type == "length":  # convert to wavelength
                self._width = width.to(u.AA).value  # pylint: disable=no-member
            else:
                raise TypeError("width must be a wavelength or frequency quantity!")
        else:
            try:
                self._width = float(width)
            except:
                raise ValueError("width must be convertible to a float if without units.")
            
    def update(self) -> None:
        """Update steps needed upon modifying child attributes."""
        pass
                           
    def __str__(self) -> str:
        """Return string representation of filter.
        Format: instrument_band.
        """
        return f"{self._instrument}_{self._band}"

    @property
    def band(self) -> str:
        """Return band of filter."""
        return self._band

    @property
    def center(self) -> u.Quantity:
        """Return center of filter,
        in Angstroms.
        """
        return self._center * u.AA  # pylint: disable=no-member
    
    @center.setter
    def center(self, center: Any) -> None:
        """If center is given without units, assume angstroms."""
        self._set_center(center)
        self.update()
        
    @property
    def width(self) -> u.Quantity:
        """Return width of filter,
        in Angstroms.
        """
        if self._width is None:
            return self._width
        return self._width * u.AA  # pylint: disable=no-member
    
    @width.setter
    def width(self, width: Any) -> None:
        """If center is given without units, assume angstroms."""
        self._set_width(width)
        self.update()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Filter):
            return False

        return (self.instrument == other.instrument) and (self.band == other.band)
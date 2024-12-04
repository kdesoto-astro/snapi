"""Stores the results of a query."""
from dataclasses import dataclass
from typing import Any, Optional

from astropy.coordinates import SkyCoord

from ..photometry import LightCurve
from ..spectrum import Spectrum


@dataclass
class QueryResult:
    """
    Class for storing query results.
    """

    objname: str = ""
    internal_names: Optional[set[str]] = None
    coordinates: Optional[SkyCoord] = None
    redshift: Optional[float] = None
    spec_class: Optional[str] = None
    light_curves: Optional[list[LightCurve]] = None
    spectra: Optional[set[Spectrum]] = None

    def __post_init__(self) -> None:
        if self.internal_names is None:
            self.internal_names = set()
        if self.light_curves is None:
            self.light_curves = []
        if self.spectra is None:
            self.spectra = set()

    def to_dict(self) -> dict[str, Any]:
        """Convert object to dictionary."""
        return {
            "objname": self.objname,
            "internal_names": self.internal_names,
            "coordinates": self.coordinates,
            "redshift": self.redshift,
            "spec_class": self.spec_class,
            "light_curves": self.light_curves,
            "spectra": self.spectra,
        }


@dataclass
class HostQueryResult(QueryResult):
    """Class for storing host galaxy
    query results.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert object to dictionary."""
        return {
            "hostname": self.objname,
            "host_internal_names": self.internal_names,
            "host_coords": self.coordinates,
            "host_redshift": self.redshift,
            "host_spectra": self.spectra,
        }

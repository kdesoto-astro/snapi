"""Stores the results of a query."""
from dataclasses import dataclass
from typing import Optional

from astropy.coordinates import SkyCoord

from ..lightcurve import LightCurve
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
    light_curves: Optional[set[LightCurve]] = None
    spectra: Optional[set[Spectrum]] = None

    def __post_init__(self) -> None:
        if self.internal_names is None:
            self.internal_names = set()
        if self.light_curves is None:
            self.light_curves = set()
        if self.spectra is None:
            self.spectra = set()

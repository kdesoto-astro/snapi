from typing import Iterable, Optional

import astropy.units as u
from astropy.coordinates import SkyCoord

from .base_classes import Base
from .lightcurve import LightCurve
from .photometry import Photometry
from .query_agents.query_result import QueryResult
from .spectroscopy import Spectroscopy
from .spectrum import Spectrum


class Transient(Base):
    """Object that stores transient information, including
    any related photometry, spectroscopy, and host galaxies.
    """

    def __init__(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        iid: Optional[str] = None,
        photometry: Optional[Photometry] = None,
        spectroscopy: Optional[Spectroscopy] = None,
        internal_names: Optional[set[str]] = None,
        spec_class: Optional[str] = None
        # host: HostGalaxy = None,
    ) -> None:
        if iid is None:
            self.id = str(id(self))
        else:
            self.id = str(iid)

        self._coord = SkyCoord(ra=ra, dec=dec, frame="icrs")

        self.photometry = photometry
        self.spectroscopy = spectroscopy
        if internal_names is None:
            self.internal_names = set()
        else:
            self.internal_names = internal_names

        self.spec_class = spec_class
        # self.host = host

    @property
    def coordinates(self) -> SkyCoord:
        """Returns the coordinates of the transient."""
        return self._coord

    def add_lightcurve(self, lightcurve: LightCurve) -> None:
        """Adds a single light curve to photometry."""
        if self.photometry is None:
            self.photometry = Photometry()  # initialize new instance

        self.photometry.add_lightcurve(lightcurve)

    def add_lightcurves(self, lightcurves: Iterable[LightCurve]) -> None:
        """Adds a set of light curves to the
        transient's associated photometry object.

        Iterates through the add_lightcurve function.
        """
        for lc in lightcurves:
            self.add_lightcurve(lc)

    def add_spectrum(self, spectrum: Spectrum) -> None:
        """Adds a single spectrum to the spectroscopy attribute."""
        if self.spectroscopy is None:
            self.spectroscopy = Spectroscopy()  # initialize new instance
        self.spectroscopy.add_spectrum(spectrum)

    def add_spectra(self, spectra: Iterable[Spectrum]) -> None:
        """Adds a set of Spectrum objects to the spectroscopy.
        Iterates through add_spectrum().
        """
        for spec in spectra:
            self.add_spectrum(spec)

    def ingest_query_info(self, result: QueryResult) -> None:
        """Ingests information from a QueryResult adds to object."""
        pass

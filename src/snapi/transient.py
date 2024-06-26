from typing import Iterable, Optional

from .base_classes import Base
from .lightcurve import LightCurve
from .photometry import Photometry
from .spectroscopy import Spectroscopy
from .spectrum import Spectrum


class Transient(Base):
    """Object that stores transient information, including
    any related photometry, spectroscopy, and host galaxies.
    """

    def __init__(
        self,
        iid: Optional[str] = None,
        photometry: Optional[Photometry] = None,
        spectroscopy: Optional[Spectroscopy] = None,
        # host: HostGalaxy = None,
    ) -> None:
        if iid is None:
            self.id = str(id(self))
        else:
            self.id = str(iid)

        self.photometry = photometry
        self.spectroscopy = spectroscopy
        # self.host = host

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

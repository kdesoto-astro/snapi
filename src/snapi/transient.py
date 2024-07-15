from typing import Any, Iterable, Optional, Type, TypeVar

import astropy.units as u
import h5py
from astropy.coordinates import SkyCoord

from .base_classes import Base
from .lightcurve import LightCurve
from .photometry import Photometry
from .spectroscopy import Spectroscopy
from .spectrum import Spectrum

TransientT = TypeVar("TransientT", bound="Transient")


class Transient(Base):
    """Object that stores transient information, including
    any related photometry, spectroscopy, and host galaxies.
    """

    def __init__(
        self,
        iid: Optional[str] = None,
        ra: Optional[u.Quantity] = None,
        dec: Optional[u.Quantity] = None,
        photometry: Optional[Photometry] = None,
        spectroscopy: Optional[Spectroscopy] = None,
        internal_names: Optional[set[str]] = None,
        spec_class: Optional[str] = None,
        redshift: Optional[float] = None,
        # host: HostGalaxy = None,
    ) -> None:
        if iid is None:
            self.id = str(id(self))
        else:
            self.id = str(iid)

        if ra is None or dec is None:
            self._coord = None
        else:
            self._coord = SkyCoord(ra=ra, dec=dec, frame="icrs")

        if photometry is None:
            self.photometry = Photometry()
        else:
            self.photometry = photometry
        self.spectroscopy = spectroscopy
        if internal_names is None:
            self.internal_names = set()
        else:
            self.internal_names = internal_names

        self.spec_class = spec_class
        self.redshift = redshift

        self._choose_main_name()
        # self.host = host

    def _choose_main_name(self) -> None:
        """Chooses the main name for the transient."""
        # check if prefix can be converted to int
        for n in self.internal_names:
            if len(n) < 5 or len(n) > 8:
                continue
            if not n[:4].isdigit() or n[:5].isdigit():
                continue
            if int(n[:4]) < 1900 or int(n[:4]) > 2100:
                continue
            self.internal_names.add(self.id)
            self.internal_names.remove(n)
            self.id = n
            break

    @property
    def coordinates(self) -> SkyCoord:
        """Returns the coordinates of the transient."""
        return self._coord

    def add_lightcurve(self, lightcurve: LightCurve) -> None:
        """Adds a single light curve to photometry."""
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

    def ingest_query_info(self, result: dict[str, Any]) -> None:
        """Ingests information from a QueryResult adds to object."""
        if result["internal_names"] is not None:
            if "" in result["internal_names"]:
                result["internal_names"].remove("")
            self.internal_names.update(result["internal_names"])

        if result["objname"] != "":
            self.internal_names.add(result["objname"])

        # prioritize object IAU name as main ID
        self._choose_main_name()

        if self.spec_class is None:  # TODO: add logging stating what was added
            self.spec_class = result["spec_class"]
        if self.redshift is None:
            self.redshift = result["redshift"]
        if self.coordinates is None:
            self._coord = result["coordinates"]
        if result["light_curves"] is not None:
            self.add_lightcurves(result["light_curves"])
        if result["spectra"] is not None:
            for spec in result["spectra"]:
                self.add_spectrum(spec)

    def save(self, filename: str) -> None:
        """Save transient object to HDF5 file."""
        with h5py.File(filename, "w") as f:
            pass

        self.photometry.save(filename, path="photometry")

        with h5py.File(filename, "a") as f:
            f.attrs["id"] = self.id
            if self.coordinates is not None:
                f.attrs["ra"] = self.coordinates.ra.deg
                f.attrs["dec"] = self.coordinates.dec.deg
            if self.redshift is not None:
                f.attrs["redshift"] = self.redshift
            if self.spec_class is not None:
                f.attrs["spec_class"] = self.spec_class
            f.attrs["internal_names"] = list(self.internal_names)

    @classmethod
    def load(cls: Type[TransientT], filename: str) -> TransientT:
        """Load transient object from HDF5 file."""
        with h5py.File(filename, "r") as f:
            photometry: Photometry = Photometry.load(filename, path="photometry")
            iid = f.attrs.get("id")
            ra = f.attrs.get("ra")
            if ra is not None:
                ra = ra * u.deg  # pylint: disable=no-member
            dec = f.attrs.get("dec")
            if dec is not None:
                dec = dec * u.deg  # pylint: disable=no-member
            redshift = f.attrs.get("redshift")
            spec_class = f.attrs.get("spec_class")
            internal_names = set(f.attrs.get("internal_names", default=set()))

        return cls(
            ra=ra,
            dec=dec,
            iid=iid,
            photometry=photometry,
            spec_class=spec_class,
            redshift=redshift,
            internal_names=internal_names,
        )

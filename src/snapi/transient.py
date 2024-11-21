from typing import Any, Iterable, Optional, Type, TypeVar
import time

import astropy.units as u
import h5py
from astropy.coordinates import SkyCoord
from astropy.units import Quantity


from .base_classes import Base
from .photometry import Photometry, LightCurve
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
        super().__init__()
        if iid is None:
            self.id = str(id(self))
        else:
            self.id = str(iid)
            
        if isinstance(ra, Quantity):
            self._ra = ra.to(u.deg).value # pylint: disable=no-member
            self._dec = dec.to(u.deg).value # pylint: disable=no-member
        elif ra is None:
            self._ra = None
            self._dec = None
        else:
            self._ra = float(ra)
            self._dec = float(dec)

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
        
        self.meta_attrs.extend(['id','_ra','_dec','internal_names','spec_class','redshift'])

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
            if self.id != str(id(self)): # don't move hash to internal_names
                self.internal_names.add(self.id)
            self.internal_names.remove(n)
            self.id = n
            break

    def __len__(self) -> int:
        """Returns the number of observations associated with
        the transient (both photometric and spectroscopic)."""
        len_phot = 0 if not self.photometry else len(self.photometry)
        len_spec = 0 if not self.spectroscopy else len(self.spectroscopy)
        
        return len_phot + len_spec
    
    def __eq__(self, other: object) -> bool:
        """Return True if there is a shared name
        between self and other, and if coordinate,
        redshift, and photometry/spectroscopy match.
        """
        if not isinstance(other, self.__class__):
            return False
        names_self = {*self.internal_names, self.id}
        names_other = {*other.internal_names, other.id}

        if len(names_self.intersection(names_other)) == 0:
            return False
                
        if self.redshift is None:
            if other.redshift is not None:
                return False
        else:
            if other.redshift is None:
                return False
            if (abs(self.redshift - other.redshift)/self.redshift) > 1e-2: # 1% precision
                return False
                        
        if self.coordinates is not None:
            if other.coordinates is None:
                return False
            if self.coordinates.separation(other.coordinates).to(u.arcsec).value > 1.: # 1 arcsec allowance
                return False
        else:
            if other.coordinates is not None:
                return False
            
        return (self.photometry == other.photometry
        ) & (
            self.spectroscopy == other.spectroscopy
        )
    
    def overlaps(self, other: object) -> bool:
        """Return True if there is a shared name
        between self and other, and if coordinates/redshift
        are self-consistent.
        """
        if not isinstance(other, self.__class__):
            return False
        
        names_self = {*self.internal_names, self.id}
        names_other = {*other.internal_names, other.id}

        if len(names_self.intersection(names_other)) == 0:
            return False
        
        if (self.redshift is not None) and (other.redshift is not None):
            if (abs(self.redshift - other.redshift)/self.redshift) > 1e-2: # 1% precision
                return False
            
        if (self.coordinates is not None) and (other.coordinates is not None):
            if self.coordinates.separation(other.coordinates).to(u.arcsec).value > 1.: # 1 arcsec allowance
                return False
            
        return True
    
    @property
    def coordinates(self) -> Optional[SkyCoord]:
        """Returns the coordinates of the transient."""
        if (self._ra is None) or (self._dec is None):
            return None
        return SkyCoord(ra=self._ra * u.deg, dec=self._dec * u.deg, frame="icrs") # pylint: disable=no-member

    def add_lightcurve(self, lightcurve: LightCurve) -> None:
        """Adds a single light curve to photometry."""
        if self.photometry is None:
            self.photometry = Photometry()
            
        self.photometry.add_lightcurve(lightcurve, inplace=True)
        
    @property
    def photometry(self) -> Photometry:
        return self._photometry
    
    @photometry.setter
    def photometry(self, photometry) -> None:
        if photometry is None:
            self.associated_objects.pop('_photometry', None)
        else:
            self.associated_objects['_photometry'] = Photometry.__name__
        self._photometry = photometry
        
    @property
    def spectroscopy(self) -> Spectroscopy:
        return self._spectroscopy
    
    @spectroscopy.setter
    def spectroscopy(self, spectroscopy) -> None:
        self._spectroscopy = spectroscopy
        if spectroscopy is None:
            self.associated_objects.pop('_spectroscopy', None)
        else:
            self.associated_objects['_spectroscopy'] = Spectroscopy.__name__

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
            self._ra = result["coordinates"].ra.to(u.deg).value
            self._dec = result["coordinates"].dec.to(u.deg).value
        if result["light_curves"] is not None:
            self.add_lightcurves(result["light_curves"])
        if result["spectra"] is not None:
            for spec in result["spectra"]:
                self.add_spectrum(spec)
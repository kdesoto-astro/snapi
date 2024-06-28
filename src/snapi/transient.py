import json
import time
from collections import OrderedDict
from typing import Any, Iterable, Optional

import astropy.units as u
import requests
from astropy.coordinates import SkyCoord

from .api_keys import TNS_API_KEY, TNS_BOT_ID, TNS_BOT_NAME
from .base_classes import Base
from .lightcurve import LightCurve
from .photometry import Photometry
from .spectroscopy import Spectroscopy
from .spectrum import Spectrum

URL_TNS_API = "https://sandbox.wis-tns.org/api/get"
TNS_HEADER = {
    "User-Agent": f'tns_marker{{"tns_id": "{TNS_BOT_ID}", "type": "bot", "name": "{TNS_BOT_NAME}"}}'
}


def get_reset_time(response: requests.Response) -> Optional[int]:
    """Determine time needed to sleep before submitting new
    TNS requests.
    """
    # If any of the '...-remaining' values is zero, return the reset time
    for name in response.headers:
        value = response.headers.get(name)
        if name.endswith("-remaining") and value == "0":
            out = response.headers.get(name.replace("remaining", "reset"))
            if out is not None:
                return int(out)
    return None


def tns_search_helper(
    ra: Optional[str] = "", dec: Optional[str] = "", internal_name: Optional[str] = ""
) -> Any:
    """Retrieve search results from TNS."""
    search_url = URL_TNS_API + "/search"
    search_query = [
        ("ra", ra),
        ("dec", dec),
        ("radius", ""),
        ("units", ""),
        ("objname", ""),
        ("objname_exact_match", 0),
        ("internal_name", internal_name),
        ("internal_name_exact_match", 1),
        ("objid", ""),
        ("public_timestamp", ""),
    ]
    json_file = OrderedDict(search_query)
    search_data = {"api_key": TNS_API_KEY, "data": json.dumps(json_file)}
    r = requests.post(search_url, headers=TNS_HEADER, data=search_data, timeout=30.0)
    if r.status_code != 200:
        print(f"ERROR {r.status_code}")
        return {}
    # sleep necessary time to abide by rate limit
    reset = get_reset_time(r)
    if reset is not None:
        time.sleep(reset + 1)
    return r.json()


def tns_object_helper(obj_name: Optional[str]) -> Any:
    """Retrieve specific object from TNS."""
    if obj_name is None:
        raise ValueError("obj_name cannot be None!")
    get_url = URL_TNS_API + "/object"
    get_query = [("objname", obj_name), ("objid", ""), ("photometry", "1"), ("spectra", "1")]
    json_file = OrderedDict(get_query)
    search_data = {"api_key": TNS_API_KEY, "data": json.dumps(json_file)}
    r = requests.post(get_url, headers=TNS_HEADER, data=search_data, timeout=30.0)
    if r.status_code != 200:
        print(f"ERROR {r.status_code}")
        return {}
    # sleep necessary time to abide by rate limit
    reset = get_reset_time(r)
    if reset is not None:
        time.sleep(reset + 1)
    return r.json()


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

    def query_tns(self) -> None:
        """Query TNS for transient by first id, then internal_names, then coordinate."""
        # first try retrieving by name
        response = tns_object_helper(self.id)
        print(response)

        # if unsuccessful, try searching by internal names.
        for i in self.internal_names:
            r = tns_search_helper(internal_name=i)
            print(r)

        # if unsuccessful, try searching by ra/dec:
        ra_formatted = self._coord.ra.to_string(
            unit=u.hourangle, sep=":", pad=True, precision=2  # pylint: disable=no-member
        )
        dec_formatted = self._coord.dec.to_string(
            unit=u.degree, sep=":", pad=True, alwayssign=True, precision=2  # pylint: disable=no-member
        )
        r = tns_search_helper(ra=ra_formatted, dec=dec_formatted)
        print(r)

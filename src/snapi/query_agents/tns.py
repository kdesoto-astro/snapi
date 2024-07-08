"""Contains TNSQueryAgent for querying transient objects from TNS."""
import json
import os
import time
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..lightcurve import Filter, LightCurve
from ..transient import Transient
from .query_agent import QueryAgent
from .query_result import QueryResult


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


class TNSQueryAgent(QueryAgent):
    """
    QueryAgent for querying transient objects from TNS.
    """

    def __init__(self) -> None:
        self._url_search = "https://www.wis-tns.org/api/get/search"
        self._url_object = "https://www.wis-tns.org/api/get/object"
        self._load_tns_credentials()
        header_phrase = f'tns_marker{{"tns_id": "{self._tns_bot_id}",'
        header_phrase += f'"type": "bot", "name": "{self._tns_bot_name}"}}'
        self._tns_header = {"User-Agent": header_phrase}
        self._timeout = 30.0  # seconds
        self._radius = 3.0  # initial search radius in arcsec

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        objname = query_result["objname"]
        ra = query_result["radeg"] * u.deg  # pylint: disable=no-member
        dec = query_result["decdeg"] * u.deg  # pylint: disable=no-member
        coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
        redshift = query_result["redshift"]
        internal_names = [q.strip() for q in query_result["internal_names"].split(",")]
        internal_names.remove("")
        light_curves = set()
        photometry_arr = query_result["photometry"]
        lc_dict: dict[str, dict[str, Any]] = {}
        for phot_dict in photometry_arr:
            filt = Filter(
                instrument=phot_dict["instrument"]["name"],
                band=phot_dict["filters"]["name"],
                center=np.nan * u.AA,  # pylint: disable=no-member
                # width=phot_dict["width"],
                # lim_mag=phot_dict["lim_mag"],
            )
            if str(filt) in lc_dict:
                lc_dict[str(filt)]["times"].append(
                    Time(phot_dict["jd"], format="jd")
                )  # pylint: disable=no-member
                lc_dict[str(filt)]["mags"].append(phot_dict["flux"])
                lc_dict[str(filt)]["mag_errs"].append(phot_dict["fluxerr"])
                # lc_dict[str(filt)]["zpts"].append(phot_dict["zpt"])
            else:
                lc_dict[str(filt)] = {
                    "times": [Time(phot_dict["jd"], format="jd")],  # pylint: disable=no-member
                    "mags": [phot_dict["flux"]],
                    "mag_errs": [phot_dict["fluxerr"]],
                    # "zpts": [phot_dict["zpt"]],
                    "filt": filt,
                }
        for phot_dict in lc_dict.values():
            phot_mags = np.array(phot_dict["mags"], dtype=object)
            phot_mags[phot_mags == ""] = np.nan
            phot_dict["mags"] = phot_mags.astype(np.float32)

            phot_mag_errs = np.array(phot_dict["mag_errs"], dtype=object)
            phot_mag_errs[phot_mag_errs == ""] = np.nan
            phot_dict["mag_errs"] = phot_mag_errs.astype(np.float32)

            light_curve = LightCurve(
                times=np.array(phot_dict["times"]),
                mags=phot_dict["mags"],
                mag_errs=phot_dict["mag_errs"],
                # zpts=phot_dict["zpts"],
                filt=phot_dict["filt"],
            )
            light_curves.add(light_curve)

        query_result_object = QueryResult(
            objname=objname,
            internal_names=set(internal_names),
            coordinates=coords,
            redshift=redshift,
            light_curves=light_curves,
        )  # TODO: incorporate spectra

        return query_result_object

    def _load_tns_credentials(self) -> None:
        """
        Load TNS credentials from environment variables.
        """
        self._tns_bot_id = os.getenv("TNS_BOT_ID")
        self._tns_bot_name = os.getenv("TNS_BOT_NAME")
        self._tns_api_key = os.getenv("TNS_API_KEY")

    def _tns_search_helper(
        self,
        ra: Optional[Any] = "",
        dec: Optional[Any] = "",
        radius: Optional[float] = 3.0,
        internal_name: Optional[str] = "",
    ) -> Any:
        """Retrieve search results from TNS."""
        search_query = [
            ("ra", ra),
            ("dec", dec),
            ("radius", radius),
            ("units", "arcsec"),
            ("objname", ""),
            ("objname_exact_match", 0),
            ("internal_name", internal_name),
            ("internal_name_exact_match", 1),
            ("objid", ""),
            ("public_timestamp", ""),
        ]
        json_file = OrderedDict(search_query)
        search_data = {"api_key": self._tns_api_key, "data": json.dumps(json_file)}
        r = requests.post(self._url_search, headers=self._tns_header, data=search_data, timeout=self._timeout)
        if r.status_code != 200:
            print(f"ERROR {r.status_code}")
            return {}
        # sleep necessary time to abide by rate limit
        reset = get_reset_time(r)
        if reset is not None:
            time.sleep(reset + 1)
        return r.json()["data"]["reply"]

    def _tns_object_helper(self, obj_name: str) -> Any:
        """Retrieve specific object from TNS."""
        get_query = [("objname", obj_name), ("objid", ""), ("photometry", "1"), ("spectra", "1")]
        json_file = OrderedDict(get_query)
        search_data = {"api_key": self._tns_api_key, "data": json.dumps(json_file)}
        r = requests.post(self._url_object, headers=self._tns_header, data=search_data, timeout=self._timeout)
        if r.status_code != 200:
            print(f"ERROR {r.status_code}")
            return {}
        # sleep necessary time to abide by rate limit
        reset = get_reset_time(r)
        if reset is not None:
            time.sleep(reset + 1)
        return r.json()["data"]["reply"]

    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by name.
        """
        super().query_by_name(names, **kwargs)  # initial checks
        names_arr = np.atleast_1d(names)
        results = []

        for name in names_arr:
            r = self._tns_object_helper(name)
            if "objname" in r:
                results.append(self._format_query_result(r))
                continue
            # if no results, try searching by internal name
            r = self._tns_search_helper(internal_name=name)
            if len(r) == 1:
                result = r[0]
                obj_query = self._tns_object_helper(result["objname"])
                results.append(self._format_query_result(obj_query))
            else:
                return [], False

        return results, True

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        """
        super().query_by_coords(coords, **kwargs)  # initial checks
        coords_arr = np.atleast_1d(coords)
        results = []
        for coord in coords_arr:
            ra_formatted = coord.ra.to_string(
                unit=u.hourangle, sep=":", pad=True, precision=2  # pylint: disable=no-member
            )
            dec_formatted = coord.dec.to_string(
                unit=u.degree, sep=":", pad=True, alwayssign=True, precision=2  # pylint: disable=no-member
            )
            rad = self._radius  # initial radius check
            r = self._tns_search_helper(ra=ra_formatted, dec=dec_formatted, radius=rad)
            while len(r) > 1:
                rad /= 2
                r = self._tns_search_helper(ra=ra_formatted, dec=dec_formatted, radius=rad)
            if len(r) == 1:
                result = r[0]
                obj_query = self._tns_object_helper(result["objname"])
                results.append(self._format_query_result(obj_query))
            else:
                return [], False

        return results, True

    def query_transient(
        self, transient: Transient, **kwargs: Mapping[str, Any]
    ) -> tuple[List[QueryResult], bool]:
        """
        Query by Transient object.
        """
        # first try retrieving by name
        name_list = list(transient.internal_names) + [
            transient.id,
        ]
        r, success = self.query_by_name(name_list, **kwargs)
        if success:
            return r, True
        # if unsuccessful, try retrieving by coordinates
        r, success = self.query_by_coords(transient.coordinates, **kwargs)
        if success:
            return r, True
        return [], False

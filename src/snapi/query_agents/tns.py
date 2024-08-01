"""Contains TNSQueryAgent for querying transient objects from TNS."""
import json
import os
import time
from collections import OrderedDict
from typing import Any, List, Optional

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time
from numpy.typing import NDArray

from ..lightcurve import Filter, LightCurve
from ..spectrum import Spectrometer, Spectrum
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
        self._data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data",
            "tns_public_objects_05_22_24.csv",
        )
        try:
            self._local_df = pd.read_csv(self._data_path)

            # in future, ensure only necessary columns are saved
            keep_cols = [
                "name_prefix",
                "name",
                "internal_names",
                "ra",
                "declination",
                "redshift",
                "type",
            ]
            self._local_df = self._local_df[keep_cols]
            self._local_df.to_csv(self._data_path, index=False)
            self._df_coords = SkyCoord(
                ra=self._local_df["ra"].values * u.deg,  # pylint: disable=no-member
                dec=self._local_df["declination"].values * u.deg,  # pylint: disable=no-member
            )
        except FileNotFoundError:
            self._local_df = None  # type: ignore
            self._df_coords = None

    def _format_light_curves(self, lc_dict: dict[str, dict[str, Any]]) -> set[LightCurve]:
        """
        Format light curves into LightCurve objects.
        """
        light_curves = set()
        for phot_dict in lc_dict.values():
            phot_mags = np.array(phot_dict["mags"], dtype=object)
            phot_mags[phot_mags == ""] = np.nan
            phot_dict["mags"] = phot_mags.astype(np.float32)

            phot_mag_errs = np.array(phot_dict["mag_errs"], dtype=object)
            phot_mag_errs[phot_mag_errs == ""] = np.nan
            phot_dict["mag_errs"] = phot_mag_errs.astype(np.float32)

            light_curve = LightCurve(
                times=phot_dict["times"],
                mags=phot_dict["mags"],
                mag_errs=phot_dict["mag_errs"],
                upper_limits=np.zeros_like(phot_dict["mags"], dtype=bool),
                # zpts=phot_dict["zpts"],
                filt=phot_dict["filt"],
            )
            light_curves.add(light_curve)
        return light_curves

    def _format_spectra(self, query_spectra: List[dict[str, Any]]) -> set[Spectrum]:
        """
        Format spectra into Spectrum objects.
        """
        # extract spectra
        spectra = set()
        for spec in query_spectra:
            file_tns_url = spec["asciifile"]
            wv, flux = self._tns_spec_helper(file_tns_url)
            if len(wv) == 0:
                continue
            spectrometer = Spectrometer(
                instrument=spec["instrument"],
                wavelength_start=wv[0],
                wavelength_delta=wv[1] - wv[0],
                num_channels=len(wv),
            )
            spectra.add(
                Spectrum(
                    time=Time(spec["jd"], format="jd"),  # pylint: disable=no-member
                    fluxes=flux,
                    errors=np.nan * np.ones(len(flux)),
                    spectrometer=spectrometer,
                )
            )
        return spectra

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        objname = query_result["objname"]
        ra = query_result["radeg"] * u.deg  # pylint: disable=no-member
        dec = query_result["decdeg"] * u.deg  # pylint: disable=no-member
        coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
        redshift = query_result["redshift"]
        internal_names = [q.strip() for q in str(query_result["internal_names"]).split(",")]
        photometry_arr = query_result["photometry"]
        lc_dict: dict[str, dict[str, Any]] = {}
        for phot_dict in photometry_arr:
            if phot_dict["instrument"]["name"] == "ZTF-Cam":
                phot_dict["instrument"]["name"] = "ZTF"  # some hard coding
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
        light_curves = self._format_light_curves(lc_dict)
        spectra = self._format_spectra(query_result["spectra"])

        if query_result["object_type"] not in ["nan", ""]:
            spec_class = str(query_result["object_type"])
        else:
            spec_class = None

        query_result_object = QueryResult(
            objname=objname,
            internal_names=set(internal_names),
            coordinates=coords,
            redshift=redshift,
            light_curves=light_curves,
            spectra=spectra,
            spec_class=spec_class,
        )

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
    ) -> List[dict[str, Any]]:
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
            return []
        # sleep necessary time to abide by rate limit
        reset = get_reset_time(r)
        if reset is not None:
            time.sleep(reset + 1)
        out: List[dict[str, Any]] = r.json()["data"]["reply"]
        return out

    def _tns_object_helper(self, obj_name: str) -> dict[str, Any]:
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
        out: dict[str, Any] = r.json()["data"]["reply"]
        return out

    def _tns_spec_helper(self, file_tns_url: str) -> NDArray[np.float32]:
        """Helper function to retrieve spectra from TNS.

        Parameters
        ----------
        file_tns_url : str
            URL to TNS file containing spectra.

        Returns
        -------
        pd.DataFrame
            DataFrame containing spectra information.
        """
        api_data = {"api_key": self._tns_api_key}
        response = requests.post(
            file_tns_url, headers=self._tns_header, data=api_data, stream=True, timeout=self._timeout
        )
        if response.status_code != 200:
            print(f"ERROR {response.status_code}")
            return np.array([[], []])
        text = response.text
        arr = [x.split() for x in text.split("\n") if (len(x) > 0 and x[0] != "#")]
        arr = [[x[0], x[1].strip("\r")] for x in arr if len(x) > 1]
        return np.array(arr).T.astype(np.float32)

    def query_by_name(self, names: Any, **kwargs: Any) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by name.
        """
        super().query_by_name(names, **kwargs)  # initial checks
        names_arr = np.atleast_1d(names)
        results = []

        if "local" in kwargs and kwargs["local"]:
            if self._local_df is None:
                return results, False
            matches = self._local_df.isin(names_arr)["name"].to_numpy()
            r_all = self._local_df[matches]
            for i, r_local in r_all.iterrows():
                objname = r_local["name"]
                coord = self._df_coords[i]
                internal_names = [q.strip() for q in str(r_local["internal_names"]).split(",")]
                if str(r_local["type"]) != "nan":
                    spec_class = str(r_local["type"])
                else:
                    spec_class = None
                results.append(
                    QueryResult(
                        objname=objname,
                        internal_names=set(internal_names),
                        coordinates=coord,
                        redshift=r_local["redshift"],
                        spec_class=spec_class,
                    )
                )
            if len(results) > 0:
                return results, True
            return results, False

        success = False
        for name in names_arr:
            r = self._tns_object_helper(name)
            if ("objname" in r) and isinstance(r["objname"], str):
                results.append(self._format_query_result(r))
                success = True
                continue
            # if no results, try searching by internal name
            r_search = self._tns_search_helper(internal_name=name)
            if len(r_search) == 1:
                result = r_search[0]
                obj_query = self._tns_object_helper(result["objname"])
                results.append(self._format_query_result(obj_query))
                success = True

        return results, success

    def query_by_coords(self, coords: Any, **kwargs: Any) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.

        If "local" kwarg is True, consults local database for object information.
        Is much faster, but also will not include most recent events or photometry/
        spectroscopy.
        """
        try:
            super().query_by_coords(coords, **kwargs)  # initial checks
        except ValueError:
            return [], False
        coords_arr = np.atleast_1d(coords)
        results = []

        if "local" in kwargs and kwargs["local"]:
            if self._local_df is None:
                return results, False
            for coord in coords_arr:
                rad = self._radius * u.arcsec  # pylint: disable=no-member
                # Perform the cone search
                separation = coord.separation(self._df_coords)
                matches = separation <= rad

                # Filter the DataFrame to get the matching entries
                r = self._local_df[matches]

                while len(r) > 1:
                    rad /= 2
                    separation = coord.separation(self._df_coords)
                    matches = separation <= rad

                    # Filter the DataFrame to get the matching entries
                    r = self._local_df[matches]

                if len(r) == 1:
                    objname = r["name"].item()
                    final_coord = self._df_coords[matches]
                    internal_names = [q.strip() for q in str(r["internal_names"].item()).split(",")]
                    if r["type"].item() not in ["nan", ""]:
                        spec_class = str(r["type"].item())
                    else:
                        spec_class = None
                    results.append(
                        QueryResult(
                            objname=objname,
                            internal_names=set(internal_names),
                            coordinates=final_coord,
                            redshift=r["redshift"].item(),
                            spec_class=spec_class,
                        )
                    )

            return results, True

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

        return results, True

    def retrieve_all_names(self, known_class: bool = True) -> NDArray[np.str_]:
        """From the local database, retrieve all object names.
        If known_class is True, only retrieve objects with known
        spectroscopic class. Requires local database to be loaded.
        """
        if self._local_df is None:
            raise ValueError("Local database not loaded.")
        if known_class:
            return self._local_df[~self._local_df.isnull()["type"]]["name"].to_numpy()
        return self._local_df["name"].to_numpy()

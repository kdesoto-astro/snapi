"""Contains ATLASQueryAgent for querying transient objects from ATLAS."""
import io
import os
import re
import sys
import time
from typing import Any, List, Mapping

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..lightcurve import Filter, LightCurve
from .query_agent import QueryAgent
from .query_result import QueryResult


class ATLASQueryAgent(QueryAgent):
    """
    QueryAgent for querying transient objects from ALeRCE.
    """

    def __init__(self) -> None:
        self._baseurl = "https://fallingstar-data.com/forcedphot"

        self._load_credentials()
        self._connect_atlas()

        self._col_renames = {"MJD": "time", "m": "mag", "dm": "mag_unc"}

        self._filt_profiles = {
            "c": [5350.0 * u.AA, 1150.0 * u.AA],  # pylint: disable=no-member
            "o": [6900.0 * u.AA, 1300.0 * u.AA],  # pylint: disable=no-member
        }  # pylint: disable=no-member

    def _load_credentials(self) -> None:
        """
        Load TNS credentials from environment variables.
        """
        self._atlas_user = os.getenv("ATLAS_USERNAME")
        self._atlas_pass = os.getenv("ATLAS_PASSWORD")

    def _connect_atlas(self) -> None:
        """Connect to ATLAS."""

        resp = requests.post(
            url=f"{self._baseurl}/api-token-auth/",
            data={
                "username": self._atlas_user,
                "password": self._atlas_pass,
            },
            timeout=180.0,  # 3 minutes
        )
        if resp.status_code == 200:
            token = resp.json()["token"]
            print(f"Token: {token}")
            headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
        else:
            raise RuntimeError(f"ERROR in connect_atlas(): {resp.status_code}")
        self._headers = headers

    def _query_atlas(self, ra: float, dec: float, min_mjd: float, max_mjd: float) -> pd.DataFrame:
        """Queries the ATLAS Forced photometry service."""
        task_url = None
        while not task_url:
            with requests.Session() as s:
                resp = s.post(
                    f"{self._baseurl}/queue/",
                    headers=self._headers,
                    data={
                        "ra": ra,
                        "dec": dec,
                        "send_email": False,
                        "mjd_min": min_mjd,
                        "mjd_max": max_mjd,
                    },
                )
                if resp.status_code == 201:
                    task_url = resp.json()["url"]
                    print(f"Task url: {task_url}")
                elif resp.status_code == 429:
                    message = resp.json()["detail"]
                    print(f"{resp.status_code} {message}")
                    t_sec = re.findall(r"available in (\d+) seconds", message)
                    t_min = re.findall(r"available in (\d+) minutes", message)
                    if t_sec:
                        waittime = int(t_sec[0])
                    elif t_min:
                        waittime = int(t_min[0]) * 60
                    else:
                        waittime = 10
                    print(f"Waiting {waittime} seconds")
                    time.sleep(waittime)
                else:
                    print(f"ERROR {resp.status_code}")
                    print(resp.text)
                    sys.exit()

        result_url = None
        taskstarted_printed = False

        while not result_url:
            with requests.Session() as s:
                resp = s.get(task_url, headers=self._headers)
                if resp.status_code == 200:
                    if resp.json()["finishtimestamp"] is not None:
                        result_url = resp.json()["result_url"]
                        print(f"Task is complete with results available at {result_url}")
                        break
                    if resp.json()["starttimestamp"]:
                        if not taskstarted_printed:
                            print(f"Task is running (started at {resp.json()['starttimestamp']})")
                            taskstarted_printed = True
                        time.sleep(2)
                    else:
                        # print(f"Waiting for job to start (queued at {resp.json()['timestamp']})")
                        time.sleep(4)
                else:
                    print(f"ERROR {resp.status_code}")
                    print(resp.text)
                    sys.exit()

        with requests.Session() as s:
            if result_url is None:
                return None
            result = s.get(result_url, headers=self._headers).text
            dfresult = pd.read_csv(io.StringIO(result.replace("###", "")), delim_whitespace=True)
        
        return dfresult

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        return QueryResult(
            coordinates=SkyCoord(
                query_result["ra"] * u.deg,  # pylint: disable=no-member
                query_result["dec"] * u.deg,  # pylint: disable=no-member
                frame="icrs",  # pylint: disable=no-member
            ),  # pylint: disable=no-member
            light_curves=query_result["light_curves"],
        )

    def _atlas_lc_helper(self, ra: float, dec: float) -> list[LightCurve]:
        """Helper function that heavily uses ATClean's LightCurve class."""
        min_mjd = 50000.0
        max_mjd = float(Time.now().mjd)
        lc_df = self._query_atlas(ra, dec, min_mjd, max_mjd)
        if lc_df is None:
            return []
        dflux_zero_mask = lc_df["duJy"] > 0
        flux_nan_mask = ~pd.isna(lc_df["uJy"])
        lc_df = lc_df.loc[dflux_zero_mask & flux_nan_mask, :]
        lc_df.rename(columns=self._col_renames, inplace=True)
        lc_df["zpt"] = 23.9

        lcs = []
        for filt, (c, w) in self._filt_profiles.items():
            single_filt_df = lc_df.loc[lc_df["F"] == filt, :]
            snapi_filt = Filter(instrument="ATLAS", band=filt, center=c, width=w)  # c=cyan, o=orange
            print(single_filt_df)
            lcs.append(LightCurve(single_filt_df, filt=snapi_filt))
        return lcs

    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by name.
        NOT IMPLEMENTED: SHOULD USE COORD QUERY.
        """
        return [], False

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        """
        try:
            super().query_by_coords(coords, **kwargs)  # initial checks
        except ValueError:
            return [], False
        coords_arr = np.atleast_1d(coords)
        results = []
        for coord in coords_arr:
            try:
                ra = coord.ra.value
                dec = coord.dec.value
                lcs = self._atlas_lc_helper(ra, dec)
                if len(lcs) == 0:
                    return [], False
                results.append(
                    self._format_query_result(
                        {
                            "ra": ra,
                            "dec": dec,
                            "light_curves": lcs,
                        }
                    )
                )
            except RuntimeError:
                results.append(QueryResult())

        return results, True

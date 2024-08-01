"""Contains TNSQueryAgent for querying transient objects from ALeRCE."""
from typing import Any, List, Mapping

import astropy.units as u
import numpy as np
import pandas as pd
from alerce.core import Alerce  # pylint: disable=import-error
from alerce.exceptions import APIError  # pylint: disable=import-error
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..lightcurve import Filter, LightCurve
from .query_agent import QueryAgent
from .query_result import QueryResult


class ALeRCEQueryAgent(QueryAgent):
    """
    QueryAgent for querying transient objects from ALeRCE.
    """

    def __init__(self) -> None:
        self._client = Alerce()
        self._radius = 3  # default search radius in arcsec
        self._int_to_band = {
            1: "g",
            2: "r",
            3: "i",
        }
        self._mag_keys = [
            "mag",
        ]
        self._mag_err_keys = [
            "e_mag",
        ]

    def _photometry_helper(self, objname: str) -> tuple[set[LightCurve], bool]:
        """
        Helper function for querying photometry data.
        """
        try:
            # Getting detections for an object
            lc = self._client.query_lightcurve(objname, format="pandas")
            dets = list(lc["detections"])[0]
            detections = pd.DataFrame.from_dict(dets)

            try:  # not available with older versions of alerce-client
                forced_phot = list(lc["forced_photometry"])[0]
                forced_detections = pd.DataFrame.from_dict(forced_phot)
                all_detections = pd.concat([detections, forced_detections], join="inner")
            except KeyError:
                all_detections = detections

            if "mjd" not in all_detections.columns:
                return set(), False

            lcs: set[LightCurve] = set()
            if len(all_detections) == 0:
                return lcs, False
            for b in np.unique(all_detections["fid"]):
                filt = Filter(
                    instrument="ZTF",
                    band=self._int_to_band[b],
                    center=np.nan * u.AA,  # pylint: disable=no-member
                )  # pylint: disable=no-member
                mask = all_detections["fid"] == b

                mags = np.nan * np.ones(len(all_detections[mask]))
                mag_errs = np.nan * np.ones(len(all_detections[mask]))
                zpts = np.nan * np.ones(len(all_detections[mask]))

                for mag_key in self._mag_keys:
                    if mag_key in all_detections.columns:
                        mags = all_detections[mask][mag_key]
                        break
                for mag_err_key in self._mag_err_keys:
                    if mag_err_key in all_detections.columns:
                        mag_errs = all_detections[mask][mag_err_key]
                        break
                for mag_err_key in self._mag_err_keys:
                    if mag_err_key in all_detections.columns:
                        mag_errs = all_detections[mask][mag_err_key]
                        break

                lc = LightCurve(
                    times=Time(all_detections[mask]["mjd"], format="mjd"),
                    mags=mags,
                    mag_errs=mag_errs,
                    upper_limits=np.zeros(len(all_detections[mask]), dtype=bool),
                    zpts=zpts,
                    filt=filt,
                )
                lcs.add(lc)
            return lcs, True

        except APIError:
            return set(), False

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        return QueryResult(
            objname=query_result["objname"],
            internal_names=set(),
            coordinates=query_result["coords"],
            redshift=query_result["redshift"],
            light_curves=query_result["light_curves"],
        )

    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by name.
        """
        super().query_by_name(names, **kwargs)  # initial checks
        names_arr = np.atleast_1d(names)
        results = []

        for name in names_arr:
            try:
                photometry, _ = self._photometry_helper(name)
                features = self._client.query_object(name)
                ra = features["meanra"]
                dec = features["meandec"]

                redshift = self._client.catshtm_redshift(ra, dec, self._radius)
                results.append(
                    self._format_query_result(
                        {
                            "objname": name,
                            "coords": SkyCoord(
                                ra * u.deg, dec * u.deg, frame="icrs"  # pylint: disable=no-member
                            ),  # pylint: disable=no-member
                            "light_curves": photometry,
                            "redshift": redshift,
                        }
                    )
                )
            except APIError:
                results.append(QueryResult())

        return results, True

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        NOT IMPLEMENTED: CAN"T QUERY ALERCE BY COORDINATES.
        """
        return [], False

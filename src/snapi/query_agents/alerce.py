"""Contains TNSQueryAgent for querying transient objects from ALeRCE."""
from typing import Any, List, Mapping

import astropy.units as u
import numpy as np
import pandas as pd
from alerce.core import Alerce  # pylint: disable=import-error
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
        self._phot_colnames = ["mjd", "fid", "mag", "e_mag", "magzpsci"]

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
            lcs = set()
            for b in np.unique(all_detections["fid"]):
                filt = Filter(instrument="ZTF", band=b, center=np.nan * u.AA)  # pylint: disable=no-member
                mask = all_detections["fid"] == b
                lc = LightCurve(
                    times=Time(all_detections[mask]["mjd"], format="mjd"),
                    mags=all_detections[mask].get("mag", np.nan * np.ones(len(all_detections[mask]))),
                    mag_errs=all_detections[mask].get("e_mag", np.nan * np.ones(len(all_detections[mask]))),
                    upper_limits=np.zeros(len(all_detections[mask])),
                    zpts=all_detections[mask].get("magzpsci", np.nan * np.ones(len(all_detections[mask]))),
                    filt=filt,
                )
                lcs.add(lc)
            return lcs, True

        except RuntimeError:
            return set(), False

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        print(query_result["light_curves"])
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
                photometry, success = self._photometry_helper(name)
                if success:
                    lcs = photometry
                else:
                    lcs = set()

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
                            "light_curves": lcs,
                            "redshift": redshift,
                        }
                    )
                )
            except RuntimeError:
                results.append(QueryResult())

        return results, True

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        NOT IMPLEMENTED: CAN"T QUERY ALERCE BY COORDINATES.
        """
        return [], False

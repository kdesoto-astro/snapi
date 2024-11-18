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
        self._band_widths = {
            1: 1317.15,
            2: 1553.43,
            3: 1488.99,
        }
        self._band_centers = {
            1: 4746.48,
            2: 6366.38,
            3: 7829.03,
        }

    def _photometry_helper(self, objname: str) -> tuple[list[LightCurve], bool]:
        """
        Helper function for querying photometry data.
        """
        if objname[:3] != "ZTF":
            return [], False
        # Getting detections for an object
        lc = self._client.query_lightcurve(objname, format="pandas")
        detections = pd.DataFrame(list(lc["detections"])[0])

        # add nondetections
        nondetections = pd.DataFrame(list(lc["non_detections"])[0])
        non_na_columns = detections.columns[detections.notna().any()]
        non_na_columns2 = nondetections.columns[nondetections.notna().any()]
        all_detections = pd.concat([detections[non_na_columns], nondetections[non_na_columns2]])

        if "forced_photometry" in lc:  # not available with older versions of alerce-client
            forced_detections = pd.DataFrame(list(lc["forced_photometry"])[0])
            non_na_columns = forced_detections.columns[forced_detections.notna().any()]
            all_detections = pd.concat([all_detections, forced_detections[non_na_columns]])

        if "mjd" not in all_detections.columns:
            return [], False
        if len(all_detections) == 0:
            return [], False

        # add extra fields
        if "extra_fields" in all_detections.columns:
            extra_fields = all_detections["extra_fields"].apply(pd.Series)
            cols_to_use = extra_fields.columns.difference(all_detections.columns)
            all_detections = pd.concat(
                [all_detections, extra_fields[cols_to_use]],
                axis=1,
            )
            all_detections.drop(columns=["extra_fields"], inplace=True)

        all_detections.reset_index(drop=True, inplace=True)

        # add missing fields
        if "mag" not in all_detections.columns:
            if "magpsf" in all_detections.columns:
                all_detections["mag"] = all_detections["magpsf"]
                all_detections["e_mag"] = all_detections["sigmapsf"]
                if "magap" in all_detections.columns:
                    all_detections["mag"].fillna(all_detections["magap"])
                    all_detections["e_mag"].fillna(all_detections["sigmagap"])
            else:
                all_detections["mag"] = all_detections["magap"]
                all_detections["e_mag"] = all_detections["sigmagap"]
        else:
            if "magpsf" in all_detections.columns:
                all_detections["mag"].fillna(all_detections["magpsf"])
                all_detections["e_mag"].fillna(all_detections["sigmapsf"])
            if "magap" in all_detections.columns:
                all_detections["mag"].fillna(all_detections["magap"])
                all_detections["e_mag"].fillna(all_detections["sigmagap"])

        all_detections["magzpsci"] = 23.90  # bandaid to make fluxes all mu-Jy

        # find non-detections
        all_detections["upper_limit"] = False
        nondetect_mask = (all_detections["mag"] == 100.0) | (all_detections["mag"].isna())
        if "diffmaglim" in all_detections.columns:
            all_detections.loc[nondetect_mask, "mag"] = all_detections.loc[nondetect_mask, "diffmaglim"]
            all_detections.loc[nondetect_mask, "e_mag"] = np.nan
            all_detections.loc[nondetect_mask, "upper_limit"] = True

        all_detections.rename(columns={
            'mjd': 'time',
            'e_mag': 'mag_unc',
            'upper_limit': 'non_detections',
            'magzpsci': 'zpt'
        }, inplace=True)

        lcs: list[LightCurve] = []

        for b in np.unique(all_detections["fid"]):
            filt = Filter(
                instrument="ZTF",
                band=self._int_to_band[b],
                center=self._band_centers[b] * u.AA,  # pylint: disable=no-member
                width=self._band_widths[b] * u.AA,  # pylint: disable=no-member
            )  # pylint: disable=no-member
            mask = all_detections["fid"] == b
            lc = LightCurve(all_detections.loc[mask,:], filt=filt)
            lcs.append(lc)
            
        return lcs, True

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

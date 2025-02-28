"""Contains TNSQueryAgent for querying transient objects from ALeRCE."""
from typing import Any, List, Mapping
from requests.exceptions import ChunkedEncodingError

import astropy.units as u
import numpy as np
import pandas as pd
from alerce.core import Alerce  # pylint: disable=import-error
from alerce.exceptions import APIError  # pylint: disable=import-error
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..photometry import Filter, LightCurve
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
        detections = pd.DataFrame(lc['detections'].item())
        detections.dropna(axis=1, inplace=True)
        
        # add nondetections
        nondetections = pd.DataFrame(lc["non_detections"].item())
        nondetections.dropna(axis=1, inplace=True)
        all_detections = pd.concat([detections, nondetections], copy=False, ignore_index=True)

        try: # forced phot available in alerce-1.3 or newer
            forced_detections = self._client.query_forced_photometry(objname, format="pandas")
            forced_detections.dropna(axis=1, inplace=True)
            all_detections = pd.concat([all_detections, forced_detections], ignore_index=True)
        except:
            pass

        if "mjd" not in all_detections.columns:
            return [], False
        if len(all_detections) == 0:
            return [], False
        
        """
        # add extra fields: seems unnecessary right now
        if "extra_fields" in all_detections.columns:
            extra_fields = all_detections["extra_fields"].apply(pd.Series)
            cols_to_use = extra_fields.columns.difference(all_detections.columns)
            all_detections = pd.concat(
                [all_detections, extra_fields[cols_to_use]],
                axis=1,
            )
            all_detections.drop(columns=["extra_fields"], inplace=True)
        """

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

        all_detections["zeropoint"] = 23.90  # bandaid to make fluxes all mu-Jy

        # find non-detections
        all_detections["upper_limit"] = False
        nondetect_mask = (all_detections["mag"] == 100.0) | (all_detections["mag"].isna())
        all_detections.loc[nondetect_mask, "e_mag"] = np.nan
        all_detections.loc[nondetect_mask, "upper_limit"] = True
        
        if "diffmaglim" in all_detections.columns:
            all_detections.loc[nondetect_mask, "mag"] = all_detections.loc[nondetect_mask, "diffmaglim"]
            
        all_detections.rename(columns={
            'e_mag': 'mag_error',
        }, inplace=True)
        
        # remove repeated non-detections + forced-detections
        nonrepeat_mask = all_detections.groupby('mjd', group_keys=False)['upper_limit'].idxmin()
        all_detections = all_detections.loc[nonrepeat_mask,['mjd', 'fid', 'mag', 'mag_error', 'upper_limit', 'zeropoint']]
        all_detections['flux'] = np.nan
        all_detections['flux_error'] = np.nan
        
        all_detections.set_index(
            pd.DatetimeIndex(
                Time(all_detections["mjd"], format="mjd").to_datetime()
            ), inplace=True
        )
        all_detections.index.name = "mjd"
        all_detections.drop(columns='mjd', inplace=True)

        lcs: list[LightCurve] = []

        for b in np.unique(all_detections["fid"]):
            filt = Filter(
                instrument="ZTF",
                band=self._int_to_band[b],
                center=self._band_centers[b],
                width=self._band_widths[b],
            )  # pylint: disable=no-member
            mask = all_detections["fid"] == b
            lc = LightCurve(all_detections.loc[mask], filt=filt, phased=False, validate=False)
            lc.update()
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
            except (ChunkedEncodingError, APIError):
                results.append(QueryResult())

        return results, True

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        NOT IMPLEMENTED: CAN"T QUERY ALERCE BY COORDINATES.
        """
        return [], False

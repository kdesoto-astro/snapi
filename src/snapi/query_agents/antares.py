"""Contains TNSQueryAgent for querying transient objects from ALeRCE."""
from typing import Any, List, Mapping

import astropy.units as u
import numpy as np
from antares_client.models import Locus  # pylint: disable=import-error
from antares_client.search import get_by_ztf_object_id, search  # pylint: disable=import-error
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn
from astropy.time import Time

from ..lightcurve import Filter, LightCurve
from .query_agent import QueryAgent
from .query_result import QueryResult


class ANTARESQueryAgent(QueryAgent):
    """
    QueryAgent for querying transient objects from ALeRCE.
    """

    def __init__(self) -> None:
        self._radius = 3 * u.arcsec  # pylint: disable=no-member
        self._ts_cols = [
            "ant_mjd",
            "ztf_magpsf",
            "ztf_sigmapsf",
            "ztf_fid",
            "ant_ra",
            "ant_dec",
            "ztf_magzpsci",
            "ant_maglim",
        ]
        self._int_to_band = {
            1: "g",
            2: "r",
            3: "i",
        }

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        return QueryResult(
            objname=query_result["objname"],
            internal_names={query_result["internal_name"]},
            coordinates=query_result["coords"],
            light_curves=query_result["light_curves"],
        )

    def _cone_search_helper(self, center: SkyCoord, radius: u.Quantity) -> list[Locus]:
        """Directly sends request to ANTARES databases to avoid outdated antares-client
        issues."""
        try:
            search_query = {
                "query": {
                    "bool": {
                        "filter": {
                            "sky_distance": {
                                "distance": f"{radius.to(u.deg).value} degree",  # pylint: disable=no-member
                                "htm16": {"center": center.to_string()},
                            },
                        },
                    },
                },
            }
            return list(search(search_query))
        except RuntimeError:
            return []

    def _locus_helper(self, locus: Locus) -> tuple[float, float, set[LightCurve]]:
        time_series = locus.timeseries[self._ts_cols]
        for col in self._ts_cols:
            if isinstance(time_series[col], MaskedColumn):
                time_series[col] = time_series[col].filled(np.nan)

        time_series["time"] = Time(time_series["ant_mjd"], format="mjd", scale="utc")
        time_series.rename_column("ztf_magpsf", "mag")
        time_series.rename_column("ztf_sigmapsf", "mag_err")
        time_series.rename_column("ztf_magzpsci", "zpt")

        bands = time_series["ztf_fid"]
        ra = np.nanmean(time_series["ant_ra"])
        dec = np.nanmean(time_series["ant_dec"])

        # handle non-detections by setting nan mags to maglim
        time_series["non_detections"] = np.isnan(time_series["mag"])
        antlims = time_series[time_series["non_detections"]]["ant_maglim"]
        time_series["mag"][time_series["non_detections"]] = antlims
        time_series.remove_columns(["ant_ra", "ant_dec", "ztf_fid", "ant_mjd", "ant_maglim"])
        lcs = set()
        for b in np.unique(bands):
            mask = bands == b
            filt = Filter(
                instrument="ZTF",
                band=self._int_to_band[b],
                center=np.nan * u.AA,  # pylint: disable=no-member
            )
            # first deal with detections
            lc = LightCurve(time_series[mask], filt=filt)
            lcs.add(lc)

        return ra, dec, lcs

    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by name.
        """
        super().query_by_name(names, **kwargs)  # initial checks
        names_arr = np.atleast_1d(names)
        results = []

        for name in names_arr:
            try:
                locus = get_by_ztf_object_id(name)
                if locus is None:
                    results.append(QueryResult())
                    continue
                ra, dec, lcs = self._locus_helper(locus)
                results.append(
                    self._format_query_result(
                        {
                            "internal_name": locus.locus_id,
                            "objname": locus.properties["ztf_object_id"],
                            "coords": SkyCoord(
                                ra * u.deg, dec * u.deg, frame="icrs"  # pylint: disable=no-member
                            ),  # pylint: disable=no-member
                            "light_curves": lcs,
                        }
                    )
                )
            except RuntimeError:
                results.append(QueryResult())

        return results, True

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by coordinates.
        """
        super().query_by_coords(coords, **kwargs)
        coords_arr = np.atleast_1d(coords)
        results = []
        for coord in coords_arr:
            try:
                for locus in self._cone_search_helper(coord, self._radius):
                    ra, dec, lcs = self._locus_helper(locus)

                    results.append(
                        self._format_query_result(
                            {
                                "internal_name": locus.locus_id,
                                "objname": locus.properties["ztf_object_id"],
                                "coords": SkyCoord(
                                    ra * u.deg, dec * u.deg, frame="icrs"  # pylint: disable=no-member
                                ),  # pylint: disable=no-member
                                "light_curves": lcs,
                            }
                        )
                    )
            except RuntimeError:
                results.append(QueryResult())

        return results, True

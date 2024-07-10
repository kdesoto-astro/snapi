"""Contains TNSQueryAgent for querying transient objects from ALeRCE."""
from typing import Any, List, Mapping

import astropy.units as u
import numpy as np
from antares.models import Locus  # pylint: disable=import-error
from antares_client.search import cone_search, get_by_ztf_object_id  # pylint: disable=import-error
from astropy.coordinates import SkyCoord

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
        ]

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

    def _locus_helper(self, locus: Locus) -> tuple[float, float, set[LightCurve]]:
        time_series = locus.timeseries[self._ts_cols]
        time_series.rename_column("ant_mjd", "time")
        time_series.rename_column("ztf_magpsf", "mag")
        time_series.rename_column("ztf_sigmapsf", "mag_err")
        time_series.rename_column("ztf_fid", "band")
        time_series.rename_column("ztf_magzpsci", "zp")

        ra = np.nanmean(time_series["ant_ra"])
        dec = np.nanmean(time_series["ant_dec"])

        time_series.remove_columns(["ant_ra", "ant_dec"])

        lcs = set()
        for b in np.unique(time_series["band"]):
            mask = time_series["band"] == b
            filt = Filter(
                instrument="ZTF",
                band="g" if b == 1 else "r",
                center=np.nan * u.AA,  # pylint: disable=no-member
            )
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
                ra, dec, lcs = self._locus_helper(locus)
                results.append(
                    self._format_query_result(
                        {
                            "internal_name": locus.alert_id,
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
                for locus in cone_search(coord, self._radius):
                    ra, dec, lcs = self._locus_helper(locus)

                    results.append(
                        self._format_query_result(
                            {
                                "internal_name": locus.alert_id,
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

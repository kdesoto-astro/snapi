"""Contains TNSQueryAgent for querying transient objects from ALeRCE."""
from typing import Any, List, Mapping
import astropy.units as u
import numpy as np
import pandas as pd
from antares_client.models import Locus  # pylint: disable=import-error
from antares_client.search import get_by_ztf_object_id, get_by_id, search  # pylint: disable=import-error
from antares_client.exceptions import AntaresException
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn
from astropy.time import Time

from ..photometry import Filter, LightCurve
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
            "ant_maglim",
            "ztf_magzpsci",
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
        internal_names = {query_result["internal_name"],}
        if query_result['id'] is not None:
            internal_names.add(query_result['id'])
        return QueryResult(
            objname=query_result["objname"],
            internal_names=internal_names,
            coordinates=query_result["coords"],
            light_curves=query_result["light_curves"],
            spec_class=query_result["spec_class"]
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

    def _locus_helper(self, locus: Locus) -> tuple[float, float, list[LightCurve]]:
        try:  # TODO: better fix for this
            time_series = locus.timeseries[self._ts_cols]
        except KeyError:  # sometimes zeropoint isn't there
            time_series = locus.timeseries[self._ts_cols[:-1]]
            time_series["ztf_magzpsci"] = np.nan

        for col in self._ts_cols:
            if isinstance(time_series[col], MaskedColumn):
                time_series[col] = time_series[col].filled(np.nan)

        times = Time(time_series["ant_mjd"], format="mjd", scale="utc")
        time_series.rename_column("ztf_magpsf", "mag")
        time_series.rename_column("ztf_sigmapsf", "mag_err")
        # time_series.rename_column("ztf_magzpsci", "zpt")
        time_series["zpt"] = 23.90  # bandaid to auto-calibrate

        bands = time_series["ztf_fid"]
        ra = float(np.nanmean(time_series["ant_ra"]))
        dec = float(np.nanmean(time_series["ant_dec"]))

        # handle non-detections by setting nan mags to maglim
        time_series["non_detections"] = np.isnan(time_series["mag"])
        antlims = time_series[time_series["non_detections"]]["ant_maglim"]
        time_series["mag"][time_series["non_detections"]] = antlims
        time_series.remove_columns(["ant_mjd", "ant_ra", "ant_dec", "ztf_fid", "ant_mjd", "ant_maglim"])

        df = pd.DataFrame(time_series.to_pandas(), index=pd.DatetimeIndex(times.to_datetime()))

        lcs = []
        for b in np.unique(bands):
            mask = bands == b
            filt = Filter(
                instrument="ZTF",
                band=self._int_to_band[b],
                center=np.nan * u.AA,  # pylint: disable=no-member
            )
            lc = LightCurve(df[mask], filt=filt)
            lcs.append(lc)
            
        # retrieve TNS name
        if 'tns_public_objects' in locus.catalog_objects:
            tns_name = locus.catalog_objects['tns_public_objects'][0]['name']
            spec_class = locus.catalog_objects['tns_public_objects'][0].get('type', None)
        else:
            tns_name = None
            spec_class = None

        return ra, dec, lcs, tns_name, spec_class

    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        """
        Query transient objects by name.
        """
        super().query_by_name(names, **kwargs)  # initial checks
        names_arr = np.atleast_1d(names)
        results = []

        for name in names_arr:
            try:
                locus1 = get_by_ztf_object_id(name)
                locus2 = get_by_id(name)
                if (locus1 is None) and (locus2 is None):
                    results.append(QueryResult())
                    continue
                else:
                    locus = locus1 if (locus2 is None) else locus2
                ra, dec, lcs, tns_name, spec_class = self._locus_helper(locus)
                results.append(
                    self._format_query_result(
                        {
                            "id": tns_name,
                            "internal_name": locus.locus_id,
                            "objname": locus.properties["ztf_object_id"],
                            "coords": SkyCoord(
                                ra * u.deg, dec * u.deg, frame="icrs"  # pylint: disable=no-member
                            ),  # pylint: disable=no-member
                            "light_curves": lcs,
                            "spec_class": spec_class,
                        }
                    )
                )
            except (RuntimeError, AntaresException):
                results.append(QueryResult())

        return results, True

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
                for locus in self._cone_search_helper(coord, self._radius):
                    ra, dec, lcs, tns_name, spec_class = self._locus_helper(locus)

                    results.append(
                        self._format_query_result(
                            {
                                "id": tns_name,
                                "internal_name": locus.locus_id,
                                "objname": locus.properties["ztf_object_id"],
                                "coords": SkyCoord(
                                    ra * u.deg, dec * u.deg, frame="icrs"  # pylint: disable=no-member
                                ),  # pylint: disable=no-member
                                "light_curves": lcs,
                                "spec_class": spec_class,
                            }
                        )
                    )
            except RuntimeError:
                results.append(QueryResult())

        return results, True

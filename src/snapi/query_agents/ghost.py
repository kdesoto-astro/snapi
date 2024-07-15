"""Contains QueryAgent for querying transient host galaxies using GHOST."""
import os
import shutil
from typing import Any, List, Mapping

import numpy as np
import pandas as pd
from astro_ghost.ghostHelperFunctions import (
    findNewHosts,
    getDBHostFromTransientCoords,
    getDBHostFromTransientName,
    getGHOST,
    getHostFromHostName,
)

from .query_agent import QueryAgent
from .query_result import HostQueryResult, QueryResult


class GHOSTQueryAgent(QueryAgent):
    """QueryAgent that uses
    GHOST to query transient host galaxies.
    """

    def __init__(self) -> None:
        self._ghost_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data",
            "ghost_db.csv",
        )
        self._tmp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "tmp"
        )
        os.makedirs(self._tmp_path, exist_ok=True)
        getGHOST(real=True, verbose=False, installpath=self._ghost_path, clobber=False)

        self._col_names = [
            "NED_name",
        ]  # TODO: fill in

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        result = HostQueryResult(
            objname=query_result["hostname"],
            internal_names=query_result["host_internal_names"],
            coordinates=query_result["host_coords"],
            redshift=query_result["host_redshift"],
            spectra=query_result["host_spectra"],
        )
        return result

    def query_by_name(self, names: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        super().query_by_name(names)

        names_arr = np.atleast_1d(names)

        result_df = getDBHostFromTransientName(names_arr, GHOSTpath=self._ghost_path)
        result_df2 = getHostFromHostName(names_arr, GHOSTpath=self._ghost_path)
        result_df = pd.concat([result_df, result_df2], ignore_index=True, join="outer")

        sub_df = result_df[self._col_names]
        result_dicts = sub_df.to_dict(orient="records")

        if len(result_df) > 0:
            return [self._format_query_result(r) for r in result_dicts], True

        return [], False

    def query_by_coords(self, coords: Any, **kwargs: Mapping[str, Any]) -> tuple[List[QueryResult], bool]:
        try:
            super().query_by_coords(coords, **kwargs)  # initial checks
        except ValueError:
            return [], False
        coords_arr = np.atleast_1d(coords)

        if "force_manual" in kwargs:
            force_manual = bool(kwargs["force_manual"])
        else:
            force_manual = False

        result_df, _ = getDBHostFromTransientCoords(coords_arr, GHOSTpath=self._ghost_path)

        if len(result_df) == 0 or force_manual:  # try manual association
            filler_names = [hash(c) for c in coords_arr]
            filler_class = ["" for c in coords_arr]
            result_df2 = findNewHosts(
                filler_names, coords_arr, filler_class, ascentMatch=True, savepath=self._tmp_path
            )
            if len(result_df2) == 0:
                result_df3 = findNewHosts(
                    filler_names, coords_arr, filler_class, ascentMatch=True, savepath=self._tmp_path, rad=150
                )
                final_results = pd.concat([result_df, result_df3], ignore_index=True)
            else:
                final_results = pd.concat([result_df, result_df2], ignore_index=True)
        else:
            final_results = result_df

        sub_df = final_results[self._col_names]
        result_dicts = sub_df.to_dict(orient="records")

        # Remove everything in self._tmp_path directory
        shutil.rmtree(self._tmp_path)
        os.makedirs(self._tmp_path, exist_ok=True)

        if len(final_results) > 0:
            return [self._format_query_result(r) for r in result_dicts], True

        return [], False

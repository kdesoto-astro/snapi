# QueryAgent for YSE DR1
import os
from typing import Any, List, Mapping

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from ..photometry import Filter, LightCurve
from .query_agent import QueryAgent
from .query_result import QueryResult


class YSEQueryAgent(QueryAgent):
    """
    QueryAgent for querying transient objects from ALeRCE.
    """

    def __init__(self, directory=None) -> None:
        self._band_centers = {
            'g': 4810.16,
            'r': 6155.47,
            'i': 7503.03,
            'z': 8668.36,
            'X': 4746.48,
            'Y': 6366.38,
        }
        self._band_widths = {
            'g': 1148.66,
            'r': 1397.73,
            'i': 1292.39,
            'z': 1038.82,
            'X': 1317.15,
            'Y': 1553.43,
        }

        if directory is None:
            raise ValueError("Directory with YSE files must be given!")
        if not os.path.exists(directory):
            raise ValueError("Directory path not found!")

        self.directory = directory


    def _find_meta_start_of_data(self, fn):
        """Helper function for YSE fn import.
        """
        obj_name = fn.split("/")[-1].split(".")[0]
        meta = {'NAME': obj_name}
        meta_keys = [
            'MWEBV',
            'REDSHIFT_FINAL',
            'SPEC_CLASS',
            'SPEC_CLASS_BROAD',
            'RA',
            'DEC',
        ]
        with open(fn, 'r', encoding='utf-8') as f:
            for i, row in enumerate(f):
                for k in meta_keys:
                    if k+":" in row:
                        meta[k] = row.split(":")[1].strip().split()[0].strip()
                if row.strip() == '':
                    return i + 6, meta

    def _yse_file_reader(self, fn):
        """Imports single YSE SNANA file.
        """
        header, meta = self._find_meta_start_of_data(fn)

        lcs = []

        df = pd.read_csv(fn, skiprows=header, delim_whitespace=True, skipfooter=1)
        df = df[['MAG', 'MAGERR', 'FLT']]
        df.rename(columns={'MJD': 'mjd', 'MAG': 'mag', 'MAGERR': 'mag_unc'})
        df['zpt'] = 23.9

        # convert to astropy Timeseries
        for b in np.unique(df['FLT']):
            if b == 'X':
                b_true = 'g'
            elif b == 'Y':
                b_true = 'r'
            else:
                b_true = b
            df_b = df[df['FLT'] == b]
            df_b = df_b.drop(columns=['FLT'])
            ts = TimeSeries.from_pandas(df_b)
            filt = Filter(
                band=b_true,
                instrument = 'ZTF' if b in ['X', 'Y'] else 'YSE',
                center=BAND_WAVELENGTHS[b] * u.AA,
                width=BAND_WIDTHS[b] * u.AA,
            )
            lc = LightCurve(
                ts, filt=filt
            )
            lcs.append(lc)
        return lcs, meta

    def _format_query_result(self, query_result: dict[str, Any]) -> QueryResult:
        """
        Format query result into QueryResult object.
        """
        meta = query_result['meta']
        return QueryResult(
            objname=query_result["name"],
            coordinates=SkyCoord(ra=meta["RA"], dec=meta["DEC"]),
            redshift=meta['REDSHIFT_FINAL'],
            spec_class=meta['SPEC_CLASS'],
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
                fn = os.path.join(
                    self.directory, name + ".snana.dat"
                )

                if not os.path.exists(fn):
                    results.append(QueryResult())
                    continue

                lcs, meta = self._yse_file_reader(fn)

                results.append(
                    self._format_query_result(
                        {
                            "name": name,
                            "meta": meta,
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
        return [], False
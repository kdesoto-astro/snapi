"""Contains ATLASQueryAgent for querying transient objects from ATLAS."""
import io
import os
import re
import sys
import time
from typing import Any, List, Mapping
from dotenv import load_dotenv

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.stats import median_abs_deviation

from ..photometry import Filter, LightCurve
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

        self._col_renames = {"MJD": "mjd", "uJy": "flux", "duJy": "flux_unc", "m": "mag", "dm": "mag_unc"}

        self._filt_profiles = {
            "c": [5350.0 * u.AA, 1150.0 * u.AA],  # pylint: disable=no-member
            "o": [6900.0 * u.AA, 1300.0 * u.AA],  # pylint: disable=no-member
        }  # pylint: disable=no-member

        # store info on past completed jobs
        jobs = self._get_all_jobs()
        self._past_jobs = {}

        for job in jobs:
            if job.get('ra') is None:
                continue
            self._past_jobs[(job.get('ra'), job.get('dec'))] = (job.get('url'), job.get('result_url'))


    def _load_credentials(self) -> None:
        """
        Load TNS credentials from environment variables.
        """
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            ".env"
        )
        load_dotenv(dotenv_path=env_path)

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
            timeout=300.0,  # 3 minutes
        )
        if resp.status_code == 200:
            token = resp.json()["token"]
            print(f"Token: {token}")
            headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
        else:
            raise RuntimeError(f"ERROR in connect_atlas(): {resp.status_code}")
        self._headers = headers

    def _get_all_jobs(self):
        all_jobs = []
        next_url = f"{self._baseurl}/queue/"

        with requests.Session() as s:
            while next_url:
                response = s.get(next_url, headers=self._headers, timeout=300.0)
                response.raise_for_status()
                if response.status_code == 200:
                    data = response.json()
                    all_jobs.extend(data['results'])
                    next_url = data.get('next')  # URL for the next page of results

        return all_jobs


    def _query_atlas(self, ra: float, dec: float, min_mjd: float, max_mjd: float) -> pd.DataFrame:
        """Queries the ATLAS Forced photometry service."""

        # Find the job with matching RA and Dec
        task_url = None
        result_url = None

        if (ra, dec) in self._past_jobs:
            task_url, result_url = self._past_jobs[(ra, dec)]
            if result_url is None:
                print(ra, dec)
                return None
        else:
            print(ra, dec)
            return None

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

        
        taskstarted_printed = False

        while not result_url:
            with requests.Session() as s:
                resp = s.get(task_url, headers=self._headers, timeout=300.0)
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
            result = s.get(result_url, headers=self._headers, timeout=300.0).text
            dfresult = pd.read_csv(io.StringIO(result.replace("###", "")), sep='\s+')
        
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
        min_mjd = 57233.0 # July 30, 2015
        max_mjd = float(Time.now().mjd)
        lc_df = self._query_atlas(ra, dec, min_mjd, max_mjd)
        if lc_df is None:
            return []
        dflux_zero_mask = lc_df["duJy"] > 0
        flux_nan_mask = ~pd.isna(lc_df["uJy"])
        lc_df = lc_df.loc[dflux_zero_mask & flux_nan_mask, :]
        lc_df, summary_df = self._subtract_fps_baseline(lc_df)
        if lc_df is None:
            return []

        fqcf_keep = summary_df.index[summary_df['flags'] == 0].unique()
        lc_df = lc_df.loc[
            (lc_df.F.isin(fqcf_keep) & (lc_df["flags"] == 0))
        ]

        lc_df.rename(columns=self._col_renames, inplace=True)
        lc_df["zpt"] = 23.9

        lcs = []
        for filt, (c, w) in self._filt_profiles.items():
            single_filt_df = lc_df.loc[lc_df["F"] == filt, :]
            snapi_filt = Filter(instrument="ATLAS", band=filt, center=c, width=w)  # c=cyan, o=orange
            lc = LightCurve(single_filt_df, filt=snapi_filt, phased=False)
            lcs.append(lc)
        return lcs
    
    def _subtract_fps_baseline(
        self,
        fp_df_orig,
        window="14D",
    ):
        """
        calculate the baseline region for an IPAC fps light curve and produce
        flux calibrated photometry

        Parameters
        ----------
        fps_df_orig : pd.DataFrame
            Dataframe with forced photometry for single coordinate.

        window : int, offset, or BaseIndexer subclass (optional, default = '10D')
            Size of the moving window. This is the number of observations used
            for calculating the rolling median.
            If its an offset then this will be the time period of each window.
            Each window will be a variable sized based on the observations
            included in the time-period. This is only valid for datetimelike
            indexes.

        Returns
        -------
        fp_df : pd.DataFrame
            Modified dataframe with flags and calibrated fluxes.
        """

        flags = {
            'pre_em': 1,
            'post_em': 2,
            'outliers': 4,
        }

        summary_flags = {
            'large_peak_scatter': 1,
            'high_scatter': 2,
            'N_baseline': 4,
            'bad_baseline': 8
        }

        fp_df = fp_df_orig.copy()
        fp_df.index = Time(fp_df.MJD, format='mjd').datetime

        fp_df.drop_duplicates(
            ['F', 'uJy', 'duJy'],
            inplace=True
        )

        # drop uncertainties > 160
        fp_df = fp_df.loc[fp_df.duJy < 160, :]

        # template shift corrections
        fp_df.loc[fp_df.MJD <= 58882.0, 'uJy'] -= 5.0
        fp_df.loc[fp_df.MJD > 58882.0, 'uJy'] += 5.0

        fp_df['flags'] = np.zeros(len(fp_df)).astype(int)

        fp_df['rolling_median'] = np.nan
        fp_df['rolling_unc'] = np.nan
        fp_df['uJy'] = fp_df['uJy'].astype(float)
        fp_df['duJy'] = fp_df['duJy'].astype(float)
        fp_df['m'] = np.nan
        fp_df['dm'] = np.nan

        summary_df = pd.DataFrame(
            columns=[
                'N_obs','t_fcqfid_max', 'median_bl', 'median_unc_bl', 'N_bl', 'snr_rms_bl',
                'median_pre', 'median_unc_pre', 'N_pre', 'snr_rms_pre',
                'median_post', 'median_unc_post', 'N_post', 'snr_rms_post', 'which_bl'
            ],
            index = fp_df.F.unique()
        )
        summary_df['flags'] = np.zeros(len(summary_df)).astype(int)
        summary_df['det_sn'] = False

        for ufid in summary_df.index:

            # only consider observations where flag < bad obs
            fcqf_df = fp_df.loc[fp_df.F == ufid]
            summary_df.loc[ufid, 'N_obs'] = len(fcqf_df)
            
            if (len(fcqf_df) < 2): # skip if fewer than two observations for this filter
                continue

            # taking rolling mean of fluxes
            flux_series = fcqf_df.uJy
            roll_med = flux_series.rolling(window, center=True).median()
            roll_unc = fcqf_df.duJy.rolling(window, center=True).median()
            fp_df.loc[fp_df.F == ufid, 'rolling_median'] = roll_med
            fp_df.loc[fp_df.F == ufid, 'rolling_unc'] = roll_unc

            # extract peak rolling flux and corresponding jd
            roll_peak = (roll_med - roll_unc).dropna().idxmax()
            flux_max = fcqf_df.loc[roll_peak, "uJy"]

            t_max = fcqf_df.loc[roll_peak, "MJD"]

            # get scatter of fluxes (single value)
            flux_scatt = median_abs_deviation(
                fcqf_df.uJy,
                scale='normal'
            )

            # variability checks
            max_over_scatt = flux_max/flux_scatt
            peak_snr = flux_max/roll_unc.loc[roll_peak]

            #if (max_over_scatt > 5 and peak_snr > 5):
            if peak_snr > 5:
                summary_df.loc[ufid, 'det_sn'] = True # SN detected in data
                summary_df.loc[ufid, 't_fcqfid_max'] = t_max # estimated peak time for that filter

        # estimate peak of light curve - use only first field? not sure why fcqfid cut is here
        t_peaks = summary_df.loc[
            summary_df['det_sn'],
            't_fcqfid_max'
        ]
        N_obs = summary_df.loc[
            summary_df['det_sn'],
            'N_obs'
        ]
        if len(N_obs) == 0: # no SN found?
            print("N_obs = 0")
            return None, None

        t_peak = (t_peaks.dropna() * N_obs.dropna()).sum() / N_obs.dropna().sum()
        summary_df['t_peak'] = t_peak
        summary_df['t_peak_scatter'] = t_peaks.std(ddof=1)

        if t_peaks.std(ddof=1) > 50:
            print("high peak scatter")
            return None, None
        
        # mask only points within 10 days of peak
        around_max_mask = (fp_df.MJD - t_peak > - 10) & (
            fp_df.MJD - t_peak < 10
        ) & (
            fp_df.uJy > 0
        )

        if around_max_mask.any():
            diff_flux_around_max = fp_df.loc[around_max_mask, "uJy"]
            mag_min = (23.9 - 2.5*np.log10(diff_flux_around_max)).dropna().min()
            #calculate time when SN signal is "gone" via Co56 decay at z ~ 0.009
            t_faded = t_peak + (22.5 - mag_min)/0.009
        else:
            t_faded = t_peak + 611. # catch strange cases where t_gmax != t_rmax

        summary_df['t_faded'] = t_faded

        # measure the baseline: either 100 days pre-peak or t_faded days post-peak
        pre_bl_mask = (t_peak - fp_df.MJD > 100)
        post_bl_mask = (fp_df.MJD > t_faded)
        bl_mask = pre_bl_mask | post_bl_mask

        for ufid in summary_df.index:

            for (mask, suffix) in zip((bl_mask, pre_bl_mask, post_bl_mask), ('bl', 'pre', 'post')):
                # get baseline values
                bl = fp_df.loc[(fp_df.F == ufid) & mask]

                base_flux = bl.uJy
                base_flux_unc = bl.duJy

                outlier_mask = (base_flux - base_flux.median()).abs() / base_flux_unc > 5

                if (suffix == 'bl') and outlier_mask.any():           
                    bl.loc[outlier_mask, "flags"] += flags['outliers']

                summary_df.loc[ufid, f'N_{suffix}'] = (~outlier_mask).sum()

                if (~outlier_mask).sum() > 1:
                    #calculate weighted mean and chisq fit to constant
                    bl_sig = bl.loc[~outlier_mask]
                    bl_flux = bl_sig.uJy
                    bl_unc = bl_sig.duJy

                    # calculate median
                    summary_df.loc[ufid, f'median_{suffix}'] = bl_flux.median()

                    # using bootstrapping to place unc on median and trim means
                    bootstraps = np.random.choice(
                        bl_flux, size=(1000, len(bl_flux)), replace=True
                    )
                    medians = np.median(bootstraps, axis=1)
                    summary_df.loc[ufid, f'median_unc_{suffix}'] = np.ptp(np.percentile(medians, (16,84))) / 2
                    
                    baseline = summary_df.loc[ufid, f'median_{suffix}']
                    baseline_comb = summary_df.loc[ufid, 'median_bl']
                    
                    snr = (bl_flux - baseline) / bl_unc
                    snr_rms = snr.quantile([0.16, 0.84]).diff().iloc[-1] / 2
                    summary_df.loc[ufid, f'snr_rms_{suffix}'] = snr_rms
                    
                    if suffix in ['pre', 'post']:
                        # calculate pre- and post-peak emission
                        em = (bl.rolling_median - baseline_comb) / bl.rolling_unc
                        if (em >= 5).sum() >= 2:
                            bl.loc[em >= 5, 'flags'] += flags[f'{suffix}_em']

                # re-attach df to master df
                fp_df.loc[(fp_df.F == ufid) & mask] = bl

            # now determine which baseline to use
            fcqf_df = fp_df.loc[fp_df.F == ufid]
            pre_em = (fcqf_df['flags'] & flags['pre_em']).astype(bool).any()
            post_em = (fcqf_df['flags'] & flags['post_em']).astype(bool).any()

            if (pre_em & post_em):
                summary_df.loc[ufid, 'which_bl'] = 'pre+post'
                summary_df.loc[ufid, 'flags'] += summary_flags['bad_baseline']
                suffix = 'bl'

            elif (pre_em and summary_df.loc[ufid, "N_post"] >= 3) or (
                (summary_df.loc[ufid, "N_pre"] < 5) and (summary_df.loc[ufid, "N_post"] >= 10)
            ):
                summary_df.loc[ufid, 'which_bl'] = 'post'
                suffix = 'post'

            elif (post_em and summary_df.loc[ufid, "N_pre"] >= 3) or (
                (summary_df.loc[ufid, "N_post"] < 5) and (summary_df.loc[ufid, "N_pre"] >= 10)
            ):
                summary_df.loc[ufid, 'which_bl'] = 'pre'
                suffix = 'pre'
            else:
                summary_df.loc[ufid, 'which_bl'] = 'pre+post'
                suffix = 'bl'
                if pre_em or post_em:
                    summary_df.loc[ufid, 'flags'] += summary_flags['bad_baseline']

            # check enough baseline points
            if summary_df.loc[ufid, f'N_{suffix}'] < 10:
                summary_df.loc[ufid, "flags"] += summary_flags['N_baseline']

            baseline = summary_df.loc[ufid, f'median_{suffix}']
            baseline_unc = summary_df.loc[ufid, f'median_unc_{suffix}']

            fp_df.loc[fp_df.F == ufid, "uJy"] -= baseline

            fp_df.loc[fp_df.F == ufid, "duJy"] = np.sqrt(
                fp_df.loc[fp_df.F == ufid, "duJy"]**2 + baseline_unc**2
            ) * summary_df.loc[ufid, f'snr_rms_{suffix}']

            pos_mask = (fp_df.uJy > 0) & (fp_df.F == ufid)
            fp_df.loc[pos_mask,'m'] = 23.9 - 2.5 * np.log10(fp_df.loc[pos_mask, "uJy"])
            fp_df.loc[pos_mask,'dm'] = (2.5 / np.log(10.)) * fp_df.loc[pos_mask, "duJy"] / fp_df.loc[pos_mask, "uJy"]

        return fp_df, summary_df

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

from typing import Sequence, TypeVar

import george
import numba
import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.stats import median_abs_deviation


T = TypeVar("T", int, float, np.float64)

def generate_gp(
    gp_vals: NDArray[np.float64], gp_errs: NDArray[np.float64], stacked_data: NDArray[np.float64]
) -> george.GP:
    """Generate a Gaussian Process object.

    Parameters
    ----------
    gp_vals: NDArray[np.float64]
        the values of the Gaussian Process
    gp_errs: NDArray[np.float64]
        the errors of the Gaussian Process
    stacked_data: NDArray[np.float64]
        the stacked data

    Returns
    -------
    george.GP
        the Gaussian Process object
    """
    kernel = np.var(gp_vals) * george.kernels.ExpSquaredKernel([100, 1], ndim=2)
    gaussian_process = george.GP(kernel)
    gaussian_process.compute(stacked_data, gp_errs)

    def neg_ln_like(params: NDArray[np.float64]) -> float:
        """Return negative log likelihood of GP."""
        gaussian_process.set_parameter_vector(params)
        return -gaussian_process.log_likelihood(gp_vals)  # type: ignore[no-any-return]

    def grad_neg_ln_like(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return gradient of negative log likelihood of GP."""
        gaussian_process.set_parameter_vector(params)
        return -gaussian_process.grad_log_likelihood(gp_vals)  # type: ignore[no-any-return]

    result = scipy.optimize.minimize(
        neg_ln_like, gaussian_process.get_parameter_vector(), jac=grad_neg_ln_like
    )
    gaussian_process.set_parameter_vector(result.x)

    return gaussian_process


# @numba.njit(parallel=True)  # type: ignore
def resample_helper(cen: Sequence[T], unc: Sequence[T], num: int) -> NDArray[np.float64]:
    """numba-enhanced helper to generate many resampled LCs."""
    rng = np.random.default_rng()
    sampled_vals = np.zeros((num, len(cen)), dtype=np.float64)
    for i in range(num):  # pylint: disable=not-an-iterable
        sampled_vals[i] += rng.normal(loc=cen, scale=unc)
    return sampled_vals


# @numba.njit(parallel=True)  # type: ignore
def update_merged_fluxes(
    keep_idxs: NDArray[np.int64], flux: NDArray[np.float64], flux_unc: NDArray[np.float64], nondetect_thresh: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Update merged fluxes with numba."""
    new_f = []
    new_ferr = []
    new_nondet = []
    for i, keep_idx in enumerate(keep_idxs):  # pylint: disable=not-an-iterable
        if i == len(keep_idxs) - 1:
            repeat_idx_subset = np.arange(keep_idx, len(flux))
        else:
            repeat_idx_subset = np.arange(keep_idx, keep_idxs[i + 1])


        weights = 1.0 / flux_unc.iloc[repeat_idx_subset] ** 2
        new_f_single = np.average(flux.iloc[repeat_idx_subset], weights=weights)
        new_var = median_abs_deviation(flux.iloc[repeat_idx_subset]) / np.sqrt(len(flux.iloc[repeat_idx_subset]))
        new_var += 1.0 / np.sqrt(np.sum(weights)) / np.sqrt(len(flux.iloc[repeat_idx_subset]))

        new_f.append(new_f_single)
        new_ferr.append(new_var)
        new_nondet.append(new_f_single / new_var < nondetect_thresh)

    return np.array(new_f), np.array(new_ferr), np.array(new_nondet)


# @numba.njit(parallel=True)
def calc_all_deltas(series: NDArray[np.float64], use_sum: bool = False) -> NDArray[np.float64]:
    """Calculate all pairwise distances between values in each set,
    assuming the first axis delineates different sets.
    """
    deltas = []
    # for i in range(series.shape[1]):
    for i in numba.prange(series.shape[1]):  # pylint: disable=not-an-iterable
        for j in range(i + 1, series.shape[1]):
            if use_sum:
                deltas.append((series[:, j] + series[:, i]).astype(np.float64))
            else:
                deltas.append((series[:, j] - series[:, i]).astype(np.float64))
    return np.vstack(deltas).T
"""Stores superclass for sampler objects and fit results."""
from typing import Optional, Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, assert_all_finite
from matplotlib.axes import Axes

from ..photometry import Photometry
from ..lightcurve import LightCurve
from ..formatter import Formatter


class SamplerResult:
    """Stores the results of a model sampler."""
    def __init__(
            self,
            fit_parameters: Any,
            sampler_name: Optional[str] = None,
        ):
        """
        Parameters
        ----------
        fit_parameters : Any
            The parameters of the fit. Array associated
            with each key must be one-dimensional.
        """
        if isinstance(fit_parameters, dict):
            for key, value in fit_parameters.items():
                if np.at_least1d(value).ndim != 1:
                    raise ValueError(f"Value associated with key {key} must be one-dimensional.")
            self._fit_params = pd.DataFrame(fit_parameters)
        elif isinstance(fit_parameters, pd.DataFrame):
            self._fit_params = fit_parameters.copy()
        else:
            raise ValueError("fit_parameters must be a dictionary or DataFrame.")

        self._sampler_name = sampler_name
        self.score = 0.0

    def __str__(self):
        return str(self._fit_params)
    
    @property
    def fit_parameters(self):
        """Returns the fit parameters."""
        return self._fit_params.copy()
    
    def save(self, save_filename: str, hdf5_path: Optional[str] = None):
        """Save the fit results to an HDF5 file.

        Parameters
        ----------
        save_filename : str
            The filename to save the fit results to.
        hdf5_path : str, optional
            The path to save the fit results to in the HDF.
        """
        if hdf5_path is None:
            if self._sampler_name is not None:
                hdf5_path = f"fit_result_{self._sampler_name}"
            else:
                hdf5_path = "fit_result"

        with pd.HDFStore(save_filename, 'a') as store:
            store.put(hdf5_path, self._fit_params)
            store.get_storer(hdf5_path).attrs.score = self.score
            store.get_storer(hdf5_path).attrs.sampler_name = self._sampler_name


    @classmethod
    def load(cls, load_filename: str, hdf5_path: Optional[str] = None):
        """Load a FitResult from an HDF5 file.

        Parameters
        ----------
        load_filename : str
            The filename to load the fit results from.
        hdf5_path : str, optional
            The path to load the fit results from in the HDF

        Returns
        -------
        FitResult
            The loaded FitResult object.
        """
        if hdf5_path is None:
            hdf5_path = 'fit_result'
        
        with pd.HDFStore(load_filename, 'r') as store:
            fit_params = store[hdf5_path]
            score = store.get_storer(hdf5_path).attrs.score
            sampler_name = store.get_storer(hdf5_path).attrs.sampler_name

        new_obj = cls(fit_params, sampler_name)
        new_obj.score = score
        return new_obj
    
class Sampler(BaseEstimator):
    """Stores the superclass for sampler objects.
    Follows scikit-learn API conventions and inherits from
    BaseEstimator.
    """
    def __init__(self):
        """Initializes the Fitter object."""
        self._mag_y = False # whether model uses magnitudes or fluxes as the y value
        self._nparams = 0 # number of parameters in the model
        self._sampler_name = '' # name of the sampler
        self.result = None

    def _validate_arrs(self, X, y):
        """Broadcasts X and y to right shape, and checks that both arrays are valid inputs.
        Allows bands to be strings, enforces floats
        in other two columns of X. Also removes any rows with NaNs."""
        X = np.atleast_2d(X)

        if X.shape[1] > 3:
            raise ValueError("X must have 3 or fewer columns.")

        if X.shape[1] == 1:
            X = np.c_[X, np.zeros(X.shape[0]), np.ones(X.shape[0])]
        elif X.shape[1] == 2:
            X = np.c_[X, np.ones(X.shape[0])]

        y = np.atleast_1d(y)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        check_array(y, ensure_2d=False, force_all_finite=False)

        # remove nan/inf rows
        mask = np.all(np.isfinite(X[:,2].astype(np.float32))) & np.isfinite(y)

        return X[mask], y[mask]

    def fit(
            self, X: NDArray[np.object_], # pylint: disable=invalid-name
            y: NDArray[np.float32],
        ) -> None: 
        """Fit the data.

        Parameters
        ----------
        X : np.ndarray
            The X data to fit. If 1d, will be reshaped to 2d.
            First column = times, second column = bands, third column = errors.
        y : np.ndarray
            The y data to fit.
        """
        self._X, self._y = self._validate_arrs(X, y) # pylint: disable=attribute-defined-outside-init, invalid-name
        self.result = None # where FitResult will be stored.
            
    def predict(self, X: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]: # pylint: disable=invalid-name
        """Returns set of modeled y values from
        set sampled parameters.

        Parameters
        ----------
        X : np.ndarray
            The x data to predict.

        Returns
        -------
        np.ndarray
            The predicted y data.
        np.ndarray
            The filtered X data.
        """
        # Check if fit has been called
        check_is_fitted(self, '_is_fitted')
        placeholder_y = np.zeros(X.shape[0])
        val_x, _ = self._validate_arrs(X, placeholder_y)

        return val_x, val_x # Placeholder for actual prediction
    
    def score(self, X: NDArray[np.object_], y: NDArray[np.float32]) -> float:
        """Returns the score of the model.

        Parameters
        ----------
        X : np.ndarray
            The x data to score.
        y : np.ndarray
            The y data to score.

        Returns
        -------
        float
            The score of the model.
        """
        # Check if fit has been called
        check_is_fitted(self, '_is_fitted')

        # Input validation
        val_x, val_y = self._validate_arrs(X, y)
        y_pred, _ = self.predict(val_x)
        return self._reduced_chi_squared(val_x, val_y, y_pred)
    
    def _convert_photometry_to_arrs(self, photometry: Photometry) -> tuple[NDArray[np.object_], NDArray[np.float32]]:
        """Converts a Photometry object to arrays for fitting."""
        # first rescale the photometry
        photometry.phase()
        #adjust so max flux = 1.0
        photometry.normalize()
        dets = photometry.detections
        if self._mag_y:
            x_arr = np.array([dets['time'].mjd, dets['filters'], dets['mag_err']], dtype=object).T
        else:
            x_arr = np.array([dets['time'].mjd, dets['filters'], dets['flux_err']], dtype=object).T

        y = dets['mag'].data if self._mag_y else dets['flux'].data
        return x_arr, y
    
    def fit_photometry(self, photometry: Photometry) -> None:
        """Fit a Photometry object. Saves information to
        Photometry object.

        Parameters
        ----------
            The Photometry object to fit.
        """
        x_phot, y_phot = self._convert_photometry_to_arrs(photometry)
        self.fit(x_phot, y_phot)

    def predict_photometry(self, photometry: Photometry) -> tuple[NDArray[np.float32], NDArray[np.float32], bool]:
        """Returns set of modeled fluxes from a real Photometry object.
        These effective y values will be fluxes or magnitudes depending on the Fitter.

        Parameters
        ----------
        photometry : Photometry
            The Photometry object to predict.

        Returns
        -------
        np.ndarray
            The predicted y data.
        bool
            Whether the y data is in magnitudes.
        """
        x_phot, _ = self._convert_photometry_to_arrs(photometry)
        return *self.predict(x_phot), self._mag_y
    
    def plot_fit(
            self, ax: Axes, formatter: Optional[Formatter] = None,
            photometry: Optional[Photometry] = None,
            X: Optional[NDArray[np.object_]] = None,
            dense: bool=True
        ) -> None:
        """Plots the model fit.

        Parameters
        ----------
        X : np.ndarray
            The x data to plot.
        y : np.ndarray
            The y data to plot.
        dense : bool, optional
            Whether to make time array dense for better plotting.
        """
        if photometry is not None:
            X, _ = self._convert_photometry_to_arrs(photometry)
        if X is None:
            X = self._X
        if formatter is None:
            formatter = Formatter()
        for b in np.unique(X[:, 1]):
            if dense:
                t_arr = np.linspace(np.min(X[:,0])-20., np.max(X[:,0])+20., 1000)
                new_x = np.repeat(t_arr[np.newaxis,:].T, 3, axis=1).astype(object)
                new_x[:,1] = b
                new_x[:,2] = 0.0 # filler
                y_pred, val_x = self.predict(new_x)
            else:
                y_pred, val_x = self.predict(X[X[:,1] == b])
            ax.plot(
                val_x[:,0], y_pred[0],
                label=f'{b}_{self._sampler_name}',
                color=formatter.edge_color,
                linewidth=formatter.line_width,
                alpha=formatter.nondetect_alpha,
            )
            for y_pred_single in y_pred[-30:]:
                ax.plot(
                    val_x[:,0], y_pred_single,
                    color=formatter.edge_color,
                    linewidth=formatter.line_width,
                    alpha=formatter.nondetect_alpha,
                )
            formatter.rotate_colors()
            formatter.rotate_markers()
        return ax
    
    def load_result(self, load_filename: str, hdf5_path: Optional[str] = None):
        """Load a FitResult from an HDF5 file.

        Parameters
        ----------
        load_filename : str
            The filename to load the fit results from.
        hdf5_path : str, optional
            The path to load the fit results from in the HDF
        """
        self.result = SamplerResult.load(load_filename, hdf5_path)
    
    def _eff_variance(self, X):
        """Returns the effective variance of the model."""
        return X[:, 2] ** 2 # default
    
    def _reduced_chi_squared(self, X, y, y_pred):
        """Returns the reduced chi-squared value of the model.

        Parameters
        ----------
        X : np.ndarray
            The x data to score.
        y : np.ndarray
            The y data to score.
        y_pred : np.ndarray
            The predicted y data.

        Returns
        -------
        float
            The reduced chi-squared value.
        """
        return np.median(
            np.sum(
                (y[np.newaxis,:] - y_pred) ** 2 / self._eff_variance(X) / (len(y) - self._nparams - 1),
                axis=1,
            )
        )



    

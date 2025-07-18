# pylint: disable=invalid-name
"""Stores superclass for sampler objects and fit results."""
import os
from typing import Any, Mapping, Optional, Type, TypeVar
import abc

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from ..formatter import Formatter
from ..photometry import Photometry
from ..base_classes import Base

ResultT = TypeVar("ResultT", bound="SamplerResult")


class SamplerPrior(Base):
    """Stores prior information for sampler. Only supports Gaussianity
    or log-Gaussianity. If log-Gaussianity, parameters are assumed of logged
    distribution."""
    
    def __init__(
        self,
        prior_info: Optional[pd.DataFrame] = None
    ):
        """Stores prior information for the Sampler."""
        super().__init__()
        if prior_info is not None:
            self._df = prior_info
            self._rng = np.random.default_rng()
            self.arr_attrs.append("_df")
            self.update()
        else:
            self._df = None
        
    def update(self) -> None:
        """Rearrange priors so correlated priors are sampled
        in the right order.
        """
        pass
        
    def __get__(self, key):
        """Retrieve prior info for a certain parameter."""
        return self._df[key, :]
    
    def __set__(self, key, val):
        """Change prior information for a certain parameter."""
        self._df.loc[key] = pd.Series(val)
        self.update()
        
    @property
    def dataframe(self):
        """Return all prior info in a dataframe."""
        return self._df.copy()
    
    @abc.abstractmethod
    def sample(self, cube, use_numpyro=False):
        """Sample from priors. If numpyro=True, then
        use the numpyro framework.
        """
        pass
            
    
class SamplerResult(Base):
    """Stores the results of a model sampler."""

    def __init__(
        self,
        fit_parameters: Any = None,
        sampler_name: Optional[str] = None,
        event_id: str = "",
    ):
        """
        Parameters
        ----------
        fit_parameters : Any
            The parameters of the fit. Array associated
            with each key must be one-dimensional.
        """
        super().__init__()
        
        self.id = event_id
        
        if isinstance(fit_parameters, dict):
            for key, value in fit_parameters.items():
                if np.atleast_1d(value).ndim != 1:
                    raise ValueError(f"Value associated with key {key} must be one-dimensional.")
            self._fit_params = pd.DataFrame(fit_parameters)
        elif isinstance(fit_parameters, pd.DataFrame):
            self._fit_params = fit_parameters.copy()
        elif fit_parameters is None:
            self._fit_params = None
        else:
            raise ValueError("fit_parameters must be a dictionary or DataFrame.")

        self._sampler_name = sampler_name
        self.score = None
        
        self.arr_attrs.append("_fit_params")
        self.meta_attrs.extend(["_id", "_sampler_name", "score"])

    def __str__(self) -> str:
        return str(self._fit_params)
    
    @property
    def sampler(self) -> str:
        """Return sampler name."""
        return self._sampler_name

    @property
    def fit_parameters(self) -> pd.DataFrame:
        """Returns the fit parameters."""
        return self._fit_params.copy()
    
    @fit_parameters.setter
    def fit_parameters(self, fit_parameters: pd.DataFrame) -> None:
        """Augments fit_parameters attribute.
        Columns should stay the same.
        """

        if (self._fit_params is not None) and np.any(
            np.intersect1d(fit_parameters.columns, self._fit_params.columns) != sorted(self._fit_params.columns)
        ):
            raise ValueError("columns must match between original and new fit_parameters")
            
        self._fit_params = fit_parameters
    
    def corner_plot(self, formatter=Formatter()):
        """Plot corner plot of fit_parameters."""
        fig = corner.corner(
            self._fit_params.to_numpy(),
            bins=20,
            labels=self._fit_params.columns,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 20},
            color=formatter.edge_color
        )
        return fig
    
    def umap(self, ax, diagnostic='pca', formatter=Formatter()):
        """Plot UMAP representation of samples."""
        import umap
        import umap.plot
        
        features = self._fit_params.to_numpy()
        
        # add jitter
        for i in range(features.shape[1]):
            features[:,i] += np.random.normal(scale=np.std(features) / 1e3, size=len(features))
            
        nan_features = np.any(np.isnan(features), axis=1)
        mapper = umap.UMAP().fit(features[~nan_features], ensure_all_finite=False)
        
        if diagnostic is None:
            ax = umap.plot.points(
                mapper,
                cmap=formatter.cmap,
                ax=ax,
            )
        else:
            ax = umap.plot.diagnostic(
                mapper,
                cmap=formatter.cmap,
                ax=ax,
                diagnostic=diagnostic
            )
        return ax
    
    def pacmap(self, ax, formatter=Formatter()):
        """Plot PACMAP representation of samples."""
        import pacmap
        
        features = self._fit_params.to_numpy()
        
        # add jitter
        for i in range(features.shape[1]):
            features[:,i] += np.random.normal(scale=np.nanstd(features) / 100, size=len(features))
        nan_features = np.any(np.isnan(features), axis=1)
        
        embedding = pacmap.PaCMAP(n_components=2)
        X_transformed = embedding.fit_transform(features[~nan_features], init="pca")

        ax.scatter(
            X_transformed[labels == l, 0],
            X_transformed[labels == l, 1],
            s=formatter.marker_size,
            color=formatter.edge_color,
            marker=formatter.marker_style,
            alpha=0.3,
        )
        return ax

    
class Sampler(BaseEstimator):  # type: ignore
    """Stores the superclass for sampler objects.
    Follows scikit-learn API conventions and inherits from
    BaseEstimator.
    """

    _X: NDArray[np.object_]
    _y: NDArray[np.float64]

    def __init__(self) -> None:
        """Initializes the Fitter object."""
        self._mag_y = False  # whether model uses magnitudes or fluxes as the y value
        self._nparams = 0  # number of parameters in the model
        self._sampler_name = ""  # name of the sampler
        self.result: Optional[SamplerResult] = None

    def _validate_arrs(
        self, X: NDArray[np.object_], y: NDArray[np.float64],
        event_indices=None,
    ) -> tuple[NDArray[np.object_], NDArray[np.float64]]:
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

        check_array(y, ensure_2d=False, ensure_all_finite=False)

        # remove nan/inf rows
        mask = np.isfinite(X[:, 2].astype(np.float64)) & np.isfinite(y)

        if event_indices is not None:
            # Prepare new index ranges
            cumsum_mask = np.cumsum(mask)

            # Prepare new index ranges
            new_index_ranges = []

            for original_start, original_end in event_indices:
                # Adjust the start and end indices based on the cumulative mask
                num_retained_before_start = cumsum_mask[original_start - 1] if original_start > 0 else 0
                num_retained_before_end = cumsum_mask[original_end - 1]

                # Append the updated range directly
                new_index_ranges.append((num_retained_before_start, num_retained_before_end))

            return X[mask], y[mask], np.array(new_index_ranges)

        return X[mask], y[mask], None
    
    @property
    def name(self) -> str:
        """Return sampler name."""
        return self._sampler_name

    def fit(
        self,
        X: NDArray[np.object_],  # pylint: disable=invalid-name
        y: NDArray[np.float64],
        event_indices=None
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
        self._X, self._y, self._idxs = self._validate_arrs(
            X, y, event_indices
        )  # pylint: disable=attribute-defined-outside-init, invalid-name
        self.result = None  # where FitResult will be stored.

    def predict(
        self,
        X: NDArray[np.object_],
        num_fits: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.object_]]:  # pylint: disable=invalid-name
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
        check_is_fitted(self, "_is_fitted")
        placeholder_y = np.zeros(X.shape[0])
        val_x, _, _ = self._validate_arrs(X, placeholder_y)

        if num_fits:
            return val_x[:num_fits, 0].astype(np.float64), val_x  # Placeholder for actual prediction
        return val_x[:, 0].astype(np.float64), val_x

    def score(self, X: NDArray[np.object_], y: NDArray[np.float64], **kwargs: Mapping[str, Any]) -> float:
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
        check_is_fitted(self, "_is_fitted")

        # Input validation
        val_x, val_y, _ = self._validate_arrs(X, y)
        y_pred, _ = self.predict(val_x)
        return self._reduced_chi_squared(val_x, val_y, y_pred, **kwargs)

    def _convert_photometry_to_arrs(
        self, photometry: Photometry,
    ) -> tuple[NDArray[np.object_], NDArray[np.float64]]:
        """Converts a Photometry object to arrays for fitting."""
        dets = photometry.detections
        dets_mjd = dets.index.total_seconds().to_numpy() / (24 * 3600)
        if self._mag_y:
            x_arr = np.array([dets_mjd, dets["filter"], dets["mag_error"]], dtype=object).T
        else:
            x_arr = np.array([dets_mjd, dets["filter"], dets["flux_error"]], dtype=object).T

        y = dets["mag"].to_numpy() if self._mag_y else dets["flux"].to_numpy()
        return x_arr, y
    
    def _convert_hierarchical_photometry_to_arrs(
        self, photometry_arr: list[Photometry],
    ) -> tuple[NDArray[np.object_], NDArray[np.float64]]:
        """Converts many Photometry object to arrays for hierarchical fitting."""
        full_x = None
        full_y = None
        start_idxs = []
        end_idxs = []

        for phot in photometry_arr:
            x_arr, y = self._convert_photometry_to_arrs(phot)

            if full_x is None:
                full_x = x_arr
                full_y = y
                start_idxs.append(0)
            else:
                start_idxs.append(len(full_y))
                full_x = np.vstack((full_x, x_arr))
                full_y = np.append(full_y, y)
                
            end_idxs.append(len(full_y))

        return full_x, full_y, np.array([start_idxs, end_idxs]).T

    def fit_photometry(self, photometry: Photometry, **kwargs) -> None:
        """Fit a Photometry object. Saves information to
        Photometry object.

        Parameters
        ----------
            The Photometry object to fit.
        """
        x_phot, y_phot = self._convert_photometry_to_arrs(photometry)
        self.fit(x_phot, y_phot, **kwargs)

    def fit_photometry_hierarchical(self, photometry_arr: list[Photometry], **kwargs) -> None:
        """Fit a Photometry object. Saves information to
        Photometry object.

        Parameters
        ----------
            The Photometry object to fit.
        """
        x_phot, y_phot, idxs = self._convert_hierarchical_photometry_to_arrs(photometry_arr)
        self.fit(x_phot, y_phot, event_indices=idxs, **kwargs)

    def predict_photometry(
        self, photometry: Photometry
    ) -> tuple[NDArray[np.float64], NDArray[np.object_], bool]:
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
        self,
        ax: Axes,
        formatter: Optional[Formatter] = None,
        photometry: Optional[Photometry] = None,
        X: Optional[NDArray[np.object_]] = None,
        dense: bool = True,
    ) -> Axes:
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

        for b in np.unique(X[:,1]):
            if dense:
                t = X[:,0].astype(float)
                t_arr = np.linspace(np.min(t) - 20.0, np.max(t) + 20.0, 1000)
                new_x = np.repeat(t_arr[np.newaxis, :].T, 3, axis=1).astype(object)
                new_x[:, 1] = b
                new_x[:, 2] = 0.0  # filler
                y_pred, val_x = self.predict(new_x)
            else:
                y_pred, val_x = self.predict(X[X[:, 1] == b])
            if len(y_pred) == 0:
                formatter.rotate_colors()
                formatter.rotate_markers()
                continue
            ax.plot(
                val_x[:, 0],
                y_pred[0],
                label=f"{b}_{self._sampler_name}",
                color=formatter.edge_color,
                linewidth=formatter.line_width,
                alpha=formatter.nondetect_alpha,
            )
            for y_pred_single in y_pred[-30:]:
                ax.plot(
                    val_x[:, 0],
                    y_pred_single,
                    color=formatter.edge_color,
                    linewidth=formatter.line_width,
                    alpha=formatter.nondetect_alpha,
                )
            formatter.rotate_colors()
            formatter.rotate_markers()
        return ax

    def load_result(self, sampler_result: SamplerResult) -> None:
        """Load a FitResult from an HDF5 file.

        Parameters
        ----------
        load_prefix : str
            The filename to load the fit results from.
        hdf5_path : str, optional
            The path to load the fit results from in the HDF
        """
        self.result = sampler_result
        self._is_fitted = True

    def _eff_variance(self, X: NDArray[np.object_]) -> NDArray[np.float64]:
        """Returns the effective variance of the model."""
        return X[:, 2].astype(np.float64) ** 2  # default

    def _reduced_chi_squared(
        self, X: NDArray[np.object_], y: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """Returns the reduced chi-squared value of each fit from the model's results.

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
        np.ndarray
            The reduced chi-squared value for each fit.
        """
        return np.sum(
            (y[np.newaxis, :] - y_pred) ** 2 / self._eff_variance(X) / (len(y) - self._nparams - 1),
            axis=1,
        )
# pylint: disable=invalid-name
"""Stores superclass for sampler objects and fit results."""
import os
from typing import Any, Mapping, Optional, Type, TypeVar
import warnings

import jax.numpy as jnp
import jax
import numpy as np
import numpyro.distributions as dist
import numpyro
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import truncnorm

from ..formatter import Formatter
from ..photometry import Photometry
from ..base_classes import Base

jax.config.update('jax_platform_name', 'cpu')
warnings.filterwarnings("ignore", category=UserWarning, message="Found vars in model but not guide: {.*}")

ResultT = TypeVar("ResultT", bound="SamplerResult")


class SamplerPrior(Base):
    """Stores prior information for sampler. Only supports Gaussianity
    or log-Gaussianity. If log-Gaussianity, parameters are assumed of logged
    distribution."""
    
    def __init__(
        self,
        prior_info: pd.DataFrame
    ):
        """Stores prior information for the Sampler."""
        super().__init__()
        for k in ['param', 'mean', 'stddev', 'min', 'max', 'logged', 'relative', 'relative_op']:
            if k not in prior_info:
                raise ValueError(f"column {k} not in prior_info!")
        self._df = prior_info
        self._df.loc[:,'logged'] = self._df.loc[:,'logged'].astype(bool)
        self._rng = np.random.default_rng()
        self.arr_attrs.append("_df")
        self.update()
        
    def update(self) -> None:
        """Rearrange priors so correlated priors are sampled
        in the right order.
        """
        rerun = False
        while rerun:
            rerun = False
            ordered_indices = {p: i for i, p in enumerate(self._df['param'])}
            for r in self._df.loc[~pd.isna(self._df['relative'])]:
                if ordered_indices[r['relative']] > r.index:
                    # move r to back of array
                    self._df.pop(r, inplace=True)
                    self._df.append(r, inplace=True)
                    rerun = True
                    
        self._relative_mask = self._df['relative'].notna().to_numpy()
        self._relative_idxs = []

        # get shuffled relative idxs
        for rel_val in self._df.loc[self._relative_mask, 'relative']:
            # Find the index where the 'param' column matches the 'relative' value
            index = self._df[self._df['param'] == rel_val].index
            self._relative_idxs.append(index[0])  # Storing the first matching index
        
        
        self._df.loc[
            self._relative_mask, 'min'
        ] = self._df.iloc[self._relative_idxs]['min'].to_numpy()
        
        self._df.loc[
            self._relative_mask, 'max'
        ] = self._df.iloc[self._relative_idxs]['max'].to_numpy()
        
                    
        self._tga = ((self._df['min'] - self._df['mean']) / self._df['stddev']).to_numpy()
        self._tgb = ((self._df['max'] - self._df['mean']) / self._df['stddev']).to_numpy()
            
        # faster sample calls
        self._logged = self._df['logged']
        self._mean = self._df['mean'].to_numpy()
        self._std = self._df['stddev'].to_numpy()
        self._numpyro_sample_arr = jnp.array(
            self._df[['min', 'max', 'mean', 'stddev']].to_numpy().astype(float).T
        )
        
        self._relative_idxs_jax = jnp.array(self._relative_idxs)
        self._logged_jax = jnp.array(self._logged)
        self._relative_mask_jax = jnp.array(self._relative_mask)
        self._params = self._df['param'].to_numpy()
        
        
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
    
    def _trunc_norm(self, fields):
        """Provides keyword parameters to numpyro's TruncatedNormal, using the fields in PriorFields.

        Parameters
        ----------
        fields : PriorFields
            The (low, high, mean, standard deviation) fields of the truncated normal distribution.

        Returns
        -------
        numpyro.distributions.TruncatedDistribution
            A truncated normal distribution.
        """
        return dist.TruncatedNormal(
            loc=fields[2], scale=fields[3], 
            low=fields[0], high=fields[1],
            validate_args=False
        )
    
    def sample(self, cube, use_numpyro=False):
        """Sample from priors. If numpyro=True, then
        use the numpyro framework.
        """
        if use_numpyro:
            num_params = self._numpyro_sample_arr.shape[1]
            # Extract necessary initialization values from the provided sample array
            min_vals, max_vals, init_loc, init_scale = self._numpyro_sample_arr
            # Start by sampling all parameters initially
            with numpyro.plate("params", num_params):
                initial_samples = numpyro.sample(
                    "initial_samples",
                    dist.TruncatedNormal(
                        loc=init_loc,
                        scale=init_scale,
                        low=min_vals,
                        high=max_vals
                    )
                )
                
                # Compute the adjustment only for relative ones
                relative_shifts = jnp.zeros(num_params)
                relative_shifts = relative_shifts.at[self._relative_mask_jax].set(
                    init_loc[self._relative_idxs_jax]
                )
                
                # New means for the next sampling, adjusted by the relative criteria
                adjusted_locs = init_loc + relative_shifts

                # Reapply the constraints to adjusted_locs to make sure they stay within bounds
                adjusted_locs_constrained = jnp.clip(adjusted_locs, min_vals, max_vals)

                # Re-sample using the adjusted means only for relative parameters
                vals = numpyro.sample(
                    "samples",
                    dist.TruncatedNormal(
                        loc=adjusted_locs_constrained,
                        scale=init_scale,
                        low=min_vals,
                        high=max_vals
                    )
                )
                vals = vals.at[self._logged_jax].set(10**vals[self._logged_jax])
            
        else:
            if cube is None:
                cube = self._rng.uniform(size=len(self._df))
            vals = np.zeros(len(cube))
            
            vals[~self._relative_mask] = truncnorm.ppf(
                cube[~self._relative_mask],
                self._tga[~self._relative_mask],
                self._tgb[~self._relative_mask],
                loc=self._mean[~self._relative_mask],
                scale=self._std[~self._relative_mask],
            )
            
            vals[self._relative_mask] = truncnorm.ppf(
                cube[self._relative_mask],
                self._tga[self._relative_mask],
                self._tgb[self._relative_mask],
                loc=self._mean[self._relative_mask] + vals[self.relative_idxs],
                scale=self._std[self._relative_mask]
            )
            
            # log transformations
            vals[self._logged] = 10**vals[self._logged]
        return vals
    
    def jax_guide(self):
        num_params = self._numpyro_sample_arr.shape[1]

        # Initialize loc to the parameter mean from the numpyro sample array
        min_vals, max_vals, init_loc, init_scale = self._numpyro_sample_arr  # Assuming mean is at index 3

        # Adjust initialization for relative parameters
        relative_shifts = jnp.zeros(num_params)
        relative_shifts = relative_shifts.at[self._relative_mask_jax].set(
            init_loc[self._relative_idxs_jax]
        )
        init_loc = init_loc + relative_shifts

        # Parameters for the guide distribution
        loc = numpyro.param("loc", init_loc,
                            constraint=dist.constraints.interval(min_vals, max_vals))

        # For scale, we'll use a fraction of the range as initial value
        scale = numpyro.param("scale", init_scale,
                              constraint=dist.constraints.positive)

        with numpyro.plate("params", num_params):
            # Sample all parameters
            samples = numpyro.sample(
                "samples",
                dist.TruncatedNormal(loc=loc, scale=scale, low=min_vals, high=max_vals)
            )
    
    def transform(self, samples):
        """Transform relative and log-Gaussian samples
        from gaussian-sampled values.
        """
        samples.loc[:,self._params[self._logged]] = 10**samples.loc[:,self._params[self._logged]]
        return samples
            
    
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
        self.score = 0.0
        
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
        if (self._fit_params is not None) and (fit_parameters.columns != self._fit_params.columns):
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
        mapper = umap.UMAP().fit(features[~nan_features], force_all_finite=False)
        
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
        self, X: NDArray[np.object_], y: NDArray[np.float64]
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

        check_array(y, ensure_2d=False, force_all_finite=False)

        # remove nan/inf rows
        mask = np.all(np.isfinite(X[:, 2].astype(np.float64))) & np.isfinite(y)

        return X[mask], y[mask]
    
    @property
    def name(self) -> str:
        """Return sampler name."""
        return self._sampler_name

    def fit(
        self,
        X: NDArray[np.object_],  # pylint: disable=invalid-name
        y: NDArray[np.float64],
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
        self._X, self._y = self._validate_arrs(
            X, y
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
        val_x, _ = self._validate_arrs(X, placeholder_y)

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
        val_x, val_y = self._validate_arrs(X, y)
        y_pred, _ = self.predict(val_x)
        return self._reduced_chi_squared(val_x, val_y, y_pred, **kwargs)

    def _convert_photometry_to_arrs(
        self, photometry: Photometry
    ) -> tuple[NDArray[np.object_], NDArray[np.float64]]:
        """Converts a Photometry object to arrays for fitting."""
        # first rescale the photometry
        photometry.phase()
        # adjust so max flux = 1.0
        photometry.normalize()
        dets = photometry.detections
        dets_mjd = dets.index.total_seconds().to_numpy() / (24 * 3600)
        if self._mag_y:
            x_arr = np.array([dets_mjd, dets["filter"], dets["mag_error"]], dtype=object).T
        else:
            x_arr = np.array([dets_mjd, dets["filter"], dets["flux_error"]], dtype=object).T

        y = dets["mag"].to_numpy() if self._mag_y else dets["flux"].to_numpy()
        return x_arr, y

    def fit_photometry(self, photometry: Photometry, **kwargs) -> None:
        """Fit a Photometry object. Saves information to
        Photometry object.

        Parameters
        ----------
            The Photometry object to fit.
        """
        x_phot, y_phot = self._convert_photometry_to_arrs(photometry)
        self.fit(x_phot, y_phot, **kwargs)

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
        for b in photometry._unique_filters:
            if dense:
                t_arr = np.linspace(np.min(X[:, 0]) - 20.0, np.max(X[:, 0]) + 20.0, 1000)
                new_x = np.repeat(t_arr[np.newaxis, :].T, 3, axis=1).astype(object)
                new_x[:, 1] = b
                new_x[:, 2] = 0.0  # filler
                y_pred, val_x = self.predict(new_x)
            else:
                y_pred, val_x = self.predict(X[X[:, 1] == b])
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
        return float(
            np.median(
                np.sum(
                    (y[np.newaxis, :] - y_pred) ** 2 / self._eff_variance(X) / (len(y) - self._nparams - 1),
                    axis=1,
                )
            )
        )
